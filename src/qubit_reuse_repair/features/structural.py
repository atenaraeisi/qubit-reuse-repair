"""Generator-neutral structural features of a VALID reuse plan.

Definitions ported from the research repo's exp4d_features.py (Experiment 4D).
No hidden GidNET imports, no routing, and no physical labels. The 23 names
of structural features match the historical implementation. The two extra
legacy columns (logical_depth, norm_depth) require an explicit materializer.

This implementation does not *prove* exact historical feature parity for every
QASM; use the provided reference-parity audit in a Qiskit environment.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from statistics import mean, pstdev
from typing import Any

from ..core import ReusePlan

STRUCTURAL_COLUMNS = (
    'quotient_unique_edges', 'quotient_density', 'max_degree',
    'max_weighted_degree', 'max_edge_weight',
    'mean_raw_jaccard_incompat', 'max_raw_jaccard_incompat', 'top2_raw_jaccard_incompat',
    'mean_raw_weighted_incompat', 'max_raw_weighted_incompat',
    'mean_chain_jaccard_incompat', 'max_chain_jaccard_incompat',
    'top2_chain_jaccard_incompat', 'mean_chain_weighted_incompat',
    'max_chain_weighted_incompat',
    'mean_first_2q_local_delay', 'mean_first_2q_layer_delay',
    'edge_weight_cv', 'max_edge_share', 'edge_weight_hhi',
    'normalized_edge_entropy', 'edge_weight_concentration', 'weighted_degree_cv',
    'extra_internal_2q_frac', 'extra_cross_2q',
)
LEGACY_RIDGE_COLUMNS = STRUCTURAL_COLUMNS[:-2] + ('logical_depth', 'norm_depth')


def _avg(vals):
    vals = list(vals)
    return float(mean(vals)) if vals else math.nan


def _max(vals):
    vals = list(vals)
    return float(max(vals)) if vals else math.nan


def _top2(vals):
    return _avg(sorted(vals, reverse=True)[:2])


def _jaccard(first, second):
    a, b = set(first), set(second)
    return float(len(a & b) / len(a | b)) if a or b else 1.0


def _weighted_jaccard(first, second):
    a, b = Counter(first), Counter(second)
    union = set(a) | set(b)
    return sum(min(a[k], b[k]) for k in union) / sum(max(a[k], b[k]) for k in union) if union else 1.0


@dataclass(frozen=True)
class InteractionEvent:
    qubits: tuple[int, ...]
    layer: int


class StructuralFeatureExtractor:
    """Prepare original logical-circuit interaction/lifetime data once.

    Only structural features are generated. Logical-to-physical mapping is
    deliberately not inferred; original logical IDs are used throughout.
    """
    def __init__(self, source_circuit: Any):
        self.source_circuit = source_circuit
        from ..validity.checker import active_qubits_from_circuit
        self.active = active_qubits_from_circuit(source_circuit)
        if not self.active:
            raise ValueError('Source circuit contains no gate-active qubits')
        measured = set()
        for inst in source_circuit.data:
            name = inst.operation.name
            if getattr(inst.operation, 'condition', None) is not None or name in (
                    'reset', 'if_else', 'while_loop', 'for_loop'):
                raise ValueError('Unsupported original circuit: conditional/reset/control-flow operation')
            qs = {source_circuit.find_bit(q).index for q in inst.qubits}
            if name == 'measure':
                if len(qs) != 1 or len(inst.clbits) not in (0,1):
                    raise ValueError('Unsupported original measurement')
                measured.update(qs)
            elif name != 'barrier' and (qs & measured or inst.clbits):
                raise ValueError('Unsupported original circuit: mid-circuit measurement/classical operands')
        self.profiles: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.timing: dict[int, tuple[float, float]] = {}
        self.twoq_pairs = []
        first_activity = {}
        first_2q_local = {}
        first_2q_layer = {}
        local_count = defaultdict(int)
        last_layer = defaultdict(lambda: -1)
        for inst in source_circuit.data:
            if inst.operation.name in ('measure', 'barrier'):
                continue
            qs = tuple(source_circuit.find_bit(q).index for q in inst.qubits)
            if not qs:
                continue
            layer = max(last_layer[q] for q in qs) + 1
            for q in qs:
                last_layer[q] = layer
                first_activity.setdefault(q, layer)
            if len(qs) == 2:
                a, b = qs
                self.twoq_pairs.append((a, b))
                for q, partner in ((a, b), (b, a)):
                    if q not in first_2q_local:
                        first_2q_local[q] = local_count[q]
                        first_2q_layer[q] = layer
                    self.profiles[q].append((partner, layer))
            for q in qs:
                local_count[q] += 1
        for q in self.active:
            self.timing[q] = (
                float(first_2q_local[q]) if q in first_2q_local else math.nan,
                float(first_2q_layer[q] - first_activity[q]) if q in first_2q_layer else math.nan,
            )
        self._cache: dict[str, dict[str, float]] = {}

    def extract(self, plan: ReusePlan) -> dict[str, float]:
        if plan.qubits != self.active:
            raise ValueError('Reuse plan active logical IDs do not match source circuit')
        if plan.signature in self._cache:
            return dict(self._cache[plan.signature])
        cmap = {q: i for i, chain in enumerate(plan.chains) for q in chain}
        width = plan.active_width
        counts = Counter()
        n_internal = 0
        for a, b in self.twoq_pairs:
            x, y = cmap[a], cmap[b]
            if x == y:
                n_internal += 1
            else:
                counts[tuple(sorted((x, y)))] += 1
        adj = {i: set() for i in range(width)}
        wdeg = [0.0 for _ in range(width)]
        for (u, v), w in counts.items():
            adj[u].add(v); adj[v].add(u)
            wdeg[u] += w; wdeg[v] += w
        weights = [float(x) for x in counts.values()]
        unique_edges = len(counts)
        out = {
            'quotient_unique_edges': float(unique_edges),
            'quotient_density': float(unique_edges / (width * (width-1)/2)) if width > 1 else 0.0,
            'max_degree': float(max((len(x) for x in adj.values()), default=0)),
            'max_weighted_degree': float(max(wdeg, default=0.0)),
            'max_edge_weight': float(max(weights, default=0)),
        }
        rj, rw, cj, cw, ld, ly = [], [], [], [], [], []
        for chain in plan.chains:
            for a, b in zip(chain, chain[1:]):
                tail = [other for other, _ in self.profiles[a][-5:]]
                head = [other for other, _ in self.profiles[b][:5]]
                ct = [cmap[q] for q in tail]
                ch = [cmap[q] for q in head]
                rj.append(1 - _jaccard(tail, head))
                rw.append(1 - _weighted_jaccard(tail, head))
                cj.append(1 - _jaccard(ct, ch))
                cw.append(1 - _weighted_jaccard(ct, ch))
                x, y = self.timing[b]
                ld.append(x); ly.append(y)
        out.update({
            'mean_raw_jaccard_incompat': _avg(rj),
            'max_raw_jaccard_incompat': _max(rj),
            'top2_raw_jaccard_incompat': _top2(rj),
            'mean_raw_weighted_incompat': _avg(rw),
            'max_raw_weighted_incompat': _max(rw),
            'mean_chain_jaccard_incompat': _avg(cj),
            'max_chain_jaccard_incompat': _max(cj),
            'top2_chain_jaccard_incompat': _top2(cj),
            'mean_chain_weighted_incompat': _avg(cw),
            'max_chain_weighted_incompat': _max(cw),
            'mean_first_2q_local_delay': _avg(x for x in ld if math.isfinite(x)),
            'mean_first_2q_layer_delay': _avg(x for x in ly if math.isfinite(x)),
        })
        if weights:
            p = [w/sum(weights) for w in weights]
            entropy = -sum(x*math.log(x) for x in p) / math.log(len(p)) if len(p) > 1 else 0.0
            out.update({
                'edge_weight_cv': float(pstdev(weights)/mean(weights)),
                'max_edge_share': max(p),
                'edge_weight_hhi': sum(x*x for x in p),
                'normalized_edge_entropy': entropy,
                'edge_weight_concentration': 1-entropy,
                'weighted_degree_cv': float(pstdev(wdeg)/mean(wdeg)) if mean(wdeg) else math.nan,
            })
        else:
            out.update({k: math.nan for k in ('edge_weight_cv', 'max_edge_share',
                'edge_weight_hhi', 'normalized_edge_entropy', 'edge_weight_concentration',
                'weighted_degree_cv')})
        n2 = len(self.twoq_pairs)
        out['extra_internal_2q_frac'] = float(n_internal/n2) if n2 else 0.0
        out['extra_cross_2q'] = float(n2 - n_internal)
        if set(out) != set(STRUCTURAL_COLUMNS):
            raise AssertionError('Structural feature schema drift')
        self._cache[plan.signature] = dict(out)
        return out
