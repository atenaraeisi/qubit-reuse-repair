"""Generator-neutral materialization of an ordered, statically valid reuse plan.

Scope: Qiskit 0.45.3, unitary Gate operations, barriers, terminal measures.
The shared compiler does *not* use any generator's internal compiler, and
therefore is not claimed to reproduce its exact gate schedule or routing costs.
It preserves logical dependencies and classical measurement destinations,
assigns each reuse chain to one circuit qubit and inserts a reset before the
first operation of each successor lifetime. No physical routing is performed.

For each source circuit, compile Original and Repaired with the SAME compiler
instance/settings. Static validity is NOT a general semantic equivalence proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from ..core.plan import ReusePlan
from ..validity.checker import ReuseValidityChecker


@dataclass(frozen=True)
class MaterializationResult:
    circuit: Any
    original_plan: ReusePlan
    logical_to_physical: dict[int, int]
    scheduled_operations: tuple[int, ...]  # source-circuit instruction indices
    reset_before_logical: tuple[int, ...]
    active_width: int
    full_width: int
    source_measurement_count: int
    compiled_measurement_count: int
    emitted_reset_count: int

    def as_metadata(self) -> dict:
        return {
            "schema_version": 1,
            "static_compilation_only": True,
            "physical_gain_measured": False,
            "active_width": self.active_width,
            "full_width": self.full_width,
            "logical_to_physical": {str(k): v for k, v in self.logical_to_physical.items()},
            "scheduled_operations": list(self.scheduled_operations),
            "reset_before_logical": list(self.reset_before_logical),
            "source_measurement_count": self.source_measurement_count,
            "compiled_measurement_count": self.compiled_measurement_count,
            "emitted_reset_count": self.emitted_reset_count,
        }


@dataclass(frozen=True)
class MaterializationPair:
    """Same-source, same-width pair for a *future* physical routing guard."""
    original: MaterializationResult
    repaired: MaterializationResult

    @property
    def same_plan(self) -> bool:
        return self.original.original_plan == self.repaired.original_plan

    @property
    def full_width(self) -> int:
        return self.original.full_width


class StaticPlanMaterializer:
    """Same-path fixed-width static-to-dynamic compiler, no Generator imports.

    Supply an existing checker to avoid repeating graph preprocessing when
    materializing many valid search candidates of the same source circuit.

    Important scope limitation: when an original logical qubit is never
    measured, its final *quantum output state* is NOT retained separately after
    reuse; this compiler only promises preservation of operations/measurements
    under the supported static-lifetime model. Semantic testing is separate.
    """

    def __init__(self, source_circuit: Any, *, checker: ReuseValidityChecker | None = None):
        self.source = source_circuit
        self.checker = checker if checker is not None else ReuseValidityChecker(source_circuit)
        if self.checker.source_circuit is not source_circuit:
            raise ValueError("checker must be constructed for exactly this source circuit")
        if self.checker.unsupported is not None:
            raise ValueError(f"unsupported source: {self.checker.unsupported.reason}")

    def __call__(self, source_circuit: Any, plan: ReusePlan) -> Any:
        """Callback interface for ExplicitChainsAdapter.compile_plan."""
        if source_circuit is not self.source:
            raise ValueError("Materializer source circuit mismatch: use a common source for both plans")
        return self.materialize(plan).circuit

    def materialize_pair(self, original: ReusePlan, repaired: ReusePlan) -> MaterializationPair:
        """Compile both using exactly one checker and one compiler configuration."""
        if original.active_width != repaired.active_width:
            raise ValueError("repair changed active width")
        before = self.materialize(original)
        after = self.materialize(repaired)
        if (before.full_width != after.full_width or
                before.circuit.num_clbits != after.circuit.num_clbits):
            raise AssertionError("original/repaired register dimensions differ")
        return MaterializationPair(before, after)

    def materialize(self, plan: ReusePlan) -> MaterializationResult:
        from qiskit import QuantumCircuit  # optional until called

        if not isinstance(plan, ReusePlan):
            raise TypeError("plan must be a ReusePlan with original logical IDs")
        verdict = self.checker.validate(plan, target_width=plan.active_width)
        if not verdict.is_valid:
            raise ValueError(f"invalid reuse plan: {verdict.reason}")

        # Preserve immutable chain-relative order; add *all* inactive declared
        # qubits as singleton physical chains so full width is accounted for.
        inactive = sorted(set(range(self.source.num_qubits)) - plan.qubits)
        full_chains = plan.chains + tuple((q,) for q in inactive)
        logical_to_physical = {
            q: physical for physical, chain in enumerate(full_chains) for q in chain
        }
        if len(logical_to_physical) != self.source.num_qubits:
            raise AssertionError("internal error: physical assignment misses logical qubits")
        full_width = len(full_chains)
        if full_width != verdict.full_width:
            raise AssertionError("internal error: checker/full-width disagreement")

        source_ops = []
        original_indices = []
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        last_q: dict[int, int] = {}
        last_c: dict[int, int] = {}
        graph = nx.DiGraph()
        source_measurements = 0

        for position, item in enumerate(self.source.data):
            if item.operation.name == "barrier":
                continue  # same semantics as the supported static checker
            qids = tuple(self.source.find_bit(q).index for q in item.qubits)
            cids = tuple(self.source.find_bit(c).index for c in item.clbits)
            source_ops.append((position, item, qids, cids))
            original_indices.append(position)
            graph.add_node(position)
            if item.operation.name == "measure":
                source_measurements += 1
            for q in qids:
                if q in last_q:
                    graph.add_edge(last_q[q], position)
                last_q[q] = position
                first.setdefault(q, position)
                last[q] = position
            for c in cids:
                if c in last_c:
                    graph.add_edge(last_c[c], position)
                last_c[c] = position

        # Valid chains contain only gate-active qubits; their first/last
        # instruction must exist. Adding these edges is the operational form
        # of output(a) -> input(b) from the independent checker.
        reset_at: dict[int, int] = {}
        for a, b in plan.reuse_edges:
            if a not in last or b not in first:
                raise AssertionError("active boundary missing a source instruction")
            graph.add_edge(last[a], first[b])
            if first[b] in reset_at:
                raise AssertionError("multiple reset boundaries before one instruction")
            reset_at[first[b]] = b
        if not nx.is_directed_acyclic_graph(graph):
            raise AssertionError("materialization order contradicts independent DAG validity")

        # Lexicographic Kahn order preserves the original sequence as much as
        # possible while satisfying quantum, classical and reuse dependencies.
        schedule = tuple(nx.lexicographical_topological_sort(graph, key=lambda position: position))
        by_position = {position: (item, qids, cids) for position, item, qids, cids in source_ops}

        compiled = (QuantumCircuit(full_width, self.source.num_clbits)
                    if self.source.num_clbits else QuantumCircuit(full_width))
        compiled.global_phase = self.source.global_phase
        compiled.metadata = dict(self.source.metadata or {})
        count_resets = 0
        count_measures = 0
        for position in schedule:
            item, qids, cids = by_position[position]
            if position in reset_at:
                logical = reset_at[position]
                compiled.reset(compiled.qubits[logical_to_physical[logical]])
                count_resets += 1
            qargs = [compiled.qubits[logical_to_physical[q]] for q in qids]
            cargs = [compiled.clbits[c] for c in cids]
            compiled.append(item.operation.copy(), qargs, cargs)
            if item.operation.name == "measure":
                count_measures += 1

        if count_measures != source_measurements:
            raise AssertionError("source measurement was lost")
        if count_resets != len(plan.reuse_edges):
            raise AssertionError("missing or extra boundary resets")
        if compiled.num_qubits != full_width or compiled.num_clbits != self.source.num_clbits:
            raise AssertionError("compiled circuit register dimensions do not match")
        return MaterializationResult(
            circuit=compiled, original_plan=plan,
            logical_to_physical=logical_to_physical,
            scheduled_operations=schedule,
            reset_before_logical=tuple(reset_at[p] for p in schedule if p in reset_at),
            active_width=plan.active_width,
            full_width=full_width,
            source_measurement_count=source_measurements,
            compiled_measurement_count=count_measures,
            emitted_reset_count=count_resets,
        )
