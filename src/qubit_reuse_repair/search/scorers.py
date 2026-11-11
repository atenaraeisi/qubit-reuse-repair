"""Pre-routing ranking interfaces. Larger score is better; ORIGINAL has score (0, ...).

These estimators NEVER receive routed costs. Scores are estimates/proxies,
not physically verified gains. Physical comparison is a later phase.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Protocol

from ..core import Objective, ReusePlan


class PlanScorer(Protocol):
    """Score a VALIDATED candidate RELATIVE TO THE ORIGINAL of this search."""
    def score(self, original: ReusePlan, candidate: ReusePlan, objective: Objective) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class StructuralScorer:
    """Topology-agnostic lexicographic quotient-interaction proxy.

    This is the Nov 2026 structural control, NOT a predictor calibrated for SWAP/depth.
    It does not optimize hardware distance; any target objective is merely used
    to label an experiment. The same feature order is used in either setting.
    """
    two_qubit_pairs: tuple[tuple[int, int], ...]

    @classmethod
    def from_circuit(cls, source_circuit: Any) -> "StructuralScorer":
        pairs = []
        for inst in source_circuit.data:
            if len(inst.qubits) == 2 and inst.operation.name not in ("barrier", "measure"):
                a, b = (source_circuit.find_bit(q).index for q in inst.qubits)
                pairs.append((a, b))
        return cls(tuple(pairs))

    def _stats(self, plan: ReusePlan) -> tuple[float, float, float]:
        chain_of = {q: i for i, chain in enumerate(plan.chains) for q in chain}
        edge = Counter()
        degree = [set() for _ in plan.chains]
        for a, b in self.two_qubit_pairs:
            if a not in chain_of or b not in chain_of:
                raise ValueError("StructuralScorer interactions include a logical qubit not in the plan")
            x, y = chain_of[a], chain_of[b]
            if x != y:
                edge[tuple(sorted((x, y)))] += 1
        for a, b in edge:
            degree[a].add(b)
            degree[b].add(a)
        total = sum(edge.values())
        hhi = sum(w * w for w in edge.values()) / (total * total) if total else 0.0
        return (float(len(edge)), float(hhi), float(max(map(len, degree), default=0)))

    def score(self, original: ReusePlan, candidate: ReusePlan, objective: Objective) -> tuple[float, ...]:
        a, b = self._stats(original), self._stats(candidate)
        # Less unique quotient edges; higher concentration; less max degree.
        return (a[0] - b[0], round(b[1] - a[1], 12), a[2] - b[2])


@dataclass(frozen=True)
class CallableScorer:
    """Injected pre-routing predictor, useful for audits and non-Ridge models.

    The callback returns *estimated gain relative to original*. It must not
    access held-out routed costs; that is the experimenter's responsibility.
    """
    predictor: Any

    def score(self, original: ReusePlan, candidate: ReusePlan, objective: Objective) -> tuple[float, ...]:
        value = float(self.predictor(original, candidate, objective))
        if not isfinite(value):
            raise ValueError("Nonfinite predicted gain")
        return (value,)


class RidgeDeltaScorer:
    """Inference for saved oldstyle Ridge JSON coefficients; NOT a trained model.

    feature_provider(plan) MUST provide the EXACT old per-circuit standardized
    features (including compilation-derived logical_depth and norm_depth)
    in the model's named columns. Passing raw features silently changes the
    meaning of the learned coefficients and is NOT supported.

    To reproduce historical 15-circuit LOCO, use its frozen model for the heldout
    circuit and its matching normalization. For new CaQR/DQCC, retrain and
    validate a feature pipeline first. Routing labels are never read here.
    """
    def __init__(self, checkpoint: Mapping[str, Any], feature_provider: Any,
                 *, expected_objective: Objective | str, heldout_circuit: str | None = None):
        self.objective = Objective(expected_objective)
        if checkpoint.get("objective") != self.objective.value:
            raise ValueError("Checkpoint objective does not match the requested objective")
        if checkpoint.get("cache_version") != "ridge_nb_loco_v1":
            raise ValueError("Unsupported Ridge checkpoint/version")
        if heldout_circuit is not None:
            if checkpoint.get("heldout_circuit") != heldout_circuit:
                raise ValueError("LOCO checkpoint not fitted for this heldout circuit")
            if heldout_circuit in checkpoint.get("training_circuits", []):
                raise ValueError("Test circuit was included in training")
        self.columns = tuple(checkpoint["columns"])
        self.mu = tuple(float(z) for z in checkpoint["feature_mean"])
        self.sd = tuple(float(z) for z in checkpoint["feature_sd"])
        self.weights = tuple(float(z) for z in checkpoint["weights"])
        self.intercept = float(checkpoint["target_mean"])
        if not self.columns or len(self.columns) != len(set(self.columns)):
            raise ValueError("Missing/duplicate feature names")
        if any(len(z) != len(self.columns) for z in (self.mu, self.sd, self.weights)):
            raise ValueError("Checkpoint feature shape mismatch")
        if not all(isfinite(v) for seq in (self.mu, self.sd, self.weights) for v in seq) or any(v <= 0 for v in self.sd) or not isfinite(self.intercept):
            raise ValueError("Invalid Ridge coefficients/scalers")
        if not callable(feature_provider):
            raise TypeError("feature_provider must be a callable supplying standardized feature mappings")
        self.feature_provider = feature_provider
        self._cache: dict[str, tuple[float, ...]] = {}

    @classmethod
    def from_file(cls, path, feature_provider, *, expected_objective, heldout_circuit=None):
        import json
        from pathlib import Path
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), feature_provider,
                   expected_objective=expected_objective, heldout_circuit=heldout_circuit)

    def _features(self, plan: ReusePlan) -> tuple[float, ...]:
        if plan.signature not in self._cache:
            raw = self.feature_provider(plan)
            if not isinstance(raw, Mapping) or any(k not in raw for k in self.columns):
                raise ValueError("Feature provider must return every checkpoint column")
            vals = tuple(float(raw[k]) for k in self.columns)
            if not all(isfinite(v) for v in vals):
                raise ValueError("Nonfinite/missing standardized features")
            self._cache[plan.signature] = vals
        return self._cache[plan.signature]

    def score(self, original: ReusePlan, candidate: ReusePlan, objective: Objective) -> tuple[float, ...]:
        if Objective(objective) != self.objective:
            raise ValueError("Ridge checkpoint cannot be used for a different objective")
        old, new = self._features(original), self._features(candidate)
        delta = (b-a for a,b in zip(old,new))
        gain = self.intercept + sum(w * ((v-mu)/sd) for v,mu,sd,w in zip(delta,self.mu,self.sd,self.weights))
        if not isfinite(gain):
            raise ValueError("Nonfinite Ridge prediction")
        return (gain,)


@dataclass(frozen=True)
class RandomUniformScorer:
    """Seeded, pseudorandom candidate priority; independent of all physical features.

Every visited candidate has an iid-like deterministic 64-bit random priority,
so the highest priority among visited candidates is a uniform random pick.
This is a model-free negative control, NOT a physical gain estimate.
"""
    seed: int = 2026

    def score(self, original: ReusePlan, candidate: ReusePlan, objective: Objective) -> tuple[float, ...]:
        data = f"{self.seed}|{objective.value}|{candidate.signature}".encode("ascii")
        number = int.from_bytes(hashlib.sha256(data).digest()[:8], "big")
        return ((number + 1) / (2**64 + 1),)
