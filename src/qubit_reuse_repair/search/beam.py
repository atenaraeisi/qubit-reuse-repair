"""Generator-agnostic pre-routing fixed-width multi-step repair search.

Historical inspiration: adaptive_repair_cached.py. This implementation differs:
all proposed parents/neighbors must pass the Phase-2 independent checker;
mutations are screened using the allowed matrix and bitset cycle closure.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
from typing import Any

from ..core import Objective, RepairConfig, ReusePlan, SearchPolicy
from ..validity import ValidityAwareEngine
from ..validity.mutations import OPERATORS
from .scorers import PlanScorer, RidgeDeltaScorer, StructuralScorer, RandomUniformScorer


@dataclass(frozen=True)
class SearchLimits:
    per_parent_keep: int = 40
    max_attempts_per_operator: int = 180
    raw_attempt_budget: int = 2400
    operators: tuple[str, ...] = OPERATORS

    def __post_init__(self):
        for name in ("per_parent_keep", "max_attempts_per_operator", "raw_attempt_budget"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.operators or len(set(self.operators)) != len(self.operators) or not set(self.operators) <= set(OPERATORS):
            raise ValueError("Invalid/duplicate repair operators")


@dataclass(frozen=True)
class SearchStep:
    signature: str
    parent_signature: str | None
    depth: int
    operator: str
    score: tuple[float, ...]

    def to_dict(self) -> dict:
        return {"signature": self.signature, "parent_signature": self.parent_signature,
                "depth": self.depth, "operator": self.operator, "score": list(self.score)}


@dataclass(frozen=True)
class SearchResult:
    original: ReusePlan
    selected: ReusePlan
    objective: Objective
    policy: SearchPolicy
    selected_score: tuple[float, ...]
    reached_depth: int
    n_scored: int
    n_valid_accepted: int
    raw_attempts: int
    stop_reason: str
    rejection_counts: dict[str, int]
    trace: tuple[SearchStep, ...]

    @property
    def changed(self) -> bool:
        return self.selected != self.original

    def to_dict(self) -> dict:
        return {
            "stage": "pre_routing_search_only", "objective": self.objective.value,
            "policy": self.policy.value, "original_plan": self.original.to_dict(),
            "selected_plan": self.selected.to_dict(), "selected_score": list(self.selected_score),
            "changed": self.changed, "reached_depth": self.reached_depth,
            "n_scored": self.n_scored, "n_valid_accepted": self.n_valid_accepted,
            "raw_attempts": self.raw_attempts, "stop_reason": self.stop_reason,
            "rejection_counts": self.rejection_counts,
            "trace": [step.to_dict() for step in self.trace],
            "physical_gain_measured": False,
        }


def _score_tuple(value: Any, *, size: int | None = None) -> tuple[float, ...]:
    if isinstance(value, (float, int)):
        value = (float(value),)
    score = tuple(float(z) for z in value)
    if not score or (size is not None and len(score) != size) or not all(math.isfinite(z) for z in score):
        raise ValueError("Scorer must return a nonempty, finite and fixed-length score tuple")
    return score


class SearchEngine:
    """Search over verified plans, never route and never use test routing labels."""

    def __init__(self, validator: ValidityAwareEngine, config: RepairConfig,
                 scorer: PlanScorer | None, *, limits: SearchLimits | None = None):
        self.validator = validator
        self.config = config
        self.limits = limits if limits is not None else SearchLimits()
        if config.search_policy is SearchPolicy.RANDOM_UNIFORM:
            if scorer is not None and not isinstance(scorer, RandomUniformScorer):
                raise TypeError("random_uniform is model-free and uses RandomUniformScorer")
            scorer = scorer or RandomUniformScorer(config.search_seed)
        self.scorer = scorer
        if config.search_policy is SearchPolicy.STRUCTURAL_BEAM:
            if not isinstance(scorer, StructuralScorer):
                raise TypeError("structural_beam requires StructuralScorer")
        elif config.search_policy is not SearchPolicy.RANDOM_UNIFORM:
            # No fake Ridge: historical random/adaptive beam need a real,
            # explicitly supplied pre-routing predictor.
            if scorer is None or not hasattr(scorer, "score"):
                raise ValueError("random_beam/adaptive_beam require an explicit trained predictor")
        if isinstance(scorer, RidgeDeltaScorer) and scorer.objective != config.objective:
            raise ValueError("Ridge scorer objective mismatch")

    def run(self, original: ReusePlan) -> SearchResult:
        cfg, caps = self.config, self.limits
        verdict = self.validator.checker.validate(original, target_width=original.active_width)
        if not verdict.is_valid:
            raise ValueError(f"Invalid original plan: {verdict.reason}")
        rng = random.Random(cfg.search_seed)
        records: dict[str, ReusePlan] = {original.signature: original}
        scores: dict[str, tuple[float, ...]] = {}
        best_sig = original.signature
        best_score: tuple[float, ...] | None = None
        trace = []
        frontier = [original.signature]
        total_stats: Counter = Counter()
        n_scored, reached = 0, 0
        stop = "depth_limit"
        for depth in range(1, cfg.max_depth + 1):
            layer: list[str] = []
            for parent_sig in frontier:
                parent = records[parent_sig]
                prepared = self.validator.prepare(parent)
                ops = list(caps.operators)
                rng.shuffle(ops)
                pool = []
                for op in ops:
                    available = caps.raw_attempt_budget - total_stats["attempted"]
                    if available <= 0:
                        stop = "raw_attempt_budget"
                        break
                    neighbors, st = prepared.scan(op,
                        max_attempts=min(caps.max_attempts_per_operator, available))
                    total_stats.update(st.__dict__)
                    pool.extend(neighbors)
                rng.shuffle(pool)
                # Per-parent keep counts distinct validated candidate plans,
                # not unchecked proposals. All scorer calls occur after full validation.
                accepted_from_parent = 0
                for neighbor in pool:
                    cand = neighbor.plan
                    sig = cand.signature
                    if sig in records:
                        total_stats["global_duplicate"] += 1
                        continue
                    if n_scored >= cfg.budget:
                        stop = "valid_candidate_budget"
                        break
                    if accepted_from_parent >= caps.per_parent_keep:
                        break
                    if self.scorer is None:
                        raise AssertionError("No scorer")
                    sc = _score_tuple(self.scorer.score(original, cand, cfg.objective),
                                      size=len(best_score) if best_score is not None else None)
                    if best_score is None:
                        best_score = tuple(0.0 for _ in sc)
                    records[sig] = cand
                    scores[sig] = sc
                    n_scored += 1
                    accepted_from_parent += 1
                    layer.append(sig)
                    trace.append(SearchStep(sig, parent_sig, depth, neighbor.mutation.operator, sc))
                    if sc > best_score:
                        best_sig, best_score = sig, sc
                if n_scored >= cfg.budget:
                    stop = "valid_candidate_budget"
                    break
                if total_stats["attempted"] >= caps.raw_attempt_budget:
                    stop = "raw_attempt_budget"
                    break
            if not layer:
                if stop == "depth_limit":
                    stop = "no_new_candidates"
                break
            reached = depth
            if cfg.search_policy in (SearchPolicy.ADAPTIVE_BEAM, SearchPolicy.STRUCTURAL_BEAM):
                layer.sort(key=lambda sig: scores[sig], reverse=True)
            else:
                rng.shuffle(layer)
            frontier = layer[:cfg.beam_width]
            if stop != "depth_limit":
                break
        if best_score is None:
            best_score = (0.0,)
        # Crucial: negative/zero predicted gain -> unchanged plan. This is a
        # pre-routing prediction guard, NOT a physical no-harm guarantee.
        return SearchResult(original, records[best_sig], cfg.objective, cfg.search_policy,
                            best_score, reached, n_scored, total_stats["accepted"],
                            total_stats["attempted"], stop,
                            {k: int(v) for k, v in total_stats.items()}, tuple(trace))


def search_plan(source_circuit: Any, original: ReusePlan, config: RepairConfig,
                scorer: PlanScorer | None = None, *, limits: SearchLimits | None = None) -> SearchResult:
    """Convenience API for original static Qiskit circuits, no native adapter yet."""
    validator = ValidityAwareEngine.from_circuit(source_circuit)
    if scorer is None and config.search_policy is SearchPolicy.STRUCTURAL_BEAM:
        scorer = StructuralScorer.from_circuit(source_circuit)
    return SearchEngine(validator, config, scorer, limits=limits).run(original)
