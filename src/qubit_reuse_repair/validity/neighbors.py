"""Validity-aware neighbor engine for use *inside* future beam search.

Generator independent.  Full independent checker is MANDATORY for each
fast-surviving candidate; unlike the old pilot, not debug-only.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..core import ReusePlan
from .checker import ReuseValidityChecker
from .closure import CompatibilityIndex
from .mutations import Mutation, OPERATORS, iter_mutation_deltas


@dataclass(frozen=True)
class ValidatedNeighbor:
    plan: ReusePlan
    mutation: Mutation


@dataclass(frozen=True)
class NeighborStats:
    attempted: int = 0
    same_plan: int = 0
    duplicate: int = 0
    fast_reject_illegal_boundary: int = 0
    fast_reject_dependency_cycle: int = 0
    full_reject: int = 0
    accepted: int = 0

    @classmethod
    def from_counter(cls, c: Counter) -> "NeighborStats":
        return cls(**{name: int(c[name]) for name in cls.__dataclass_fields__})


class PreparedPlan:
    def __init__(self, engine: "ValidityAwareEngine", parent: ReusePlan):
        self.engine = engine
        self.parent = parent
        self.width = parent.active_width
        engine.checker.spec.check_plan_integrity(parent, target_active_width=self.width)
        verdict = engine.checker.validate(parent, target_width=self.width)
        if not verdict.is_valid:
            raise ValueError(f"Initial/parent plan is not statically valid: {verdict.reason}")
        self.base_edges = frozenset(parent.reuse_edges)
        if engine.index.closure_of(self.base_edges) is None:
            raise AssertionError("Checker and incremental closure disagree on parent validity")
        self._closure_without: dict[frozenset, tuple[int, ...]] = {}
        self._seen: set[frozenset[tuple[int, int]]] = set()

    def _fast_status(self, mutation: Mutation) -> tuple[str, frozenset]:
        if not mutation.removed.issubset(self.base_edges):
            raise AssertionError("Mutation delta removes edge missing from parent")
        resulting = (self.base_edges - mutation.removed) | mutation.added
        if len(resulting) != len(self.base_edges):
            raise AssertionError("Mutation changed reuse edge count; width not preserved")
        if not all(self.engine.index.legal_boundary(a, b) for a, b in mutation.added):
            return "illegal_boundary", resulting
        if mutation.removed not in self._closure_without:
            closure = self.engine.index.closure_of(self.base_edges - mutation.removed)
            if closure is None:
                raise AssertionError("Deleting edges produced a dependency cycle")
            self._closure_without[mutation.removed] = closure
        status = self.engine.index.candidate_allowed(
            self.base_edges, mutation.removed, mutation.added,
            closure_without=self._closure_without[mutation.removed])
        return status, resulting

    def scan(self, operator: str, *, max_attempts: int = 0) -> tuple[list[ValidatedNeighbor], NeighborStats]:
        """Scan one operator. max_attempts=0 means exhaustive; cap counts RAW attempts.

        _seen persists across different operator scans on the same PreparedPlan.
        A search engine should create a new PreparedPlan when changing parent.
        """
        if operator not in OPERATORS:
            raise ValueError(f"Unsupported operator: {operator}")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0:
            raise ValueError("max_attempts must be nonnegative int (0=unlimited)")
        stats = Counter()
        survivors: list[ValidatedNeighbor] = []
        for index, mut in enumerate(iter_mutation_deltas(self.parent, operator)):
            if max_attempts and index >= max_attempts:
                break
            stats["attempted"] += 1
            resulting = (self.base_edges - mut.removed) | mut.added
            if len(resulting) != len(self.base_edges):
                raise AssertionError("Invalid width-preserving delta")
            if resulting == self.base_edges:
                stats["same_plan"] += 1
                continue
            if resulting in self._seen:
                stats["duplicate"] += 1
                continue
            self._seen.add(resulting)
            status, _ = self._fast_status(mut)
            if status != "valid":
                stats["fast_reject_" + status] += 1
                continue
            candidate = mut.materialize(self.parent)
            if candidate.active_width != self.width or frozenset(candidate.reuse_edges) != resulting:
                raise AssertionError("Mutation materialization disagrees with edge delta or width")
            # Strict: no candidate can ever be passed to Search without this check.
            verdict = self.engine.checker.validate(candidate, target_width=self.width)
            if not verdict.is_valid:
                stats["full_reject"] += 1
                raise AssertionError(f"Fast/static independent checker mismatch: {verdict.reason}")
            survivors.append(ValidatedNeighbor(candidate, mut))
            stats["accepted"] += 1
        return survivors, NeighborStats.from_counter(stats)

    def scan_all(self, *, operators: tuple[str, ...] = OPERATORS,
                 max_attempts_per_operator: int = 0) -> tuple[list[ValidatedNeighbor], NeighborStats]:
        """Deduplicate between operators and aggregate all reasons."""
        total = Counter()
        neighbors: list[ValidatedNeighbor] = []
        if len(set(operators)) != len(operators):
            raise ValueError("Duplicate operators in scan_all")
        for op in operators:
            new, count = self.scan(op, max_attempts=max_attempts_per_operator)
            neighbors.extend(new)
            total.update(count.__dict__)
        return neighbors, NeighborStats.from_counter(total)


class ValidityAwareEngine:
    def __init__(self, checker: ReuseValidityChecker):
        self.checker = checker
        self.index = CompatibilityIndex(checker)  # exactly once per circuit

    @classmethod
    def from_circuit(cls, circuit) -> "ValidityAwareEngine":
        return cls(ReuseValidityChecker(circuit))

    def prepare(self, plan: ReusePlan) -> PreparedPlan:
        return PreparedPlan(self, plan)

    def neighbors(self, plan: ReusePlan, *, operators: tuple[str, ...] = OPERATORS,
                  max_attempts_per_operator: int = 0) -> tuple[list[ValidatedNeighbor], NeighborStats]:
        return self.prepare(plan).scan_all(
            operators=operators, max_attempts_per_operator=max_attempts_per_operator)
