"""Circuit-specific allowed-boundary matrix and bitset DAG transitive closure.

Based on `PlanSpace` in experiments/reuse_audit_fixed/pilot_code/plan_space.py.
The matrix is necessary but NOT sufficient: multiple legal boundaries may
create a global dependency cycle, which is checked incrementally.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx

from .checker import ReuseValidityChecker


class CompatibilityIndex:
    """Immutable per-circuit state shared by every prepared parent plan."""

    def __init__(self, checker: ReuseValidityChecker):
        if checker.unsupported is not None:
            raise ValueError(f"Unsupported circuit: {checker.unsupported.reason}")
        self.checker = checker
        self.active = tuple(sorted(checker.active_qubits))
        self.pos = {q: i for i, q in enumerate(self.active)}
        n = len(self.active)
        event_nodes = []
        for q in self.active:
            event_nodes.extend((checker.input_nodes[q], checker.output_nodes[q]))
        if len(set(event_nodes)) != 2 * n:
            raise ValueError("Every active input/output event must be distinct")
        event_pos = {node: k for k, node in enumerate(event_nodes)}
        reach = []
        for event in event_nodes:
            mask = 0
            for descendant in nx.descendants(checker.original_graph, event):
                j = event_pos.get(descendant)
                if j is not None:
                    mask |= 1 << j
            reach.append(mask)
        self.reach0 = tuple(reach)
        self.allowed = tuple(tuple(checker.is_boundary_legal(a, b) for b in self.active)
                             for a in self.active)
        self.allowed_mask = tuple(sum((1 << j) for j, allowed in enumerate(row) if allowed)
                                  for row in self.allowed)

    def legal_boundary(self, a: int, b: int) -> bool:
        return a in self.pos and b in self.pos and self.allowed[self.pos[a]][self.pos[b]]

    @staticmethod
    def _add_edge(reach: tuple[int, ...], a: int, b: int) -> tuple[int, ...] | None:
        """a and b are active positions; add finish(a)->start(b) to closure."""
        u, v = 2 * a + 1, 2 * b
        if u == v or ((reach[v] >> u) & 1):
            return None
        descendants = reach[v] | (1 << v)
        incoming = 1 << u
        nxt = list(reach)
        for x in range(len(nxt)):
            if x == u or (nxt[x] & incoming):
                nxt[x] |= descendants
        return tuple(nxt)

    def closure_of(self, edges: Iterable[tuple[int, int]]) -> tuple[int, ...] | None:
        state = self.reach0
        for a, b in sorted(edges):
            if a not in self.pos or b not in self.pos:
                raise ValueError("Edge references inactive/unknown logical qubit")
            state = self._add_edge(state, self.pos[a], self.pos[b])
            if state is None:
                return None
        return state

    def preserves_acyclicity(self, edges: Iterable[tuple[int, int]]) -> bool:
        return self.closure_of(edges) is not None

    def candidate_allowed(self, base_edges: frozenset[tuple[int, int]],
                          removed: frozenset[tuple[int, int]],
                          added: frozenset[tuple[int, int]],
                          closure_without: tuple[int, ...] | None = None) -> str:
        """Return valid, illegal_boundary, or dependency_cycle (no materialization)."""
        if not removed.issubset(base_edges):
            raise AssertionError("Mutation removes a nonexistent reuse edge")
        resulting = (base_edges - removed) | added
        if len(resulting) != len(base_edges):
            raise AssertionError("Mutation changed number of reuse edges / active width")
        if not all(self.legal_boundary(a, b) for a, b in added):
            return "illegal_boundary"
        state = closure_without if closure_without is not None else self.closure_of(base_edges - removed)
        if state is None:
            raise AssertionError("Removing edges from valid plan must not introduce cycle")
        for a, b in sorted(added):
            state = self._add_edge(state, self.pos[a], self.pos[b])
            if state is None:
                return "dependency_cycle"
        return "valid"
