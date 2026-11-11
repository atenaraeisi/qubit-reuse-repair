"""Delta-first fixed-width mutation operators, adapted from the audited pilot.

Only the two selected production operators are exposed.  Complete plans are
materialized only after the fast dependency filter accepts edge deltas.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

from ..core import ReusePlan

OPERATORS = ("suffix_swap", "chain_transfer")


@dataclass(frozen=True)
class Mutation:
    operator: str
    source_chain: int
    destination_chain: int
    a: int
    b: int
    removed: frozenset[tuple[int, int]]
    added: frozenset[tuple[int, int]]

    def materialize(self, plan: ReusePlan) -> ReusePlan:
        chunks = list(plan.chains)
        i, j, a, b = self.source_chain, self.destination_chain, self.a, self.b
        if self.operator == "suffix_swap":
            left, right = plan.chains[i], plan.chains[j]
            chunks[i] = left[:a] + right[b:]
            chunks[j] = right[:b] + left[a:]
        elif self.operator == "chain_transfer":
            src, dst = plan.chains[i], plan.chains[j]
            q = src[a]
            chunks[i] = src[:a] + src[a + 1:]
            chunks[j] = dst[:b] + (q,) + dst[b:]
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
        return ReusePlan.from_chains(chunks)


def iter_mutation_deltas(plan: ReusePlan, operator: str) -> Iterator[Mutation]:
    """Deterministic raw mutation enumeration, legacy pilot operator parity."""
    chunks = plan.chains
    if operator == "suffix_swap":
        for i, j in combinations(range(len(chunks)), 2):
            l, r = chunks[i], chunks[j]
            for a in range(1, len(l) + 1):
                for b in range(1, len(r) + 1):
                    old, new = [], []
                    if a < len(l):
                        old.append((l[a-1], l[a]))
                        new.append((r[b-1], l[a]))
                    if b < len(r):
                        old.append((r[b-1], r[b]))
                        new.append((l[a-1], r[b]))
                    yield Mutation(operator, i, j, a, b, frozenset(old), frozenset(new))
    elif operator == "chain_transfer":
        for i, src in enumerate(chunks):
            if len(src) <= 1:
                continue
            for a, q in enumerate(src):
                for j, dst in enumerate(chunks):
                    if i == j:
                        continue
                    for b in range(len(dst)+1):
                        old, new = [], []
                        if a:
                            old.append((src[a-1], q))
                        if a+1 < len(src):
                            old.append((q, src[a+1]))
                        if a and a+1 < len(src):
                            new.append((src[a-1], src[a+1]))
                        if b and b < len(dst):
                            old.append((dst[b-1], dst[b]))
                        if b:
                            new.append((dst[b-1], q))
                        if b < len(dst):
                            new.append((q, dst[b]))
                        yield Mutation(operator, i, j, a, b, frozenset(old), frozenset(new))
    else:
        raise ValueError(f"Unsupported operator {operator!r}, expected one of {OPERATORS}")
