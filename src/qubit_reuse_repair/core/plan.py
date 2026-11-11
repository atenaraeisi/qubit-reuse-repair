"""Canonical, generator-independent representation of a qubit-reuse plan.

Only structural *well-formedness* is checked here. A plan that passes these
checks may still violate quantum dependencies or compiler semantics. That is
checked by a separate circuit-aware validity layer in the next stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ReusePlan:
    """Immutable collection of ordered logical-qubit reuse chains.

    Chain order is semantic (reuse a qubit from first to last); order *between*
    chains is not semantic and is canonicalized. IDs always refer to logical
    qubits in the original source circuit, never physical qubits.

    Only gate-active logical qubits belong to these chains. Declared but
    inactive qubits are handled by CircuitSpec when computing full width.
    """

    chains: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        try:
            raw = tuple(tuple(chain) for chain in self.chains)
        except (TypeError, ValueError) as exc:
            raise ValueError("chains must be an iterable of nonempty iterables") from exc
        if not raw or any(not chain for chain in raw):
            raise ValueError("reuse plan must contain nonempty chains")
        flat = []
        for chain in raw:
            for q in chain:
                if isinstance(q, bool) or not isinstance(q, Integral):
                    raise TypeError("logical qubit IDs must be integers (not bools)")
                if q < 0:
                    raise ValueError("logical qubit IDs must be nonnegative")
                flat.append(int(q))
        if len(flat) != len(set(flat)):
            raise ValueError("a logical qubit must occur exactly once")
        canon = tuple(sorted(tuple(int(q) for q in chain) for chain in raw))
        object.__setattr__(self, "chains", canon)

    @classmethod
    def from_chains(cls, chains: Iterable[Iterable[int]]) -> "ReusePlan":
        return cls(tuple(tuple(chain) for chain in chains))

    @classmethod
    def from_json(cls, raw: str) -> "ReusePlan":
        """Read a JSON array of chains; no eval or native-object assumptions."""
        data = json.loads(raw)
        if not isinstance(data, list) or any(not isinstance(c, list) for c in data):
            raise ValueError("plan JSON must be a list of nonempty lists")
        return cls.from_chains(data)

    def to_json(self) -> str:
        return json.dumps(self.chains, separators=(",", ":"), ensure_ascii=True)

    @property
    def active_width(self) -> int:
        return len(self.chains)

    @property
    def qubits(self) -> frozenset[int]:
        return frozenset(q for chain in self.chains for q in chain)

    @property
    def n_active_qubits(self) -> int:
        return sum(len(chain) for chain in self.chains)

    @property
    def reuse_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple((a, b) for chain in self.chains for a, b in zip(chain, chain[1:]))

    @property
    def signature(self) -> str:
        """Stable signature of the complete canonical plan; not generator-specific."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def same_active_width_as(self, other: "ReusePlan") -> bool:
        return self.active_width == other.active_width

    def to_dict(self) -> dict:
        return {"schema_version": 1, "chains": [list(c) for c in self.chains]}

    @classmethod
    def from_dict(cls, obj: dict) -> "ReusePlan":
        if not isinstance(obj, dict) or obj.get("schema_version") != 1:
            raise ValueError("unsupported ReusePlan schema version")
        return cls.from_chains(obj["chains"])
