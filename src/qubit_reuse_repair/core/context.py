"""Circuit identity and width bookkeeping, without importing any generator."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from .plan import ReusePlan


@dataclass(frozen=True)
class CircuitSpec:
    """Logical-qubit identities in the *original* (unreused) circuit.

    active_qubits are the IDs included in a reuse plan. Inactive declared
    qubits are deliberately NOT silently discarded; they add to full width.
    The eventual Qiskit adapter will derive this spec from the circuit.
    """

    n_declared_qubits: int
    active_qubits: frozenset[int]

    def __post_init__(self) -> None:
        n = self.n_declared_qubits
        if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
            raise ValueError("n_declared_qubits must be a positive integer")
        active = frozenset(self.active_qubits)
        if not active or any(isinstance(q, bool) or not isinstance(q, Integral)
                             or q < 0 or q >= n for q in active):
            raise ValueError("active_qubits must be nonempty IDs in [0, n_declared_qubits)")
        object.__setattr__(self, "active_qubits", frozenset(int(q) for q in active))
        object.__setattr__(self, "n_declared_qubits", int(n))

    @property
    def n_inactive_qubits(self) -> int:
        return self.n_declared_qubits - len(self.active_qubits)

    def check_plan_integrity(self, plan: ReusePlan, *, target_active_width: int | None = None) -> None:
        """Raise if malformed/incorrect membership/width; NOT a DAG validity test."""
        if plan.qubits != self.active_qubits:
            missing = sorted(self.active_qubits - plan.qubits)
            extra = sorted(plan.qubits - self.active_qubits)
            raise ValueError(f"active logical-qubit mismatch: missing={missing}, extra={extra}")
        if target_active_width is not None and plan.active_width != target_active_width:
            raise ValueError(f"active width changed: {target_active_width} -> {plan.active_width}")

    def full_width(self, plan: ReusePlan) -> int:
        self.check_plan_integrity(plan)
        return plan.active_width + self.n_inactive_qubits
