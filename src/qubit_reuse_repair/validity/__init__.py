"""Static fixed-width reuse validation and delta-first neighbor generation."""
from .checker import ReuseValidityChecker, ValidationResult, active_qubits_from_circuit
from .closure import CompatibilityIndex
from .mutations import Mutation, OPERATORS, iter_mutation_deltas
from .neighbors import ValidatedNeighbor, NeighborStats, ValidityAwareEngine, PreparedPlan

__all__ = [
    "ReuseValidityChecker", "ValidationResult", "active_qubits_from_circuit",
    "CompatibilityIndex", "Mutation", "OPERATORS", "iter_mutation_deltas",
    "ValidatedNeighbor", "NeighborStats", "ValidityAwareEngine", "PreparedPlan",
]
