"""Run: PYTHONPATH=src python examples/normalize_plan.py"""
from qubit_reuse_repair import ReusePlan, CircuitSpec, RepairConfig

original = ReusePlan.from_chains([(2, 4), (0, 3), (1,)])
spec = CircuitSpec(n_declared_qubits=6, active_qubits=frozenset({0, 1, 2, 3, 4}))
spec.check_plan_integrity(original, target_active_width=3)

print("canonical chains:", original.chains)
print("active width:", original.active_width)
print("full width (including inactive declared q5):", spec.full_width(original))
print("reuse edges:", original.reuse_edges)
print("signature:", original.signature[:16])
print("config:", RepairConfig())
