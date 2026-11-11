import pytest

qiskit = pytest.importorskip('qiskit')
from qubit_reuse_repair import ReusePlan, RepairConfig, SearchLimits, search_plan


def test_real_qiskit_static_search_pre_routing():
    qc = qiskit.QuantumCircuit(4, 4)
    for q in range(4):
        qc.h(q)
        qc.measure(q, q)
    original = ReusePlan.from_chains(((0,1),(2,3)))
    result = search_plan(qc, original,
        RepairConfig(search_policy='structural_beam', model='none', budget=20, max_depth=3),
        limits=SearchLimits(raw_attempt_budget=150, max_attempts_per_operator=30))
    assert result.n_scored > 0
    assert not result.to_dict()['physical_gain_measured']
    assert result.selected.active_width == 2
    assert result.selected.qubits == original.qubits


def test_real_qiskit_model_free_baseline():
    qc = qiskit.QuantumCircuit(4)
    for q in range(4): qc.h(q)
    orig = ReusePlan.from_chains(((0,1),(2,3)))
    result = search_plan(qc, orig,
        RepairConfig(search_policy='random_uniform', model='none', budget=10, max_depth=2),
        limits=SearchLimits(raw_attempt_budget=120, max_attempts_per_operator=30))
    assert result.n_scored > 0
    assert result.changed
