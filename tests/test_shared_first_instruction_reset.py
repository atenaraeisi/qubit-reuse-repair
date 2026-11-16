"""Regression test for multiple reused lifetimes starting at one gate.

Place this file in the project's tests/ directory. Requires Qiskit 0.45.
"""
import pytest
pytest.importorskip('qiskit')
from qiskit import QuantumCircuit
from qubit_reuse_repair import ReusePlan, StaticPlanMaterializer


def test_two_resets_before_shared_successor_cx():
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.x(1)
    qc.cx(2, 3)
    for q in range(4):
        qc.measure(q, q)
    p = ReusePlan.from_chains([[0, 2], [1, 3]])
    m = StaticPlanMaterializer(qc)
    assert m.checker.validate(p, target_width=2).is_valid
    out = m.materialize(p)
    assert out.full_width == 2
    assert out.emitted_reset_count == 2
    assert set(out.reset_before_logical) == {2, 3}
    assert out.compiled_measurement_count == 4
    names = [item.operation.name for item in out.circuit.data]
    cx = names.index('cx')
    assert names[cx - 2:cx] == ['reset', 'reset']
