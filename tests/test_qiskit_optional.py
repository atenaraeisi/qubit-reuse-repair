"""Real Qiskit checks, skipped when Qiskit is not installed locally."""
import pytest
from qubit_reuse_repair import ReusePlan, ReuseValidityChecker, ValidityAwareEngine

qiskit = pytest.importorskip('qiskit')
from qiskit import QuantumCircuit


def test_static_terminal_measurement_and_inactive_width():
    qc = QuantumCircuit(4, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(2)
    qc.measure(0, 0)
    qc.measure(1, 1)
    # active q0,q1,q2; declared q3 is inactive
    ck = ReuseValidityChecker(qc)
    assert ck.active_qubits == frozenset((0,1,2))
    original = ReusePlan.from_chains([[0],[1],[2]])
    assert ck.validate(original).is_valid
    assert ck.validate(original).full_width == 4
    assert not ck.is_boundary_legal(0,1)  # same two-qubit operation
    assert not ck.validate([[0,1],[2]]).is_valid
    engine = ValidityAwareEngine(ck)
    assert engine.index.allowed == tuple(tuple(ck.is_boundary_legal(a,b) for b in sorted(ck.active_qubits)) for a in sorted(ck.active_qubits))


def test_dynamic_inputs_rejected_explicitly():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.measure_all()
    qc.reset(0)
    ck = ReuseValidityChecker(qc)
    assert not ck.validate([[0],[1]]).is_valid
    assert ck.unsupported is not None
