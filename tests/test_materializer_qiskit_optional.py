"""Qiskit 0.45.3 optional tests: static plan materialization & exact readout parity."""
from __future__ import annotations

import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qubit_reuse_repair import (
    ReusePlan, StaticPlanMaterializer, ReuseValidityChecker,
    GidNETChainsAdapter, CaQRChainsAdapter, DQCCChainsAdapter,
)


def prepare_pair():
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.cx(2, 3)
    for i in range(4):
        qc.measure(i, i)
    plan = ReusePlan.from_chains([[0, 2], [1, 3]])
    return qc, plan


def test_reuse_compilation_keeps_measurements_and_reset_per_boundary():
    qc, plan = prepare_pair()
    compiler = StaticPlanMaterializer(qc)
    result = compiler.materialize(plan)
    output = result.circuit
    assert result.full_width == 2
    assert output.num_qubits == 2
    assert output.num_clbits == 4
    assert result.logical_to_physical == {0: 0, 2: 0, 1: 1, 3: 1}
    assert result.emitted_reset_count == 2
    assert result.source_measurement_count == result.compiled_measurement_count == 4
    assert [item.operation.name for item in output.data].count('reset') == 2
    assert [item.operation.name for item in output.data].count('measure') == 4
    assert len(result.scheduled_operations) == 8
    assert result.as_metadata()['physical_gain_measured'] is False
    assert output.depth() >= 1  # not routed depth


def test_same_materializer_for_all_three_adapters_and_both_plans():
    qc, plan = prepare_pair()
    compiler = StaticPlanMaterializer(qc)
    outs = []
    for adapter_type in (GidNETChainsAdapter, CaQRChainsAdapter, DQCCChainsAdapter):
        adapter = adapter_type(compile_plan=compiler)
        common = adapter.to_common(qc, [[0, 2], [1, 3]])
        assert common == plan
        outs.append(adapter.materialize(qc, common))
    assert all(out == outs[0] for out in outs)  # same compilation regardless of label
    # Original and repaired are compiled using identical code/settings.
    alt = ReusePlan.from_chains([[0, 3], [1, 2]])
    assert compiler.checker.validate(alt).is_valid
    other = compiler.materialize(alt)
    assert other.full_width == 2
    assert other.emitted_reset_count == 2
    assert other.circuit.num_clbits == outs[0].num_clbits


def test_invalid_plan_and_unsupported_source_are_rejected():
    qc, plan = prepare_pair()
    compiler = StaticPlanMaterializer(qc)
    with pytest.raises(ValueError, match='invalid reuse plan'):
        compiler.materialize(ReusePlan.from_chains([[0, 1], [2, 3]]))
    other = qc.copy()
    with pytest.raises(ValueError, match='source circuit mismatch'):
        compiler(other, plan)
    dynamic = QuantumCircuit(2, 1)
    dynamic.h(0)
    dynamic.measure(0, 0)
    dynamic.x(0)
    with pytest.raises(ValueError, match='unsupported source'):
        StaticPlanMaterializer(dynamic)


def test_measurement_only_inactive_qubit_kept():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.x(1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.measure(2, 2)  # inactive but still measured
    plan = ReusePlan.from_chains([[0, 1]])
    compiled = StaticPlanMaterializer(qc).materialize(plan)
    assert compiled.active_width == 1
    assert compiled.full_width == 2
    assert compiled.logical_to_physical[2] == 1
    assert compiled.compiled_measurement_count == 3


def _exact_classical_probs(qc):
    """Exact statevector branching over measure/reset for tiny circuits only."""
    import numpy as np
    from qiskit.quantum_info import Statevector
    from qiskit.circuit.library import XGate
    branches = [(1.0, Statevector.from_label('0' * qc.num_qubits), (0,) * qc.num_clbits)]
    for item in qc.data:
        name = item.operation.name
        if name == 'barrier':
            continue
        qargs = [qc.find_bit(q).index for q in item.qubits]
        cargs = [qc.find_bit(c).index for c in item.clbits]
        updated = []
        for weight, state, bits in branches:
            if name in ('measure', 'reset'):
                q = qargs[0]
                for outcome in (0, 1):
                    projected = state.data.copy()
                    for index in range(len(projected)):
                        if (index >> q) & 1 != outcome:
                            projected[index] = 0
                    p = float(np.vdot(projected, projected).real)
                    if p < 1e-14:
                        continue
                    collapsed = Statevector(projected / np.sqrt(p))
                    if name == 'reset' and outcome:
                        collapsed = collapsed.evolve(XGate(), qargs=[q])
                    next_bits = list(bits)
                    if name == 'measure':
                        next_bits[cargs[0]] = outcome
                    updated.append((weight * p, collapsed, tuple(next_bits)))
            else:
                updated.append((weight, state.evolve(item.operation, qargs=qargs), bits))
        branches = updated
    out = {}
    for prob, _, bits in branches:
        out[bits] = out.get(bits, 0.0) + prob
    assert sum(out.values()) == pytest.approx(1)
    return out


def test_exact_classical_readout_distribution_on_entangled_pairs():
    qc, plan = prepare_pair()
    compiled = StaticPlanMaterializer(qc).materialize(plan)
    a = _exact_classical_probs(qc)
    b = _exact_classical_probs(compiled.circuit)
    for bits in set(a) | set(b):
        assert a.get(bits, 0.0) == pytest.approx(b.get(bits, 0.0), abs=1e-10)


def test_barrier_dropped_and_original_gate_order_respected_on_chain():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.barrier()
    qc.x(1)
    qc.z(2)
    for logical in range(3):
        qc.measure(logical, logical)
    p = ReusePlan.from_chains([[0, 2], [1]])
    out = StaticPlanMaterializer(qc).materialize(p)
    names = [item.operation.name for item in out.circuit.data]
    assert 'barrier' not in names
    assert names.count('h') == 1
    assert names.count('x') == 1
    assert names.count('z') == 1
    assert names.count('reset') == 1
    assert names.count('measure') == 3


def test_no_clbits_and_materialization_pair_width_guard():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(1)
    qc.z(2)
    compiler = StaticPlanMaterializer(qc)
    orig = ReusePlan.from_chains([[0, 1], [2]])
    alt = ReusePlan.from_chains([[0, 2], [1]])
    pair = compiler.materialize_pair(orig, alt)
    assert pair.full_width == 2
    assert pair.same_plan is False
    assert pair.original.circuit.num_clbits == 0
    assert pair.repaired.circuit.num_clbits == 0
    with pytest.raises(ValueError, match='repair changed active width'):
        compiler.materialize_pair(orig, ReusePlan.from_chains([[0], [1], [2]]))
