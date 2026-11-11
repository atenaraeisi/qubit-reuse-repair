import pytest
from qubit_reuse_repair import (ReusePlan, ReuseValidityChecker,
    StructuralFeatureExtractor, CaQRChainsAdapter)

def test_qiskit_real_circuit_pipeline_pre_routing():
    qiskit=pytest.importorskip('qiskit')
    qc=qiskit.QuantumCircuit(4,4)
    qc.h(0);qc.h(1);qc.h(2);qc.h(3)
    qc.cx(0,1);qc.cx(2,3)
    for i in range(4):qc.measure(i,i)
    p=ReusePlan.from_chains([[0],[1],[2],[3]])
    assert ReuseValidityChecker(qc).validate(p,target_width=4).is_valid
    assert StructuralFeatureExtractor(qc).extract(p)['quotient_unique_edges']==2.0
    assert CaQRChainsAdapter().to_common(qc,[[0],[1],[2],[3]])==p

def test_qiskit_feature_reject_bad_plan_membership():
    qiskit=pytest.importorskip('qiskit')
    qc=qiskit.QuantumCircuit(3)
    qc.h(0);qc.h(1)
    ex=StructuralFeatureExtractor(qc)
    with pytest.raises(ValueError,match='active logical'):
        ex.extract(ReusePlan.from_chains([[0,1],[2]]))
