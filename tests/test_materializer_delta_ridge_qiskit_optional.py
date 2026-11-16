import pytest
pytest.importorskip('qiskit')
from qiskit import QuantumCircuit
from qubit_reuse_repair import ReusePlan, StaticPlanMaterializer
from qubit_reuse_repair.features.materializer_raw import MaterializerRawFeatureProvider
from qubit_reuse_repair.search.materializer_ridge import MaterializerDeltaRidge
from pathlib import Path


def test_end_to_end_feature_extract_and_score_without_routing():
    qc=QuantumCircuit(4,4)
    qc.h(0)
    qc.x(1)
    qc.cx(2,3)
    for i in range(4): qc.measure(i,i)
    plan_a=ReusePlan.from_chains(((0,2),(1,3)))
    plan_b=ReusePlan.from_chains(((0,3),(1,2)))
    materializer=StaticPlanMaterializer(qc)
    for plan in (plan_a,plan_b):
        assert materializer.checker.validate(plan,target_width=2).is_valid
    provider=MaterializerRawFeatureProvider(qc,materializer=materializer)
    assert len(provider(plan_a))==25
    assert provider(plan_a)['logical_depth']>=0
    assert provider(plan_b)['norm_depth']>=0
    root=Path(__file__).resolve().parents[1]
    for obj in ('swap_count','routed_depth'):
        model=MaterializerDeltaRidge.from_file(
            root/'pilot_models'/f'{obj}_one_step_MATERIALIZER.json',
            provider,objective=obj)
        assert len(model.score(plan_a,plan_b,obj))==1
