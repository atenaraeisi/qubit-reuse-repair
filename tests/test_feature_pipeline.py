"""Generator-agnostic feature and historical-model bridge unit tests."""
from __future__ import annotations
from dataclasses import dataclass
import math
import pytest
from qubit_reuse_repair import (ReusePlan, StructuralFeatureExtractor, STRUCTURAL_COLUMNS,
    LEGACY_RIDGE_COLUMNS, FrozenFeatureNormalizer, LegacyFeatureBridge,
    RidgeDeltaScorer, GidNETChainsAdapter, CaQRChainsAdapter, DQCCChainsAdapter,
    Objective)

@dataclass(frozen=True)
class Op:
    name: str

@dataclass(frozen=True)
class Item:
    operation: Op
    qubits: tuple[int, ...]
    clbits: tuple = ()

class Circuit:
    num_qubits = 4
    def __init__(self):
        self.data = [Item(Op('h'),(0,)),Item(Op('cx'),(0,1)),
                     Item(Op('cx'),(2,3)),Item(Op('cx'),(0,2)),
                     Item(Op('measure'),(2,))]
    def find_bit(self,x):
        return type('BitInfo',(),{'index':x})()


def two_plans():
    return (ReusePlan.from_chains([[0,1],[2,3]]),
            ReusePlan.from_chains([[0,2],[1,3]]))


def test_feature_schema_exact_historical_dimensions():
    assert len(STRUCTURAL_COLUMNS)==25
    assert len(LEGACY_RIDGE_COLUMNS)==25
    assert 'logical_depth' not in STRUCTURAL_COLUMNS
    assert LEGACY_RIDGE_COLUMNS[-2:]==('logical_depth','norm_depth')
    assert STRUCTURAL_COLUMNS[:23]==LEGACY_RIDGE_COLUMNS[:23]


def test_structural_features_known_quotient_counts():
    plan,_=two_plans()
    f=StructuralFeatureExtractor(Circuit()).extract(plan)
    assert set(f)==set(STRUCTURAL_COLUMNS)
    assert f['quotient_unique_edges']==1.0
    assert f['quotient_density']==1.0
    assert f['max_degree']==1.0
    assert f['max_weighted_degree']==1.0
    assert f['max_edge_weight']==1.0
    assert f['extra_internal_2q_frac']==pytest.approx(2/3)
    assert f['extra_cross_2q']==1.0
    assert f['edge_weight_hhi']==1.0
    assert f['edge_weight_concentration']==1.0
    assert f['mean_first_2q_local_delay']==0.0


def test_features_depend_on_chains_not_generator_or_order():
    p,q=two_plans()
    x=StructuralFeatureExtractor(Circuit())
    assert x.extract(p)==x.extract(ReusePlan.from_chains([[2,3],[0,1]]))
    assert x.extract(q)['extra_internal_2q_frac']==pytest.approx(1/3)
    assert x.extract(q)['extra_cross_2q']==2.0
    with pytest.raises(ValueError,match='active logical'):
        x.extract(ReusePlan.from_chains([[0,1],[2]]))


def test_normalizer_freeze_and_zero_std_and_nan():
    pool={'a':{'x':1.0,'y':math.nan,'z':9.0},
          'b':{'x':3.0,'y':math.nan,'z':9.0}}
    norm=FrozenFeatureNormalizer.fit(pool,['x','y','z'])
    assert norm.transform({'x':1,'y':math.nan,'z':9})=={'x':-1.0,'y':0.0,'z':0.0}
    assert norm.transform({'x':5,'y':math.nan,'z':9})['x']==3.0
    assert norm.fingerprint==FrozenFeatureNormalizer.fit(dict(reversed(list(pool.items()))),['x','y','z']).fingerprint
    with pytest.raises(ValueError,match='nonempty'):FrozenFeatureNormalizer.fit({},['x'])
    with pytest.raises(ValueError,match='Missing feature'):norm.transform({'x':1})


def test_frozen_bridge_compiled_depths_explicit_and_cached():
    p,q=two_plans()
    calls=[]
    def compile_depths(plan):
        calls.append(plan.signature)
        return (10.0 if plan==p else 15.0, 8.0 if plan==p else 12.0)
    bridge=LegacyFeatureBridge(StructuralFeatureExtractor(Circuit()),compile_depths,[p,q])
    assert len(bridge.columns)==25
    assert set(bridge(p))==set(LEGACY_RIDGE_COLUMNS)
    assert bridge(p)['logical_depth']==-1.0
    assert bridge(q)['norm_depth']==1.0
    assert len(calls)==2  # compiled each reference plan once, cache reused
    assert bridge.normalizer.reference_signatures==tuple(sorted([p.signature,q.signature]))


def test_bridge_refuses_missing_or_invalid_compilation():
    p,q=two_plans()
    with pytest.raises(TypeError,match='required'):
        LegacyFeatureBridge(StructuralFeatureExtractor(Circuit()),None,[p,q])
    with pytest.raises(ValueError,match='two depths'):
        LegacyFeatureBridge(StructuralFeatureExtractor(Circuit()),lambda p:(5,),[p])
    with pytest.raises(ValueError,match='negative'):
        LegacyFeatureBridge(StructuralFeatureExtractor(Circuit()),lambda p:(5,-1),[p])


def test_bridge_to_ridge_with_exact_feature_names():
    p,q=two_plans()
    bridge=LegacyFeatureBridge(StructuralFeatureExtractor(Circuit()),
                               lambda p:(10,8) if p.chains[0][0]==0 else (12,9),[p,q])
    coeff=[0.0]*25
    coeff[0]=1.0
    ckpt={'cache_version':'ridge_nb_loco_v1','objective':'swap_count',
          'heldout_circuit':'example.qasm','training_circuits':['training.qasm'],
          'columns':list(LEGACY_RIDGE_COLUMNS),'feature_mean':[0]*25,
          'feature_sd':[1]*25,'weights':coeff,'target_mean':0.0}
    predictor=RidgeDeltaScorer(ckpt,bridge,expected_objective='swap_count',heldout_circuit='example.qasm')
    assert math.isfinite(predictor.score(p,q,Objective.SWAP_COUNT)[0])
    with pytest.raises(ValueError,match='different objective'):
        predictor.score(p,q,Objective.ROUTED_DEPTH)
    with pytest.raises(ValueError,match='LOCO'):
        RidgeDeltaScorer(ckpt,bridge,expected_objective='swap_count',heldout_circuit='unknown.qasm')


def test_explicit_chain_adapters_and_mapping_fail_closed():
    p,_=two_plans()
    qc=Circuit()
    for cls in (GidNETChainsAdapter,CaQRChainsAdapter,DQCCChainsAdapter):
        a=cls()
        assert a.to_common(qc,[[2,3],[0,1]])==p
        assert a.to_common(qc,{'chains':[[0,1],[2,3]]})==p
        with pytest.raises(NotImplementedError,match='materializer'):
            a.materialize(qc,p)
        with pytest.raises(ValueError,match='Only an explicit'):
            a.to_common(qc,{'arbitrary':[[0,1],[2,3]]})
    a=DQCCChainsAdapter(native_to_logical={10:0,11:1,12:2,13:3})
    assert a.to_common(qc,[[10,11],[12,13]])==p
    with pytest.raises(ValueError,match='Missing native'):
        a.to_common(qc,[[10,11],[12,14]])
    with pytest.raises(ValueError,match='mismatch'):
        CaQRChainsAdapter().to_common(qc,[[0,1],[2]])


def test_materializer_always_receives_common_plan():
    qc=Circuit();p,_=two_plans()
    seen=[]
    a=CaQRChainsAdapter(compile_plan=lambda qc,plan:(seen.append(plan), 'compiled')[1])
    assert a.materialize(qc,p)=='compiled'
    assert seen==[p]

def test_feature_extractor_rejects_mid_circuit_measurement():
    qc=Circuit()
    qc.data.insert(2,Item(Op('measure'),(0,), (0,)))
    with pytest.raises(ValueError,match='mid-circuit'):
        StructuralFeatureExtractor(qc)
