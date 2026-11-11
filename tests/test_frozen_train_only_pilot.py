import json
import math
from pathlib import Path
import pytest
from qubit_reuse_repair.core import ReusePlan, Objective
from qubit_reuse_repair.search.train_only import TrainOnlyDeltaRidge
from qubit_reuse_repair.features.pilot_bridge import PandasFrozenNormalizer
from qubit_reuse_repair.features.structural import LEGACY_RIDGE_COLUMNS

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('objective',["swap_count","routed_depth"])
def test_models_equal_real_schema_and_deltas(objective):
    ck=json.loads((ROOT/'pilot_models'/f'{objective}_one_step_TRAIN_ONLY.json').read_text())
    o=ReusePlan.from_chains([(0,1),(2,3)])
    c=ReusePlan.from_chains([(0,3),(2,1)])
    # fixed standardized vectors controlled independently of physical labels
    def features(p):
        return {k:(0.0 if p==o else 0.1) for k in LEGACY_RIDGE_COLUMNS}
    model=TrainOnlyDeltaRidge(ck,features,objective=objective)
    expected=ck['intercept']+sum(w * ((0.1-m)/sd)
                for w,m,sd in zip(ck['weights'],ck['feature_mean'],ck['feature_sd']))
    assert math.isclose(model.score(o,c,Objective(objective))[0],expected,rel_tol=1e-13)
    with pytest.raises(ValueError,match='other physical objective'):
        model.score(o,c,Objective('routed_depth' if objective=='swap_count' else 'swap_count'))


def test_corrupted_model_cannot_silently_run():
    ck=json.loads((ROOT/'pilot_models'/'swap_count_one_step_TRAIN_ONLY.json').read_text())
    o=ReusePlan.from_chains([(0,1),(2,3)])
    f=lambda _: {k:0 for k in LEGACY_RIDGE_COLUMNS}
    altered=dict(ck,raw_feature_order=list(reversed(ck['raw_feature_order'])))
    with pytest.raises(ValueError,match='feature names'):
        TrainOnlyDeltaRidge(altered,f,objective='swap_count')
    altered=dict(ck,objective='routed_depth')
    with pytest.raises(ValueError,match='objective'):
        TrainOnlyDeltaRidge(altered,f,objective='swap_count')


def test_frozen_pool_fit_and_schema():
    pytest.importorskip('pandas')
    rows={
        'x':{k:0.0 for k in LEGACY_RIDGE_COLUMNS},
        'y':{k:2.0 for k in LEGACY_RIDGE_COLUMNS}}
    n=PandasFrozenNormalizer.fit(rows)
    assert n.fingerprint==PandasFrozenNormalizer.fit(dict(reversed(list(rows.items())))).fingerprint
    assert set(n.transform(rows['x']))==set(LEGACY_RIDGE_COLUMNS)
    assert all(abs(z+1.0)<1e-12 for z in n.transform(rows['x']).values())
    with pytest.raises(ValueError,match='At least 2'):
        PandasFrozenNormalizer.fit({'x':rows['x']})
