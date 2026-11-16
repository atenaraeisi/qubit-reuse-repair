"""Frozen MATERIALIZER Ridge inference parity against the checkpoint equation."""
import copy
import json
import math
from pathlib import Path

import pytest

from qubit_reuse_repair.core import Objective, ReusePlan
from qubit_reuse_repair.search.materializer_ridge import MaterializerDeltaRidge
from qubit_reuse_repair.features.structural import LEGACY_RIDGE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def fixture(obj):
    return json.loads((ROOT / 'pilot_models' /
                       f'{obj}_one_step_MATERIALIZER.json').read_text(encoding='utf8'))


def plans():
    return (ReusePlan.from_chains(((0,1),(2,3))),
            ReusePlan.from_chains(((0,3),(2,1))))


@pytest.mark.parametrize('objective',('swap_count','routed_depth'))
def test_materializer_checkpoint_predicts_original_difference(objective):
    ck=fixture(objective)
    old,new=plans()
    before={k:100.0+i for i,k in enumerate(LEGACY_RIDGE_COLUMNS)}
    after={k:v+((i%3)-1)*0.25 for i,(k,v) in enumerate(before.items())}
    # Train-time raw imputation occurs on each row BEFORE subtracting.
    before['mean_raw_weighted_incompat'] = float('nan')
    after['mean_chain_jaccard_incompat'] = None
    rows={old.signature:before,new.signature:after}
    model=MaterializerDeltaRidge(ck,lambda p:rows[p.signature],objective=objective)
    impute=dict(zip(ck['raw_feature_order'],ck['raw_impute']))
    expected_delta=[]
    for k in ck['raw_feature_order']:
        a,b=before[k],after[k]
        a=impute[k] if a is None or (isinstance(a,float) and math.isnan(a)) else a
        b=impute[k] if b is None or (isinstance(b,float) and math.isnan(b)) else b
        expected_delta.append(b-a)
    expected=ck['intercept']+sum(w*(delta-mean)/sd for w,delta,mean,sd in
        zip(ck['weights'],expected_delta,ck['feature_mean'],ck['feature_sd']))
    actual=model.score(old,new,Objective(objective))[0]
    assert math.isclose(actual,expected,abs_tol=1e-10,rel_tol=1e-11)
    assert model.score(old,new,objective)==(actual,)  # cached, deterministic
    with pytest.raises(ValueError, match='other physical objective'):
        model.score(old,new,'routed_depth' if objective=='swap_count' else 'swap_count')


def test_do_not_apply_old_per_circuit_normalizer_to_new_model():
    ck=fixture('swap_count')
    old,new=plans()
    rows={old.signature:{k:0.0 for k in ck['raw_feature_order']},
          new.signature:{k:1.0 for k in ck['raw_feature_order']}}
    model=MaterializerDeltaRidge(ck,lambda p:rows[p.signature],objective='swap_count')
    predicted=model.score(old,new,'swap_count')[0]
    expected=ck['intercept']+sum(w*(1.0-mu)/sd for w,mu,sd in
                                  zip(ck['weights'],ck['feature_mean'],ck['feature_sd']))
    assert math.isclose(predicted,expected,rel_tol=1e-12)


@pytest.mark.parametrize('field,value',[
    ('protocol','fixed_train_family_cv_25delta_ridge_v1'),
    ('feature_protocol','wrong'),('objective','routed_depth'),
    ('raw_feature_order', ['other']*25), ('weights',[0.0]*24),
    ('feature_sd',[0.0]*25), ('raw_impute',[float('inf')]*25),
    ('train_only',False),('arm','multi_step'),
])
def test_fail_closed_on_bad_checkpoint(field,value):
    ck=copy.deepcopy(fixture('swap_count'))
    ck[field]=value
    with pytest.raises((ValueError,KeyError)):
        MaterializerDeltaRidge(ck,lambda _: {},objective='swap_count')


def test_reject_missing_and_infinite_raw_features():
    ck=fixture('swap_count')
    old,new=plans()
    model=MaterializerDeltaRidge(ck,lambda _: {},objective='swap_count')
    with pytest.raises(ValueError,match='missing'):
        model.score(old,new,'swap_count')
    rows={k:0.0 for k in ck['raw_feature_order']}
    rows['max_degree']=float('inf')
    model=MaterializerDeltaRidge(ck,lambda _: rows,objective='swap_count')
    with pytest.raises(ValueError,match='infinite'):
        model.score(old,new,'swap_count')
