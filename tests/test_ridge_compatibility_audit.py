"""Auditor regression checks. Purely observational; never enables legacy inference."""
import importlib.util
from pathlib import Path
import json
import pandas as pd
from qubit_reuse_repair.features import LEGACY_RIDGE_COLUMNS

SCRIPT=Path(__file__).parents[1]/'scripts'/'audit_ridge_compatibility.py'
sp=importlib.util.spec_from_file_location('audit_ridge_compatibility',str(SCRIPT))
mod=importlib.util.module_from_spec(sp)
sp.loader.exec_module(mod)


def test_original_json_checkpoint_schema(tmp_path):
    p=tmp_path/'ckpt.json'
    d={'cache_version':'ridge_nb_loco_v1','objective':'swap_count','heldout_circuit':'held.qasm',
       'training_circuits':['other.qasm'],'columns':list(LEGACY_RIDGE_COLUMNS),
       'weights':[0.0]*25,'feature_mean':[0.0]*25,'feature_sd':[1.0]*25,'target_mean':0.0}
    p.write_text(json.dumps(d),encoding='utf-8')
    assert mod.audit_checkpoint(p)['valid_schema']
    d['columns'][0]='renamed_column'
    p.write_text(json.dumps(d),encoding='utf-8')
    assert not mod.audit_checkpoint(p)['valid_schema']


def test_normalization_drift_on_nearly_constant_reference(tmp_path):
    f=tmp_path/'oldfeatures.csv'
    a=[]
    for i,v in enumerate([0.4444444444444443,0.4444444444444446,
                          0.4444444444444444,0.4444444444444445]):
        row={c:float(i+1) for c in LEGACY_RIDGE_COLUMNS[:23]}
        row['mean_chain_jaccard_incompat']=v
        row.update(circuit='test.qasm',sig=str(i))
        a.append(row)
    pd.DataFrame(a).to_csv(f,index=False)
    report,_=mod.audit_reference(f)
    assert report['n_plans']==4
    assert any(x['feature']=='mean_chain_jaccard_incompat' for x in report['tiny_std_features'])
    assert report['normalizer_max_abs_z_difference'] >= 0.0


def test_training_only_multistep_format_is_distinct(tmp_path):
    p=tmp_path/'swap_count_multi_step_TRAIN_ONLY.json'
    d={'protocol':'fixed_train_family_cv_25delta_ridge_v1',
       'objective':'swap_count','arm':'multi_step','raw_feature_order':list(LEGACY_RIDGE_COLUMNS),
       'feature_names':['d_'+x for x in LEGACY_RIDGE_COLUMNS],
       'weights':[0.0]*25,'feature_mean':[0.0]*25,'feature_sd':[1.0]*25,
       'intercept':0.0,'scope':'TRAIN ONLY; not validated', 'training_circuits':['A.qasm']}
    p.write_text(json.dumps(d),encoding='utf-8')
    assert mod.audit_multistep_checkpoint(p)['valid_schema']
    assert not mod.audit_checkpoint(p)['valid_schema']
