#!/usr/bin/env python3
"""Read-only historical Ridge compatibility audit (never trains or routes).

Expected original source files from the `experiments` GitHub branch:
  exp4d_features_all.csv (23 historical structural + 2 unused extras)
  exp4c1_all/features.csv (logical_depth, norm_depth, circuit, sig)
  adaptive_ridge_models/*.json (30 LOCO checkpoint files)

Run from repository root with existing [legacy_parity] dependencies installed.
No mutable model state, labels, or routing data is read here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd
import numpy as np

from qubit_reuse_repair.features import LEGACY_RIDGE_COLUMNS, FrozenFeatureNormalizer

COLUMNS=tuple(LEGACY_RIDGE_COLUMNS)
STRUCT23=COLUMNS[:23]
OBJECTIVES={'swap_count','routed_depth'}


def audit_checkpoint(path:Path)->dict:
    try:
        c=json.loads(path.read_text(encoding='utf-8-sig'))
        errors=[]
        if c.get('cache_version') != 'ridge_nb_loco_v1':errors.append('wrong cache_version')
        if c.get('objective') not in OBJECTIVES:errors.append('wrong objective')
        if tuple(c.get('columns',[]))!=COLUMNS:errors.append('wrong ordered feature schema')
        n=len(COLUMNS)
        for key in ('weights','feature_mean','feature_sd'):
            a=c.get(key,[])
            if len(a)!=n or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in a):
                errors.append('invalid '+key)
        if len(c.get('feature_sd',[]))==n and any(v<=0 for v in c['feature_sd']):
            errors.append('nonpositive feature_sd')
        if not isinstance(c.get('target_mean'),(float,int)) or not math.isfinite(c['target_mean']):
            errors.append('invalid target_mean')
        if c.get('heldout_circuit') in c.get('training_circuits',[]):
            errors.append('train/heldout overlap')
        if not c.get('heldout_circuit') or not c.get('training_circuits'):
            errors.append('missing LOCO metadata')
        return {'name':path.name,'valid_schema':not errors,'errors':errors,
                'objective':c.get('objective'),'heldout_circuit':c.get('heldout_circuit'),
                'n_training_circuits':len(c.get('training_circuits',[])),
                'n_training_pairs':c.get('n_training_pairs')}
    except Exception as e:
        return {'name':path.name,'valid_schema':False,'errors':[f'{type(e).__name__}: {e}']}


def audit_multistep_checkpoint(path: Path) -> dict:
    """Separate format: 2026 TRAIN-ONLY one-/multi-step models (not legacy LOCO)."""
    try:
        c=json.loads(path.read_text(encoding='utf-8-sig'))
        errors=[]
        if c.get('protocol')!='fixed_train_family_cv_25delta_ridge_v1':
            errors.append('unexpected training protocol')
        if c.get('objective') not in OBJECTIVES:errors.append('invalid objective')
        if c.get('arm') not in ('one_step','multi_step'):errors.append('invalid arm')
        if tuple(c.get('raw_feature_order',[]))!=COLUMNS:
            errors.append('raw feature order mismatch')
        if tuple(c.get('feature_names',[]))!=tuple('d_'+x for x in COLUMNS):
            errors.append('delta feature order mismatch')
        for key in ('weights','feature_mean','feature_sd'):
            a=c.get(key,[])
            if len(a)!=25 or any(not isinstance(z,(int,float)) or not math.isfinite(z) for z in a):
                errors.append('invalid '+key)
        if len(c.get('feature_sd',[]))==25 and any(z<=0 for z in c['feature_sd']):
            errors.append('nonpositive feature_sd')
        if not isinstance(c.get('intercept'),(int,float)) or not math.isfinite(c['intercept']):
            errors.append('invalid intercept')
        if 'TRAIN ONLY' not in c.get('scope',''):
            errors.append('unexpected checkpoint scope')
        return {'name':path.name,'valid_schema':not errors,'errors':errors,
                'objective':c.get('objective'),'arm':c.get('arm'),
                'n_training_rows':c.get('n_training_rows'),
                'n_training_circuits':len(c.get('training_circuits',[]))}
    except Exception as e:
        return {'name':path.name,'valid_schema':False,'errors':[f'{type(e).__name__}: {e}']}


def _validate_joined_features(struct:pd.DataFrame,compiled:pd.DataFrame):
    if not {'circuit','sig','logical_depth','norm_depth'}<=set(compiled.columns):
        raise ValueError('Compiled CSV must have circuit, sig, logical_depth, norm_depth')
    if compiled.duplicated(['circuit','sig']).any():
        raise ValueError('Compiled CSV has duplicate circuit/sig')
    m=struct.merge(compiled[['circuit','sig','logical_depth','norm_depth']],
                   on=['circuit','sig'],how='left',validate='one_to_one',indicator=True)
    if not (m['_merge']=='both').all():
        raise ValueError('Original compiled depth features missing for some structural rows')
    if m[['logical_depth','norm_depth']].isna().any().any():
        raise ValueError('Original compiled depth fields contain null values')
    return m.drop(columns=['_merge'])


def audit_reference(feature_csv:Path,compiled_csv:Path|None=None,tol:float=1e-9):
    df=pd.read_csv(feature_csv,encoding='utf-8-sig',low_memory=False)
    if not {'circuit','sig',*STRUCT23}<=set(df.columns):
        raise ValueError('Structural feature CSV missing expected columns')
    if df.duplicated(['circuit','sig']).any():
        raise ValueError('Structural feature CSV has duplicate circuit/sig')
    result={'n_circuits':int(df.circuit.nunique()),'n_plans':len(df),
            'structural_feature_columns':len(STRUCT23),'compiled_features_supplied':compiled_csv is not None,
            'normalizer_drift':{},'tiny_std_features':[]}
    max_diff=0.0;diff_count=0
    for circuit,g in df.groupby('circuit',sort=True):
        # This mirrors historical `full.groupby('circuit')` and pandas .mean/.std(ddof=0).
        historical=(g[list(STRUCT23)]-g[list(STRUCT23)].mean()) / g[list(STRUCT23)].std(ddof=0).replace(0,np.nan)
        historical=historical.fillna(0)
        rows={str(r['sig']):r[list(STRUCT23)].to_dict() for _,r in g.iterrows()}
        new=FrozenFeatureNormalizer.fit(rows,STRUCT23)
        current=pd.DataFrame([new.transform(rows[str(s)]) for s in g.sig],index=g.index)
        discrepancies=(historical-current).abs().to_numpy(dtype=float)
        local=float(np.nanmax(discrepancies)) if discrepancies.size else 0.0
        count=int(np.sum(discrepancies>tol))
        max_diff=max(max_diff,local);diff_count+=count
        if count:
            result['normalizer_drift'][str(circuit)]={'max_abs_z_difference':local,'n_z_differences_over_tol':count}
        for name,sd in g[list(STRUCT23)].std(ddof=0).items():
            if 0<sd<1e-12:
                result['tiny_std_features'].append({'circuit':str(circuit),'feature':name,'historical_sd':float(sd)})
    result['normalizer_max_abs_z_difference']=max_diff
    result['normalizer_n_z_differences_over_tol']=diff_count
    result['reference_input_fingerprint_sha256']=hashlib.sha256(feature_csv.read_bytes()).hexdigest()

    stats=None
    if compiled_csv is not None:
        merged=_validate_joined_features(df,pd.read_csv(compiled_csv,encoding='utf-8-sig'))
        stats={'schema':'legacy_circuit_reference_scalers_v1','warning':
               'Historical reference only: NOT a proof of dynamic compiler parity or out-of-distribution validity.',
               'columns':COLUMNS,'structural_source_sha256':result['reference_input_fingerprint_sha256'],
               'compiled_source_sha256':hashlib.sha256(compiled_csv.read_bytes()).hexdigest(),
               'circuits':{}}
        for circ,g in merged.groupby('circuit',sort=True):
            values=g[list(COLUMNS)].astype(float)
            mu=values.mean(axis=0)
            sd=values.std(axis=0,ddof=0).replace(0,np.nan)
            stats['circuits'][str(circ)]={
                'n_plans':len(g),'mean':[None if pd.isna(mu[c]) else float(mu[c]) for c in COLUMNS],
                'sd':[None if pd.isna(sd[c]) else float(sd[c]) for c in COLUMNS],
            }
        result['compiled_depth_feature_coverage']={'n_plans':len(merged),'n_circuits':len(stats['circuits'])}
    return result,stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,help='Directory with historical adaptive_ridge_models/*.json')
    p.add_argument('--multistep-checkpoint-dir',type=Path,help='Directory with ml_ridge_multistep25_v1/*TRAIN_ONLY.json')
    p.add_argument('--legacy-structural-csv',type=Path,help='Historical exp4d_features_all.csv (1267 plans)')
    p.add_argument('--legacy-compiled-csv',type=Path,help='Historical exp4c1_all/features.csv')
    p.add_argument('--out',type=Path,default=Path('ridge_compatibility_report.json'))
    p.add_argument('--emit-reference-stats',type=Path,help='Write exact pandas original catalog normalization stats')
    p.add_argument('--tol',type=float,default=1e-9)
    a=p.parse_args()
    if not a.checkpoint_dir and not a.multistep_checkpoint_dir and not a.legacy_structural_csv:
        p.error('Give at least --checkpoint-dir, --multistep-checkpoint-dir or --legacy-structural-csv')
    if a.legacy_compiled_csv and not a.legacy_structural_csv:
        p.error('--legacy-compiled-csv requires --legacy-structural-csv')
    if a.emit_reference_stats and not a.legacy_compiled_csv:
        p.error('--emit-reference-stats requires both historical CSVs')
    out={'schema':'research_ridge_compatibility_audit_v1',
         'read_only':True,'physical_gain_measured':False,
         'model_reuse_approved':False,
         'warnings':['A matching checkpoint schema does NOT establish feature or compiler parity.',
                     'Old checkpoints are 15 leave-one-circuit-out models, not a general model for CaQR/DQCC.',
                     'Compiled-depth parity is NOT checked by this audit: compare old and new materializers separately.']}
    if a.checkpoint_dir:
        models=sorted(a.checkpoint_dir.glob('*.json'))
        if not models:raise ValueError('Checkpoint directory has no JSON models')
        checks=[audit_checkpoint(f) for f in models]
        out['checkpoints']={'n_files':len(checks),'n_pass':sum(x['valid_schema'] for x in checks),
                            'n_fail':sum(not x['valid_schema'] for x in checks),'models':checks}
    if a.multistep_checkpoint_dir:
        ms=sorted(a.multistep_checkpoint_dir.glob('*_TRAIN_ONLY.json'))
        if not ms:raise ValueError('No *_TRAIN_ONLY.json in --multistep-checkpoint-dir')
        checks=[audit_multistep_checkpoint(f) for f in ms]
        out['multistep_checkpoints']={'n_files':len(checks),'n_pass':sum(x['valid_schema'] for x in checks),
                                      'n_fail':sum(not x['valid_schema'] for x in checks),'models':checks}
    if a.legacy_structural_csv:
        ref,stats=audit_reference(a.legacy_structural_csv,a.legacy_compiled_csv,a.tol)
        out['historical_reference']=ref
        if a.emit_reference_stats:
            a.emit_reference_stats.parent.mkdir(parents=True,exist_ok=True)
            a.emit_reference_stats.write_text(json.dumps(stats,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
            out['historical_reference']['exported_reference_stats']=str(a.emit_reference_stats)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Checkpoint schema:',out.get('checkpoints',{}).get('n_pass','not supplied'), '/',out.get('checkpoints',{}).get('n_files','not supplied'))
    if 'multistep_checkpoints' in out:
        print('New TRAIN-only schemas:',out['multistep_checkpoints']['n_pass'],'/',out['multistep_checkpoints']['n_files'])
    if 'historical_reference' in out:
        ref=out['historical_reference']
        print('Legacy reference:',ref['n_circuits'],'circuits;',ref['n_plans'],'plans')
        print('Normalizer max |z-old - z-current|:',ref['normalizer_max_abs_z_difference'])
        print('Normalizer differences > tol:',ref['normalizer_n_z_differences_over_tol'])
        print('Tiny nonzero SD features:',len(ref['tiny_std_features']))
    print('NOT APPROVED for production; routing/compiled-depth parity and external validation still needed.')
    print('Report:',a.out)

if __name__=='__main__':main()
