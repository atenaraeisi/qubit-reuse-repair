"""Repeat frozen TRAIN_ONLY Ridge pilot on an explicitly listed set of plan/QASM pairs.

A JSON manifest has entries {generator, qasm, plan}. All paths relative to
repository root. No automatic generator calling or native format guessing.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import traceback

from fast_ridge_pilot import build_parser, run

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--objectives',default='swap_count,routed_depth')
    ap.add_argument('--arm',choices=['one_step'],default='one_step')
    ap.add_argument('--model-set',choices=['materializer','legacy'],default='materializer',
                    help='Select newly trained raw-delta models or historical exploratory models')
    ap.add_argument('--policy',choices=['random_beam','adaptive_beam'],default='random_beam')
    ap.add_argument('--topology',choices=['line','ring','grid','heavy_hex7'],default='heavy_hex7')
    ap.add_argument('--physical-qubits',type=int)
    ap.add_argument('--seed',type=int,default=200)
    ap.add_argument('--control-second-seed',type=int,default=201)
    ap.add_argument('--budget',type=int,default=120)
    ap.add_argument('--max-depth',type=int,default=5)
    ap.add_argument('--reference-attempts',type=int,default=150)
    ap.add_argument('--allow-training-circuit',action='store_true')
    ap.add_argument('--out-dir',type=Path,default=Path('pilot_output'))
    ap.add_argument('--overwrite',action='store_true')
    a=ap.parse_args()
    entries=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    if not isinstance(entries,list) or not entries:
        ap.error('Expected a nonempty JSON list')
    objectives=[v.strip() for v in a.objectives.split(',')]
    if not objectives or len(set(objectives))!=len(objectives) or not set(objectives)<= {'swap_count','routed_depth'}:
        ap.error('Invalid objective list')
    if a.out_dir.exists() and any(a.out_dir.iterdir()) and not a.overwrite:
        ap.error('Output directory is not empty; choose another or --overwrite')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for item in entries:
        if not isinstance(item,dict) or any(k not in item for k in ('generator','qasm','plan')):
            ap.error('Each entry must have generator, qasm and plan fields')
        for obj in objectives:
            case=Path(item['qasm']).stem
            folder=a.out_dir/item['generator']/case/obj
            command=[
                '--qasm',str(ROOT/item['qasm']),'--plan',str(ROOT/item['plan']),
                '--generator',str(item['generator']),
                '--checkpoint',str(ROOT/'pilot_models'/(
                     f'{obj}_{a.arm}_MATERIALIZER.json' if a.model_set=='materializer'
                     else f'{obj}_{a.arm}_TRAIN_ONLY.json')),
                '--objective',obj,'--arm',a.arm,'--policy',a.policy,
                '--topology',a.topology,'--seed',str(a.seed),'--budget',str(a.budget),
                '--max-depth',str(a.max_depth),'--reference-attempts',str(a.reference_attempts),
                '--out-dir',str(folder)]
            if a.physical_qubits is not None:command+=['--physical-qubits',str(a.physical_qubits)]
            if a.control_second_seed is not None: command+=['--control-second-seed',str(a.control_second_seed)]
            if a.allow_training_circuit: command+=['--allow-training-circuit']
            if a.overwrite: command+=['--overwrite']
            row=dict(generator=item['generator'],circuit=case,objective=obj,policy=a.policy,
                     model_set=a.model_set)
            try:
                args=build_parser().parse_args(command)
                data=run(args)
                s=data['selection']; orig=s['original_route']; prop=s['repaired_route']
                row.update(status='ok',proposed_changed=data['search']['changed'],
                    selected_source=s['selected_source'],n_scored=data['search']['n_scored'],
                    baseline_swap=orig['swap_count'],candidate_swap=prop['swap_count'],
                    final_swap=s['selected_route']['swap_count'],
                    baseline_depth=orig['routed_depth'],candidate_depth=prop['routed_depth'],
                    final_depth=s['selected_route']['routed_depth'],
                    guard_gain_pct=s['actual_gain_pct'],guard_gain_abs=s['actual_gain_abs'],
                    best_of_two_gain_abs=(s['original_best_of_two_control'] or {}).get('control_gain_abs'),
                    observed_guard_minus_control_abs=(s['original_best_of_two_control'] or {}).get('guard_minus_control_abs'),
                    observed_proposed_gain_abs=orig[obj]-prop[obj],error='')
            except Exception as e:
                row.update(status='failed',error=type(e).__name__+': '+str(e))
                folder.mkdir(parents=True,exist_ok=True)
                (folder/'failure_trace.txt').write_text(traceback.format_exc(),encoding='utf-8')
                print('[FAILED]',item['generator'],case,obj,':',e,flush=True)
            rows.append(row)
            print('[DONE]',row['generator'],case,obj,row['status'],flush=True)

    all_fields=list(dict.fromkeys(k for r in rows for k in r))
    with (a.out_dir/'pilot_rows.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)
    groups=defaultdict(list)
    for r in rows:
        groups[(r['generator'],r['objective'])].append(r)
    summary=[]
    for (gen,obj),group in sorted(groups.items()):
        ok=[r for r in group if r['status']=='ok']
        defined=[float(r['guard_gain_pct']) for r in ok if r.get('guard_gain_pct') is not None]
        summary.append(dict(generator=gen,objective=obj,n_total=len(group),n_succeeded=len(ok),
            n_failed=len(group)-len(ok),n_percent_defined=len(defined),
            exploratory_mean_guard_gain_pct=sum(defined)/len(defined) if defined else None,
            n_positive_guard=sum(int(r['guard_gain_abs']>0) for r in ok),
            n_harmful_proposed=sum(int(r['observed_proposed_gain_abs']<0) for r in ok),
            n_better_than_best_of_two=sum(int(r['observed_guard_minus_control_abs']>0)
              for r in ok if r.get('observed_guard_minus_control_abs') is not None)))
    (a.out_dir/'pilot_summary.json').write_text(json.dumps(dict(
        schema='pilot_not_final_test_v2',model_set=a.model_set,groups=summary,
        notice='Descriptive pilot only. Separate training-overlap cases and audit exact feature extraction parity before independent generalization claims.'),
        indent=2,ensure_ascii=False),encoding='utf-8')
    print('Summary:',a.out_dir/'pilot_summary.json')
    print('Rows:',a.out_dir/'pilot_rows.csv')
    for x in summary: print(x)
    if any(r['status']=='failed' for r in rows):
        raise SystemExit(1)


if __name__=='__main__':main()
