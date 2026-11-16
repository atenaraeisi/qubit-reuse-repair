#!/usr/bin/env python3
"""Frozen 162-circuit CaQR -> valid reuse plan -> Ridge+Routing campaign.

Subcommands: roster, screen, evaluate, summary.
Designed to be copied into the root of the existing qubit-reuse-repair project.
All jobs persist one small JSON checkpoint, so interrupted runs can resume.
No routing or generator simulation is performed by this script: both use the
real user's Qiskit 0.45 environment. The two final-test cases stay reserved.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from campaign.roster_source import roster, MODEL_TRAIN_FILES

FIELDS=['circuit','family_heuristic','previous_split','training_filename_overlap',
        'reserved_final_test','status','error','declared_qubits','active_qubits',
        'active_width','reuse_edges','valid_one_step_neighbors','raw_attempts',
        'width_reduction_pct','caqr_elapsed_seconds','qasm_sha256','plan_sha256',
        'qasm_path','plan_path','native_chain_path','caqr_reused_qasm_path']


def write_json(path:Path, content):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(content,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_config(outdir):
    data={'schema':'caqr_campaign_162_roster_v1','selected':roster(),
          'exclusion_policy':'exclude five test*.qasm and alu-v0_27_example.qasm',
          'final_test_policy':'wim_266 and squar5_261 reserved until final method freeze',
          'note':'Known previous splits are preserved; all other circuits stay unassigned.'}
    raw=json.dumps(data,sort_keys=True,ensure_ascii=False).encode()
    fingerprint=hashlib.sha256(raw).hexdigest()
    target=outdir/'frozen_roster.json'
    if target.exists():
        old=read_json(target)
        if old.get('roster_sha256')!=fingerprint:
            raise RuntimeError('Frozen roster differs. Do NOT overwrite an existing campaign; choose a fresh --out-dir')
    else:
        write_json(target,dict(data,roster_sha256=fingerprint))
    return target


def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def screen_record_path(outdir,filename):
    return outdir/'screen'/'cases'/(Path(filename).stem+'.json')


def screen_records(outdir):
    result=[]
    for entry in roster():
        p=screen_record_path(outdir,entry['circuit'])
        row=dict(entry)
        if p.exists():row.update(read_json(p))
        else:row['status']='final_test_reserved' if entry['reserved_final_test'] else 'not_screened'
        result.append(row)
    return result


def native_chain_to_plan(chain_file,original):
    """CaQR writes only multi-lifetime chains. Complete with active singletons."""
    from qubit_reuse_repair import ReusePlan
    from qubit_reuse_repair.validity.checker import active_qubits_from_circuit
    active=set(active_qubits_from_circuit(original))
    if not active: raise ValueError('No gate-active qubits')
    chains=[]; used=[]
    for lineno,line in enumerate(chain_file.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():continue
        if not re.fullmatch(r'\s*\d+(?:\s*->\s*\d+)*\s*',line):
            raise ValueError(f'Malformed CaQR chain line {lineno}: {line!r}')
        chain=[int(n) for n in re.findall(r'\d+',line)]
        chains.append(chain);used.extend(chain)
    if len(used)!=len(set(used)):
        raise ValueError('CaQR chain has duplicate logical-qubit IDs')
    if not set(used)<=active:
        raise ValueError(f'CaQR chain includes inactive/out-of-bounds IDs: {set(used)-active}')
    chains.extend([[q] for q in sorted(active-set(used))])
    return ReusePlan.from_chains(chains)


def screen_one(entry,caqr_root:Path,outdir:Path,timeout:int,attempts:int):
    from qiskit import QuantumCircuit
    from qubit_reuse_repair import ValidityAwareEngine
    from qubit_reuse_repair.validity.checker import active_qubits_from_circuit
    fn=entry['circuit'];stem=Path(fn).stem
    base=dict(entry,status='failed',error='')
    source=caqr_root/'benchmarks'/fn
    if not source.is_file():
        base.update(status='missing_qasm',error=str(source))
        return base
    base['qasm_sha256']=sha256(source)
    try:
        qc=QuantumCircuit.from_qasm_file(str(source))
        active=active_qubits_from_circuit(qc)
        base.update(declared_qubits=qc.num_qubits,active_qubits=len(active))
        if len(active)==0:
            base.update(status='unsupported_input',error='No active gate qubits')
            return base
        output=caqr_root/'output';output.mkdir(exist_ok=True)
        chain=output/(stem+'_reuse_chain.txt')
        dyn=output/(stem+'_reuse.qasm')
        before=chain.stat().st_mtime_ns if chain.exists() else None
        start=time.perf_counter()
        done=subprocess.run([sys.executable,'main.py','-b','benchmarks/'+fn,'-v','0'],
                 cwd=str(caqr_root),capture_output=True,text=True,encoding='utf-8',errors='replace',
                 timeout=timeout)
        base['caqr_elapsed_seconds']=round(time.perf_counter()-start,3)
        logdir=outdir/'screen'/'caqr_logs';logdir.mkdir(parents=True,exist_ok=True)
        (logdir/(stem+'.txt')).write_text('STDOUT:\n'+done.stdout+'\nSTDERR:\n'+done.stderr,encoding='utf-8')
        if done.returncode!=0:
            base.update(status='caqr_failed',error='Exit '+str(done.returncode)+': '+done.stderr[-1000:])
            return base
        if not chain.is_file() or not dyn.is_file():
            base.update(status='caqr_failed',error='Native CaQR output chain or QASM not written')
            return base
        if before is not None and chain.stat().st_mtime_ns==before:
            base.update(status='caqr_failed',error='Unchanged native chain timestamp (stale output refused)')
            return base
        plan=native_chain_to_plan(chain,qc)
        engine=ValidityAwareEngine.from_circuit(qc)
        validity=engine.checker.validate(plan,target_width=plan.active_width)
        base.update(active_width=plan.active_width,reuse_edges=len(plan.reuse_edges),
                    width_reduction_pct=100.0*(len(active)-plan.active_width)/len(active),
                    plan_sha256=hashlib.sha256(plan.to_json().encode()).hexdigest())
        if not validity.is_valid:
            base.update(status='invalid_plan',error=str(validity.reason))
            return base
        neighbors,stats=engine.neighbors(plan,max_attempts_per_operator=attempts)
        base.update(valid_one_step_neighbors=len(neighbors),raw_attempts=stats.attempted)
        dst=ROOT/'pilot_cases'/'CaQR_162';dst.mkdir(parents=True,exist_ok=True)
        qasm_dst=dst/fn
        plan_dst=dst/(stem+'.plan.json')
        native_dst=dst/(stem+'.native_caqr_chain.txt')
        dyn_dst=dst/(stem+'.native_caqr_output.qasm')
        shutil.copyfile(source,qasm_dst)
        plan_dst.write_text(plan.to_json(),encoding='utf-8')
        shutil.copyfile(chain,native_dst)
        shutil.copyfile(dyn,dyn_dst)
        base.update(qasm_path=str(qasm_dst),plan_path=str(plan_dst),native_chain_path=str(native_dst),
                    caqr_reused_qasm_path=str(dyn_dst))
        if not plan.reuse_edges:base['status']='no_reuse'
        elif not neighbors:base['status']='no_observed_neighbor'
        else:base['status']='ready'
        return base
    except subprocess.TimeoutExpired as e:
        base.update(status='caqr_timeout',error=f'CaQR > {timeout}s')
        return base
    except Exception as e:
        base.update(status='failed',error=f'{type(e).__name__}: {e}')
        return base


def cmd_roster(a):
    cfg=make_config(a.out_dir)
    rows=screen_records(a.out_dir)
    write_csv(a.out_dir/'roster_162.csv',rows)
    c=Counter(x['previous_split'] for x in rows)
    print('Frozen roster:',cfg,'; 162 benchmark files,',dict(c))
    if a.caqr_root:
        missing=[x['circuit'] for x in rows if not (a.caqr_root/'benchmarks'/x['circuit']).is_file()]
        print('Local CaQR QASM missing:',len(missing))
        if missing:print('First missing:',missing[:15])


def cmd_screen(a):
    make_config(a.out_dir)
    if a.force and (a.out_dir/'evaluation_protocol.json').exists():
        raise SystemExit('Cannot force-rescreen cases after evaluation is frozen; use a new --out-dir')
    if not (a.caqr_root/'main.py').is_file():
        raise SystemExit('CaQR main.py missing: '+str(a.caqr_root))
    sys.path.insert(0,str(ROOT/'src'))
    pending=[]
    for entry in roster():
        if entry['reserved_final_test'] and not a.release_final_test:continue
        p=screen_record_path(a.out_dir,entry['circuit'])
        if p.exists() and not a.force:
            old=read_json(p)
            if not a.retry_failed or old.get('status') in ('ready','no_reuse','no_observed_neighbor'):
                continue
        pending.append(entry)
    if a.batch_size:pending=pending[:a.batch_size]
    print(f'CaQR screening: {len(pending)} new circuits (Qiskit executable {sys.executable})',flush=True)
    for i,entry in enumerate(pending,1):
        row=screen_one(entry,a.caqr_root,a.out_dir,a.timeout,a.attempts)
        write_json(screen_record_path(a.out_dir,entry['circuit']),row)
        write_csv(a.out_dir/'screening_162.csv',screen_records(a.out_dir),FIELDS)
        print(f"[{i}/{len(pending)}] {entry['circuit']}: {row['status']}, "
              f"active={row.get('active_qubits','-')}, width={row.get('active_width','-')}, "
              f"neighbors={row.get('valid_one_step_neighbors','-')}",flush=True)
        if row.get('error'):print(' ERROR:',row['error'][:300],flush=True)
    write_csv(a.out_dir/'screening_162.csv',screen_records(a.out_dir),FIELDS)
    print('Screening counts:',dict(Counter(r['status'] for r in screen_records(a.out_dir))),flush=True)


def evaluation_record_path(outdir,name,objective,seed):
    return outdir/'evaluation'/'cases'/Path(name).stem/objective/f'seed_{seed}'/'job.json'


def evaluate_one(entry,screenrow,objective,seed,a):
    jobdir=evaluation_record_path(a.out_dir,entry['circuit'],objective,seed).parent
    out=jobdir/'routed'
    objpath=ROOT/'pilot_models'/(
        f'{objective}_one_step_MATERIALIZER.json' if a.model_set=='materializer'
        else f'{objective}_one_step_TRAIN_ONLY.json')
    result=dict(circuit=entry['circuit'],family_heuristic=entry['family_heuristic'],
                previous_split=entry['previous_split'],training_filename_overlap=entry['training_filename_overlap'],
                model_set=a.model_set,objective=objective,seed=seed,status='failed',error='')
    if not objpath.is_file():
        result.update(error='Missing checkpoint: '+str(objpath))
        return result
    if sha256(screenrow['qasm_path']) != screenrow.get('qasm_sha256'):
        result.update(error='QASM hash changed after screening; must freeze a new campaign')
        return result
    if sha256(screenrow['plan_path']) != screenrow.get('plan_sha256'):
        result.update(error='ReusePlan hash changed after screening; must freeze a new campaign')
        return result
    args=[sys.executable,str(ROOT/'scripts'/'fast_ridge_pilot.py'),
          '--qasm',str(screenrow['qasm_path']),'--plan',str(screenrow['plan_path']),
          '--generator','CaQR','--checkpoint',str(objpath),
          '--objective',objective,'--arm','one_step','--policy','random_beam',
          '--mode','route_guarded','--budget',str(a.budget),
          '--max-depth',str(a.max_depth),'--reference-attempts',str(a.reference_attempts),
          '--topology',a.topology,'--seed',str(seed),
          '--control-second-seed',str(seed+a.control_seed_offset),
          '--out-dir',str(out)]
    # Use the ACTUAL checkpoint training list, which can differ from the old
    # roster metadata; never silently allow a truly held-out circuit.
    model_training=set(read_json(objpath).get('training_circuits',[]))
    if entry['circuit'] in {str(x).split('::')[-1] for x in model_training}:
        args.append('--allow-training-circuit')
        result['training_overlap_with_selected_model']=True
    else:
        result['training_overlap_with_selected_model']=False
    result['model_set']=a.model_set
    if (a.force or a.retry_failed) and out.exists():
        args.append('--overwrite')
    env=dict(os.environ)
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    start=time.perf_counter()
    try:
        done=subprocess.run(args,cwd=str(ROOT),env=env,capture_output=True,
              text=True,encoding='utf-8',errors='replace',timeout=a.routing_timeout)
        result['elapsed_seconds']=round(time.perf_counter()-start,3)
        (jobdir/'console.txt').write_text('STDOUT:\n'+done.stdout+'\nSTDERR:\n'+done.stderr,encoding='utf-8')
        if done.returncode!=0:
            result.update(status='failed',error=f'Exit {done.returncode}: '+done.stderr[-1100:]+done.stdout[-400:])
            return result
        report=out/'routing_report.json'
        if not report.is_file():
            result.update(status='failed',error='Expected routing_report.json missing')
            return result
        data=read_json(report)
        if data.get('objective')!=objective or data.get('source_qasm_filename')!=entry['circuit']:
            result.update(status='failed',error='Routing report does not match scheduled case')
            return result
        sel=data['selection']
        result.update(status='ok',error='',report_path=str(report),
                      selected_source=sel['selected_source'],
                      original_swap=sel['original_route']['swap_count'],
                      proposed_swap=sel['repaired_route']['swap_count'],
                      final_swap=sel['selected_route']['swap_count'],
                      original_depth=sel['original_route']['routed_depth'],
                      proposed_depth=sel['repaired_route']['routed_depth'],
                      final_depth=sel['selected_route']['routed_depth'],
                      gain_pct=sel['actual_gain_pct'],gain_abs=sel['actual_gain_abs'],
                      proposed_gain_abs=(sel['original_route'][objective]-sel['repaired_route'][objective]),
                      guard_minus_best2_abs=(sel.get('original_best_of_two_control') or {}).get('guard_minus_control_abs'),
                      n_scored=data['search']['n_scored'],
                      plan_changed=data['search']['changed'])
        return result
    except subprocess.TimeoutExpired:
        result.update(status='timeout',error=f'Routing > {a.routing_timeout}s')
        return result
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        return result


def parse_seeds(s):
    values=[int(x.strip()) for x in s.split(',') if x.strip()]
    if not values or len(values)!=len(set(values)) or any(x<0 or x>4294960000 for x in values):
        raise ValueError('Seeds must be unique positive uint32 values')
    return values


def cmd_evaluate(a):
    make_config(a.out_dir)
    seeds=parse_seeds(a.seeds)
    if any(s+a.control_seed_offset>2**32-1 or s+a.control_seed_offset==s for s in seeds):
        raise SystemExit('Invalid control seed offset')
    if not (ROOT/'scripts'/'fast_ridge_pilot.py').is_file():
        raise SystemExit('Existing fast_ridge_pilot.py not found in the project root')
    hashfiles=['scripts/fast_ridge_pilot.py',
        'src/qubit_reuse_repair/search/beam.py',
        'src/qubit_reuse_repair/features/structural.py',
        'src/qubit_reuse_repair/compiler/static.py',
        'src/qubit_reuse_repair/features/legacy_reset.py',
        'src/qubit_reuse_repair/routing/backend.py',
        'src/qubit_reuse_repair/routing/selection.py']
    if a.model_set=='materializer':
        hashfiles += ['src/qubit_reuse_repair/features/materializer_raw.py',
                      'src/qubit_reuse_repair/search/materializer_ridge.py',
                      'pilot_models/swap_count_one_step_MATERIALIZER.json',
                      'pilot_models/routed_depth_one_step_MATERIALIZER.json']
    else:
        hashfiles += ['src/qubit_reuse_repair/features/pilot_bridge.py',
                      'src/qubit_reuse_repair/search/train_only.py',
                      'pilot_models/swap_count_one_step_TRAIN_ONLY.json',
                      'pilot_models/routed_depth_one_step_TRAIN_ONLY.json']
    missing=[n for n in hashfiles if not (ROOT/n).is_file()]
    if missing: raise SystemExit('Missing pipeline/checkpoint files: '+str(missing))
    hashes={name:sha256(ROOT/name) for name in hashfiles}
    protocol=dict(file_hashes=hashes,model_set=a.model_set,
                  policy='random_beam',arm='one_step',topology=a.topology,
                  objectives=['swap_count','routed_depth'],seeds=seeds,
                  control_seed_offset=a.control_seed_offset,budget=a.budget,
                  max_depth=a.max_depth,reference_attempts=a.reference_attempts)
    cfg=a.out_dir/'evaluation_protocol.json'
    if cfg.exists() and read_json(cfg)!=protocol:
        raise SystemExit('Experiment protocol differs from frozen run; use a NEW --out-dir')
    write_json(cfg,protocol)
    previous={r['circuit']:r for r in screen_records(a.out_dir)}
    todo=[]
    for entry in roster():
        if entry['reserved_final_test'] and not a.release_final_test:continue
        sc=previous[entry['circuit']]
        if sc['status']!='ready':continue
        for obj in ['swap_count','routed_depth']:
            for seed in seeds:
                p=evaluation_record_path(a.out_dir,entry['circuit'],obj,seed)
                if p.exists() and not a.force:
                    old=read_json(p)
                    if not a.retry_failed or old.get('status')=='ok':continue
                todo.append((entry,sc,obj,seed,p))
    if a.batch_size:todo=todo[:a.batch_size]
    print('Routing jobs scheduled:',len(todo),'(each includes matched guard + Best-of-2 control)',flush=True)
    for i,(entry,sc,obj,seed,p) in enumerate(todo,1):
        p.parent.mkdir(parents=True,exist_ok=True)
        row=evaluate_one(entry,sc,obj,seed,a)
        write_json(p,row)
        print(f"[{i}/{len(todo)}] {entry['circuit']} {obj} seed={seed}: {row['status']} gain={row.get('gain_pct')}",flush=True)
        if row.get('error'):print(' ERROR:',row['error'][:350],flush=True)
    cmd_summary(a)


def summary_rows(outdir,seeds):
    screen={r['circuit']:r for r in screen_records(outdir)}
    out=[]
    for e in roster():
        sc=screen[e['circuit']]
        for objective in ('swap_count','routed_depth'):
            for seed in seeds:
                row=dict(e,objective=objective,seed=seed,screen_status=sc['status'])
                job=evaluation_record_path(outdir,e['circuit'],objective,seed)
                if job.exists():row.update(read_json(job))
                else:row['status']='not_evaluated' if sc['status']=='ready' else sc['status']
                out.append(row)
    return out


def mean(values):
    return sum(values)/len(values) if values else None


def aggregate_circuits(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['circuit'],r['objective'])].append(r)
    out=[]
    for (circ,obj),rs in sorted(groups.items()):
        ok=[r for r in rs if r['status']=='ok']
        gains=[float(r['gain_pct']) for r in ok if r.get('gain_pct') is not None]
        prop=[float(r['proposed_gain_abs']) for r in ok if r.get('proposed_gain_abs') is not None]
        control=[float(r['guard_minus_best2_abs']) for r in ok if r.get('guard_minus_best2_abs') is not None]
        out.append(dict(circuit=circ,objective=obj,previous_split=rs[0]['previous_split'],
              family_heuristic=rs[0]['family_heuristic'],
              training_filename_overlap=rs[0]['training_filename_overlap'],
              screen_status=rs[0]['screen_status'],
              n_seeds_requested=len(rs),n_seeds_succeeded=len(ok),
              n_percent_defined=len(gains),mean_guard_gain_pct=mean(gains),
              mean_proposed_gain_abs=mean(prop),
              mean_guard_minus_best2_abs=mean(control),
              n_seeds_guard_positive=sum(r.get('gain_abs',0)>0 for r in ok),
              n_seeds_proposed_harmful=sum(r.get('proposed_gain_abs',0)<0 for r in ok),
              n_seeds_guard_beats_best2=sum(r.get('guard_minus_best2_abs',0)>0 for r in ok),
              status='routed' if ok else (rs[0]['screen_status'] if rs[0]['screen_status']!='ready'
                       else 'routing_incomplete')))
    return out


def cmd_summary(a):
    make_config(a.out_dir)
    if (a.out_dir/'evaluation_protocol.json').exists():
        seeds=read_json(a.out_dir/'evaluation_protocol.json')['seeds']
    else:seeds=parse_seeds(getattr(a,'seeds','200'))
    by_job=summary_rows(a.out_dir,seeds)
    by_circuit=aggregate_circuits(by_job)
    write_csv(a.out_dir/'per_circuit_seed.csv',by_job)
    write_csv(a.out_dir/'per_circuit_objective.csv',by_circuit)
    groups=[]
    for objective in ('swap_count','routed_depth'):
        for split in ['known_train','known_validation','unassigned','final_test_reserved','ALL_EXPLORATORY']:
            selected=[r for r in by_circuit if r['objective']==objective and
                      (r['previous_split']==split if split!='ALL_EXPLORATORY'
                       else r['previous_split']!='final_test_reserved')]
            complete=[r for r in selected if r['n_seeds_succeeded']==r['n_seeds_requested']]
            defined=[r['mean_guard_gain_pct'] for r in complete if r['mean_guard_gain_pct'] is not None]
            groups.append(dict(objective=objective,split=split,
                  n_roster=len(selected),n_ready=sum(r['screen_status']=='ready' for r in selected),
                  n_fully_evaluated=len(complete),n_gain_percent_defined=len(defined),
                  mean_gain_pct_over_complete_macro=mean(defined),
                  n_any_positive=sum(r['n_seeds_guard_positive']>0 for r in complete),
                  n_any_negative_proposal=sum(r['n_seeds_proposed_harmful']>0 for r in complete),
                  n_any_beats_best2=sum(r['n_seeds_guard_beats_best2']>0 for r in complete)))
    summ=dict(schema='caqr_162_exploratory_campaign_v1',
       roster_count=162,screen_counts=dict(Counter(x['status'] for x in screen_records(a.out_dir))),
       routing_job_counts=dict(Counter(x['status'] for x in by_job)),
       groups=groups,
       warnings=['No final-test tuning; two reserved filenames locked by default',
         'Check selected model_set in evaluation_protocol.json; old and new results MUST NOT be pooled',
         'Materializer Ridge uses fixed TRAIN medians and delta mean/sd; parity of feature extraction with trainer source is not independently verified',
         'One-step supervised model applied during deeper beam search introduces distribution shift',
         'Unassigned split does not imply training independence; whole data are not independent validation',
         'Group by circuit before averaging across circuits; report family dependence',
         'The saved per-circuit table contains all 162 circuits including failures and no-neighbor cases',
         'No-neighbor / unscreened / failures are not assigned fabricated physical gains',
         'Final Guard nonnegative selected-seed gain does NOT establish expected improvement or outperform Best-of-2'])
    write_json(a.out_dir/'summary_162.json',summ)
    print('Screen:',summ['screen_counts'],'; Routing:',summ['routing_job_counts'],flush=True)
    print('Reports:',a.out_dir/'per_circuit_objective.csv',a.out_dir/'summary_162.json',flush=True)
    for x in groups:
        if x['split']=='ALL_EXPLORATORY':
            print('DESCRIPTIVE ONLY',x,flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    for action in ('roster','screen','evaluate','summary'):
        s=sub.add_parser(action)
        s.add_argument('--out-dir',type=Path,default=ROOT/'caqr_162_results')
        if action in ('roster','screen'):
            s.add_argument('--caqr-root',type=Path,required=True)
        if action=='screen':
            s.add_argument('--batch-size',type=int,default=20,help='0 means all remaining')
            s.add_argument('--timeout',type=int,default=300,help='seconds for each native CaQR run')
            s.add_argument('--attempts',type=int,default=150)
            s.add_argument('--force',action='store_true')
            s.add_argument('--retry-failed',action='store_true',help='Retry only failed/timeout jobs, not successes')
            s.add_argument('--release-final-test',action='store_true')
        if action=='evaluate':
            s.add_argument('--model-set',choices=['materializer','legacy'],default='materializer')
            s.add_argument('--batch-size',type=int,default=20,help='number of routing jobs; 0 means all remaining')
            s.add_argument('--seeds',default='200,211,223')
            s.add_argument('--control-seed-offset',type=int,default=1000)
            s.add_argument('--topology',choices=['heavy_hex7','line','ring','grid'],default='heavy_hex7')
            s.add_argument('--budget',type=int,default=120)
            s.add_argument('--max-depth',type=int,default=5)
            s.add_argument('--reference-attempts',type=int,default=150)
            s.add_argument('--routing-timeout',type=int,default=420)
            s.add_argument('--force',action='store_true')
            s.add_argument('--retry-failed',action='store_true',help='Retry only failed/timeout jobs, not successes')
            s.add_argument('--release-final-test',action='store_true')
        if action=='summary':
            s.add_argument('--seeds',default='200,211,223')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.action=='roster':return cmd_roster(a)
    if a.action=='screen':return cmd_screen(a)
    if a.action=='evaluate':return cmd_evaluate(a)
    return cmd_summary(a)

if __name__=='__main__':main()
