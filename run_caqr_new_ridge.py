#!/usr/bin/env python3
"""One-command entrypoint for the updated CaQR MATERIALIZER-Ridge campaign.

Usage (from project root, in the Qiskit 0.45 virtual environment):
    python run_caqr_new_ridge.py prepare
    python run_caqr_new_ridge.py evaluate
    python run_caqr_new_ridge.py summary
    python run_caqr_new_ridge.py all

Old screening results can be reused because screening does not run learned
Ridge or the plan materializer. Old ROUTING results must never be reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DEFAULT_OLD=ROOT/'caqr_162_results'
DEFAULT_NEW=ROOT/'caqr_162_results_materializer_ridge_v2'
SCRIPT=ROOT/'scripts'/'caqr_campaign.py'


def prepare(old:Path,new:Path):
    old=old.resolve();new=new.resolve()
    if old==new:
        raise SystemExit('The new results directory MUST differ from the old one.')
    if not (old/'screen'/'cases').is_dir():
        raise SystemExit(f'Old screening records not found: {old / "screen" / "cases"}')
    if not (old/'frozen_roster.json').is_file():
        raise SystemExit('Old frozen_roster.json not found')
    if (new/'evaluation_protocol.json').exists():
        cfg=json.loads((new/'evaluation_protocol.json').read_text(encoding='utf-8-sig'))
        if cfg.get('model_set')!='materializer':
            raise SystemExit('Destination contains a different experimental protocol!')
    new.mkdir(parents=True,exist_ok=True)
    src_roster=(old/'frozen_roster.json').read_bytes()
    dest_roster=new/'frozen_roster.json'
    if dest_roster.exists() and dest_roster.read_bytes()!=src_roster:
        raise SystemExit('Different frozen roster: refusing to merge records.')
    if not dest_roster.exists():
        shutil.copyfile(old/'frozen_roster.json',dest_roster)
    src_screen=old/'screen'
    dest_screen=new/'screen'
    if dest_screen.exists():
        print(f'Screening cache already exists: {dest_screen}. Nothing overwritten.')
    else:
        shutil.copytree(src_screen,dest_screen)
        print(f'Copied previous screening only: {dest_screen}')
    print('Old ROUTING files were NOT copied.')
    cases=list((dest_screen/'cases').glob('*.json'))
    missing=[]
    for path in cases:
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        if data.get('status')=='ready':
            for key in ('qasm_path','plan_path'):
                if not Path(data.get(key,'')).is_file():
                    missing.append((path.name,key,data.get(key)))
    print(f'Screened circuit records: {len(cases)}; ready-case missing paths: {len(missing)}')
    if missing:
        print('First missing paths:',missing[:5])
        raise SystemExit('The original QASM/plan files are missing. Restore pilot_cases/CaQR_162 or re-screen.')


def campaign(action,new:Path,batch_size:int,timeout:int, retry_failed:bool=False):
    cmd=[sys.executable,str(SCRIPT),action,'--out-dir',str(new)]
    if action=='evaluate':
        cmd += ['--model-set','materializer','--batch-size',str(batch_size),
                '--seeds','200,211,223','--topology','heavy_hex7',
                '--budget','120','--max-depth','5','--routing-timeout',str(timeout)]
        if retry_failed:cmd.append('--retry-failed')
    print('RUN:',' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','evaluate','summary','all'])
    p.add_argument('--old-results',type=Path,default=DEFAULT_OLD)
    p.add_argument('--new-results',type=Path,default=DEFAULT_NEW)
    p.add_argument('--batch-size',type=int,default=0,help='0 = all remaining jobs')
    p.add_argument('--routing-timeout',type=int,default=420)
    p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args(argv)
    if a.batch_size<0 or a.routing_timeout<=0:p.error('Batch/timeout out of range')
    if a.action in ('prepare','all'):
        prepare(a.old_results,a.new_results)
    if a.action in ('evaluate','all'):
        campaign('evaluate',a.new_results,a.batch_size,a.routing_timeout,a.retry_failed)
    if a.action in ('summary','all'):
        campaign('summary',a.new_results,a.batch_size,a.routing_timeout)


if __name__=='__main__':main()
