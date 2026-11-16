#!/usr/bin/env python3
"""External QASM2 benchmark campaign using the existing CaQR + Repair pipeline.

All benchmark discovery, conversion requirements, and evaluation outputs are
kept separate from the frozen 162-circuit CaQR experiment. Requires the
existing scripts/caqr_campaign.py in the same project and the native CaQR checkout.

The source benchmark set must be original (unrouted) QASM2 circuits. Never
silently convert QASM3 or change the source circuit semantics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_internal_campaign():
    spec = importlib.util.spec_from_file_location(
        'internal_caqr_campaign', ROOT / 'scripts' / 'caqr_campaign.py'
    )
    if spec is None or spec.loader is None:
        raise RuntimeError('Missing scripts/caqr_campaign.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def normalize_name(value):
    value = re.sub(r'[^A-Za-z0-9]+', '_', value)
    return value.strip('_').lower()[:45] or 'case'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def scan_source(path, max_qubits, max_gates):
    content = path.read_bytes()
    try:
        source = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        return None, 'not_utf8'
    version = re.search(r'\bOPENQASM\s+([0-9.]+)\s*;', source)
    if not version:
        return None, 'missing_openqasm_header'
    if version.group(1) != '2.0':
        return None, 'openqasm3_requires_conversion'
    if re.search(r'\breset\b|\bif\s*\(', source):
        return None, 'dynamic_or_reset_input'
    # Gate counts exclude declarations, includes and terminal readout; the
    # Qiskit preflight below is authoritative when Qiskit is available.
    qregs = re.findall(r'(?m)^\s*qreg\s+[A-Za-z_][\w]*\[(\d+)\]\s*;', source)
    if not qregs:
        return None, 'no_qreg'
    nq = sum(int(x) for x in qregs)
    if nq < 2 or nq > max_qubits:
        return None, 'outside_qubit_limit'
    code = re.sub(r'(?m)//[^\n]*', '', source)
    rough_operations = len(re.findall(r';', code)) - len(re.findall(
        r'\b(?:OPENQASM|include|qreg|creg|measure|barrier)\b[^;]*;', code
    ))
    if rough_operations > max_gates:
        return None, 'over_gate_limit'
    if re.search(r'(?m)^\s*if\s*\(', code):
        return None, 'classical_condition'
    # After first measurement, only other measurements/barriers are allowed.
    first_measure = re.search(r'(?m)^\s*measure\b', code)
    if first_measure:
        remainder = code[first_measure.start():]
        lines = [line.strip() for line in remainder.splitlines() if line.strip()]
        if any(not re.match(r'^(?:measure\b|barrier\b|//)', line) for line in lines):
            return None, 'mid_circuit_measurement'
    return {'qasm_sha256': hash_bytes(content),
            'declared_qubits_est': nq, 'operations_est': max(0, rough_operations)}, None


def optional_qiskit_check(path, max_qubits, max_gates):
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        return None, None
    try:
        qc = QuantumCircuit.from_qasm_file(str(path))
        if qc.num_qubits > max_qubits:
            return None, 'qiskit_qubit_limit'
        if len(qc.data) > max_gates:
            return None, 'qiskit_gate_limit'
        # Distinguish support before actually invoking the CaQR generator.
        return {'qiskit_qubits': qc.num_qubits, 'qiskit_instructions': len(qc.data)}, None
    except Exception as exc:
        return None, f'qiskit_parse_failed:{type(exc).__name__}:{str(exc)[:140]}'


def prepare(args):
    source = args.source_dir.resolve()
    caqr = args.caqr_root.resolve()
    if not source.is_dir():
        raise SystemExit(f'QASM source directory not found: {source}')
    if not (caqr / 'main.py').is_file():
        raise SystemExit(f'Native CaQR main.py not found: {caqr}')
    files = sorted(source.rglob('*.qasm'), key=lambda p: str(p.relative_to(source)).casefold())
    if args.categories:
        cats = {x.lower() for x in args.categories.split(',') if x}
        files = [p for p in files if p.relative_to(source).parts[0].lower() in cats]
    entries, excluded = [], []
    seen_sha = set()
    old_training = set()
    # Content-level overlap with the legacy CaQR input pool is recorded even
    # when filenames differ; this is not proof of actual TRAIN membership.
    legacy_caqr_hashes = set()
    for old in (caqr / 'benchmarks').glob('*.qasm'):
        if old.name.startswith('ex_'):
            continue
        try:
            legacy_caqr_hashes.add(hash_bytes(old.read_bytes()))
        except OSError:
            continue
    for ck in ('swap_count', 'routed_depth'):
        p = ROOT / 'pilot_models' / f'{ck}_one_step_MATERIALIZER.json'
        if p.is_file():
            try:
                old_training.update(str(x).split('::')[-1].lower() for x in read_json(p).get('training_circuits', []))
            except Exception:
                pass
    for path in files:
        rel = path.relative_to(source).as_posix()
        # QASMBench includes pre-transpiled counterparts of original inputs.
        # Only original, unmapped circuits belong in an unbiased routing test.
        if 'transpiled' in path.stem.lower():
            excluded.append({'source_path': rel, 'reason': 'already_transpiled'})
            continue
        record, err = scan_source(path, args.max_qubits, args.max_gates)
        if err:
            excluded.append({'source_path': rel, 'reason': err})
            continue
        if record['qasm_sha256'] in seen_sha:
            excluded.append({'source_path': rel, 'reason': 'duplicate_qasm_bytes'})
            continue
        qiskit_meta, err = optional_qiskit_check(path, args.max_qubits, args.max_gates)
        if err:
            excluded.append({'source_path': rel, 'reason': err})
            continue
        if args.max_circuits and len(entries) >= args.max_circuits:
            excluded.append({'source_path': rel, 'reason': 'max_circuits_limit'})
            continue
        seen_sha.add(record['qasm_sha256'])
        category = path.relative_to(source).parts[0] if len(path.relative_to(source).parts) > 1 else 'root'
        # Preserve the precise file SHA in the copied name to prevent stale
        # generator output and avoid collisions among repeated source filenames.
        name = (f'ex_{normalize_name(args.suite)}_'
                f'{normalize_name(category)}_{normalize_name(path.stem)}_'
                f'{record["qasm_sha256"][:10]}.qasm')
        family = re.sub(r'_indep.*$', '', path.stem, flags=re.I)
        family = re.sub(r'[_-](?:n?\d+)\b.*$', '', family)
        family = normalize_name(family)
        # Group-level holdout, determined by source-family, *before results*.
        group_hash = int(hash_bytes(f'{args.suite}/{family}'.encode())[:8], 16) % 100
        reserved = group_hash < args.holdout_percent
        known_train = path.name.lower() in old_training
        entries.append(dict(
            circuit=name, family_heuristic=f'{args.suite}/{family}',
            previous_split='final_test_reserved' if reserved else 'unassigned',
            training_filename_overlap=known_train,
            overlaps_caqr_input_sha=record['qasm_sha256'] in legacy_caqr_hashes,
            reserved_final_test=reserved,
            source_suite=args.suite,
            source_relpath=rel,
            source_path=str(path.resolve()),
            source_sha256=record['qasm_sha256'],
            declared_qubits_est=record['declared_qubits_est'],
            operations_est=record['operations_est'],
            **(qiskit_meta or {})
        ))
    if not entries:
        breakdown = dict(Counter(e['reason'] for e in excluded))
        raise SystemExit('No compatible OpenQASM2 files selected. Exclusions: '
                         + str(breakdown) + '\nFor MQT Bench, convert QASM3 in a separate modern Qiskit environment first.')
    outdir = args.out_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    roster_data = dict(schema='external_qasm2_caqr_campaign_v1', source_suite=args.suite,
                       selected=entries, selection=dict(categories=args.categories,
                       max_qubits=args.max_qubits,max_gates=args.max_gates,
                       holdout_percent=args.holdout_percent,max_circuits=args.max_circuits),
                       notes=['Research outcomes are conditioned on valid reuse plans and observed neighbors.',
                              'The training overlap flag based on filenames is incomplete; audit by QASM hash and circuit families.',
                              'Only unmodified QASM2 source circuits were selected; no semantic transformation performed.'])
    roster_data['roster_sha256'] = hash_bytes(json.dumps(
        {k:v for k,v in roster_data.items() if k!='roster_sha256'},
        sort_keys=True, ensure_ascii=False).encode())
    target = outdir / 'frozen_roster.json'
    if target.is_file() and read_json(target) != roster_data:
        raise SystemExit('Frozen roster already differs from discovered sources. Use a fresh --out-dir; results must not be pooled.')
    if (outdir / 'evaluation_protocol.json').is_file() and not target.is_file():
        raise SystemExit('Cannot prepare into an evaluated directory without a frozen roster.')
    write_json(target, roster_data)
    write_csv(outdir/'external_roster.csv', entries)
    write_csv(outdir/'excluded_inputs.csv', excluded)
    target_dir = caqr/'benchmarks'
    target_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        dst = target_dir / entry['circuit']
        if dst.is_file():
            if hash_bytes(dst.read_bytes()) != entry['source_sha256']:
                raise SystemExit(f'Name collision / source drift at {dst}')
        else:
            shutil.copyfile(entry['source_path'], dst)
    print('Prepared:', len(entries), 'QASM2 circuits;',
          sum(e['reserved_final_test'] for e in entries), 'reserved holdout')
    print('Excluded:', len(excluded), dict(Counter(r['reason'] for r in excluded)))
    print('Frozen roster:', target)
    print('Source circuits copied into:', target_dir)


def activate_campaign(outdir):
    cfg = outdir/'frozen_roster.json'
    if not cfg.is_file():
        raise SystemExit(f'Missing frozen external roster: {cfg}. Run prepare first.')
    data = read_json(cfg)
    selected = data['selected']
    if not selected or len({e['circuit'] for e in selected}) != len(selected):
        raise SystemExit('Invalid external roster / duplicate prepared names')
    camp = load_internal_campaign()
    # Explicitly supply our immutable external roster to every reused pipeline
    # function, rather than ever falling back to CaQR's 162 historical files.
    camp.roster = lambda: selected
    camp.make_config = lambda expected_outdir: (
        cfg if expected_outdir.resolve() == outdir.resolve()
        else (_ for _ in ()).throw(ValueError('Output-directory / roster mismatch'))
    )
    return camp, data


def publish_external_summary(camp, outdir, data):
    original = outdir/'summary_162.json'
    if not original.is_file():
        return
    summary = camp.read_json(original)
    summary['schema'] = 'external_qasm2_caqr_campaign_summary_v1'
    summary['roster_count'] = len(data['selected'])
    summary['source_suite'] = data['source_suite']
    summary['selection'] = data['selection']
    summary.setdefault('warnings', []).extend([
        'External benchmarks: original QASM2 and native CaQR reuse plans; not 162 historical CaQR benchmarks.',
        'A zero gain after guard does not prove the prediction model is accurate.',
        'Same exact benchmark or family may have been used for model training; audit source-content hashes.',
    ])
    camp.write_json(outdir/'summary_external.json', summary)
    camp.write_csv(outdir/'screening_external.csv', camp.screen_records(outdir))
    print('External summary:', outdir/'summary_external.json')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    for action in ('prepare','screen','evaluate','summary'):
        s = sub.add_parser(action)
        s.add_argument('--out-dir', required=True, type=Path)
        if action=='prepare':
            s.add_argument('--source-dir',required=True,type=Path)
            s.add_argument('--caqr-root',required=True,type=Path)
            s.add_argument('--suite',default='QASMBench')
            s.add_argument('--categories',default='small,medium',help='Comma-separated top-level source directories; blank scans all')
            s.add_argument('--max-qubits',type=int,default=16)
            s.add_argument('--max-gates',type=int,default=3000)
            s.add_argument('--max-circuits',type=int,default=0)
            s.add_argument('--holdout-percent',type=int,default=20)
        if action=='screen':
            s.add_argument('--caqr-root',required=True,type=Path)
            s.add_argument('--batch-size',type=int,default=0)
            s.add_argument('--timeout',type=int,default=300)
            s.add_argument('--attempts',type=int,default=150)
            s.add_argument('--retry-failed',action='store_true')
        if action=='evaluate':
            s.add_argument('--model-set',choices=['materializer','legacy'],default='materializer')
            s.add_argument('--batch-size',type=int,default=0)
            s.add_argument('--seeds',default='200,211,223')
            s.add_argument('--control-seed-offset',type=int,default=1000)
            s.add_argument('--topology',default='heavy_hex7',choices=['heavy_hex7','line','ring','grid'])
            s.add_argument('--budget',type=int,default=120)
            s.add_argument('--max-depth',type=int,default=5)
            s.add_argument('--reference-attempts',type=int,default=150)
            s.add_argument('--routing-timeout',type=int,default=420)
            s.add_argument('--retry-failed',action='store_true')
        if action=='summary':
            s.add_argument('--seeds',default='200,211,223')
    a = p.parse_args(argv)
    if a.action=='prepare':
        if not 0<=a.holdout_percent<=100: p.error('--holdout-percent must be 0..100')
        return prepare(a)
    outdir = a.out_dir.resolve()
    camp,data = activate_campaign(outdir)
    if a.action=='screen':
        caqr = a.caqr_root.resolve()
        for e in data['selected']:
            pth=caqr/'benchmarks'/e['circuit']
            if not pth.exists() or hash_bytes(pth.read_bytes())!=e['source_sha256']:
                raise SystemExit(f'Prepared CaQR benchmark missing or modified: {pth}. Run prepare again with same source & settings.')
        a.caqr_root=caqr
        a.force=False
        a.release_final_test=False
        result=camp.cmd_screen(a)
        publish_external_summary(camp,outdir,data) if (outdir/'evaluation_protocol.json').is_file() else None
        return result
    if a.action=='evaluate':
        a.force=False
        a.release_final_test=False
        camp.cmd_evaluate(a)
    else:
        camp.cmd_summary(a)
    publish_external_summary(camp,outdir,data)


if __name__=='__main__':
    main()
