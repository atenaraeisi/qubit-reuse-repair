"""Run in user's Qiskit 0.45.3 environment.

QASM + plan corpus parity audit against an INDEPENDENT direct-mutation path.
The new fast engine must produce the same ORDERED statically valid plans as
slow legacy-style enumerate/materialize/full-check, using same RAW cap.

Manifest CSV columns: qasm_path,plan_json
Each row is one original circuit QASM and common ReusePlan JSON (active IDs).

  python scripts/verify_real_corpus.py --manifest manifest.csv \
    --out audit_phase2.json --max-attempts-per-op 180

This does NOT check semantic equivalence of routed dynamic circuits.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

from qubit_reuse_repair import ReusePlan, ReuseValidityChecker, ValidityAwareEngine
from qubit_reuse_repair.validity import OPERATORS


def raw_direct(plan, op):
    chains = plan.chains
    if op == 'suffix_swap':
        for i, j in combinations(range(len(chains)), 2):
            l, r = chains[i], chains[j]
            for a in range(1, len(l)+1):
                for b in range(1, len(r)+1):
                    c = list(chains)
                    c[i], c[j] = l[:a] + r[b:], r[:b] + l[a:]
                    yield ReusePlan.from_chains(c)
    elif op == 'chain_transfer':
        for i, src in enumerate(chains):
            if len(src) <= 1:
                continue
            for a, q in enumerate(src):
                for j, dst in enumerate(chains):
                    if i == j:
                        continue
                    for b in range(len(dst)+1):
                        c = list(chains)
                        c[i] = src[:a] + src[a+1:]
                        c[j] = dst[:b] + (q,) + dst[b:]
                        yield ReusePlan.from_chains(c)
    else:
        raise ValueError(op)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('audit_phase2.json'))
    p.add_argument('--max-attempts-per-op', type=int, default=180)
    a = p.parse_args()
    if a.max_attempts_per_op < 0:
        p.error('max-attempts-per-op must be >=0 (0=unlimited)')
    from qiskit import QuantumCircuit
    rows = list(csv.DictReader(a.manifest.open(encoding='utf-8-sig',newline='')))
    if not rows or not {'qasm_path','plan_json'}.issubset(rows[0]):
        p.error('Manifest needs columns: qasm_path,plan_json')
    report = {'rows': [], 'counts': {'total': len(rows),'pass':0,'fail':0},
              'explanation': 'Static structural parity only; no routed/semantic equivalence.'}
    for ix, row in enumerate(rows):
        result = {'row':ix, 'qasm_path':row['qasm_path']}
        try:
            path = Path(row['qasm_path'])
            if not path.is_absolute():
                path = a.manifest.resolve().parent / path
            qc = QuantumCircuit.from_qasm_file(str(path))
            ck = ReuseValidityChecker(qc)
            plan = ReusePlan.from_json(row['plan_json'])
            valid = ck.validate(plan,target_width=plan.active_width)
            if not valid.is_valid:
                raise ValueError('Invalid original: '+valid.reason)
            prep = ValidityAwareEngine(ck).prepare(plan)
            old_seen = set()
            counts = {}
            for op in OPERATORS:
                raw_attempts = 0
                expected = []
                for proposal in raw_direct(plan,op):
                    if a.max_attempts_per_op and raw_attempts >= a.max_attempts_per_op:
                        break
                    raw_attempts += 1
                    if proposal == plan or proposal in old_seen:
                        continue
                    old_seen.add(proposal)
                    if ck.validate(proposal,target_width=plan.active_width).is_valid:
                        expected.append(proposal)
                actual, stats = prep.scan(op,max_attempts=a.max_attempts_per_op)
                if raw_attempts != stats.attempted or expected != [x.plan for x in actual]:
                    raise AssertionError(f'Parity mismatch: {op} expected={len(expected)} actual={len(actual)} attempts={raw_attempts}/{stats.attempted}')
                counts[op] = stats.__dict__
            result.update(status='PASS', active_width=plan.active_width, counts=counts)
            report['counts']['pass'] += 1
        except Exception as exc:
            result.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}')
            report['counts']['fail'] += 1
        report['rows'].append(result)
        print(f"{result['status']}: {result['qasm_path']} {result.get('reason','')}",flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report['counts']))
    if report['counts']['fail']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
