"""Check independent Phase-4 structural feature definitions against original Experiment-4D source.

Arguments: original QASM, JSON list of plans, ORIGINAL exp4d_features.py from
experiments branch. The legacy pipeline compacts active qubits; this tool
fails closed unless compaction preserves original active logical indices.
Does not compare logical_depth/norm_depth: those require dynamic compilation.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
from qubit_reuse_repair import ReusePlan, ReuseValidityChecker
from qubit_reuse_repair.features import StructuralFeatureExtractor, STRUCTURAL_COLUMNS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qasm',required=True,type=Path)
    ap.add_argument('--plans',required=True,type=Path,help='JSON array of plans; each plan is an array of chains')
    ap.add_argument('--legacy-features-py',required=True,type=Path)
    ap.add_argument('--tol',default=1e-9,type=float)
    a=ap.parse_args()
    from qiskit import QuantumCircuit
    qc=QuantumCircuit.from_qasm_file(str(a.qasm))
    spec=importlib.util.spec_from_file_location('historical_exp4d_features',a.legacy_features_py.resolve())
    if spec is None or spec.loader is None:raise RuntimeError('Could not import historical exp4d_features.py')
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    qc2,mapping=lib.prepare_gidnet_input(qc)
    if any(k!=v for k,v in mapping.items()) or set(mapping)!=set(range(qc.num_qubits)):
        raise ValueError('Legacy compaction changes logical IDs; parity requires an identity mapping')
    profiles,timing=lib.build_interaction_profiles(qc2)
    new=StructuralFeatureExtractor(qc)
    ck=ReuseValidityChecker(qc)
    plans=json.loads(a.plans.read_text(encoding='utf-8-sig'))
    diffs=[]; n=0
    for raw in plans:
        plan=ReusePlan.from_chains(raw)
        verdict=ck.validate(plan,target_width=plan.active_width)
        if not verdict.is_valid:
            raise ValueError('Plan invalid: '+verdict.reason)
        old=lib.plan_features(qc2,profiles,timing,plan.chains)
        fresh=new.extract(plan)
        for name in STRUCTURAL_COLUMNS:
            x,y=float(old[name]),float(fresh[name])
            match=(math.isnan(x) and math.isnan(y)) or (math.isfinite(x) and math.isfinite(y) and abs(x-y)<=a.tol)
            if not match:diffs.append((n,name,x,y))
        n+=1
    print(f'Compared {n} plans x {len(STRUCTURAL_COLUMNS)} features, mismatches={len(diffs)}')
    for item in diffs[:10]:print('MISMATCH',item)
    if diffs:raise SystemExit(1)
    print('PASS: structural feature parity against historical Experiment 4D source')

if __name__=='__main__':main()
