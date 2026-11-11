"""Inspect real-QASM structural features without routing or a fake Ridge model.

PowerShell: python examples/inspect_features_qasm.py examples/sample_static.qasm examples/sample_plan.json --out phase4_features.json
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from qubit_reuse_repair import ReusePlan, ValidityAwareEngine
from qubit_reuse_repair.features import StructuralFeatureExtractor, STRUCTURAL_COLUMNS


def safe_json(x):
    if isinstance(x,float) and not math.isfinite(x): return None
    if isinstance(x,dict): return {k:safe_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [safe_json(v) for v in x]
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('qasm',type=Path)
    parser.add_argument('plan',type=Path)
    parser.add_argument('--out',type=Path,default=Path('phase4_features.json'))
    args=parser.parse_args()
    from qiskit import QuantumCircuit
    qc=QuantumCircuit.from_qasm_file(str(args.qasm))
    plan=ReusePlan.from_json(args.plan.read_text(encoding='utf-8-sig'))
    engine=ValidityAwareEngine.from_circuit(qc)
    result=engine.checker.validate(plan,target_width=plan.active_width)
    if not result.is_valid:
        raise ValueError(f'Invalid input ReusePlan: {result.reason}')
    extractor=StructuralFeatureExtractor(qc)
    features=extractor.extract(plan)
    payload={'stage':'structural_feature_inspection_only','plan':plan.to_dict(),
             'structural_features': features, 'columns':list(STRUCTURAL_COLUMNS),
             'physical_gain_measured':False,
             'note':'No routing, physical depth, or SWAP gain has been measured. NaNs are exported as null.'}
    args.out.write_text(json.dumps(safe_json(payload),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Plan valid:',result.is_valid)
    print('Structural features:',len(features))
    print('Quotient unique edges:',features['quotient_unique_edges'])
    print('Output:',args.out)
    print('NO ROUTING; NO PHYSICAL GAIN MEASURED')

if __name__=='__main__':main()
