"""CLI: common-plan materialization only (NO ROUTING, NO physical gain)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qiskit import QuantumCircuit
from qubit_reuse_repair import ReusePlan, StaticPlanMaterializer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('qasm', type=Path, help='Original static QASM 2.0 circuit')
    p.add_argument('original_plan', type=Path, help='Original ordered chains JSON')
    p.add_argument('--candidate-plan', type=Path, help='Optional repaired ordered chains JSON')
    p.add_argument('--out-dir', type=Path, default=Path('phase5_compiled'))
    args = p.parse_args()
    if not args.qasm.is_file() or not args.original_plan.is_file():
        p.error('QASM and original plan must exist')
    original_qc = QuantumCircuit.from_qasm_file(str(args.qasm))
    compiler = StaticPlanMaterializer(original_qc)
    original_plan = ReusePlan.from_json(args.original_plan.read_text(encoding='utf-8-sig'))
    candidate = (ReusePlan.from_json(args.candidate_plan.read_text(encoding='utf-8-sig'))
                 if args.candidate_plan else None)
    if candidate is not None:
        pair = compiler.materialize_pair(original_plan, candidate)
        original_result, candidate_result = pair.original, pair.repaired
    else:
        original_result = compiler.materialize(original_plan)
        candidate_result = None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, result in [('original', original_result), ('repaired', candidate_result)]:
        if result is None:
            continue
        qasm_path = args.out_dir / (name + '.qasm')
        qasm_path.write_text(result.circuit.qasm(), encoding='utf-8')
        rows[name] = {**result.as_metadata(),
                      'compiled_unrouted_depth': result.circuit.depth(),
                      'compiled_operation_count': result.circuit.size(),
                      'qasm_path': str(qasm_path)}
        print(f'{name}: width={result.full_width}  resets={result.emitted_reset_count} '
              f'measures={result.compiled_measurement_count} '
              f'unrouted_depth={result.circuit.depth()} -> {qasm_path}')
    summary = {'schema_version': 1,
               'original_qasm': str(args.qasm),
               'objective': None,
               'no_routing': True,
               'physical_gain_measured': False,
               'same_compiler_for_both': candidate_result is not None,
               'plans': rows}
    target = args.out_dir / 'materialization_report.json'
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Report:', target)
    print('NO ROUTING: compiler depth is NOT routed depth; no SWAP or physical gain claim.')


if __name__ == '__main__':
    main()
