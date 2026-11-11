"""Real-QASM Phase 3 SEARCH ONLY, with model-free structural control.

Example (PowerShell):
    python examples/search_qasm.py examples/smoke.qasm examples/smoke_plan.json `
      --objective swap_count --policy structural_beam --budget 120 --out search_out.json

No dynamic circuit is built; no routing is executed; objective is only a label
for this structural control. Ridge-guided search must use the library API with
an actually trained, feature-compatible RidgeDeltaScorer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from qubit_reuse_repair import ReusePlan, RepairConfig, SearchLimits, search_plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('qasm', type=Path)
    ap.add_argument('plan_json', type=Path)
    ap.add_argument('--objective', choices=('swap_count', 'routed_depth'), default='swap_count')
    ap.add_argument('--policy', choices=('structural_beam', 'random_uniform'), default='structural_beam')
    ap.add_argument('--budget', type=int, default=120)
    ap.add_argument('--depth', type=int, default=5)
    ap.add_argument('--beam-width', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--raw-budget', type=int, default=2400)
    ap.add_argument('--max-attempts-per-operator', type=int, default=180)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()

    from qiskit import QuantumCircuit
    qc = QuantumCircuit.from_qasm_file(str(args.qasm))
    original = ReusePlan.from_json(args.plan_json.read_text(encoding='utf-8-sig'))
    config = RepairConfig(objective=args.objective, search_policy=args.policy, model='none',
                          max_depth=args.depth, budget=args.budget, beam_width=args.beam_width,
                          search_seed=args.seed)
    limits = SearchLimits(max_attempts_per_operator=args.max_attempts_per_operator,
                          raw_attempt_budget=args.raw_budget)
    result = search_plan(qc, original, config, limits=limits)
    print('Original plan:', result.original.chains)
    print('Selected plan:', result.selected.chains)
    print('Valid candidate plans scored:', result.n_scored)
    print('Raw mutation attempts:', result.raw_attempts)
    print('Changed:', result.changed, '; stop:', result.stop_reason)
    print('Score:', result.selected_score)
    print('NOTE: no physical SWAP/depth improvement has been measured (NO routing).')
    if args.out is not None:
        args.out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        print('Wrote:', args.out)


if __name__ == '__main__':
    main()
