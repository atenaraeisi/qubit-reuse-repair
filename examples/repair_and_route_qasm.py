"""Phase 6 END-TO-END demo: model-free Search -> Materializer -> SABRE -> Guard.

For now only structural_beam/random_uniform can be selected from this CLI.
Ridge-based search is available in the Python API when you explicitly supply
compatible trained model + feature extraction; no silent pseudo-Ridge fallback.

Example (PowerShell):
  python examples/repair_and_route_qasm.py examples/sample_static.qasm `
    examples/sample_plan.json --policy random_uniform --objective swap_count `
    --topology line --physical-qubits 3 --out-dir phase6_e2e_demo
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from route_qasm_pair import add_router_args,create_router,prepare_out_dir,save_routes,print_result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('qasm',type=Path)
    p.add_argument('original_plan',type=Path)
    p.add_argument('--policy',choices=('structural_beam','random_uniform'),default='structural_beam')
    p.add_argument('--search-seed',type=int,default=2026)
    p.add_argument('--budget',type=int,default=120)
    p.add_argument('--max-depth',type=int,default=5)
    p.add_argument('--beam-width',type=int,default=5)
    p.add_argument('--raw-budget',type=int,default=2400)
    p.add_argument('--max-attempts-per-operator',type=int,default=180)
    add_router_args(p)
    args=p.parse_args()
    from qiskit import QuantumCircuit
    from qubit_reuse_repair import (ReusePlan,RepairConfig,SearchLimits,repair_and_route)
    if not args.qasm.is_file() or not args.original_plan.is_file():
        p.error('Source QASM and original plan JSON must exist')
    circuit=QuantumCircuit.from_qasm_file(str(args.qasm))
    plan=ReusePlan.from_json(args.original_plan.read_text(encoding='utf-8-sig'))
    # All legal moves preserve active width and inactive declared qubits.
    inactive=circuit.num_qubits-len(plan.qubits)
    backend,cmap=create_router(args,plan.active_width+inactive)
    config=RepairConfig(objective=args.objective,search_policy=args.policy,model='none',
                        budget=args.budget,max_depth=args.max_depth,beam_width=args.beam_width,
                        search_seed=args.search_seed,final_selection=args.mode)
    limits=SearchLimits(raw_attempt_budget=args.raw_budget,
                       max_attempts_per_operator=args.max_attempts_per_operator)
    prepare_out_dir(args.out_dir,args.overwrite)
    outcome=repair_and_route(circuit,plan,config=config,backend=backend,limits=limits,
                             evaluation_seed=args.seed,control_second_seed=args.control_second_seed)
    print('Search:',args.policy,'scored:',outcome.search.n_scored,'stop:',outcome.search.stop_reason)
    print('Original plan:',outcome.search.original.chains)
    print('Proposed plan:',outcome.search.selected.chains)
    print_result(outcome.physical)
    report={
        'schema':'phase6_search_route_v1',
        'source_qasm':str(args.qasm),
        'source_qasm_sha256':hashlib.sha256(args.qasm.read_bytes()).hexdigest(),
        'hardware':{'topology':args.topology,'physical_qubits':cmap.size(),
                    'directed_edges':list(map(list,cmap.get_edges()))},
        'routing_settings':backend.settings.to_dict(),
        'search':outcome.search.to_dict(),
        'timing':outcome.to_dict()['timing'],
        'selection':outcome.physical.to_dict(),
        'is_independent_model_validation':False,
        'warning':'model-free pilot, not trained Ridge; do not report as ML gain',
    }
    save_routes(outcome.physical,args.out_dir,report=report)


if __name__=='__main__':main()
