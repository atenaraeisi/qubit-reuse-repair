"""Phase 6: physical comparison of TWO explicit valid reuse plans (no Search).

Example (Windows PowerShell, Qiskit 0.45.3 environment):
  python examples/route_qasm_pair.py examples/feature_parity_static.qasm `
    examples/feature_parity_plan.json examples/phase5_alternative_plan.json `
    --topology line --physical-qubits 3 --objective swap_count `
    --mode route_guarded --seed 200 --control-second-seed 201 `
    --out-dir phase6_pair_demo

Results are observations for the frozen evaluation seeds, NOT heldout generalization
claims. All circuits are built by the SAME Phase-5 materializer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def add_router_args(p):
    p.add_argument('--topology', choices=('line','ring','grid','heavy_hex7'), default='line')
    p.add_argument('--physical-qubits', type=int, default=None)
    p.add_argument('--grid-rows', type=int)
    p.add_argument('--grid-cols', type=int)
    p.add_argument('--optimization-level', type=int, choices=(0,1,2,3), default=1)
    p.add_argument('--seed', type=int, default=200, help='Frozen final evaluation seed (NOT a search seed)')
    p.add_argument('--control-second-seed', type=int, help='Optional third route: Original Best-of-2')
    p.add_argument('--objective', choices=('swap_count','routed_depth'), default='swap_count')
    p.add_argument('--mode', choices=('predict_only','route_guarded'), default='route_guarded')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--overwrite', action='store_true', help='Explicitly overwrite an earlier result report')


def create_router(args, circuit_width):
    from qubit_reuse_repair import make_coupling_map, RoutingSettings, QiskitSabreRouter
    width = args.physical_qubits
    if args.topology in ('line','ring') and width is None:
        width = max(circuit_width, 2 if args.topology=='line' else 3)
    cmap = make_coupling_map(args.topology, physical_qubits=width,
                             grid_rows=args.grid_rows, grid_cols=args.grid_cols)
    router=QiskitSabreRouter(cmap,RoutingSettings(optimization_level=args.optimization_level))
    if cmap.size() < circuit_width:
        raise ValueError(f'Hardware has {cmap.size()} qubits but materialized plans need {circuit_width}')
    return router, cmap


def prepare_out_dir(path, overwrite):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f'{path} is nonempty; use a fresh --out-dir (or --overwrite)')
    path.mkdir(parents=True,exist_ok=True)


def save_routes(result, outdir, *, report):
    """Save exactly the actual routed circuits used to make the decision."""
    paths = {}
    for role, observation in [('original',result.original_route),
                              ('candidate',result.repaired_route),
                              ('selected',result.selected_route)]:
        if observation is None:
            continue
        path = outdir / f'{role}_routed.qasm'
        path.write_text(observation.circuit.qasm(), encoding='utf-8')
        paths[role] = str(path)
    control = result.original_best_of_two
    if control is not None:
        path=outdir/'control_original_second_routed.qasm'
        path.write_text(control.second_original.circuit.qasm(),encoding='utf-8')
        paths['control_original_second']=str(path)
    report['routed_qasm_paths']=paths
    (outdir/'routing_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Final routed circuit:',paths['selected'])
    print('Report:',outdir/'routing_report.json')


def print_result(result):
    print('Objective:',result.objective.value,' Mode:',result.mode.value)
    print('Selected:',result.selected_source,' Main routing jobs:',result.routes_for_main_method)
    if result.original_route is not None:
        print('Original (SWAP, depth):',result.original_route.swap_count,result.original_route.routed_depth)
    print('Candidate (SWAP, depth):',result.repaired_route.swap_count,result.repaired_route.routed_depth)
    print('Selected (SWAP, depth):',result.selected_route.swap_count,result.selected_route.routed_depth)
    print('Actual gain in chosen objective:',result.actual_gain_abs,'; %:',result.actual_gain_pct)
    if result.original_best_of_two is not None:
        ctl = result.original_best_of_two
        print('Original Best-of-2 objective gain:',ctl.control_gain_abs,
              '; guarded method minus control benefit:',ctl.guard_minus_control_abs)
    print('NOTE: two-route selection does not demonstrate expected gain on independent seeds.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('qasm',type=Path)
    p.add_argument('original_plan',type=Path)
    p.add_argument('candidate_plan',type=Path)
    add_router_args(p)
    a=p.parse_args()
    from qiskit import QuantumCircuit
    from qubit_reuse_repair import ReusePlan, StaticPlanMaterializer, select_routed_plan
    for src in (a.qasm,a.original_plan,a.candidate_plan):
        if not src.is_file(): p.error(f'File does not exist: {src}')
    qc=QuantumCircuit.from_qasm_file(str(a.qasm))
    before=ReusePlan.from_json(a.original_plan.read_text(encoding='utf-8-sig'))
    after=ReusePlan.from_json(a.candidate_plan.read_text(encoding='utf-8-sig'))
    pair=StaticPlanMaterializer(qc).materialize_pair(before,after)
    backend,cmap=create_router(a,pair.full_width)
    prepare_out_dir(a.out_dir,a.overwrite)
    result=select_routed_plan(pair,backend,objective=a.objective,mode=a.mode,
                              evaluation_seed=a.seed,control_second_seed=a.control_second_seed)
    print_result(result)
    report={
        'schema':'phase6_explicit_pair_v1',
        'source_qasm':str(a.qasm),
        'source_qasm_sha256':hashlib.sha256(a.qasm.read_bytes()).hexdigest(),
        'source_plan_signatures':{'original':before.signature,'candidate':after.signature},
        'materializer':'StaticPlanMaterializer_v1',
        'materialized':{'original':pair.original.as_metadata(),'candidate':pair.repaired.as_metadata()},
        'hardware':{'topology':a.topology,'physical_qubits':cmap.size(),
                    'directed_edges':list(map(list,cmap.get_edges()))},
        'routing_settings':backend.settings.to_dict(),
        'selection':result.to_dict(),
        'is_independent_model_validation':False,
    }
    save_routes(result,a.out_dir,report=report)


if __name__=='__main__':main()
