#!/usr/bin/env python3
"""One frozen pretrained TRAIN_ONLY Ridge pilot on an explicitly supplied reuse plan.

CAUTION: This is an EXPERIMENTAL cross-compiler transfer test, NOT a replay
of old LOCO results and NOT yet a generalization claim. No routing is used
for search, reference normalization, or checkpoint training. Routing occurs
only after search selects a plan. Generator-specific provenance is a label,
not a guarantee that a plan actually came from that generator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def _hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    from qiskit import QuantumCircuit
    from qubit_reuse_repair import (ReusePlan, RepairConfig, SearchLimits,
        StaticPlanMaterializer, StructuralFeatureExtractor, repair_and_route,
        make_coupling_map, RoutingSettings, QiskitSabreRouter, ValidityAwareEngine)
    from qubit_reuse_repair.features.pilot_bridge import PilotFeatureBridge
    from qubit_reuse_repair.features.legacy_reset import normalize_reset_for_depth
    from qubit_reuse_repair.search.train_only import TrainOnlyDeltaRidge

    qasm=Path(args.qasm)
    source=QuantumCircuit.from_qasm_file(str(qasm))
    original=ReusePlan.from_json(Path(args.plan).read_text(encoding='utf-8-sig'))
    engine=ValidityAwareEngine.from_circuit(source)
    valid=engine.checker.validate(original,target_width=original.active_width)
    if not valid.is_valid:
        raise ValueError(f'Original input plan invalid: {valid.reason}')
    if original.n_active_qubits < original.active_width:
        raise AssertionError('Bad source active width')
    checkpoint=json.loads(Path(args.checkpoint).read_text(encoding='utf-8-sig'))
    if checkpoint.get('objective') != args.objective:
        raise ValueError('Wrong objective checkpoint')
    if checkpoint.get('arm') != args.arm:
        raise ValueError('Wrong one_step/multi_step checkpoint')
    # The training list uses labels CaQR/QASMBench for benchmark sources.
    # For a new generator using the SAME logical QASM, treat identical filenames
    # as possible family overlap, irrespective of generator label.
    if any(str(x).split('::')[-1] == qasm.name for x in checkpoint.get('training_circuits',[])):
        if not args.allow_training_circuit:
            raise ValueError('This circuit filename occurs in model TRAINING. '
              'Choose an unseen QASM or explicitly use --allow-training-circuit for a '
              'development-only diagnostic (not an independent pilot).')
        training_overlap=True
    else:
        training_overlap=False

    t0=time.perf_counter()
    neighbors,refstats=engine.neighbors(original,max_attempts_per_operator=args.reference_attempts)
    refs={p.signature:p for p in [original]+[n.plan for n in neighbors]}
    if len(refs) < 2:
        raise ValueError('No alternative valid plan in the frozen reference pool; '
           'cannot run Ridge on this start. Report as no-repair-possible, not a gain.')
    reference_plans=[refs[k] for k in sorted(refs)[:args.max_reference]]
    if original.signature not in {p.signature for p in reference_plans}:
        reference_plans=[original]+reference_plans[:args.max_reference-1]

    materializer=StaticPlanMaterializer(source,checker=engine.checker)
    compile_cache={}
    def compile_depths(plan):
        if plan.signature not in compile_cache:
            dyn=materializer.materialize(plan).circuit
            normalized=normalize_reset_for_depth(dyn)
            compile_cache[plan.signature]=(float(dyn.depth() or 0),float(normalized.depth() or 0))
        return compile_cache[plan.signature]
    if checkpoint.get('protocol') == 'materializer25_rawdelta_group_ridge_v1':
        # The newly trained model consumes RAW plan features, then subtracts
        # and standardizes deltas using TRAINING-ONLY fixed coefficients.
        # DO NOT reuse the old per-circuit frozen reference-pool normalizer.
        from qubit_reuse_repair.features.materializer_raw import MaterializerRawFeatureProvider
        from qubit_reuse_repair.search.materializer_ridge import MaterializerDeltaRidge
        provider=MaterializerRawFeatureProvider(
            source,materializer=materializer,extractor=StructuralFeatureExtractor(source))
        model=MaterializerDeltaRidge(checkpoint,provider,objective=args.objective)
        model_info={
            'path':str(args.checkpoint),'sha256':_hash(args.checkpoint),
            'protocol':checkpoint['protocol'],'arm':args.arm,
            'training_overlap_filename':training_overlap,
            'n_train_circuits':len(checkpoint.get('training_circuits',[])),
            'n_train_pairs':checkpoint.get('n_training_examples'),
            'feature_definition':'23 raw structural + common materializer logical_depth and reset-lowered norm_depth',
            'normalization':'raw plan imputation using TRAIN medians; Candidate - Original raw delta; TRAIN delta mean/sd',
            'reference_pool_size':len(reference_plans),
            'reference_raw_attempts':refstats.attempted,
            'warning':'Feature extraction source parity with friend training dataset not yet independently verified; one-step model used in multi-step beam is distribution shift',
        }
    elif checkpoint.get('protocol') == 'fixed_train_family_cv_25delta_ridge_v1':
        bridge=PilotFeatureBridge(StructuralFeatureExtractor(source),compile_depths,reference_plans)
        model=TrainOnlyDeltaRidge(checkpoint,bridge,objective=args.objective)
        model_info={
            'path':str(args.checkpoint),'sha256':_hash(args.checkpoint),
            'protocol':checkpoint['protocol'],'arm':args.arm,
            'training_overlap_filename':training_overlap,
            'n_train_circuits':len(checkpoint.get('training_circuits',[])),
            'feature_definition':'historical 23 structural + common materializer logical depth + historical reset-lowering normalized depth',
            'normalization':'frozen, deterministic one-step valid neighbor pool; pandas mean and std(ddof=0)',
            'normalization_fingerprint':bridge.normalizer.fingerprint,
            'reference_pool_size':len(reference_plans),
            'reference_raw_attempts':refstats.attempted,
            'warning':'The reference pool/compiler differs from training; pretrained weights are exploratory transfer ONLY',
        }
    else:
        raise ValueError('Unsupported checkpoint protocol: '+str(checkpoint.get('protocol')))
    print('Frozen reference plans:',len(reference_plans),'prepared in',round(time.perf_counter()-t0,2),'s')
    print('Checkpoint:',checkpoint['objective'],checkpoint['arm'],'TRAIN_ONLY; no retraining')

    if args.topology=='heavy_hex7':
        cmap=make_coupling_map('heavy_hex7')
    elif args.topology=='grid':
        cmap=make_coupling_map('grid',grid_rows=args.grid_rows,grid_cols=args.grid_cols)
    else:
        n=args.physical_qubits or max(2,original.active_width+source.num_qubits-len(original.qubits))
        cmap=make_coupling_map(args.topology,physical_qubits=n)
    if cmap.size() < original.active_width + source.num_qubits-len(original.qubits):
        raise ValueError('Hardware has fewer qubits than source plan materialized width')
    backend=QiskitSabreRouter(cmap,RoutingSettings(optimization_level=args.optimization_level))
    cfg=RepairConfig(objective=args.objective,search_policy=args.policy,model='ridge',
                     max_depth=args.max_depth,budget=args.budget,beam_width=args.beam_width,
                     search_seed=args.search_seed,final_selection=args.mode)
    caps=SearchLimits(max_attempts_per_operator=args.search_attempts,
                      raw_attempt_budget=args.raw_budget,per_parent_keep=args.per_parent_keep)

    output=Path(args.out_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError('Output directory nonempty; provide a new --out-dir or --overwrite')
    output.mkdir(parents=True,exist_ok=True)
    res=repair_and_route(source,original,config=cfg,backend=backend,scorer=model,
                         limits=caps,evaluation_seed=args.seed,
                         control_second_seed=args.control_second_seed)
    from qiskit import QuantumCircuit
    for label,observation in [('original',res.physical.original_route),
        ('candidate',res.physical.repaired_route),('selected',res.physical.selected_route)]:
        if observation is not None:
            (output/(label+'_routed.qasm')).write_text(observation.circuit.qasm(),encoding='utf-8')
    if res.physical.original_best_of_two is not None:
        second=res.physical.original_best_of_two.second_original
        (output/'original_control_second_routed.qasm').write_text(second.circuit.qasm(),encoding='utf-8')
    report={
        'schema':'generator_neutral_materializer_ridge_pilot_v2',
        'generator_label_user_supplied':args.generator,
        'source_qasm':str(qasm),'source_qasm_filename':qasm.name,'source_sha256':_hash(qasm),
        'plan_sha256':_hash(args.plan),'objective':args.objective,
        'checkpoint':model_info,
        'policy':args.policy,'routing_seed':args.seed,
        'topology':args.topology,'physical_qubits':cmap.size(),
        'hardware_directed_edges':list(map(list,cmap.get_edges())),
        'routing_settings':backend.settings.to_dict(),
        'search':res.search.to_dict(),
        'selection':res.physical.to_dict(),
        'timing':res.to_dict()['timing'],
        'independent_validation':False if training_overlap else 'unseen-name pilot; not independently validated model/feature transfer',
        'cautions':['No routing labels were read during search',
          model_info['warning'],
          'One observed routing seed cannot support a generalization or no-harm claim',
          'Generator label is provenance supplied by caller; native generator parsing not yet verified',
          'Do not mix compiler revisions or compare different router settings as matched results']
    }
    (output/'routing_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    a,b=res.physical.original_route,res.physical.repaired_route
    print('Policy',args.policy,'original:',original.chains,'proposed:',res.search.selected.chains)
    print('Scored candidates:',res.search.n_scored,'predicted gain:',res.search.selected_score,
          'actual proposed cost (SWAP,depth):',None if b is None else (b.swap_count,b.routed_depth))
    print('Original (SWAP,depth):',None if a is None else (a.swap_count,a.routed_depth))
    print('Final source:',res.physical.selected_source,'selected metric gain:',res.physical.actual_gain_pct)
    print('Report:',output/'routing_report.json')
    return report


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qasm',required=True,type=Path)
    p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--generator',choices=['CaQR','DQCC','GidNET'],required=True)
    p.add_argument('--checkpoint',required=True,type=Path,
        help='A MATERIALIZER raw-delta or legacy TRAIN_ONLY Ridge JSON checkpoint')
    p.add_argument('--arm',choices=['one_step','multi_step'],default='one_step')
    p.add_argument('--objective',choices=['swap_count','routed_depth'],required=True)
    p.add_argument('--policy',choices=['random_beam','adaptive_beam'],default='random_beam')
    p.add_argument('--mode',choices=['route_guarded','predict_only'],default='route_guarded')
    p.add_argument('--budget',type=int,default=120)
    p.add_argument('--beam-width',type=int,default=5)
    p.add_argument('--max-depth',type=int,default=5)
    p.add_argument('--per-parent-keep',type=int,default=40)
    p.add_argument('--search-attempts',type=int,default=180)
    p.add_argument('--raw-budget',type=int,default=2400)
    p.add_argument('--reference-attempts',type=int,default=150)
    p.add_argument('--max-reference',type=int,default=120)
    p.add_argument('--search-seed',type=int,default=2026)
    p.add_argument('--topology',choices=['line','ring','grid','heavy_hex7'],default='heavy_hex7')
    p.add_argument('--physical-qubits',type=int,default=None)
    p.add_argument('--grid-rows',type=int,default=4)
    p.add_argument('--grid-cols',type=int,default=4)
    p.add_argument('--optimization-level',type=int,choices=[0,1,2,3],default=1)
    p.add_argument('--seed',type=int,default=200)
    p.add_argument('--control-second-seed',type=int,default=None)
    p.add_argument('--allow-training-circuit',action='store_true')
    p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--overwrite',action='store_true')
    return p

if __name__=='__main__':
    args=build_parser().parse_args()
    if args.max_reference < 2 or args.reference_attempts < 1:
        raise SystemExit('Reference caps must be positive, max-reference >=2')
    run(args)
