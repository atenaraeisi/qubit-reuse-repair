"""Real Qiskit end-to-end pilot smoke; skipped if no Qiskit installed."""
from pathlib import Path
import json
import pytest
pytest.importorskip('qiskit')

ROOT=Path(__file__).resolve().parents[1]


def test_train_only_model_end_to_end_smoke(tmp_path):
    # Uses a synthetic QASM and its known valid source plan. The aim is
    # execution wiring, not a physical gain claim or heldout model validation.
    from scripts.fast_ridge_pilot import build_parser,run
    args=build_parser().parse_args([
        '--qasm',str(ROOT/'examples'/'sample_static.qasm'),
        '--plan',str(ROOT/'examples'/'sample_plan.json'),
        '--generator','CaQR',
        '--checkpoint',str(ROOT/'pilot_models'/'routed_depth_one_step_TRAIN_ONLY.json'),
        '--objective','routed_depth','--topology','line','--physical-qubits','3',
        '--budget','8','--max-depth','1','--reference-attempts','20',
        '--out-dir',str(tmp_path/'pilot')])
    data=run(args)
    assert data['checkpoint']['protocol']=='fixed_train_family_cv_25delta_ridge_v1'
    assert data['selection']['selected_route']['routed_depth']>=0
    assert (tmp_path/'pilot'/'routing_report.json').exists()
    assert data['selection']['routes_for_main_method'] in (1,2)
    assert data['selection']['actual_gain_abs']>=0
