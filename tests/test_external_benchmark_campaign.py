"""Unit tests for external input selection; native Qiskit integration is separate."""
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'external_caqr_campaign_under_test', ROOT/'scripts'/'external_caqr_campaign.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_accepts_static_qasm2(tmp_path):
    q=tmp_path/'good.qasm'
    q.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\nh q[0];\ncx q[0],q[1];\n')
    d,reason=mod.scan_source(q,16,3000)
    assert reason is None
    assert d['declared_qubits_est']==4


def test_rejects_qasm3_and_midcircuit_measure(tmp_path):
    q=tmp_path/'v3.qasm'
    q.write_text('OPENQASM 3.0;\nqubit[4] q;\nh q[0];\n')
    _,reason=mod.scan_source(q,16,3000)
    assert reason=='openqasm3_requires_conversion'
    q=tmp_path/'mid.qasm'
    q.write_text('OPENQASM 2.0;\nqreg q[4];\ncreg c[4];\n'
                 'h q[0];\nmeasure q[0] -> c[0];\nx q[0];\n')
    _,reason=mod.scan_source(q,16,3000)
    assert reason=='mid_circuit_measurement'


def test_excludes_gate_reset_and_over_qubit_limit(tmp_path):
    q=tmp_path/'bad.qasm'
    q.write_text('OPENQASM 2.0;\nqreg q[4];\nreset q[0];\n')
    _,reason=mod.scan_source(q,16,3000)
    assert reason=='dynamic_or_reset_input'
    q=tmp_path/'big.qasm'
    q.write_text('OPENQASM 2.0;\nqreg q[99];\nh q[0];\n')
    _,reason=mod.scan_source(q,16,3000)
    assert reason=='outside_qubit_limit'
