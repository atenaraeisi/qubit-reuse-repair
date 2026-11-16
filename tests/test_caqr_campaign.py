import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load_path(module, path):
    spec=importlib.util.spec_from_file_location(module,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

r=load_path('campaign_roster',ROOT/'campaign'/'roster_source.py')
c=load_path('campaign_runner',ROOT/'scripts'/'caqr_campaign.py')


def test_frozen_selection_exact_counts():
    items=r.roster()
    assert len(items)==162
    assert len({x['circuit'] for x in items})==162
    assert len(r.NAMES)==168
    assert len(r.EXCLUDE)==6
    assert Counter(x['previous_split'] for x in items)=={
        'known_train':38,'known_validation':12,'final_test_reserved':2,'unassigned':110}
    assert len([x for x in items if x['training_filename_overlap']])==14
    assert {x['circuit'] for x in items if x['reserved_final_test']}=={'wim_266.qasm','squar5_261.qasm'}
    assert not any(x['circuit'] in r.EXCLUDE for x in items)


def test_holdout_does_not_autoscreen(tmp_path):
    rows=c.screen_records(tmp_path)
    assert len(rows)==162
    assert {x['status'] for x in rows}=={'not_screened','final_test_reserved'}
    assert len(c.summary_rows(tmp_path,[200]))==324


def test_freeze_immutable_across_restarts(tmp_path):
    target=c.make_config(tmp_path)
    assert c.read_json(target)['roster_sha256']==c.read_json(target)['roster_sha256']
    c.make_config(tmp_path)
    bad=c.read_json(target)
    bad['roster_sha256']='tampered'
    c.write_json(target,bad)
    with pytest.raises(RuntimeError,match='Frozen roster differs'):
        c.make_config(tmp_path)


def test_seed_and_aggregation_is_per_circuit_not_per_seed():
    assert c.parse_seeds('200,211,223')==[200,211,223]
    for s in ('','1,1','-1','4294967296'):
        with pytest.raises(ValueError): c.parse_seeds(s)
    rows=[]
    for name,gains in [('a.qasm',[2,4,6]),('b.qasm',[10,20,30])]:
        for seed,gain in zip([200,211,223],gains):
            rows.append(dict(circuit=name,objective='swap_count',seed=seed,
               previous_split='known_validation',family_heuristic='family',
               training_filename_overlap=False,screen_status='ready',
               status='ok',gain_pct=float(gain),gain_abs=gain,
               proposed_gain_abs=-1,guard_minus_best2_abs=1))
    out=c.aggregate_circuits(rows)
    assert len(out)==2
    assert {o['circuit']:o['mean_guard_gain_pct'] for o in out}=={'a.qasm':4.0,'b.qasm':20.0}
    assert sum(x['n_seeds_proposed_harmful'] for x in out)==6


def test_unevaluated_rows_dont_fabricate_zero_gain(tmp_path):
    c.make_config(tmp_path)
    row=next(x for x in r.roster() if x['previous_split']=='known_validation')
    path=c.screen_record_path(tmp_path,row['circuit'])
    c.write_json(path,dict(row,status='no_observed_neighbor',valid_one_step_neighbors=0))
    records=c.summary_rows(tmp_path,[200,211,223])
    item=next(x for x in records if x['circuit']==row['circuit'] and x['objective']=='swap_count')
    assert item['status']=='no_observed_neighbor'
    assert 'gain_pct' not in item
    agg=c.aggregate_circuits(records)
    item2=next(x for x in agg if x['circuit']==row['circuit'] and x['objective']=='swap_count')
    assert item2['mean_guard_gain_pct'] is None
    assert item2['n_seeds_succeeded']==0


def test_notebook_syntax_and_record_sheet():
    nb=json.loads((ROOT/'CaQR_162_benchmark_campaign.ipynb').read_text(encoding='utf-8'))
    assert nb['nbformat']==4
    assert len(nb['cells'])>=10
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),f'notebook cell {i}','exec')
    import csv
    with (ROOT/'campaign'/'benchmark_roster_162.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==162
    assert sum(x['reserved_final_test']=='True' for x in rows)==2


def test_qiskit_optional_parse_native_chain(tmp_path):
    pytest.importorskip('qiskit')
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(4)
    for q in [0,1,2,3]:qc.h(q)
    f=tmp_path/'chains.txt'
    f.write_text('0 -> 2\n')
    plan=c.native_chain_to_plan(f,qc)
    assert plan.chains==((0,2),(1,),(3,))
    f.write_text('0 -> 2\n2 -> 3\n')
    with pytest.raises(ValueError,match='duplicate'):
        c.native_chain_to_plan(f,qc)
