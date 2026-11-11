"""Actual Qiskit 0.45.x SABRE transpiler exercises; skipped without Qiskit."""
from __future__ import annotations

import pytest
qiskit = pytest.importorskip('qiskit')

from qubit_reuse_repair import (ReusePlan, StaticPlanMaterializer, QiskitSabreRouter,
    make_coupling_map, select_routed_plan, RoutingSettings, RepairConfig,
    SearchLimits, repair_and_route)


def source_pair():
    qc = qiskit.QuantumCircuit(4,4)
    qc.h(0);qc.cx(0,1)
    qc.h(2);qc.cx(2,3)
    for i in range(4): qc.measure(i,i)
    original=ReusePlan.from_chains([[0,2],[1,3]])
    candidate=ReusePlan.from_chains([[0,3],[1,2]])
    return qc, StaticPlanMaterializer(qc).materialize_pair(original,candidate)


def test_physical_router_and_route_guard_are_real():
    qc,pair=source_pair()
    router=QiskitSabreRouter(make_coupling_map('line',physical_qubits=3))
    out=select_routed_plan(pair,router,objective='swap_count',evaluation_seed=200,
                           control_second_seed=201)
    assert out.routes_for_main_method==2
    assert out.original_route.seed==200 and out.repaired_route.seed==200
    assert out.original_best_of_two.second_original.seed==201
    assert out.actual_gain_abs >= 0
    assert out.selected_route.swap_count == out.selected_route.circuit.count_ops().get('swap',0)
    assert out.selected_route.routed_depth == out.selected_route.circuit.depth()
    assert out.selected_circuit.num_clbits == qc.num_clbits
    assert out.original_route.routing_time_sec >= 0


def test_exact_same_plan_one_routing_and_separate_objectives():
    qc,pair=source_pair()
    materializer=StaticPlanMaterializer(qc)
    pair_same=materializer.materialize_pair(pair.original.original_plan,pair.original.original_plan)
    out=select_routed_plan(pair_same,QiskitSabreRouter(make_coupling_map('line',physical_qubits=2)),
                           objective='routed_depth',evaluation_seed=202)
    assert out.routes_for_main_method == 1 and out.actual_gain_abs == 0
    assert out.original_route is out.repaired_route
    assert out.selected_route.routed_depth >= 0


def test_integrated_model_free_search_to_real_routing():
    qc,_=source_pair()
    plan=ReusePlan.from_chains([[0,2],[1,3]])
    config=RepairConfig(objective='swap_count', search_policy='random_uniform',
                        model='none', budget=16, max_depth=2, beam_width=3,
                        final_selection='route_guarded', search_seed=7)
    out=repair_and_route(qc, plan, config=config,
       backend=QiskitSabreRouter(make_coupling_map('line',physical_qubits=2)),
       limits=SearchLimits(raw_attempt_budget=120,max_attempts_per_operator=20),
       evaluation_seed=204)
    assert out.final_plan.active_width == plan.active_width
    assert out.physical.actual_gain_abs >= 0
    assert out.to_dict()['search']['physical_gain_measured'] is False
    assert out.to_dict()['physical']['selected_route']['swap_count'] >= 0
    assert out.to_dict()['timing']['search_time_sec'] >= 0


def test_qiskit_hardware_width_and_bad_topology_fail_early():
    cmap=make_coupling_map('line',physical_qubits=2)
    router=QiskitSabreRouter(cmap,RoutingSettings(optimization_level=1))
    with pytest.raises(ValueError,match='exceeds hardware width'):
        router.route(qiskit.QuantumCircuit(3),seed=200)
    with pytest.raises(ValueError,match='requires'):
        make_coupling_map('grid',physical_qubits=4)
    with pytest.raises(ValueError,match='>=2'):
        make_coupling_map('line',physical_qubits=1)
