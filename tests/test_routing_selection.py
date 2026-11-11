"""No-Qiskit tests: exactly what two physical routes and guarded selection mean."""
from __future__ import annotations

from dataclasses import dataclass
import pytest

from qubit_reuse_repair import (ReusePlan, MaterializationResult, MaterializationPair,
    RouteObservation, RoutingSettings, select_routed_plan, repair_and_route, RepairConfig)


@dataclass
class FakeCircuit:
    name: str
    num_qubits: int = 2
    num_clbits: int = 4


class FakeBackend:
    def __init__(self, costs):
        self.costs = costs
        self.calls = []

    def route(self, circuit, *, seed):
        self.calls.append((circuit.name, seed))
        swaps, depth = self.costs[(circuit.name, seed)]
        return RouteObservation(circuit=FakeCircuit(circuit.name + f"_routed_{seed}"),
            seed=seed, swap_count=swaps, routed_depth=depth,
            routing_time_sec=0.02, operation_counts={'swap': swaps, 'cx': 1})


def make_pair(same=False):
    original = ReusePlan.from_chains([[0, 1], [2, 3]])
    candidate = original if same else ReusePlan.from_chains([[0, 2], [1, 3]])
    def make(name, plan):
        return MaterializationResult(circuit=FakeCircuit(name), original_plan=plan,
            logical_to_physical={0: 0, 1: 0, 2: 1, 3: 1},
            scheduled_operations=(), reset_before_logical=(), active_width=2,
            full_width=2, source_measurement_count=4, compiled_measurement_count=4,
            emitted_reset_count=2)
    return MaterializationPair(make('original', original), make('repaired', candidate))


def test_guarded_swap_selection_ignores_depth_regression_and_reports_it():
    backend = FakeBackend({('original',200):(10, 100), ('repaired',200):(7, 130)})
    result = select_routed_plan(make_pair(), backend, objective='swap_count', evaluation_seed=200)
    assert result.selected_source == 'repaired'
    assert result.selected_plan == result.candidate_plan
    assert result.actual_gain_abs == 3
    assert result.actual_gain_pct == 30.0
    assert result.selected_route.routed_depth == 130
    assert result.selected_circuit.name == 'repaired_routed_200'
    assert backend.calls == [('original',200), ('repaired',200)]
    assert result.to_dict()['routes_for_main_method'] == 2


def test_guarded_depth_selection_ignores_swap_regression_and_reports_it():
    backend = FakeBackend({('original',200):(10, 100), ('repaired',200):(20, 90)})
    result = select_routed_plan(make_pair(), backend, objective='routed_depth')
    assert result.changed and result.actual_gain_abs == 10 and result.actual_gain_pct == 10
    assert result.selected_route.swap_count == 20


def test_guarded_keeps_original_on_harm_or_tie():
    for cost in ((11,80), (10,10)):
        backend = FakeBackend({('original',200):(10,100),('repaired',200):cost})
        result = select_routed_plan(make_pair(), backend, objective='swap_count')
        assert result.selected_source == 'original_guarded'
        assert result.selected_route is result.original_route
        assert result.actual_gain_abs == 0


def test_same_plan_avoids_duplicate_route_and_reports_zero_observed_gain():
    backend = FakeBackend({('original',200):(12,80)})
    result = select_routed_plan(make_pair(same=True), backend, objective='swap_count')
    assert backend.calls == [('original',200)]
    assert result.original_route is result.repaired_route is result.selected_route
    assert result.actual_gain_abs == 0 and result.routes_for_main_method == 1
    assert result.selected_source == 'original_unchanged'


def test_best_of_two_uses_two_original_routes_and_is_not_part_of_guard():
    backend = FakeBackend({('original',200):(10,100),('repaired',200):(7,101),('original',201):(6,102)})
    result = select_routed_plan(make_pair(), backend, objective='swap_count',
                                evaluation_seed=200, control_second_seed=201)
    assert result.actual_gain_abs == 3 and result.selected_source == 'repaired'
    assert result.original_best_of_two.control_gain_abs == 4
    assert result.original_best_of_two.guard_minus_control_abs == -1  # control better than guard
    assert backend.calls == [('original',200),('repaired',200),('original',201)]
    assert result.routes_for_main_method == 2
    assert result.to_dict()['additional_control_routes'] == 1
    assert result.to_dict()['original_best_of_two_control']['best_original']['seed'] == 201


def test_zero_original_cost_reports_undefined_percent_and_no_fake_gain():
    backend = FakeBackend({('original',200):(0,100),('repaired',200):(1,80)})
    result = select_routed_plan(make_pair(), backend, objective='swap_count')
    assert result.actual_gain_abs == 0
    assert result.actual_gain_pct is None
    assert not result.to_dict()['gain_pct_defined']


def test_predict_only_routes_selected_only_without_claiming_gain():
    backend = FakeBackend({('repaired',200):(7,90)})
    result = select_routed_plan(make_pair(), backend, objective='swap_count', mode='predict_only')
    assert backend.calls == [('repaired',200)]
    assert result.original_route is None and result.routes_for_main_method == 1
    assert result.actual_gain_abs is None and result.actual_gain_pct is None
    assert result.to_dict()['zero_observed_harm_guarantee_scope'] is None


def test_control_not_permitted_for_predict_only_or_same_seed():
    backend = FakeBackend({})
    with pytest.raises(ValueError, match='requires route_guarded'):
        select_routed_plan(make_pair(), backend, objective='swap_count', mode='predict_only',control_second_seed=201)
    with pytest.raises(ValueError, match='differ'):
        select_routed_plan(make_pair(), backend, objective='swap_count',control_second_seed=200)
    assert not backend.calls


def test_reject_width_mismatch_before_routing():
    pair = make_pair()
    pair.repaired.circuit.num_clbits = 3
    backend = FakeBackend({})
    with pytest.raises(ValueError,match='widths or clbits'):
        select_routed_plan(pair,backend,objective='swap_count')
    assert not backend.calls


def test_backend_wrong_seed_and_wrong_result_type_fail_closed():
    class BadSeed:
        def route(self, circuit, *, seed):
            return RouteObservation(circuit=circuit, seed=seed + 1,
                swap_count=0, routed_depth=1, routing_time_sec=0, operation_counts={})
    with pytest.raises(ValueError,match='wrong seed'):
        select_routed_plan(make_pair(),BadSeed(),objective='swap_count')
    class BadType:
        def route(self, circuit, *, seed): return {'swap_count': 0}
    with pytest.raises(TypeError,match='RouteObservation'):
        select_routed_plan(make_pair(),BadType(),objective='swap_count')


@pytest.mark.parametrize('seed', [-1, 2**32, 1.5, True])
def test_bad_evaluation_seed_fails_closed(seed):
    with pytest.raises(ValueError,match='routing seed'):
        select_routed_plan(make_pair(),FakeBackend({}),objective='swap_count',evaluation_seed=seed)


def test_bad_observation_values_and_settings_rejected():
    with pytest.raises(ValueError,match='swap_count disagrees'):
        RouteObservation(FakeCircuit('x'),200,3,7,0,{'swap':2})
    with pytest.raises(ValueError,match='nonnegative'):
        RouteObservation(FakeCircuit('x'),200,-1,7,0,{'swap':-1})
    with pytest.raises(ValueError,match='optimization_level'):
        RoutingSettings(optimization_level=7)
    assert RoutingSettings().to_dict()['routing_method'] == 'sabre'


def test_no_silent_negative_gain_when_routing_worsens_other_metric():
    backend=FakeBackend({('original',200):(5,10),('repaired',200):(0,999)})
    out=select_routed_plan(make_pair(),backend,objective='swap_count')
    assert out.selected_source=='repaired'
    assert out.selected_route.routed_depth==999
    assert out.actual_gain_abs == 5


def test_failing_repair_router_does_not_report_success():
    backend=FakeBackend({('original',200):(5,10)})
    with pytest.raises(KeyError):
        select_routed_plan(make_pair(),backend,objective='swap_count')
    assert backend.calls == [('original',200),('repaired',200)]
