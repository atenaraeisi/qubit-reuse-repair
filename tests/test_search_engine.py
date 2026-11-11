"""Phase 3 structural/pre-routing search tests; no routing labels available."""
import json
import math
import random

import networkx as nx
import pytest

from qubit_reuse_repair import (ReusePlan, ReuseValidityChecker, ValidityAwareEngine,
                                SearchEngine, SearchLimits, RepairConfig, StructuralScorer,
                                CallableScorer, RidgeDeltaScorer, Objective, SearchPolicy)


def engine_with_independent_wires(n=5):
    ins = {q: ("in", q) for q in range(n)}
    outs = {q: ("out", q) for q in range(n)}
    dag = nx.DiGraph()
    dag.add_edges_from((ins[q], outs[q]) for q in range(n))
    checker = ReuseValidityChecker.from_graphs(
        num_qubits=n, active_qubits=range(n), full_graph=dag, full_inputs=ins, full_outputs=outs)
    return ValidityAwareEngine(checker)


def starting_plan():
    return ReusePlan.from_chains([[0, 1], [2, 3], [4]])


def scorer():
    # Purely syntactic, for end-to-end tests; does not imply physical gain.
    return CallableScorer(lambda origin, cand, target: 1.0 + 0.1 * len(cand.chains[0]) if origin != cand else 0.0)


def config(policy='random_beam', obj='swap_count', seed=123, max_depth=3, budget=25):
    return RepairConfig(search_policy=policy, model='none' if policy in ('structural_beam', 'random_uniform') else 'ridge',
                        objective=obj, max_depth=max_depth, budget=budget, beam_width=3,
                        search_seed=seed)


def limits():
    return SearchLimits(per_parent_keep=8, max_attempts_per_operator=30,
                        raw_attempt_budget=500)


@pytest.mark.parametrize('policy', ['random_beam', 'adaptive_beam'])
@pytest.mark.parametrize('obj', ['swap_count', 'routed_depth'])
def test_search_returns_valid_equal_width(policy, obj):
    engine = engine_with_independent_wires()
    original = starting_plan()
    result = SearchEngine(engine, config(policy=policy, obj=obj), scorer(), limits=limits()).run(original)
    assert result.changed
    assert result.n_scored > 0
    assert result.n_scored <= 25
    assert result.raw_attempts <= 500
    assert result.objective.value == obj
    assert result.selected.qubits == original.qubits
    assert result.selected.active_width == original.active_width
    assert engine.checker.validate(result.selected, target_width=original.active_width).is_valid
    assert result.to_dict()['physical_gain_measured'] is False
    assert all(r.operator in ('suffix_swap', 'chain_transfer') for r in result.trace)


def test_reproducible_and_does_not_route():
    engine = engine_with_independent_wires()
    worker = SearchEngine(engine, config(seed=987), scorer(), limits=limits())
    x = worker.run(starting_plan())
    y = worker.run(starting_plan())
    assert x.to_dict() == y.to_dict()
    assert json.loads(json.dumps(x.to_dict())) == x.to_dict()


def test_no_positive_predicted_gain_returns_original():
    x = SearchEngine(engine_with_independent_wires(), config(),
                     CallableScorer(lambda *args: -5.0), limits=limits()).run(starting_plan())
    assert not x.changed
    assert x.selected_score == (0.0,)
    assert x.n_scored > 0


def test_bad_original_rejected_before_search():
    with pytest.raises(ValueError, match='Invalid original'):
        SearchEngine(engine_with_independent_wires(5), config(), scorer()).run(
            ReusePlan.from_chains([[0, 1], [2, 3]]))


def test_missing_predictor_is_not_silently_replaced_with_proxy():
    with pytest.raises(ValueError, match='predictor'):
        SearchEngine(engine_with_independent_wires(), config(), None)


def test_structural_beam_needs_structural_scorer():
    engine = engine_with_independent_wires()
    with pytest.raises(TypeError):
        SearchEngine(engine, config(policy='structural_beam'), scorer())
    s = StructuralScorer(((0, 2), (0, 3), (1, 2), (1, 4)))
    z = SearchEngine(engine, config(policy='structural_beam'), s, limits=limits()).run(starting_plan())
    assert z.n_scored > 0
    assert len(z.selected_score) == 3
    assert z.selected.active_width == 3
    assert z.policy is SearchPolicy.STRUCTURAL_BEAM


def test_no_predictions_used_on_raw_or_fast_rejected_moves(monkeypatch):
    engine = engine_with_independent_wires()
    original = starting_plan()
    calls = []
    original_validate = engine.checker.validate
    def checked(plan, *args, **kwargs):
        calls.append(plan.signature)
        return original_validate(plan, *args, **kwargs)
    monkeypatch.setattr(engine.checker, 'validate', checked)
    result = SearchEngine(engine, config(), scorer(), limits=limits()).run(original)
    # Every candidate scored was first passed to the full independent checker.
    assert set(s.signature for s in result.trace).issubset(set(calls))
    assert result.n_scored == len(result.trace)


def test_no_router_or_physical_label_injected():
    engine = engine_with_independent_wires()
    result = SearchEngine(engine, config(obj='routed_depth'), scorer(), limits=limits()).run(starting_plan())
    assert result.to_dict()['stage'] == 'pre_routing_search_only'
    assert 'routed_depth_gain_pct' not in result.to_dict()


def test_scoring_budget_and_raw_budget_enforced():
    engine = engine_with_independent_wires()
    c = config(budget=2, max_depth=5)
    r = SearchEngine(engine, c, scorer(), limits=limits()).run(starting_plan())
    assert r.n_scored == 2
    assert r.stop_reason == 'valid_candidate_budget'
    c2 = config(budget=200, max_depth=5)
    r2 = SearchEngine(engine, c2, scorer(), limits=SearchLimits(
        per_parent_keep=10, max_attempts_per_operator=10, raw_attempt_budget=11)).run(starting_plan())
    assert r2.raw_attempts == 11
    assert r2.stop_reason == 'raw_attempt_budget'


def test_bad_predictions_fail_closed():
    for value in [float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='Nonfinite'):
            SearchEngine(engine_with_independent_wires(), config(),
                CallableScorer(lambda *args: value), limits=limits()).run(starting_plan())


def checkpoint(objective='swap_count'):
    return dict(cache_version='ridge_nb_loco_v1', objective=objective, heldout_circuit='x.qasm',
                training_circuits=['y.qasm'], columns=['first','second'],
                feature_mean=[0,0], feature_sd=[1,2], weights=[1,2], target_mean=0.0)


def test_ridge_checkpoint_math_and_objective_guard():
    o = starting_plan()
    new = ReusePlan.from_chains([[0,2],[1,3],[4]])
    f = lambda plan: {'first': 2.0 if plan == new else 0.0,
                      'second': 4.0 if plan == new else 0.0}
    scorer = RidgeDeltaScorer(checkpoint(), f, expected_objective='swap_count', heldout_circuit='x.qasm')
    assert scorer.score(o, new, Objective.SWAP_COUNT) == (6.0,)
    with pytest.raises(ValueError, match='different objective'):
        scorer.score(o, new, Objective.ROUTED_DEPTH)
    with pytest.raises(ValueError, match='does not match'):
        RidgeDeltaScorer(checkpoint(), f, expected_objective='routed_depth')
    with pytest.raises(ValueError, match='LOCO'):
        RidgeDeltaScorer(checkpoint(), f, expected_objective='swap_count', heldout_circuit='z.qasm')


def test_ridge_incompatible_feature_provider_rejected():
    z = RidgeDeltaScorer(checkpoint(), lambda plan: {'first': 1.0}, expected_objective='swap_count')
    with pytest.raises(ValueError, match='every checkpoint column'):
        z.score(starting_plan(), ReusePlan.from_chains([[0,2],[1,3],[4]]), Objective.SWAP_COUNT)
    bad = checkpoint()
    bad['feature_sd']=[0,1]
    with pytest.raises(ValueError, match='Invalid Ridge'):
        RidgeDeltaScorer(bad, lambda p: {}, expected_objective='swap_count')


def test_mismatched_score_dimensions_fail_closed():
    with pytest.raises(ValueError, match='score tuple'):
        _force_dimensions()


def _force_dimensions():
    # Simulates a broken scorer without wrapping it in CallableScorer.
    class Bad:
        def score(self, original, candidate, objective):
            return (1.0,) if candidate.chains[0][0] == 0 else (2.0, 3.0)
    SearchEngine(engine_with_independent_wires(), config(), Bad(), limits=limits()).run(starting_plan())


def test_limits_reject_bad_values():
    with pytest.raises(ValueError): SearchLimits(raw_attempt_budget=0)
    with pytest.raises(ValueError): SearchLimits(operators=('suffix_swap','suffix_swap'))


def test_model_free_random_control_reproducible():
    from qubit_reuse_repair import RandomUniformScorer
    assert RepairConfig(search_policy='random_uniform', model='none').model == 'none'
    with pytest.raises(ValueError):
        RepairConfig(search_policy='random_uniform', model='ridge')
    engine = engine_with_independent_wires()
    cfg = config(policy='random_uniform', obj='routed_depth', seed=222)
    x = SearchEngine(engine, cfg, None, limits=limits()).run(starting_plan())
    y = SearchEngine(engine, cfg, None, limits=limits()).run(starting_plan())
    assert x.to_dict() == y.to_dict()
    assert x.policy is SearchPolicy.RANDOM_UNIFORM
    assert x.changed
    with pytest.raises(TypeError):
        SearchEngine(engine, cfg, scorer())
