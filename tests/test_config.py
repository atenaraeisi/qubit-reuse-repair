import pytest
from qubit_reuse_repair import RepairConfig, Objective, FinalSelection, SearchPolicy


def test_two_separate_objectives_and_final_guard():
    x = RepairConfig(objective="swap_count", final_selection="route_guarded")
    y = RepairConfig(objective="routed_depth", final_selection="predict_only")
    assert x.objective is Objective.SWAP_COUNT
    assert y.objective is Objective.ROUTED_DEPTH
    assert x.final_selection is FinalSelection.ROUTE_GUARDED
    assert y.final_selection is FinalSelection.PREDICT_ONLY
    assert x.budget == 120 and x.max_depth == 5


def test_reject_unimplemented_composite():
    with pytest.raises(ValueError):
        RepairConfig(objective="both")


def test_supported_search_contract():
    assert RepairConfig(search_policy="structural_beam", model="none").search_policy is SearchPolicy.STRUCTURAL_BEAM
    with pytest.raises(ValueError):
        RepairConfig(search_policy="structural_beam", model="ridge")
    with pytest.raises(ValueError):
        RepairConfig(model="none", search_policy="random_beam")
    with pytest.raises(ValueError):
        RepairConfig(budget=0)
