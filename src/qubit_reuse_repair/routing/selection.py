"""Frozen physical final selection: two-route guard, one-route predict-only,
and optional equal-budget Original Best-of-2 control.

Never use these observed routing costs as a predictor during Search or to
choose search policy on test circuits. This module only acts AFTER the pre-
routing search has frozen a proposed plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..compiler import MaterializationPair
from ..core import FinalSelection, Objective, ReusePlan
from .backend import MAX_SEED, RouteObservation, RoutingBackend, RoutingSettings, _seed


def _metric(obs: RouteObservation, objective: Objective) -> int:
    return getattr(obs, objective.value)


def _validated_route(backend: RoutingBackend, circuit: Any, seed: int) -> RouteObservation:
    value = backend.route(circuit, seed=seed)
    if not isinstance(value, RouteObservation):
        raise TypeError("Routing backend must return a RouteObservation (no silent fake metrics)")
    if value.seed != seed:
        raise ValueError("Routing backend returned the wrong seed")
    return value


@dataclass(frozen=True)
class OriginalBestOfTwoControl:
    """Control uses TWO Original routes, sharing the first with guarded mode."""

    second_original: RouteObservation
    best_original: RouteObservation
    control_gain_abs: int
    control_gain_pct: float | None
    guard_minus_control_abs: int

    def to_dict(self) -> dict:
        return {
            "second_original": self.second_original.to_dict(),
            "best_original": self.best_original.to_dict(),
            "control_gain_abs": self.control_gain_abs,
            "control_gain_pct": self.control_gain_pct,
            "guard_minus_control_abs": self.guard_minus_control_abs,
        }


@dataclass(frozen=True)
class PhysicalSelectionResult:
    original_plan: ReusePlan
    candidate_plan: ReusePlan
    selected_plan: ReusePlan
    objective: Objective
    mode: FinalSelection
    original_route: RouteObservation | None
    repaired_route: RouteObservation
    selected_route: RouteObservation
    selected_source: str
    actual_gain_abs: int | None
    actual_gain_pct: float | None
    routes_for_main_method: int
    original_best_of_two: OriginalBestOfTwoControl | None = None

    @property
    def selected_circuit(self) -> Any:
        return self.selected_route.circuit

    @property
    def changed(self) -> bool:
        return self.selected_plan != self.original_plan

    def to_dict(self) -> dict:
        return {
            "schema": "physical_selection_v1",
            "objective": self.objective.value,
            "selection_mode": self.mode.value,
            "search_selection_was_original": self.candidate_plan == self.original_plan,
            "selected_source": self.selected_source,
            "output_is_original_plan": not self.changed,
            "original_plan": self.original_plan.to_dict(),
            "candidate_plan": self.candidate_plan.to_dict(),
            "selected_plan": self.selected_plan.to_dict(),
            "routes_for_main_method": self.routes_for_main_method,
            "additional_control_routes": int(self.original_best_of_two is not None),
            "original_route": self.original_route.to_dict() if self.original_route is not None else None,
            "repaired_route": self.repaired_route.to_dict(),
            "selected_route": self.selected_route.to_dict(),
            "actual_gain_abs": self.actual_gain_abs,
            "actual_gain_pct": self.actual_gain_pct,
            "gain_pct_defined": self.actual_gain_pct is not None,
            "both_metrics_are_recorded": True,
            "zero_observed_harm_guarantee_scope": (
                "only the measured matched-seed routes in route_guarded mode"
                if self.mode is FinalSelection.ROUTE_GUARDED else None),
            "original_best_of_two_control": (
                self.original_best_of_two.to_dict()
                if self.original_best_of_two is not None else None),
        }


def select_routed_plan(pair: MaterializationPair, backend: RoutingBackend, *,
                       objective: Objective | str, mode: FinalSelection | str = FinalSelection.ROUTE_GUARDED,
                       evaluation_seed: int = 200,
                       control_second_seed: int | None = None) -> PhysicalSelectionResult:
    """Route only when needed; keep the *actual routed winner*, not a re-transpilation.

    route_guarded: Original and Repaired each routed on identical evaluation_seed.
    If the plans are identical, route Original once and reuse that observation.
    Tie-break always retains Original. Gain percent is undefined when Original
    cost is zero; no epsilon/positive-only exclusion is silently applied.

    predict_only: ONLY route the preselected candidate. Physical improvement
    cannot be inferred without routing Original; therefore gain is None.

    control_second_seed: only in guarded mode; one EXTRA Original routing is
    performed. Its baseline method uses 2 Original routes vs 2 main-method
    routes (when changed). It is never used to influence the main selection.
    """
    objective = Objective(objective)
    mode = FinalSelection(mode)
    evaluation_seed = _seed(evaluation_seed)
    if control_second_seed is not None:
        control_second_seed = _seed(control_second_seed)
        if mode is not FinalSelection.ROUTE_GUARDED:
            raise ValueError("Original Best-of-2 control requires route_guarded mode")
        if control_second_seed == evaluation_seed:
            raise ValueError("control_second_seed must differ from evaluation_seed")
    if pair.original.original_plan.active_width != pair.repaired.original_plan.active_width:
        raise ValueError("Original/Repaired active widths differ")
    if (pair.original.full_width != pair.repaired.full_width or
            pair.original.circuit.num_qubits != pair.repaired.circuit.num_qubits or
            pair.original.circuit.num_clbits != pair.repaired.circuit.num_clbits):
        raise ValueError("Original/Repaired compiled widths or clbits differ")
    orig_plan = pair.original.original_plan
    cand_plan = pair.repaired.original_plan
    if mode is FinalSelection.PREDICT_ONLY:
        rep = _validated_route(backend, pair.repaired.circuit, evaluation_seed)
        return PhysicalSelectionResult(orig_plan, cand_plan, cand_plan, objective, mode,
                None, rep, rep, 'repaired_prediction' if not pair.same_plan else 'unchanged',
                None, None, 1)

    orig = _validated_route(backend, pair.original.circuit, evaluation_seed)
    if pair.same_plan:
        rep = orig
        chosen = orig
        chosen_source = 'original_unchanged'
        routes_main = 1
    else:
        rep = _validated_route(backend, pair.repaired.circuit, evaluation_seed)
        # Lower is better. Equal costs mean Original is preferred.
        if _metric(rep, objective) < _metric(orig, objective):
            chosen = rep
            chosen_source = 'repaired'
        else:
            chosen = orig
            chosen_source = 'original_guarded'
        routes_main = 2
    base_cost = _metric(orig, objective)
    gain = base_cost - _metric(chosen, objective)
    if gain < 0:
        raise AssertionError("A route-guarded result increased its selected physical cost")
    gain_pct = 100.0 * gain / base_cost if base_cost else None
    control = None
    if control_second_seed is not None:
        extra = _validated_route(backend, pair.original.circuit, control_second_seed)
        best_orig = min((orig, extra), key=lambda obs: _metric(obs, objective))
        control_gain = base_cost - _metric(best_orig, objective)
        control = OriginalBestOfTwoControl(
            second_original=extra, best_original=best_orig,
            control_gain_abs=control_gain,
            control_gain_pct=100.0 * control_gain / base_cost if base_cost else None,
            guard_minus_control_abs=_metric(best_orig, objective) - _metric(chosen, objective),
        )
    return PhysicalSelectionResult(
        orig_plan, cand_plan,
        cand_plan if chosen is rep and not pair.same_plan else orig_plan,
        objective, mode, orig, rep, chosen, chosen_source,
        gain, gain_pct, routes_main, control,
    )
