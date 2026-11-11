"""End-to-end, explicit-plan static repair -> select -> route.

Ridge remains opt-in and requires a feature-compatible scorer. This does not
parse native CaQR/DQCC formats or claim semantic equivalence in general.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from ..compiler import StaticPlanMaterializer
from ..core import RepairConfig, ReusePlan
from ..search import SearchLimits, SearchResult, search_plan
from ..search.scorers import PlanScorer
from .backend import RoutingBackend
from .selection import PhysicalSelectionResult, select_routed_plan


@dataclass(frozen=True)
class RepairPipelineResult:
    search: SearchResult
    physical: PhysicalSelectionResult
    search_time_sec: float
    materialization_time_sec: float
    physical_selection_time_sec: float

    @property
    def final_plan(self) -> ReusePlan:
        return self.physical.selected_plan

    @property
    def final_routed_circuit(self) -> Any:
        return self.physical.selected_circuit

    def to_dict(self) -> dict:
        return {
            "schema": "generator_neutral_repair_and_route_v1",
            "search": self.search.to_dict(),
            "physical": self.physical.to_dict(),
            "timing": {
                "search_time_sec": self.search_time_sec,
                "materialization_time_sec": self.materialization_time_sec,
                "physical_selection_time_sec": self.physical_selection_time_sec,
            },
            "warning": "static input subset only; no general semantic-equivalence guarantee",
        }


def repair_and_route(source_circuit: Any, original_plan: ReusePlan, *,
                     config: RepairConfig, backend: RoutingBackend,
                     scorer: PlanScorer | None = None,
                     limits: SearchLimits | None = None,
                     evaluation_seed: int = 200,
                     control_second_seed: int | None = None) -> RepairPipelineResult:
    """Generator-neutral API with explicit common ordered-chain reuse plan.

    Structural search has NO access to measured routing values. The plan is
    frozen before compiling and routing. All materialized output uses the
    same new materializer and must be statically valid and fixed width.
    """
    t0 = time.perf_counter()
    sr = search_plan(source_circuit, original_plan, config, scorer=scorer, limits=limits)
    t1 = time.perf_counter()
    materializer = StaticPlanMaterializer(source_circuit)
    pair = materializer.materialize_pair(sr.original, sr.selected)
    t2 = time.perf_counter()
    physical = select_routed_plan(pair, backend, objective=config.objective,
        mode=config.final_selection, evaluation_seed=evaluation_seed,
        control_second_seed=control_second_seed)
    t3 = time.perf_counter()
    return RepairPipelineResult(search=sr, physical=physical,
        search_time_sec=t1 - t0, materialization_time_sec=t2 - t1,
        physical_selection_time_sec=t3 - t2)
