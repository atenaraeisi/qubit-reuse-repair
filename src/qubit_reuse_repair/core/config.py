"""Experiment-safe repair configuration: one objective at a time."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Objective(str, Enum):
    SWAP_COUNT = "swap_count"
    ROUTED_DEPTH = "routed_depth"


class FinalSelection(str, Enum):
    PREDICT_ONLY = "predict_only"
    ROUTE_GUARDED = "route_guarded"


class SearchPolicy(str, Enum):
    RANDOM_BEAM = "random_beam"
    ADAPTIVE_BEAM = "adaptive_beam"
    STRUCTURAL_BEAM = "structural_beam"
    RANDOM_UNIFORM = "random_uniform"


@dataclass(frozen=True)
class RepairConfig:
    objective: Objective = Objective.SWAP_COUNT
    search_policy: SearchPolicy = SearchPolicy.RANDOM_BEAM
    model: str = "ridge"
    max_depth: int = 5
    budget: int = 120
    beam_width: int = 5
    final_selection: FinalSelection = FinalSelection.ROUTE_GUARDED
    search_seed: int = 2026

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective", Objective(self.objective))
        object.__setattr__(self, "search_policy", SearchPolicy(self.search_policy))
        object.__setattr__(self, "final_selection", FinalSelection(self.final_selection))
        if self.model not in ("ridge", "none"):
            raise ValueError("model must be 'ridge' or 'none'")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (self.max_depth, self.budget, self.beam_width)):
            raise ValueError("depth, budget, and beam_width must be positive integers")
        if isinstance(self.search_seed, bool) or not isinstance(self.search_seed, int):
            raise ValueError("search_seed must be an integer")
        if self.search_policy in (SearchPolicy.STRUCTURAL_BEAM, SearchPolicy.RANDOM_UNIFORM) and self.model != "none":
            raise ValueError("structural_beam/random_uniform require model='none'")
        if self.search_policy in (SearchPolicy.RANDOM_BEAM, SearchPolicy.ADAPTIVE_BEAM) and self.model != "ridge":
            raise ValueError("historical random/adaptive beam currently require model='ridge'")
