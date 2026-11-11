"""Generator-agnostic, fixed-width qubit-reuse repair and physical selection."""
from .core import ReusePlan, CircuitSpec, RepairConfig, Objective, SearchPolicy, FinalSelection
from .validity import (ReuseValidityChecker, ValidationResult, CompatibilityIndex,
                       ValidityAwareEngine, PreparedPlan, ValidatedNeighbor, NeighborStats)
from .features import (StructuralFeatureExtractor, STRUCTURAL_COLUMNS, LEGACY_RIDGE_COLUMNS,
                       FrozenFeatureNormalizer, LegacyFeatureBridge)
from .adapters import (ExplicitChainsAdapter, GidNETChainsAdapter, CaQRChainsAdapter, DQCCChainsAdapter)
from .compiler import StaticPlanMaterializer, MaterializationResult, MaterializationPair
from .search import (SearchEngine, SearchLimits, SearchResult, SearchStep, search_plan,
                     StructuralScorer, CallableScorer, RidgeDeltaScorer, RandomUniformScorer)
from .routing import (RoutingSettings, RouteObservation, RoutingBackend, RoutingFailure,
                      QiskitSabreRouter, make_coupling_map, OriginalBestOfTwoControl,
                      PhysicalSelectionResult, select_routed_plan, RepairPipelineResult,
                      repair_and_route)

__version__ = "0.6.0"
__all__ = [
    "ReusePlan", "CircuitSpec", "RepairConfig", "Objective", "SearchPolicy", "FinalSelection",
    "ReuseValidityChecker", "ValidationResult", "CompatibilityIndex",
    "ValidityAwareEngine", "PreparedPlan", "ValidatedNeighbor", "NeighborStats",
    "SearchEngine", "SearchLimits", "SearchResult", "SearchStep", "search_plan",
    "StructuralScorer", "CallableScorer", "RidgeDeltaScorer", "RandomUniformScorer",
    "StructuralFeatureExtractor", "STRUCTURAL_COLUMNS", "LEGACY_RIDGE_COLUMNS",
    "FrozenFeatureNormalizer", "LegacyFeatureBridge",
    "StaticPlanMaterializer", "MaterializationResult", "MaterializationPair",
    "ExplicitChainsAdapter", "GidNETChainsAdapter", "CaQRChainsAdapter", "DQCCChainsAdapter",
    "RoutingSettings", "RouteObservation", "RoutingBackend", "RoutingFailure",
    "QiskitSabreRouter", "make_coupling_map", "OriginalBestOfTwoControl",
    "PhysicalSelectionResult", "select_routed_plan", "RepairPipelineResult", "repair_and_route",
]
