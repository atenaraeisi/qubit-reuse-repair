"""Raw 25-feature provider for the common static materializer.

No per-circuit normalization is fitted here. Missing structural values are
returned as NaN for model-specific TRAIN-median imputation. This reproduces
the documented schema, but independent parity with the trainer's source
feature implementation requires reference examples from model training.
"""
from __future__ import annotations

import math
from typing import Any

from ..core import ReusePlan
from ..compiler.static import StaticPlanMaterializer
from .structural import LEGACY_RIDGE_COLUMNS, StructuralFeatureExtractor
from .legacy_reset import normalize_reset_for_depth


class MaterializerRawFeatureProvider:
    FEATURE_PROTOCOL = 'common_materializer_raw25_v1'

    def __init__(self, source_circuit: Any, *, materializer=None, extractor=None):
        self.extractor = extractor or StructuralFeatureExtractor(source_circuit)
        self.materializer = materializer or StaticPlanMaterializer(source_circuit)
        self._cache: dict[str, dict[str, float]] = {}

    def __call__(self, plan: ReusePlan) -> dict[str, float]:
        if plan.signature not in self._cache:
            row = self.extractor.extract(plan)
            dynamic = self.materializer.materialize(plan).circuit
            normalized = normalize_reset_for_depth(dynamic)
            row['logical_depth'] = float(dynamic.depth() or 0)
            row['norm_depth'] = float(normalized.depth() or 0)
            if any(c not in row for c in LEGACY_RIDGE_COLUMNS):
                raise ValueError('Common raw materializer extractor missing named features')
            self._cache[plan.signature] = {k:float(row[k]) for k in LEGACY_RIDGE_COLUMNS}
        return dict(self._cache[plan.signature])
