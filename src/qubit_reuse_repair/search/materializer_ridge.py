"""Inference for the frozen *raw-delta* materializer-trained Ridge checkpoints.

Unlike the historical train-only model, this model consumes RAW, per-plan
features, imputes missing values with TRAIN-only medians, computes the
candidate-minus-original difference, then uses TRAIN-only delta mean/std.

No routing measurements or candidate labels are used during inference.
The feature extraction path must be checked for exact training parity separately.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from ..core import Objective, ReusePlan
from ..features.structural import LEGACY_RIDGE_COLUMNS


class MaterializerDeltaRidge:
    PROTOCOL = 'materializer25_rawdelta_group_ridge_v1'
    FEATURE_PROTOCOL = 'common_materializer_raw25_v1'

    def __init__(self, checkpoint: Mapping[str, Any], feature_provider: Callable,
                 *, objective: Objective | str):
        self.objective = Objective(objective)
        if checkpoint.get('protocol') != self.PROTOCOL:
            raise ValueError('Unsupported materializer Ridge protocol')
        if checkpoint.get('feature_protocol') != self.FEATURE_PROTOCOL:
            raise ValueError('Unsupported materializer feature protocol')
        if checkpoint.get('objective') != self.objective.value:
            raise ValueError('Ridge checkpoint objective mismatch')
        if checkpoint.get('arm') != 'one_step' or checkpoint.get('train_only') is not True:
            raise ValueError('Expected a one_step TRAIN-only Ridge checkpoint')
        if not callable(feature_provider):
            raise TypeError('feature_provider must return raw per-plan feature mappings')
        columns = tuple(checkpoint['raw_feature_order'])
        if columns != LEGACY_RIDGE_COLUMNS:
            raise ValueError('Checkpoint raw feature names/order differ from extractor')
        if tuple(checkpoint['feature_names']) != tuple('d_' + x for x in columns):
            raise ValueError('Checkpoint delta feature names/order mismatch')
        self.raw_impute = tuple(float(x) for x in checkpoint['raw_impute'])
        self.mean = tuple(float(x) for x in checkpoint['feature_mean'])
        self.std = tuple(float(x) for x in checkpoint['feature_sd'])
        self.weights = tuple(float(x) for x in checkpoint['weights'])
        self.intercept = float(checkpoint['intercept'])
        if any(len(x) != len(columns) for x in (self.raw_impute, self.mean, self.std, self.weights)):
            raise ValueError('Checkpoint vector length mismatch')
        if any(not math.isfinite(x) for arr in
               (self.raw_impute, self.mean, self.std, self.weights) for x in arr):
            raise ValueError('Checkpoint contains nonfinite numbers')
        if any(sd <= 0 for sd in self.std) or not math.isfinite(self.intercept):
            raise ValueError('Checkpoint contains invalid scaling/intercept')
        self.columns = columns
        self.feature_provider = feature_provider
        self.training_circuits = tuple(checkpoint.get('training_circuits', ()))
        self._cache: dict[str, tuple[float, ...]] = {}

    @classmethod
    def from_file(cls, path: str | Path, feature_provider: Callable,
                  *, objective: Objective | str) -> 'MaterializerDeltaRidge':
        checkpoint = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return cls(checkpoint, feature_provider, objective=objective)

    def raw_vector(self, plan: ReusePlan) -> tuple[float, ...]:
        """Return the fixed-median-imputed 25 raw features, cached by plan."""
        if plan.signature not in self._cache:
            row = self.feature_provider(plan)
            if not isinstance(row, Mapping) or any(k not in row for k in self.columns):
                raise ValueError('Raw feature provider missing checkpoint features')
            out = []
            for feature, median in zip(self.columns, self.raw_impute):
                value = row[feature]
                if value is None:
                    number = float('nan')
                else:
                    number = float(value)
                if math.isinf(number):
                    raise ValueError(f'Raw feature {feature} is infinite')
                out.append(median if math.isnan(number) else number)
            self._cache[plan.signature] = tuple(out)
        return self._cache[plan.signature]

    def delta_z(self, original: ReusePlan, candidate: ReusePlan) -> tuple[float, ...]:
        before, after = self.raw_vector(original), self.raw_vector(candidate)
        return tuple((new-old-mean)/sd for old,new,mean,sd in
                     zip(before,after,self.mean,self.std))

    def score(self, original: ReusePlan, candidate: ReusePlan,
              objective: Objective | str) -> tuple[float, ...]:
        if Objective(objective) != self.objective:
            raise ValueError('Ridge checkpoint cannot score the other physical objective')
        z = self.delta_z(original, candidate)
        predicted_percent_gain = self.intercept + sum(w*v for w,v in zip(self.weights,z))
        if not math.isfinite(predicted_percent_gain):
            raise ValueError('Nonfinite predicted percentage gain')
        return (predicted_percent_gain,)
