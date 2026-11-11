"""Frozen TRAIN_ONLY 25-delta Ridge inference; no training or routing during scoring.

This consumes the four ml_ridge_multistep25_v1/*TRAIN_ONLY.json checkpoints.
Preliminary exploratory policy only: frozen model ≠ independently validated
performance, and a frozen per-circuit reference pool must be specified/recorded.
"""
from __future__ import annotations
import json
from pathlib import Path
import math
from typing import Mapping, Callable
from ..core import ReusePlan, Objective
from ..features.structural import LEGACY_RIDGE_COLUMNS


class TrainOnlyDeltaRidge:
    def __init__(self, checkpoint: dict, feature_provider: Callable, *, objective: str):
        obj = Objective(objective)
        if checkpoint.get('protocol') != 'fixed_train_family_cv_25delta_ridge_v1':
            raise ValueError('Not the supported frozen train-only checkpoint')
        if checkpoint.get('objective') != obj.value:
            raise ValueError('Model objective mismatch')
        if checkpoint.get('scope', '').upper().startswith('TRAIN ONLY') is False:
            raise ValueError('Expected train-only checkpoint')
        columns = tuple(checkpoint['raw_feature_order'])
        if columns != tuple(LEGACY_RIDGE_COLUMNS):
            raise ValueError('Incorrect feature names/order')
        if tuple(checkpoint['feature_names']) != tuple('d_' + x for x in columns):
            raise ValueError('Checkpoint delta feature schema mismatch')
        self.columns = columns
        self.mu = tuple(float(x) for x in checkpoint['feature_mean'])
        self.sd = tuple(float(x) for x in checkpoint['feature_sd'])
        self.weights = tuple(float(x) for x in checkpoint['weights'])
        self.intercept = float(checkpoint['intercept'])
        if any(len(x) != len(columns) for x in (self.mu, self.sd, self.weights)):
            raise ValueError('Checkpoint vector width mismatch')
        if (not all(math.isfinite(x) for s in (self.mu, self.sd, self.weights) for x in s)
                or any(x <= 0 for x in self.sd) or not math.isfinite(self.intercept)):
            raise ValueError('Invalid checkpoint coefficients')
        if not callable(feature_provider):
            raise TypeError('Must supply frozen, per-circuit standardized features')
        self.feature_provider = feature_provider
        self.objective = obj
        self.arm = checkpoint['arm']
        self.training_circuits = tuple(checkpoint.get('training_circuits', []))
        self._cache = {}

    @classmethod
    def from_file(cls, path, feature_provider, *, objective):
        return cls(json.loads(Path(path).read_text(encoding='utf-8-sig')),
                   feature_provider, objective=objective)

    def score(self, original: ReusePlan, candidate: ReusePlan, objective: Objective):
        if Objective(objective) != self.objective:
            raise ValueError('Model cannot score the other physical objective')
        def get(p):
            if p.signature not in self._cache:
                row = self.feature_provider(p)
                v = tuple(float(row[x]) for x in self.columns)
                if not all(math.isfinite(x) for x in v):
                    raise ValueError('Nonfinite standardized features')
                self._cache[p.signature] = v
            return self._cache[p.signature]
        before, after = get(original), get(candidate)
        delta = (a-b for a, b in zip(after, before))
        gain = self.intercept + sum(w * ((d-m)/s) for d,m,s,w in zip(delta,self.mu,self.sd,self.weights))
        if not math.isfinite(gain):
            raise ValueError('Nonfinite predicted gain')
        return (float(gain),)
