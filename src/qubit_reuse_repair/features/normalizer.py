"""Frozen unlabeled, per-circuit feature standardization (Experiment 4D style).

Mean/std computed on a *declared, frozen candidate/reference plan pool*, not
updated on-the-fly during search. Thus search order cannot alter predictions.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence
from ..core import ReusePlan
from .structural import LEGACY_RIDGE_COLUMNS, StructuralFeatureExtractor


@dataclass(frozen=True)
class FrozenFeatureNormalizer:
    columns: tuple[str, ...]
    mean: tuple[float, ...]
    sd: tuple[float, ...]
    reference_signatures: tuple[str, ...]
    fingerprint: str

    @classmethod
    def fit(cls, rows: Mapping[str, Mapping[str, float]], columns: Sequence[str]):
        """rows keyed by canonical plan signature, never labels; ddof=0.

        Mimics historical fillna(0) *after* standardization: NaN -> z=0.
        """
        if not rows:
            raise ValueError('A nonempty, explicitly frozen reference pool is required')
        cols = tuple(columns)
        if not cols or len(set(cols)) != len(cols):
            raise ValueError('Missing/duplicate feature columns')
        keys = tuple(sorted(rows))
        if any(set(cols) - set(rows[k]) for k in keys):
            raise ValueError('Feature row is missing a checkpoint column')
        means, sds = [], []
        for col in cols:
            vals = [float(rows[k][col]) for k in keys]
            finite = [v for v in vals if math.isfinite(v)]
            if not finite:
                # Matches Experiment 4D all-NaN normalized-to-zero semantics.
                means.append(math.nan); sds.append(math.nan)
                continue
            mu = sum(finite)/len(finite)
            sd = (sum((v-mu)**2 for v in finite)/len(finite))**0.5
            means.append(mu); sds.append(sd if sd != 0 else math.nan)
        payload = {'columns': cols, 'reference_signatures': keys,
                   'mean': [None if math.isnan(x) else x for x in means],
                   'sd': [None if math.isnan(x) else x for x in sds]}
        fingerprint = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        return cls(cols,tuple(means),tuple(sds),keys,fingerprint)

    def transform(self, row: Mapping[str, float]) -> dict[str, float]:
        if set(self.columns) - set(row):
            raise ValueError('Missing feature in normalization input')
        out = {}
        for col, mu, sd in zip(self.columns, self.mean, self.sd):
            v = float(row[col])
            out[col] = (v-mu)/sd if math.isfinite(v) and math.isfinite(mu) and math.isfinite(sd) and sd>0 else 0.0
        return out


class LegacyFeatureBridge:
    """Historical 25-column input for RidgeDeltaScorer, with explicit compile depths.

    compile_depths(plan) -> (logical_depth, norm_depth). The caller MUST use
    the same dynamic compiler and normalization recipe for reference pool and
    live candidates. Neither score nor normalization uses physical routing.
    """
    def __init__(self, extractor: StructuralFeatureExtractor,
                 compile_depths: Callable[[ReusePlan], tuple[float, float]],
                 reference_plans: Sequence[ReusePlan],
                 *, required_columns: Sequence[str] = LEGACY_RIDGE_COLUMNS):
        if not callable(compile_depths):
            raise TypeError('Exact dynamic compilation depth provider is required')
        self.extractor = extractor
        self.compile_depths = compile_depths
        self.columns = tuple(required_columns)
        if set(self.columns) - set(LEGACY_RIDGE_COLUMNS):
            raise ValueError('Unsupported columns for legacy 25-feature model')
        self.raw_cache: dict[str, dict[str, float]] = {}
        plans = {p.signature: p for p in reference_plans}
        if not plans:
            raise ValueError('Cannot standardize without frozen reference plans')
        rows = {sig:self.raw(p) for sig,p in plans.items()}
        self.normalizer = FrozenFeatureNormalizer.fit(rows,self.columns)

    def raw(self, plan: ReusePlan) -> dict[str, float]:
        if plan.signature not in self.raw_cache:
            row = self.extractor.extract(plan)
            pair = self.compile_depths(plan)
            if not isinstance(pair, (tuple, list)) or len(pair)!=2:
                raise ValueError('compile_depths must return exactly two depths')
            depth, norm = (float(x) for x in pair)
            if not all(math.isfinite(x) and x >= 0 for x in (depth,norm)):
                raise ValueError('Non-finite/negative dynamic circuit depths')
            row.update(logical_depth=depth,norm_depth=norm)
            self.raw_cache[plan.signature]=row
        return dict(self.raw_cache[plan.signature])

    def __call__(self, plan: ReusePlan) -> dict[str, float]:
        return self.normalizer.transform(self.raw(plan))
