"""Pre-routing, frozen legacy-style feature bridge for exploratory pilots.

Uses pandas per-circuit mean/std(ddof=0), matching the historical normalization
convention. A deterministic reference pool is frozen BEFORE search. Replacing
historical GidNET candidate pools or compiler changes the model input
statistics; the caller must report this as distribution shift.
"""
from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..core import ReusePlan
from .normalizer import LegacyFeatureBridge
from .structural import LEGACY_RIDGE_COLUMNS, StructuralFeatureExtractor


@dataclass(frozen=True)
class PandasFrozenNormalizer:
    columns: tuple[str, ...]
    mean: tuple[float, ...]
    sd: tuple[float, ...]
    reference_signatures: tuple[str, ...]
    fingerprint: str

    @classmethod
    def fit(cls, rows: Mapping[str, Mapping[str, float]], columns=LEGACY_RIDGE_COLUMNS):
        import pandas as pd
        if len(rows) < 2:
            raise ValueError('At least 2 distinct reference plans required for learned scoring')
        cols = tuple(columns)
        keys = tuple(sorted(rows))
        df = pd.DataFrame([{c:float(rows[k][c]) for c in cols} for k in keys],columns=cols)
        means = df.mean(axis=0)
        sds = df.std(axis=0,ddof=0).replace(0, float('nan'))
        # Finite positive SD is not guaranteed for flat / all-NaN columns.
        mu = tuple(float(means[c]) for c in cols)
        sd = tuple(float(sds[c]) for c in cols)
        hash_payload = {'cols':cols,'keys':keys,
             'mean':[x if math.isfinite(x) else None for x in mu],
             'sd':[x if math.isfinite(x) else None for x in sd]}
        fp=hashlib.sha256(json.dumps(hash_payload,sort_keys=True).encode()).hexdigest()
        return cls(cols,mu,sd,keys,fp)

    def transform(self,row):
        out={}
        for c,mu,sd in zip(self.columns,self.mean,self.sd):
            v=float(row[c])
            out[c]=(v-mu)/sd if all(map(math.isfinite,(v,mu,sd))) and sd>0 else 0.0
        return out


class PilotFeatureBridge(LegacyFeatureBridge):
    def __init__(self, extractor:StructuralFeatureExtractor, compile_depths,
                 reference_plans:Sequence[ReusePlan]):
        super().__init__(extractor,compile_depths,reference_plans,
                         required_columns=LEGACY_RIDGE_COLUMNS)
        self.normalizer=PandasFrozenNormalizer.fit(self.raw_cache, self.columns)
