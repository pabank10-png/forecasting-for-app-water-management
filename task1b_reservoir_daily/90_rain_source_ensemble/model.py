"""Run one inflow model once per forecast-rain source (ECMWF, GFS, ICON, JMA, TMD) and combine.

    m = Model(base="05_lightgbm_daily").fit(df)
    fc = m.predict(df, 7, {"ECMWF": fut1, "GFS": fut2, ...})   # one future frame per source
    m.overflow_probability(storage_now, capacity, outflow_per_day)

Weights: inverse MAE of each source's past forecasts (set_source_mae), else equal.
P10 / P90 of the result = the widest of (weighted P10/P90, spread between sources).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common import ensemble
from common.base import result
from common.registry import load

TASK = "task1b_reservoir_daily"


class Model:
    def __init__(self, base="05_lightgbm_daily", shrink=0.5, **base_kw):
        self.base = base if not isinstance(base, str) else load(f"{TASK}/{base}").Model(**base_kw)
        self.shrink, self.source_mae = shrink, None

    def set_source_mae(self, maes):
        self.source_mae = maes
        return self

    def fit(self, df):
        self.base.fit(df)
        return self

    def predict(self, df, horizon=7, future=None):
        if not isinstance(future, dict):
            return self.base.predict(df, horizon, future)
        self.members = {s: self.base.predict(df, horizon, f) for s, f in future.items()}
        self.weights = ensemble.inverse_mae_weights({s: self.source_mae[s] for s in self.members}, self.shrink)             if self.source_mae else {s: 1 / len(self.members) for s in self.members}
        c = ensemble.combine(self.members, self.weights)
        p50s = np.array([m["p50"].to_numpy() for m in self.members.values()])
        lo = np.minimum(c["p10"].to_numpy(), p50s.min(0))
        hi = np.maximum(c["p90"].to_numpy(), p50s.max(0))
        return result(lo, c["p50"], hi)

    def overflow_probability(self, storage, capacity, outflow_per_day=0.0):
        """Weighted share of rain sources whose P50 inflow fills the reservoir within the horizon."""
        end = {s: storage + np.cumsum(m["p50"].to_numpy() - outflow_per_day).max() for s, m in self.members.items()}
        return ensemble.exceed_probability(end, self.weights, capacity)
