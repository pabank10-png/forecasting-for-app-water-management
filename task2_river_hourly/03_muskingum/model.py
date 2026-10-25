"""Muskingum routing: downstream discharge from the upstream (or reservoir release) hydrograph.

y = downstream discharge (m3/s), column `inflow` = upstream discharge (m3/s).
Future inflow (e.g. spill forecast from task 1B, or an upstream forecast)
goes in `future[inflow]`. Convert levels <-> discharge with
common.routing.fit_rating / level / discharge.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.base import interval_from_errors
from common.routing import fit_muskingum, muskingum


class Model:
    def __init__(self, inflow="inflow", dt=1.0, horizon=48, local=True):
        self.inflow, self.dt, self.horizon, self.local = inflow, dt, horizon, local

    def fit(self, df):
        d = df[[self.inflow, "y"]].interpolate(limit=6).dropna()
        I, O = d[self.inflow].to_numpy(float), d["y"].to_numpy(float)
        self.K, self.X = fit_muskingum(I, O, self.dt)
        sim = muskingum(I, self.K, self.X, self.dt, O[0])
        self.lateral = float(np.median(O - sim)) if self.local else 0.0  # local inflow between the stations
        res = O - sim - self.lateral
        self.errors = [res[h:] - res[:-h] if h < len(res) else res for h in range(1, self.horizon + 1)]
        return self

    def predict(self, df, horizon=48, future=None):
        n = len(df) - 1
        I = df[self.inflow].to_numpy(float)[:n]
        fut = future[self.inflow].to_numpy(float)[:horizon] if future is not None and self.inflow in future             else np.full(horizon, I[-1])
        I = np.concatenate([[I[-1]], fut])
        o0 = df["y"].to_numpy(float)[n - 1] - self.lateral
        p50 = muskingum(I, self.K, self.X, self.dt, o0)[1:] + self.lateral
        return interval_from_errors(p50, self.errors)
