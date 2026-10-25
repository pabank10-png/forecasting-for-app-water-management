"""ARX: one linear equation per hour ahead (least squares), predicting the change in level
from recent levels, upstream levels, recent rain and the rain still to fall up to that hour."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.base import interval_from_errors
from common.river import STEPS, river_builder


class Model:
    def __init__(self, upstream=(), horizon=48, steps=STEPS, ridge=1e-3):
        self.builder = river_builder(upstream)
        self.horizon, self.steps, self.ridge = horizon, [s for s in steps if s <= horizon], ridge

    def _X(self, f, togo):
        z = np.nan_to_num(f.to_numpy(float) - self.mean)  # missing input = its average
        return np.column_stack([np.ones(len(f)), z, togo])

    def fit(self, df):
        f = self.builder(df)
        self.mean = np.nanmean(f.to_numpy(float), 0)
        base = f["y_lag1"].to_numpy(float)
        rain = df["rain"].fillna(0)
        self.coef, self.errors = {}, {}
        for h in self.steps:
            togo = rain[::-1].rolling(h, min_periods=1).sum()[::-1].to_numpy()
            t = df["y"].shift(-(h - 1)).to_numpy(float) - base
            ok = ~np.isnan(t)
            A = self._X(f, togo)[ok]
            w = np.linalg.solve(A.T @ A + self.ridge * np.eye(A.shape[1]), A.T @ t[ok])
            self.coef[h] = w
            self.errors[h] = t[ok] - A @ w
        return self

    def predict(self, df, horizon=48, future=None):
        f = self.builder(df).iloc[[-1]]
        base = float(f["y_lag1"].iloc[0])
        r = np.nan_to_num(future["rain"].to_numpy(float)) if future is not None and "rain" in future else np.zeros(1)
        pts = {h: base + float((self._X(f, [r[:h].sum()]) @ w)[0]) for h, w in self.coef.items()}
        ks = sorted(pts)
        p50 = np.interp(np.arange(1, horizon + 1), ks, [pts[k] for k in ks])
        err = [self.errors[min([k for k in ks if k >= h] or [ks[-1]])] for h in range(1, horizon + 1)]
        return interval_from_errors(p50, err, floor=None)
