"""Travel time: water seen upstream arrives downstream about tau hours later.

tau = lag with the highest correlation between upstream and downstream hourly
changes; a = how much of the upstream change shows up downstream.
level_down(t+h) = level_down(t) + a * (up(t+h-tau) - up(t-tau)).
Beyond tau hours ahead it needs an upstream forecast in `future[upstream]`
(otherwise upstream is held at its last value).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.base import interval_from_errors


class Model:
    def __init__(self, upstream="up1", max_lag=24, horizon=48):
        self.up, self.max_lag, self.horizon = upstream, max_lag, horizon

    def _pred(self, y, u, t, h):
        a = u[t + h - self.tau] if t + h - self.tau < len(u) else u[-1]
        return y[t] + self.a * (a - u[t - self.tau])

    def fit(self, df):
        y, u = df["y"].to_numpy(float), df[self.up].to_numpy(float)
        dy, du = np.diff(y), np.diff(u)
        best = (-2, 1)
        for lag in range(0, self.max_lag + 1):
            a, b = du[:len(du) - lag], dy[lag:]
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() > 30:
                r = np.corrcoef(a[ok], b[ok])[0, 1]
                best = max(best, (r, lag))
        self.corr, self.tau = best
        k = max(self.tau, 1)
        a, b = u[k:] - u[:-k], y[k:] - y[:-k]
        b = np.roll(b, -self.tau)[:len(a) - self.tau] if self.tau else b
        a = a[:len(b)]
        ok = ~(np.isnan(a) | np.isnan(b))
        self.a = float(np.sum(a[ok] * b[ok]) / max(np.sum(a[ok] ** 2), 1e-9))
        n = len(y)
        ts = np.arange(self.tau + 2, n - 1)[-3000::3]
        self.errors = [np.array([y[t + h] - self._pred(y[:t + 1], u[:t + 1], t, h) for t in ts if t + h < n])
                       for h in range(1, self.horizon + 1)]
        return self

    def predict(self, df, horizon=48, future=None):
        n = len(df) - 1
        y = df["y"].to_numpy(float)[:n]
        u = df[self.up].to_numpy(float)[:n]
        if future is not None and self.up in future:
            u = np.concatenate([u, future[self.up].to_numpy(float)])
        t = n - 1
        p50 = [self._pred(y, u, t, h) for h in range(1, horizon + 1)]
        return interval_from_errors(p50, self.errors, floor=None)
