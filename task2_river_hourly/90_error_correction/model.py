"""Error correction (AR(1) data assimilation) around any task-2 model.

e(t) = y(t) - yhat(t), where yhat(t) is the base model's 1-hour forecast made an
hour earlier. Forecast: yhat_corr(t+h) = yhat(t+h) + phi^h * e(t). phi is fitted
on past origins (least squares over all steps). P10/P90 shift with P50.

    Model(base="05_lightgbm", upstream=("up1",)).fit(df).predict(df, 48, future)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.base import result
from common.registry import load

TASK = "task2_river_hourly"


class Model:
    def __init__(self, base="05_lightgbm", horizon=48, n_origins=150, every=6, window=400, **base_kw):
        self.make = base if callable(base) else (lambda: load(f"{TASK}/{base}").Model(**base_kw))
        self.horizon, self.n_origins, self.every, self.window = horizon, n_origins, every, window

    def _cut(self, df, i):
        """df rows up to i, row i becomes the forecast row."""
        h = df.iloc[max(0, i - self.window):i + 1].copy()
        h.loc[h.index[-1], "y"] = np.nan
        return h.reset_index(drop=True)

    def _future(self, df, i, n):
        cols = [c for c in df.columns if c not in ("y", "date")]
        return df.iloc[i:i + n][cols].reset_index(drop=True)

    def last_error(self, df, base, n):
        """y at row n-1 minus base's 1-hour forecast of it (made from rows before n-1)."""
        y_obs = df["y"].iloc[n - 1]
        f = base.predict(self._cut(df, n - 1), 1, self._future(df, n - 1, 1))
        return float(y_obs - f["p50"].iloc[0])

    def fit(self, df):
        y = df["y"].to_numpy(float)
        n_obs = int(np.where(~np.isnan(y))[0][-1]) + 1
        first = n_obs - self.horizon - self.every * self.n_origins
        origins = [i for i in range(max(self.window, first), n_obs - self.horizon, self.every)]
        cal = self.make().fit(df.iloc[:origins[0]]) if origins else None
        e, E = [], []
        for i in origins:
            ei = self.last_error(df, cal, i)
            fc = cal.predict(self._cut(df, i), self.horizon, self._future(df, i, self.horizon))
            e.append(ei)
            E.append(y[i:i + self.horizon] - fc["p50"].to_numpy())
        e, E = np.array(e), np.array(E)
        hs = np.arange(1, self.horizon + 1)
        grid = np.linspace(0, 0.99, 100)
        cost = [np.nanmean((E - (p ** hs)[None, :] * e[:, None]) ** 2) for p in grid] if len(e) else [0]
        self.phi = float(grid[int(np.argmin(cost))]) if len(e) else 0.0
        self.base = self.make().fit(df)
        return self

    def predict(self, df, horizon=48, future=None):
        n = len(df) - 1 if np.isnan(df["y"].iloc[-1]) else len(df)
        fc = self.base.predict(df, horizon, future)
        self.e = self.last_error(df, self.base, n)
        c = self.phi ** np.arange(1, horizon + 1) * self.e
        return result(fc["p10"] + c, fc["p50"] + c, fc["p90"] + c, floor=None)
