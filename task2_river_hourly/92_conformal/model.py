"""Conformal (CQR) wrapper for task-2 models: calibrate P10-P90 to cover about 80% of actual levels.

How: fit the base model once on data before the calibration window, forecast
from origins every `every` hours through that window, measure how far actual
levels fell outside the band per hour ahead (q-hat), then refit on all data
and widen (or narrow) the band by q-hat.

    Model(base="05_lightgbm", upstream=("up1",)).fit(df).predict(df, 48, future)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common import conformal
from common.registry import load

TASK = "task2_river_hourly"


class Model:
    def __init__(self, base="05_lightgbm", horizon=48, calib_hours=24 * 90, every=6, window=400,
                 coverage=0.8, **base_kw):
        self.make = base if callable(base) else (lambda: load(f"{TASK}/{base}").Model(**base_kw))
        self.horizon, self.calib, self.every, self.window, self.coverage = horizon, calib_hours, every, window, coverage

    def fit(self, df):
        y = df["y"].to_numpy(float)
        n_obs = int(np.where(~np.isnan(y))[0][-1]) + 1
        start = max(self.window, n_obs - self.calib - self.horizon)
        cal = self.make().fit(df.iloc[:start])
        cols = [c for c in df.columns if c not in ("y", "date")]
        s = []
        for i in range(start, n_obs - self.horizon, self.every):
            h = df.iloc[i - self.window:i + 1].copy().reset_index(drop=True)
            h.loc[h.index[-1], "y"] = np.nan
            fc = cal.predict(h, self.horizon, df.iloc[i:i + self.horizon][cols].reset_index(drop=True))
            s.append(conformal.scores(y[i:i + self.horizon], fc["p10"], fc["p90"]))
        s = np.array(s)
        self.q = np.array([conformal.qhat(s[:, k], self.coverage) for k in range(self.horizon)])
        self.base = self.make().fit(df)
        return self

    def predict(self, df, horizon=48, future=None):
        fc = self.base.predict(df, horizon, future)
        return conformal.apply(fc, self.q[:horizon], floor=None)
