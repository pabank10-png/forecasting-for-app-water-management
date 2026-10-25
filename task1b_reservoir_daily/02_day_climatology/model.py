"""Day-of-year climatology: P10/P50/P90 of all past years within +-7 days of the target date."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from common.base import result


class Model:
    def __init__(self, half_window=7):
        self.half = half_window

    def fit(self, df):
        d = df.dropna(subset=["y"])
        self.doy = pd.to_datetime(d["date"]).dt.dayofyear.to_numpy()
        self.y = d["y"].to_numpy(float)
        return self

    def predict(self, df, horizon=7, future=None):
        start = pd.to_datetime(df["date"].iloc[-1])
        q = []
        for h in range(horizon):
            t = (start + pd.Timedelta(days=h)).dayofyear
            dist = np.abs(self.doy - t)
            v = self.y[np.minimum(dist, 366 - dist) <= self.half]
            q.append(np.quantile(v, [0.1, 0.5, 0.9]) if len(v) >= 10 else [np.nan] * 3)
        q = np.array(q)
        return result(q[:, 0], q[:, 1], q[:, 2])
