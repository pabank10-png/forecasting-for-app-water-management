"""Climatology: P10/P50/P90 of the same calendar month in past years."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from common.base import result


def months(df, horizon):
    """Calendar month of each forecast step (the last row of df is step 1)."""
    m0 = pd.to_datetime(df["date"].iloc[-1]).month
    return [(m0 - 1 + h) % 12 + 1 for h in range(horizon)]


class Model:
    def __init__(self, min_years=3):
        self.min_years = min_years

    def fit(self, df):
        d = df.dropna(subset=["y"])
        m = pd.to_datetime(d["date"]).dt.month
        self.by_month = {k: g.to_numpy(float) for k, g in d["y"].groupby(m)}
        return self

    def predict(self, df, horizon=1, future=None):
        q = []
        for m in months(df, horizon):
            v = self.by_month.get(m, np.array([]))
            q.append(np.quantile(v, [0.1, 0.5, 0.9]) if len(v) >= self.min_years else [np.nan] * 3)
        q = np.array(q)
        return result(q[:, 0], q[:, 1], q[:, 2])
