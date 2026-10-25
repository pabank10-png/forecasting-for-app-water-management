"""Seasonal naive: each month = the same month last year. P10/P90 from past year-on-year changes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.base import history, interval_from_errors


class Model:
    def __init__(self, season=12):
        self.season = season

    def fit(self, df):
        y = history(df)
        self.errors = [y[self.season:] - y[:-self.season]]
        return self

    def predict(self, df, horizon=1, future=None):
        y, s = history(df), self.season
        n = len(df) - 1  # index of step 1
        p50 = []
        for h in range(horizon):
            i = n + h - s
            while i >= len(y):
                i -= s
            p50.append(y[i] if i >= 0 else np.nan)
        return interval_from_errors(p50, self.errors)
