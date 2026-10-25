"""Persistence with damped trend: level(t+h) = y(t) + h * d * (y(t) - y(t-1)).
d = 0 is plain persistence. P10/P90 from past errors of the same rule per hour ahead."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.base import history, interval_from_errors


def rule(y, h, d):
    return y[-1] + h * d * (y[-1] - y[-2])


class Model:
    def __init__(self, horizon=48, damping=0.7, sample=2000):
        self.horizon, self.d, self.sample = horizon, damping, sample

    def fit(self, df):
        y = history(df)
        idx = np.arange(2, len(y) - 1)[-self.sample * 5::5]
        self.errors = []
        for h in range(1, self.horizon + 1):
            i = idx[idx + h - 1 < len(y)]
            self.errors.append(np.array([y[k + h - 1] - rule(y[:k], h, self.d) for k in i]))
        return self

    def predict(self, df, horizon=48, future=None):
        y = history(df)
        return interval_from_errors([rule(y, h, self.d) for h in range(1, horizon + 1)], self.errors, floor=None)
