"""Persistence: inflow on every day ahead = the latest observed day. P10/P90 from past
changes over the same number of days."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.base import history, interval_from_errors


class Model:
    def __init__(self, horizon=7):
        self.horizon = horizon

    def fit(self, df):
        y = history(df)
        self.errors = [y[h:] - y[:-h] for h in range(1, self.horizon + 1)]
        return self

    def predict(self, df, horizon=7, future=None):
        y = history(df)
        return interval_from_errors([y[-1]] * horizon, self.errors)
