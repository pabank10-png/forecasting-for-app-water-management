"""LightGBM (main model): one quantile model per hours-ahead step (1, 2, 3, 6, 12, 24, 36, 48;
steps in between are interpolated), predicting the change from the latest level.

Inputs: level lags 1-24 h, upstream station lags, rain sums 1-72 h, hour of
day, plus rain at and up to the target hour from `future["rain"]`, and any
`extra` columns such as reservoir release from task 1B.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.river import STEPS, river_builder
from common.tabular import DirectQuantile


class Model:
    def __init__(self, upstream=(), extra=(), horizon=48, steps=STEPS, params=None):
        self.kw = dict(backend="lightgbm", horizon=horizon, builder=river_builder(upstream, extra=extra),
                       future_features=("rain",) + tuple(extra), future_mode="both", transform=None,
                       base_col="y_lag1", floor=None, steps=[s for s in steps if s <= horizon], params=params)

    def fit(self, df):
        self.m = DirectQuantile(**self.kw).fit(df)
        return self

    def predict(self, df, horizon=48, future=None):
        return self.m.predict(df, horizon, future)
