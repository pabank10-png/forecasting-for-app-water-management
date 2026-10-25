"""LightGBM daily: one quantile model per day ahead (1-7).

Inputs: inflow lags 1/2/3/7 days, rain sums of the last 1/3/7/30 days,
yesterday's storage, day-of-year sin/cos, and the forecast rain of the
target day (`future["rain"]`; run once per rain source for 90_rain_source_ensemble).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from functools import partial

from common.features import build
from common.tabular import DirectQuantile

builder = partial(build, lags=(1, 2, 3, 7), sums={"rain": (1, 3, 7, 30)}, keep=("storage",), calendar="doy")


class Model:
    def __init__(self, horizon=7, params=None):
        self.kw = dict(backend="lightgbm", horizon=horizon, builder=builder, future_features=("rain",), params=params)

    def fit(self, df):
        self.m = DirectQuantile(**self.kw).fit(df)
        return self

    def predict(self, df, horizon=7, future=None):
        return self.m.predict(df, horizon, future)
