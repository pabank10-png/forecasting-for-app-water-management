"""LightGBM quantile regression, one model per step ahead (direct) and per quantile.

Inputs: every numeric LAG-table column (inflow lags, rain lags, ENSO/IOD lags,
soil moisture, storage, month sin/cos). future_features=("rain_lag0_mm",)
adds the target month's forecast rain, which must then be given in `future`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.features import monthly_features
from common.tabular import DirectQuantile


class Model:
    def __init__(self, horizon=12, future_features=(), params=None):
        self.kw = dict(backend="lightgbm", horizon=horizon, future_features=future_features, params=params)

    def fit(self, df):
        self.m = DirectQuantile(features=monthly_features(df), **self.kw).fit(df)
        return self

    def predict(self, df, horizon=1, future=None):
        return self.m.predict(df, horizon, future)

    def importance(self, h=1):
        return self.m.importance(h)
