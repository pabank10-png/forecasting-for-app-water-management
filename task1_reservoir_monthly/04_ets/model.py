"""ETS / Holt-Winters (additive season on log(1 + inflow)), refit on the history at each forecast."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings

import numpy as np
import pandas as pd

from common.base import history, result


class Model:
    def __init__(self, season=12, trend=None, damped=False):
        self.season, self.trend, self.damped = season, trend, damped

    def _fit(self, y):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        z = np.log1p(np.maximum(y, 0))
        z = pd.Series(np.where(np.isnan(z), np.nanmean(z), z))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ETSModel(z, error="add", trend=self.trend, damped_trend=self.damped and self.trend is not None,
                            seasonal="add", seasonal_periods=self.season).fit(disp=False)

    def fit(self, df):
        self.res = self._fit(history(df))
        return self

    def predict(self, df, horizon=1, future=None):
        y = history(df)
        res = self._fit(y)
        n = len(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = res.get_prediction(start=n, end=n + horizon - 1).summary_frame(alpha=0.2)
        return result(np.expm1(f["pi_lower"]), np.expm1(f["mean"]), np.expm1(f["pi_upper"]))
