"""SARIMAX on log(1 + inflow): AR + seasonal AR(12) + exogenous inputs (the X).

Default exog: last month's rain, 3-month rain, soil moisture, ENSO (ONI) and IOD (DMI),
all lag 1, so step 1 uses values already known at the start of the month. Later steps
take values from `future` when given, otherwise fill_future (calendar-month mean for
rain and soil moisture, last value for ENSO/IOD).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings

import numpy as np

from common.base import result
from common.features import LAG_COVARIATES, fill_future


class Model:
    def __init__(self, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), exog=LAG_COVARIATES):
        self.order, self.seasonal_order, self.exog = order, seasonal_order, list(exog)

    def _x(self, df):
        return df[self.exog].ffill().bfill().to_numpy(float) if self.exog else None

    def fit(self, df):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        d = df.iloc[df["y"].first_valid_index():df["y"].last_valid_index() + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = SARIMAX(np.log1p(d["y"].to_numpy(float)), exog=self._x(d), order=self.order,
                               seasonal_order=self.seasonal_order, trend="c").fit(disp=False)
        self.start = df["y"].first_valid_index()
        return self

    def predict(self, df, horizon=1, future=None):
        d = df.iloc[self.start:len(df) - 1]  # history rows
        x_hist = self._x(d)
        x_fut = None
        if self.exog:
            filled = fill_future(df, self.exog, horizon)
            x_fut = np.array([[future[c].iloc[h] if future is not None and c in future and h < len(future)
                               else filled[c].iloc[h] for c in self.exog] for h in range(horizon)], float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = self.res.apply(np.log1p(d["y"].to_numpy(float)), exog=x_hist)
            f = r.get_forecast(horizon, exog=x_fut).summary_frame(alpha=0.2)
        return result(np.expm1(f["mean_ci_lower"]), np.expm1(f["mean"]), np.expm1(f["mean_ci_upper"]))
