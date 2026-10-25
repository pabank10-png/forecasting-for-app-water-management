"""Conformal (CQR) wrapper: calibrate any model's P10–P90 to cover about 80% of actual values.

How: run a rolling backtest of the base model over the last `calib_years`,
measure how far actual values fell outside its band per step ahead
(q-hat), then fit the base model on all data and widen (or narrow) its band
by q-hat.

    Model(base="05_lightgbm_quantile").fit(df).predict(df, 3)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import backtest, conformal
from common.registry import load

TASK = "task1_reservoir_monthly"


class Model:
    def __init__(self, base="01_climatology", calib_years=5, horizon=12, coverage=0.8, refit_every=12, **base_kw):
        self.make = base if callable(base) else (lambda: load(f"{TASK}/{base}").Model(**base_kw))
        self.calib_years, self.horizon, self.coverage, self.refit_every = calib_years, horizon, coverage, refit_every

    def fit(self, df):
        n_obs = df["y"].last_valid_index() + 1
        start = max(36, n_obs - 12 * self.calib_years - self.horizon)
        origins = list(range(start, n_obs - 1))
        bt = backtest.run(self.make, df.iloc[:n_obs], origins, self.horizon, refit_every=self.refit_every)
        self.q = conformal.from_backtest(bt, self.coverage)
        self.base = self.make().fit(df)
        return self

    def predict(self, df, horizon=1, future=None):
        fc = self.base.predict(df, horizon, future)
        return conformal.apply(fc, self.q[:horizon] if len(self.q) >= horizon else self.q[-1])
