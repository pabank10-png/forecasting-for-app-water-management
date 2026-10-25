"""Ensemble: weighted average of several models, weights = inverse backtest MAE shrunk
halfway toward equal weights (lambda = 0.5). Champion rule: the ensemble is used only
if its backtest MAE beats the best single model by at least 2%; otherwise that model is used.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import backtest, ensemble
from common.metrics import mae
from common.registry import load

TASK = "task1_reservoir_monthly"


class Model:
    def __init__(self, members=("01_climatology", "02_seasonal_naive", "05_lightgbm_quantile"),
                 calib_years=5, horizon=12, shrink=0.5, margin=0.02, refit_every=12):
        self.makers = {m: (lambda m=m: load(f"{TASK}/{m}").Model()) if isinstance(m, str) else m for m in members}
        self.calib_years, self.horizon, self.shrink, self.margin, self.refit_every =             calib_years, horizon, shrink, margin, refit_every

    def fit(self, df):
        n_obs = df["y"].last_valid_index() + 1
        start = max(36, n_obs - 12 * self.calib_years - self.horizon)
        origins = list(range(start, n_obs - 1))
        bts = {k: backtest.run(f, df.iloc[:n_obs], origins, self.horizon, refit_every=self.refit_every)
               for k, f in self.makers.items()}
        self.maes = {k: mae(b["actual"], b["p50"]) for k, b in bts.items()}
        self.weights = ensemble.inverse_mae_weights(self.maes, self.shrink)
        blend = sum(bts[k]["p50"].to_numpy() * w for k, w in self.weights.items())
        self.ensemble_mae = mae(next(iter(bts.values()))["actual"], blend)
        self.choice = ensemble.choose(self.maes, self.ensemble_mae, self.margin)
        self.models = {k: f().fit(df) for k, f in self.makers.items()}
        return self

    def predict(self, df, horizon=1, future=None):
        fcs = {k: m.predict(df, horizon, future) for k, m in self.models.items()}
        if self.choice != "ensemble":
            return fcs[self.choice]
        return ensemble.combine(fcs, self.weights)
