"""Chronos-2 (Amazon) zero-shot: no training on our data. Needs internet the first time
to download amazon/chronos-2 (pip install "chronos-forecasting>=2.0" torch)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import foundation


class Model:
    def __init__(self, covariates=(), model_id="amazon/chronos-2"):
        self.covariates, self.model_id = covariates, model_id

    def fit(self, df):
        return self  # zero-shot

    def predict(self, df, horizon=1, future=None):
        return foundation.chronos2(df, horizon, future, self.covariates, freq="ME", model_id=self.model_id)
