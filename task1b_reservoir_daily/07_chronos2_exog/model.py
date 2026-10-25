"""Chronos-2 zero-shot with rain as a covariate (past rain + forecast rain in `future`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import foundation


class Model:
    def __init__(self, covariates=("rain",), model_id="amazon/chronos-2"):
        self.covariates, self.model_id = covariates, model_id

    def fit(self, df):
        return self  # zero-shot

    def predict(self, df, horizon=7, future=None):
        return foundation.chronos2(df, horizon, future, self.covariates, freq="D", model_id=self.model_id)
