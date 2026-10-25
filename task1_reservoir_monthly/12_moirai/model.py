"""Moirai 2.0 (Salesforce) zero-shot; can take covariates whose future values are in `future`
(pip install uni2ts)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import foundation


class Model:
    def __init__(self, covariates=(), model_id="Salesforce/moirai-2.0-R-small"):
        self.covariates, self.model_id = covariates, model_id

    def fit(self, df):
        return self  # zero-shot

    def predict(self, df, horizon=1, future=None):
        return foundation.moirai(df, horizon, future, self.covariates, freq="ME", model_id=self.model_id)
