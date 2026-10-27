"""Moirai 2.0 (Salesforce) zero-shot on inflow with past rain, soil moisture, ENSO and IOD as
past covariates; can also take covariates whose future values are in `future` (pip install uni2ts)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import foundation
from common.features import SEQ_COVARIATES


class Model:
    def __init__(self, covariates=(), model_id="Salesforce/moirai-2.0-R-small", past_covariates=SEQ_COVARIATES):
        self.covariates, self.model_id, self.past_covariates = covariates, model_id, past_covariates

    def fit(self, df):
        return self  # zero-shot

    def predict(self, df, horizon=1, future=None):
        return foundation.moirai(df, horizon, future, self.covariates, freq="ME", model_id=self.model_id,
                                 past_covariates=self.past_covariates)
