"""TimesFM 2.5 (Google) zero-shot on inflow, plus an in-context regression on lag-1 rain, soil
moisture, ENSO and IOD (pip install timesfm torch)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import foundation
from common.features import LAG_COVARIATES


class Model:
    def __init__(self, model_id="google/timesfm-2.5-200m-pytorch", covariates=LAG_COVARIATES):
        self.model_id, self.covariates = model_id, covariates

    def fit(self, df):
        return self  # zero-shot

    def predict(self, df, horizon=1, future=None):
        return foundation.timesfm(df, horizon, self.model_id, covariates=self.covariates)
