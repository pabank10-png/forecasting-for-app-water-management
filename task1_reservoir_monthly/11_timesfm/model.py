"""TimesFM 2.5 (Google) zero-shot on the inflow series only (pip install timesfm torch)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import foundation


class Model:
    def __init__(self, model_id="google/timesfm-2.5-200m-pytorch"):
        self.model_id = model_id

    def fit(self, df):
        return self  # zero-shot

    def predict(self, df, horizon=1, future=None):
        return foundation.timesfm(df, horizon, self.model_id)
