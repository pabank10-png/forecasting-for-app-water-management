"""Chronos-2 fine-tuned on our reservoirs: a few hundred low learning-rate steps on the
training years only (never the test year), then forecast like 10_chronos2.

Uses Chronos2Pipeline.fit from chronos-forecasting >= 2.0; not yet run here
(weights could not be downloaded in this environment).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common import foundation
from common.base import history


class Model:
    def __init__(self, horizon=12, steps=300, learning_rate=1e-5, extra_series=(), model_id="amazon/chronos-2"):
        self.horizon, self.steps, self.lr = horizon, steps, learning_rate
        self.extra_series, self.model_id = list(extra_series), model_id

    def fit(self, df):
        """extra_series: other reservoirs' DataFrames (same cut-off date) to train on together."""
        from chronos import BaseChronosPipeline
        base = BaseChronosPipeline.from_pretrained(self.model_id, device_map="cpu")
        if not hasattr(base, "fit"):
            raise RuntimeError("chronos-forecasting รุ่นนี้ยังไม่มี fine-tune (ต้องเป็น >= 2.0)")
        inputs = [{"target": history(d).astype(np.float32)} for d in [df] + self.extra_series]
        self.pipeline = base.fit(inputs=inputs, prediction_length=self.horizon,
                                 num_steps=self.steps, learning_rate=self.lr)
        return self

    def predict(self, df, horizon=1, future=None):
        return foundation.chronos2(df, horizon, future, freq="ME", pipeline=self.pipeline)
