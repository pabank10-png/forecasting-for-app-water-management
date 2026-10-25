"""Encoder-decoder LSTM: the encoder reads the last 60 days (inflow + observed rain),
the decoder reads forecast rain day by day and outputs P10/P50/P90 for days 1-7."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.nn import LSTMSeq2Seq, WindowForecaster


class Model:
    def __init__(self, window=60, horizon=7, epochs=30, **kw):
        self.kw = dict(window=window, horizon=horizon, past_cols=("rain",), future_cols=("rain",),
                       epochs=epochs, batch=256, **kw)

    def fit(self, df):
        self.m = WindowForecaster(LSTMSeq2Seq, **self.kw).fit(df)
        return self

    def predict(self, df, horizon=7, future=None):
        return self.m.predict(df, horizon, future)
