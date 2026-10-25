"""LSTM: reads the last `window` months of inflow + rain + soil moisture + ENSO in order,
keeps a memory (cell state) of wet/dry spells, then outputs P10/P50/P90 for each month ahead.

past_cols are read for past months only. seq2seq=True uses an encoder-decoder
that steps through `future_cols` (e.g. forecast rain) month by month.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.nn import LSTMDirect, LSTMSeq2Seq, WindowForecaster


class Model:
    def __init__(self, window=24, horizon=12, past_cols=("rain_lag0_mm", "sm_root_lag1", "oni_lag1"),
                 future_cols=(), seq2seq=False, epochs=200, **kw):
        self.net = LSTMSeq2Seq if seq2seq else LSTMDirect
        self.kw = dict(window=window, horizon=horizon, past_cols=past_cols, future_cols=future_cols,
                       epochs=epochs, **kw)

    def fit(self, df):
        self.m = WindowForecaster(self.net, **self.kw).fit(df)
        return self

    def predict(self, df, horizon=1, future=None):
        return self.m.predict(df, horizon, future)
