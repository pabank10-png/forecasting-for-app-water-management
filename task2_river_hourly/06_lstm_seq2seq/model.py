"""Encoder-decoder LSTM (the usual Deep Learning flood model): the encoder reads the last 72 hours
of level, upstream levels and rain; the decoder reads forecast rain hour by hour up to 48 h.
Train one model on all stations together (stack their rows) when data allows."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.nn import LSTMSeq2Seq, WindowForecaster


class Model:
    def __init__(self, upstream=(), window=72, horizon=48, epochs=20, **kw):
        self.kw = dict(window=window, horizon=horizon, past_cols=tuple(upstream) + ("rain",),
                       future_cols=("rain",), transform=None, floor=None, epochs=epochs, batch=256, **kw)

    def fit(self, df):
        self.m = WindowForecaster(LSTMSeq2Seq, **self.kw).fit(df)
        return self

    def predict(self, df, horizon=48, future=None):
        return self.m.predict(df, horizon, future)
