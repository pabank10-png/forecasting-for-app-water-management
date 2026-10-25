"""N-HiTS for 1-48 h: coarse-to-fine blocks (max-pool 8, 4, 1 over the last 168 hours of level)
plus forecast rain for all 48 hours as an input, predicting every hour at once."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.nn import NHits, WindowForecaster


class Model:
    def __init__(self, window=168, horizon=48, pools=(8, 4, 1), epochs=20, **kw):
        self.pools = pools
        self.kw = dict(window=window, horizon=horizon, future_cols=("rain",), transform=None, floor=None,
                       epochs=epochs, batch=256, **kw)

    def fit(self, df):
        net = lambda W, H, cp, cf: NHits(W, H, cp, cf, pools=self.pools)
        self.m = WindowForecaster(net, **self.kw).fit(df)
        return self

    def predict(self, df, horizon=48, future=None):
        return self.m.predict(df, horizon, future)
