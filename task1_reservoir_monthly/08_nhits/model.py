"""N-HiTS: blocks that look at the inflow window at coarse-to-fine resolution (max-pool 4, 2, 1)
and interpolate a few predicted points to every month. Good for the 12-month horizon."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.nn import NHits, WindowForecaster


class Model:
    def __init__(self, window=36, horizon=12, pools=(4, 2, 1), epochs=300, **kw):
        self.pools = pools
        self.kw = dict(window=window, horizon=horizon, epochs=epochs, **kw)

    def fit(self, df):
        net = lambda W, H, cp, cf: NHits(W, H, cp, cf, pools=self.pools)
        self.m = WindowForecaster(net, **self.kw).fit(df)
        return self

    def predict(self, df, horizon=1, future=None):
        return self.m.predict(df, horizon, future)
