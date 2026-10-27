"""N-HiTS: blocks that look at the inflow window at coarse-to-fine resolution (max-pool 4, 2, 1)
and interpolate a few predicted points to every month. Good for the 12-month horizon.
Past rain, soil moisture, ENSO and IOD are pooled the same way and fed to every block."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.features import SEQ_COVARIATES
from common.nn import NHits, WindowForecaster


class Model:
    def __init__(self, window=36, horizon=12, pools=(4, 2, 1), epochs=300, past_cols=SEQ_COVARIATES, **kw):
        self.pools = pools
        self.kw = dict(window=window, horizon=horizon, epochs=epochs, past_cols=past_cols, **kw)

    def fit(self, df):
        net = lambda W, H, cp, cf: NHits(W, H, cp, cf, pools=self.pools)
        self.m = WindowForecaster(net, **self.kw).fit(df)
        return self

    def predict(self, df, horizon=1, future=None):
        return self.m.predict(df, horizon, future)
