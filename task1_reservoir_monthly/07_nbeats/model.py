"""N-BEATSx (interpretable): stacked trend and seasonality blocks on the inflow window, with the
past window of rain, soil moisture, ENSO and IOD fed to every block as extra inputs (the "x").

Each block reads what earlier blocks could not explain, predicts a backcast
(removed from the input) and a forecast (added to the output). The trend
block uses a polynomial basis, the seasonality block a Fourier basis with
period 12. Output is P10/P50/P90 trained with the pinball loss.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.features import SEQ_COVARIATES
from common.nn import WindowForecaster, mlp, nn, torch


def _poly(n, degree, offset=0, scale=1):
    t = (np.arange(n) + offset) / scale
    return np.stack([t ** p for p in range(degree + 1)])


def _fourier(n, period, harmonics, offset=0):
    t = np.arange(n) + offset
    rows = [np.ones(n)]
    for k in range(1, harmonics + 1):
        rows += [np.cos(2 * np.pi * k * t / period), np.sin(2 * np.pi * k * t / period)]
    return np.stack(rows)


class NBeats(nn.Module if nn else object):
    def __init__(self, W, H, c_past, c_fut, period=12, degree=2, harmonics=4, hidden=128, blocks=(2, 2)):
        super().__init__()
        self.W, self.H = W, H
        bases = [(_poly(W, degree, -W, H), _poly(H, degree, 0, H))] * blocks[0] +                 [(_fourier(W, period, harmonics, -W), _fourier(H, period, harmonics, 0))] * blocks[1]
        self.bb = [torch.tensor(b, dtype=torch.float32) for b, _ in bases]
        self.bf = [torch.tensor(f, dtype=torch.float32) for _, f in bases]
        self.blocks = nn.ModuleList(mlp(W * c_past, b.shape[0] + 3 * f.shape[0], hidden) for b, f in bases)

    def forward(self, past, fut):
        res, out = past[:, :, 0], 0
        cov = past[:, :, 1:].flatten(1)  # covariate window, unchanged from block to block
        for blk, bb, bf in zip(self.blocks, self.bb, self.bf):
            th = blk(torch.cat([res, cov], 1))
            nb, nf = bb.shape[0], bf.shape[0]
            res = res - th[:, :nb] @ bb
            out = out + (th[:, nb:].view(-1, 3, nf) @ bf).permute(0, 2, 1)
        return out


class Model:
    def __init__(self, window=36, horizon=12, epochs=300, past_cols=SEQ_COVARIATES, **kw):
        self.kw = dict(window=window, horizon=horizon, epochs=epochs, past_cols=past_cols, **kw)

    def fit(self, df):
        self.m = WindowForecaster(NBeats, **self.kw).fit(df)
        return self

    def predict(self, df, horizon=1, future=None):
        return self.m.predict(df, horizon, future)
