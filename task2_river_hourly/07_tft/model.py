"""Temporal Fusion Transformer (compact version of Lim et al. 2021).

Past inputs (level, upstream levels, rain) and known future inputs
(forecast rain, hour) each go through a Variable Selection Network whose
softmax weights say how much each input matters; an LSTM encoder/decoder
reads them in order; masked self-attention lets each forecast hour look back
at the most relevant hours; the output is P10/P50/P90.
After predict(), Model.importance() gives the average input weights.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.nn import WindowForecaster, nn, torch


class GRN(nn.Module if nn else object):
    """Gated residual network: ELU layer, gated linear unit, skip connection, layer norm."""

    def __init__(self, i, d, o=None, dropout=0.1):
        super().__init__()
        o = o or d
        self.skip = nn.Linear(i, o) if i != o else nn.Identity()
        self.l1, self.l2 = nn.Linear(i, d), nn.Linear(d, 2 * o)
        self.drop, self.norm = nn.Dropout(dropout), nn.LayerNorm(o)

    def forward(self, x):
        a = self.l2(self.drop(nn.functional.elu(self.l1(x))))
        return self.norm(self.skip(x) + nn.functional.glu(a, -1))


class VSN(nn.Module if nn else object):
    """Variable selection: embed each input separately, weight them with a softmax."""

    def __init__(self, n, d):
        super().__init__()
        self.emb = nn.ModuleList(nn.Linear(1, d) for _ in range(n))
        self.grn = nn.ModuleList(GRN(d, d) for _ in range(n))
        self.weights = GRN(n * d, d, n)

    def forward(self, x):
        e = [g(m(x[..., i:i + 1])) for i, (m, g) in enumerate(zip(self.emb, self.grn))]
        w = torch.softmax(self.weights(torch.cat(e, -1)), -1)
        return (torch.stack(e, -1) * w.unsqueeze(-2)).sum(-1), w


class TFT(nn.Module if nn else object):
    def __init__(self, W, H, c_past, c_fut, d=32, heads=4):
        super().__init__()
        self.H = H
        self.vp, self.vf = VSN(c_past, d), VSN(c_fut + 1, d)  # +1: position of the hour ahead
        self.enc, self.dec = nn.LSTM(d, d, batch_first=True), nn.LSTM(d, d, batch_first=True)
        self.gate = GRN(d, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.post = GRN(d, d)
        self.out = nn.Linear(d, 3)
        mask = torch.triu(torch.ones(W + H, W + H, dtype=torch.bool), 1)
        self.register_buffer("mask", mask)

    def forward(self, past, fut):
        B = past.shape[0]
        pos = torch.linspace(0, 1, self.H, device=past.device).view(1, -1, 1).expand(B, -1, -1)
        xp, wp = self.vp(past)
        xf, wf = self.vf(torch.cat([fut, pos], -1))
        self.last_weights = (wp.mean((0, 1)).detach(), wf.mean((0, 1)).detach())
        e, state = self.enc(xp)
        dd, _ = self.dec(xf, state)
        seq = self.gate(torch.cat([e, dd], 1))
        a, _ = self.attn(seq[:, -self.H:], seq, seq, attn_mask=self.mask[-self.H:])
        z = self.post(a + seq[:, -self.H:])
        return past[:, -1:, :1] + self.out(z)


class Model:
    def __init__(self, upstream=(), window=72, horizon=48, epochs=15, **kw):
        self.past = ("y",) + tuple(upstream) + ("rain",)
        self.kw = dict(window=window, horizon=horizon, past_cols=tuple(upstream) + ("rain",),
                       future_cols=("rain",), transform=None, floor=None, epochs=epochs, batch=128,
                       lr=1e-3, **kw)

    def fit(self, df):
        self.m = WindowForecaster(TFT, **self.kw).fit(df)
        return self

    def predict(self, df, horizon=48, future=None):
        return self.m.predict(df, horizon, future)

    def importance(self):
        """Average variable-selection weights from the last predict() call."""
        wp, wf = self.m.net.last_weights
        return {"past": dict(zip(self.past, np.round(wp.numpy(), 3))),
                "future": dict(zip(("rain", "hour_ahead"), np.round(wf.numpy(), 3)))}
