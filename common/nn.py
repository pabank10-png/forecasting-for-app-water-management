"""PyTorch training loop shared by the Deep Learning models (N-BEATS, N-HiTS, LSTM, TFT ...).

A network maps
    past (batch, window, 1 + len(past_cols))   y and past features, oldest first
    fut  (batch, horizon, len(future_cols))     known future features (may have 0 columns)
to (batch, horizon, 3) = P10/P50/P90 in scaled units. Training minimises the
pinball (quantile) loss; targets that are missing are masked out.
"""
import numpy as np

from .base import QUANTILES, result

try:
    import torch
    from torch import nn
except ImportError:  # the Baseline / Statistics models still work without torch
    torch, nn = None, None


def _need_torch():
    if torch is None:
        raise ImportError("ต้องติดตั้ง torch ก่อน: pip install torch")


def pinball_loss(pred, y):
    q = torch.tensor(QUANTILES, dtype=pred.dtype, device=pred.device)
    d = y.unsqueeze(-1) - pred
    loss = torch.maximum(q * d, (q - 1) * d)
    mask = ~torch.isnan(y)
    loss = torch.where(mask.unsqueeze(-1), loss, torch.zeros_like(loss))
    return loss.sum() / (mask.sum() * len(QUANTILES)).clamp(min=1)


class WindowForecaster:
    def __init__(self, net_fn, window, horizon, past_cols=(), future_cols=(), transform="log1p",
                 epochs=300, lr=1e-3, batch=64, seed=0, val_frac=0.15, floor=0.0):
        _need_torch()
        self.net_fn, self.window, self.horizon = net_fn, window, horizon
        self.past_cols, self.future_cols = list(past_cols), list(future_cols)
        self.transform, self.epochs, self.lr, self.batch = transform, epochs, lr, batch
        self.seed, self.val_frac, self.floor = seed, val_frac, floor

    # scaling ---------------------------------------------------------------
    def _t(self, y):
        return np.log1p(np.maximum(y, 0)) if self.transform == "log1p" else y

    def _it(self, y):
        return np.expm1(y) if self.transform == "log1p" else y

    def _arrays(self, df):
        y = (self._t(df["y"].to_numpy(float)) - self.mu) / self.sd
        p = np.column_stack([y] + [(df[c].to_numpy(float) - self.cm[c]) / self.cs[c] for c in self.past_cols])
        f = np.column_stack([(df[c].to_numpy(float) - self.cm[c]) / self.cs[c] for c in self.future_cols]) \
            if self.future_cols else np.zeros((len(df), 0))
        return y, p, f

    def fit(self, df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        ty = self._t(df["y"].to_numpy(float))
        self.mu, self.sd = np.nanmean(ty), np.nanstd(ty) or 1.0
        self.cm = {c: np.nanmean(df[c]) for c in self.past_cols + self.future_cols}
        self.cs = {c: (np.nanstd(df[c]) or 1.0) for c in self.past_cols + self.future_cols}
        y, p, f = self._arrays(df)
        p, f = np.nan_to_num(p), np.nan_to_num(f)
        W, H = self.window, self.horizon
        P, F, Y = [], [], []
        for r in range(W, len(df)):
            tgt = np.full(H, np.nan)
            seg = y[r:r + H]
            tgt[:len(seg)] = seg
            if np.isnan(tgt).all() or np.isnan(y[r - W:r]).mean() > 0.2:
                continue
            fut = np.zeros((H, f.shape[1]))
            fseg = f[r:r + H]
            fut[:len(fseg)] = fseg
            P.append(p[r - W:r]), F.append(fut), Y.append(tgt)
        if len(Y) < 8:
            raise ValueError(f"ข้อมูลน้อยเกินไปสำหรับ window={W} horizon={H}")
        P, F, Y = (torch.tensor(np.array(a), dtype=torch.float32) for a in (P, F, Y))
        n_val = int(len(Y) * self.val_frac) if len(Y) >= 40 else 0
        tr, va = slice(0, len(Y) - n_val), slice(len(Y) - n_val, len(Y))
        self.net = self.net_fn(W, H, P.shape[2], F.shape[2])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        best, best_state = np.inf, None
        for _ in range(self.epochs):
            self.net.train()
            idx = torch.randperm(tr.stop)
            for i in range(0, len(idx), self.batch):
                b = idx[i:i + self.batch]
                opt.zero_grad()
                loss = pinball_loss(self.net(P[b], F[b]), Y[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            if n_val:
                self.net.eval()
                with torch.no_grad():
                    v = pinball_loss(self.net(P[va], F[va]), Y[va]).item()
                if v < best:
                    best, best_state = v, {k: t.clone() for k, t in self.net.state_dict().items()}
        if best_state:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, df, horizon=None, future=None):
        horizon = horizon or self.horizon
        n = len(df) - 1  # forecast row
        d = df.copy()
        if future is not None:
            for c in self.future_cols:
                if c in future:
                    vals = future[c].to_numpy(float)[:self.horizon]
                    d = d.reindex(range(n + max(len(vals), 1)))
                    d.loc[n:n + len(vals) - 1, c] = vals
        y, p, f = self._arrays(d)
        past = np.nan_to_num(pd_ffill(p[n - self.window:n]))
        fut = np.zeros((self.horizon, f.shape[1]))
        fseg = np.nan_to_num(f[n:n + self.horizon])
        fut[:len(fseg)] = fseg
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(past[None], dtype=torch.float32),
                           torch.tensor(fut[None], dtype=torch.float32))[0].numpy()
        out = self._it(out * self.sd + self.mu)[:horizon]
        return result(out[:, 0], out[:, 1], out[:, 2], self.floor)


def pd_ffill(a):
    """Forward-fill NaN rows (then back-fill the start) in a 2-D array."""
    import pandas as pd
    return pd.DataFrame(a).ffill().bfill().to_numpy()


def mlp(i, o, hidden=128, layers=2, dropout=0.1):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


if nn is not None:
    class LSTMDirect(nn.Module):
        """Encoder LSTM over the past window; last hidden state + future features -> all steps at once."""

        def __init__(self, W, H, c_past, c_fut, hidden=64, layers=1):
            super().__init__()
            self.H = H
            self.enc = nn.LSTM(c_past, hidden, layers, batch_first=True)
            self.head = mlp(hidden + H * c_fut, H * 3, hidden=hidden)

        def forward(self, past, fut):
            _, (h, _) = self.enc(past)
            z = torch.cat([h[-1], fut.flatten(1)], 1)
            return self.head(z).view(-1, self.H, 3)

    class LSTMSeq2Seq(nn.Module):
        """Encoder reads the past; decoder steps through future features (e.g. forecast rain) one step at a time."""

        def __init__(self, W, H, c_past, c_fut, hidden=64, layers=1):
            super().__init__()
            self.H = H
            self.enc = nn.LSTM(c_past, hidden, layers, batch_first=True)
            self.step = nn.Embedding(H, 8)
            self.dec = nn.LSTM(c_fut + 8, hidden, layers, batch_first=True)
            self.out = nn.Linear(hidden, 3)

        def forward(self, past, fut):
            _, state = self.enc(past)
            steps = self.step(torch.arange(self.H, device=past.device)).expand(past.shape[0], -1, -1)
            o, _ = self.dec(torch.cat([fut, steps], 2), state)
            base = past[:, -1:, :1]  # last observed (scaled) value
            return base + self.out(o)

    class NHits(nn.Module):
        """N-HiTS: each block looks at a max-pooled (coarser) copy of the window, predicts a few
        coarse points that are interpolated to every step, and passes the unexplained rest on."""

        def __init__(self, W, H, c_past, c_fut, pools=(4, 2, 1), hidden=128):
            super().__init__()
            self.W, self.H = W, H
            self.pools = [min(k, W) for k in pools]
            self.blocks = nn.ModuleList()
            self.ncoef = []
            for k in self.pools:
                n_in = -(-W // k) * c_past + H * c_fut
                nc = max(2, -(-H // k))
                self.ncoef.append(nc)
                self.blocks.append(mlp(n_in, W + 3 * nc, hidden=hidden))

        def forward(self, past, fut):
            res = past[:, :, 0]
            out = 0
            for k, nc, blk in zip(self.pools, self.ncoef, self.blocks):
                pooled = nn.functional.max_pool1d(res.unsqueeze(1), k, k, ceil_mode=True).squeeze(1)
                cov = nn.functional.max_pool1d(past[:, :, 1:].permute(0, 2, 1), k, k, ceil_mode=True).flatten(1) \
                    if past.shape[2] > 1 else past[:, :0, 0]  # rain, soil, ENSO, IOD at the same resolution
                th = blk(torch.cat([pooled, cov, fut.flatten(1)], 1))
                back, coef = th[:, :self.W], th[:, self.W:].view(-1, 3, nc)
                fc = nn.functional.interpolate(coef, size=self.H, mode="linear", align_corners=True)
                res = res - back
                out = out + fc.permute(0, 2, 1)
            return out
