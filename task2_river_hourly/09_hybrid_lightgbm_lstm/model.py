"""Hybrid: LightGBM (05) forecasts first, then a small LSTM learns LightGBM's recent mistakes
(the sequence of its 1-hour-ahead errors) and adds a correction to every hour ahead.

The LSTM is trained on the last `resid_frac` of the data, where LightGBM
(fitted on the earlier part) had not seen the answers; LightGBM is then
refitted on everything.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.base import result
from common.nn import nn, torch
from common.registry import load


class ResidualLSTM(nn.Module if nn else object):
    def __init__(self, H, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, H)

    def forward(self, e):
        _, (h, _) = self.lstm(e.unsqueeze(-1))
        return self.out(h[-1])


class Model:
    def __init__(self, upstream=(), horizon=48, window=24, resid_frac=0.3, epochs=60):
        self.lgb = lambda: load("task2_river_hourly/05_lightgbm").Model(upstream=upstream, horizon=horizon,
                                                                        steps=range(1, horizon + 1))
        self.H, self.W, self.frac, self.epochs = horizon, window, resid_frac, epochs

    def fit(self, df):
        cut = int(len(df) * (1 - self.frac))
        early = self.lgb().fit(df.iloc[:cut])
        P = early.m.p50_matrix(df)
        y = df["y"].to_numpy(float)
        n, H, W = len(df), self.H, self.W
        R = np.full((n, H), np.nan)
        for h in range(H):
            R[:n - h, h] = y[h:] - P[:n - h, h]
        e1 = R[:, 0]
        X, Y = [], []
        for r in range(max(cut, W + 1), n - H):
            win = e1[r - W:r]
            if not np.isnan(win).any() and not np.isnan(R[r]).any():
                X.append(win), Y.append(R[r])
        torch.manual_seed(0)
        self.net = ResidualLSTM(H)
        self.scale = float(np.nanstd(e1[cut:]) or 1.0)
        if len(X) >= 20:
            X = torch.tensor(np.array(X) / self.scale, dtype=torch.float32)
            Y = torch.tensor(np.array(Y) / self.scale, dtype=torch.float32)
            opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.net(X), Y)
                loss.backward()
                opt.step()
        self.trained = len(X) >= 20
        self.main = self.lgb().fit(df)
        self.early = early
        return self

    def predict(self, df, horizon=48, future=None):
        fc = self.main.predict(df, horizon, future)
        if not self.trained:
            return fc
        n = len(df) - 1
        y = df["y"].to_numpy(float)
        P1 = self.main.m.p50_matrix(df.iloc[max(0, n - self.W - 80):], 1)[:, 0]
        yy = y[max(0, n - self.W - 80):]
        e1 = (yy - P1)[-self.W - 1:-1]  # 1-hour-ahead errors of the last W hours
        if np.isnan(e1).any():
            return fc
        with torch.no_grad():
            c = self.net(torch.tensor(e1[None] / self.scale, dtype=torch.float32))[0].numpy() * self.scale
        c = c[:horizon]
        return result(fc["p10"] + c, fc["p50"] + c, fc["p90"] + c, floor=None)
