"""Direct multi-step quantile regression with gradient-boosted trees.

One model per step ahead h and per quantile (P10/P50/P90). For step h the
inputs are the features of the first forecast row plus `future_features`
taken from the target row (for example forecast rain of that day).
"""
import numpy as np
import pandas as pd

from .base import QUANTILES, result


def _transform(kind, y):
    return np.log1p(np.maximum(y, 0)) if kind == "log1p" else y


def _inverse(kind, y):
    return np.expm1(y) if kind == "log1p" else y


class DirectQuantile:
    def __init__(self, backend="lightgbm", horizon=12, features=None, future_features=(),
                 builder=None, transform="log1p", base_col=None, params=None, floor=0.0,
                 future_mode="value", steps=None):
        """future_mode: "value" = the target step's value, "sum" = total from step 1 to the
        target step (e.g. rain still to fall), "both". steps: train only these steps ahead
        (e.g. 1, 2, 3, 6, 12, 24, 48) and interpolate the others."""
        self.backend, self.horizon = backend, horizon
        self.future_mode, self.steps = future_mode, steps
        self.features, self.future_features = features, tuple(future_features)
        self.builder, self.transform, self.base_col = builder, transform, base_col
        self.params, self.floor = params or {}, floor

    def _x(self, df):
        f = self.builder(df) if self.builder else df[self.features]
        if self.features is None:
            self.features = list(f.columns)
        return f[self.features]

    def _target(self, df, x, h):
        y = _transform(self.transform, df["y"].shift(-(h - 1)).to_numpy(float))
        if self.base_col:
            y = y - x[self.base_col].to_numpy(float)
        return y

    def _design(self, x, df, h):
        xh = x.copy()
        for c in self.future_features:
            if self.future_mode in ("value", "both"):
                xh[f"{c}_target"] = df[c].shift(-(h - 1)).to_numpy(float)
            if self.future_mode in ("sum", "both"):
                xh[f"{c}_tosum"] = df[c][::-1].rolling(h, min_periods=1).sum()[::-1].to_numpy(float)
        return xh

    def _future_row(self, xh, df, future, h):
        for c in self.future_features:
            if future is not None and c in future and len(future):
                vals = future[c].to_numpy(float)[:h]
            elif c in df:
                vals = df[c].to_numpy(float)[-1:]
            else:
                vals = np.array([np.nan])
            if self.future_mode in ("value", "both"):
                xh[f"{c}_target"] = vals[h - 1] if len(vals) >= h else np.nan
            if self.future_mode in ("sum", "both"):
                xh[f"{c}_tosum"] = np.nansum(vals) if len(vals) else np.nan
        return xh

    def _new(self, tau):
        if self.backend == "lightgbm":
            import lightgbm as lgb
            p = dict(objective="quantile", alpha=tau, n_estimators=300, learning_rate=0.03,
                     num_leaves=15, min_child_samples=5, subsample=0.9, subsample_freq=1,
                     colsample_bytree=0.9, verbose=-1)
            p.update(self.params)
            return lgb.LGBMRegressor(**p)
        import xgboost as xgb
        p = dict(objective="reg:quantileerror", quantile_alpha=tau, n_estimators=300,
                 learning_rate=0.03, max_depth=4, min_child_weight=2, subsample=0.9,
                 colsample_bytree=0.9, reg_lambda=1.0)
        p.update(self.params)
        return xgb.XGBRegressor(**p)

    def fit(self, df):
        x = self._x(df)
        self.models = {}
        for h in (self.steps or range(1, self.horizon + 1)):
            xh, y = self._design(x, df, h), self._target(df, x, h)
            ok = ~np.isnan(y)
            if self.base_col:
                ok &= ~np.isnan(x[self.base_col].to_numpy(float))
            if ok.sum() < 10:
                continue
            self.models[h] = [self._new(t).fit(xh[ok], y[ok]) for t in QUANTILES]
        return self

    def predict(self, df, horizon=None, future=None):
        horizon = horizon or self.horizon
        x = self._x(df).iloc[[-1]]
        trained = sorted(self.models)
        need = sorted({min([s for s in trained if s >= h] or [trained[-1]]) for h in range(1, horizon + 1)} |
                      {max([s for s in trained if s <= h] or [trained[0]]) for h in range(1, horizon + 1)})
        qs = {}
        for h in need:
            xh = self._future_row(x.copy(), df, future, h)
            q = np.array([m.predict(xh)[0] for m in self.models[h]])
            if self.base_col:
                q = q + float(x[self.base_col].iloc[0])
            qs[h] = _inverse(self.transform, q)
        ks = np.array(need)
        a = np.array([[np.interp(h, ks, [qs[k][j] for k in need]) for j in range(3)] for h in range(1, horizon + 1)])
        return result(a[:, 0], a[:, 1], a[:, 2], self.floor)

    def p50_matrix(self, df, horizon=None):
        """P50 for every row of df used as the first forecast step (rows x steps); for hybrid models."""
        horizon = horizon or self.horizon
        x = self._x(df)
        out = np.full((len(df), horizon), np.nan)
        trained = sorted(self.models)
        for h in range(1, horizon + 1):
            k = min([s for s in trained if s >= h] or [trained[-1]])
            p = self.models[k][1].predict(self._design(x, df, k))
            if self.base_col:
                p = p + x[self.base_col].to_numpy(float)
            out[:, h - 1] = _inverse(self.transform, p)
        return out

    def importance(self, h=1):
        """Feature importance of the P50 model for step h (for explaining the forecast)."""
        m = self.models[h][1]
        names = list(self._design(pd.DataFrame(columns=self.features, dtype=float),
                                  pd.DataFrame(columns=list(self.future_features), dtype=float), h).columns)
        return pd.Series(m.feature_importances_, index=names).sort_values(ascending=False)
