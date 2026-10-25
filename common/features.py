"""Feature builders. A feature at row r only uses rows before r (see base.py)."""
import numpy as np
import pandas as pd

# Columns of the monthly LAG table that are not model inputs.
NOT_FEATURES = {"y", "date", "reservoir", "rain_source", "is_partial_month", "year_ce",
                "rain_lag0_mm", "inflow_clim_month"}


def monthly_features(df):
    """Numeric LAG-table columns usable as inputs (rain_lag0 is excluded: it is the target month's own rain)."""
    return [c for c in df.columns if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(df[c])]


def build(df, lags=(1, 2, 3), lag_cols=None, sums=None, means=None, keep=(), calendar=None):
    """Lag / rolling features at each row from earlier rows.

    lags:     lags of y (1 = previous row)
    lag_cols: {column: lags} for other series (e.g. upstream level)
    sums:     {column: windows} rolling sums ending at the previous row (e.g. rain)
    means:    {column: windows} rolling means ending at the previous row
    keep:     columns copied from the previous row (e.g. storage)
    calendar: "doy" (daily) or "hour" (hourly) adds sin/cos of the row's own date
    """
    out = pd.DataFrame(index=df.index)
    for l in lags:
        out[f"y_lag{l}"] = df["y"].shift(l)
    if len(lags) >= 2:
        out["y_diff1"] = out[f"y_lag{lags[0]}"] - out[f"y_lag{lags[1]}"]
    for c, ls in (lag_cols or {}).items():
        for l in ls:
            out[f"{c}_lag{l}"] = df[c].shift(l)
    for c, ws in (sums or {}).items():
        prev = df[c].shift(1)
        for w in ws:
            out[f"{c}_sum{w}"] = prev.rolling(w, min_periods=max(1, w // 2)).sum()
    for c, ws in (means or {}).items():
        prev = df[c].shift(1)
        for w in ws:
            out[f"{c}_mean{w}"] = prev.rolling(w, min_periods=max(1, w // 2)).mean()
    for c in keep:
        out[f"{c}_lag1"] = df[c].shift(1)
    if calendar and "date" in df.columns:
        d = pd.to_datetime(df["date"])
        if calendar == "doy":
            a = 2 * np.pi * d.dt.dayofyear / 365.25
        else:
            a = 2 * np.pi * d.dt.hour / 24
        out[f"{calendar}_sin"], out[f"{calendar}_cos"] = np.sin(a), np.cos(a)
    return out
