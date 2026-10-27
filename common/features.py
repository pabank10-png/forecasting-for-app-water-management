"""Feature builders. A feature at row r only uses rows before r (see base.py)."""
import numpy as np
import pandas as pd

# Columns of the monthly LAG table that are not model inputs.
NOT_FEATURES = {"y", "date", "reservoir", "rain_source", "is_partial_month", "year_ce",
                "rain_lag0_mm", "inflow_clim_month", "rain_mm", "sm_root", "sm_top", "oni", "dmi"}

# Monthly inputs shared by every non-Baseline model:
# - SEQ_COVARIATES: own-month series that sequence models read over their past window
#   (the window itself is the lag: 24-36 months of rain, soil moisture, ENSO, IOD)
# - LAG_COVARIATES: lag-1 columns known at the start of the target month, used as
#   regressors by SARIMAX and TimesFM (later steps are filled by fill_future)
SEQ_COVARIATES = ("rain_mm", "sm_root", "oni", "dmi")
LAG_COVARIATES = ("rain_lag1_mm", "rain_sum3_lag1_mm", "sm_root_lag1", "oni_lag1", "dmi_lag1")
PERSISTENT = ("oni", "dmi")  # slow climate indices: hold the last value for later steps


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


def fill_future(df, cols, horizon):
    """Values of lag-type regressors for each forecast step, using only what is known at the origin.

    df ends with the forecast row (its lag-1 columns are known). Step 1 takes that row;
    later steps hold the last value for ENSO/IOD and use the calendar-month mean of the
    history for rain and soil moisture (their real values are not known yet).
    """
    last = df.iloc[-1]
    hist = df.iloc[:-1]
    month = pd.to_datetime(hist["date"]).dt.month
    start = pd.Timestamp(last["date"])
    rows = []
    for h in range(horizon):
        m = (start + pd.offsets.MonthEnd(h)).month
        row = {}
        for c in cols:
            known = df[c].ffill().iloc[-1]
            if h == 0 and pd.notna(last[c]):
                row[c] = last[c]
            elif c.split("_")[0] in PERSISTENT:
                row[c] = known
            else:
                v = hist[c][month == m].mean()
                row[c] = v if pd.notna(v) else known
        rows.append(row)
    return pd.DataFrame(rows, columns=list(cols))
