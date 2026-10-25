"""Shared contract for every model in this repo.

Every model.py exposes `class Model` with two methods:

    fit(df)                          -> self
    predict(df, horizon, future=None) -> DataFrame(step, p10, p50, p90)

`df` is a pandas DataFrame in time order with the target in column "y" and
any features in other columns. Its LAST row is the first step to forecast:
"y" is empty there and the other columns hold what is already known for that
step (in the monthly LAG table this is the current, partial month). Rows
before it are history. `future` holds known values for the next `horizon`
steps (for example forecast rain), one row per step, same column names.

Use `next_row(df, future)` to append that first empty row when your data
ends at the last observation (daily and hourly data).
"""
import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
COLS = ("p10", "p50", "p90")


def result(p10, p50, p90, floor=0.0):
    """Build the standard output, fixing crossed quantiles and negatives."""
    a = np.column_stack([np.asarray(v, float).ravel() for v in (p10, p50, p90)])
    a = np.sort(a, axis=1)
    if floor is not None:
        a = np.maximum(a, floor)
    return pd.DataFrame({"step": np.arange(1, len(a) + 1), "p10": a[:, 0], "p50": a[:, 1], "p90": a[:, 2]})


def history(df, col="y"):
    """Observed target values before the forecast row (trailing empties dropped)."""
    y = df[col].to_numpy(float)
    n = len(y)
    while n and np.isnan(y[n - 1]):
        n -= 1
    return y[:n]


def next_row(df, future=None):
    """Append the first step to forecast (empty y) if df ends with an observation."""
    if len(df) and pd.isna(df["y"].iloc[-1]):
        return df
    row = {c: np.nan for c in df.columns}
    if future is not None and len(future):
        for c in future.columns:
            row[c] = future[c].iloc[0]
    if "date" in df.columns and len(df) >= 2:
        row["date"] = df["date"].iloc[-1] + (df["date"].iloc[-1] - df["date"].iloc[-2])
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def interval_from_errors(p50, errors, floor=0.0):
    """Add P10/P90 to point forecasts using past errors (actual - forecast).

    errors: list with one array per step ahead (or one array for all steps).
    """
    p50 = np.asarray(p50, float)
    lo, hi = [], []
    for h in range(len(p50)):
        e = np.asarray(errors[min(h, len(errors) - 1)], float)
        e = e[~np.isnan(e)]
        q = np.quantile(e, [0.1, 0.9]) if len(e) >= 5 else (0.0, 0.0)
        lo.append(p50[h] + q[0])
        hi.append(p50[h] + q[1])
    return result(lo, p50, hi, floor)


def pinball(y, q, tau):
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.nanmean(np.maximum(tau * d, (tau - 1) * d)))
