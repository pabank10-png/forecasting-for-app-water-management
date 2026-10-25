"""Rolling-origin backtest: refit on data before each origin, forecast `horizon` steps, compare."""
import numpy as np
import pandas as pd

from .metrics import coverage, mae


def run(make_model, df, origins, horizon, future_cols=(), refit_every=1):
    """make_model() -> new Model. origins: row indices of the first forecast step.

    Returns one row per (origin, step) with actual and P10/P50/P90.
    """
    rows, model = [], None
    for k, i in enumerate(origins):
        if model is None or k % refit_every == 0:
            model = make_model().fit(df.iloc[:i])
        hist = df.iloc[:i + 1].copy()
        hist.loc[hist.index[-1], "y"] = np.nan
        fut = df.iloc[i:i + horizon][list(future_cols)].reset_index(drop=True) if future_cols else None
        fc = model.predict(hist, horizon, fut)
        act = df["y"].iloc[i:i + horizon].to_numpy(float)
        for h in range(min(horizon, len(act))):
            rows.append({"origin": df["date"].iloc[i] if "date" in df else i, "step": h + 1, "actual": act[h],
                         "p10": fc["p10"].iloc[h], "p50": fc["p50"].iloc[h], "p90": fc["p90"].iloc[h]})
    return pd.DataFrame(rows)


def summary(bt):
    """MAE of P50 and P10–P90 coverage per step."""
    out = []
    for h, b in bt.groupby("step"):
        b = b.dropna(subset=["actual"])
        out.append({"step": h, "n": len(b), "mae": mae(b["actual"], b["p50"]),
                    "coverage_80": coverage(b["actual"], b["p10"], b["p90"])})
    return pd.DataFrame(out)


def yearly_origins(df, first_year, last_year, month=1):
    """Row index of each January (or `month`) from first_year to last_year, for rolling yearly tests."""
    d = pd.to_datetime(df["date"])
    return [i for i in df.index[(d.dt.month == month) & d.dt.year.between(first_year, last_year)]]
