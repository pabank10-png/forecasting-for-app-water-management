"""Load the prepared datasets. Data is never stored in this repo.

The folder is read from FORECAST_DATA_DIR, or by default from the `pilot`
checkout next to this repo: ../pilot/forecasting/1-reservoir-inflow/data/csv
"""
import csv
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO.parent / "pilot" / "forecasting" / "1-reservoir-inflow" / "data" / "csv"
RESERVOIRS = ("DK", "KY", "NPL", "THREE", "PS")


def data_dir():
    d = Path(os.environ.get("FORECAST_DATA_DIR", DEFAULT_DIR))
    if not d.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ข้อมูล {d} (ตั้ง FORECAST_DATA_DIR ได้)")
    return d


def _num(v):
    if v in ("", None):
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


def read_csv(name):
    with open(data_dir() / name, encoding="utf-8-sig", newline="") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def lag_table(reservoir=None):
    """Monthly LAG sheet: one row per reservoir-month with target + lag features."""
    rows = read_csv("lag.csv")
    return [r for r in rows if reservoir is None or r["reservoir"] == reservoir]


def monthly_inflow(reservoir):
    """[(date 'YYYY-MM-DD', month 1..12, inflow_mcm or None, is_partial)] for every month; inflow is None for a partial month."""
    return [(r["date"], int(r["month"]), r["target_inflow_mcm"], r["is_partial_month"])
            for r in lag_table(reservoir)]


def monthly_frame(reservoir):
    """LAG table as a DataFrame with y = monthly inflow (MCM); the partial current month has y empty."""
    df = pd.read_csv(data_dir() / "lag.csv", encoding="utf-8-sig", parse_dates=["date"])
    df = df[df["reservoir"] == reservoir].sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"target_inflow_mcm": "y"})
    # Own-month series for the sequence models (they read past rows only). A lag-1 column on
    # the next row is this month's value, so shift(-1) moves it back to its own month.
    df["rain_mm"] = df["rain_lag0_mm"]
    for c in ("sm_root", "sm_top", "oni", "dmi"):
        df[c] = df[f"{c}_lag1"].shift(-1)
    return df


def daily_frame(reservoir):
    """Daily table: y = inflow (MCM/day, RID where present else dam workbook), rain, storage, outflow."""
    df = pd.read_csv(data_dir() / f"daily_{reservoir}.csv", encoding="utf-8-sig", parse_dates=["date"])
    col = lambda c: df[c] if c in df else pd.Series(float("nan"), index=df.index)  # PS has RID columns only
    out = pd.DataFrame({"date": df["date"]})
    out["y"] = col("inflow_mcm_rid").fillna(col("inflow_mcm_dam"))
    out["rain"] = col("rain_mm")
    out["storage"] = col("storage_mcm")
    dam_out = col("outlet_mcm_dam").fillna(0) + col("spillway_mcm_dam").fillna(0)
    out["outflow"] = col("outflow_mcm_rid").fillna(dam_out.where(col("outlet_mcm_dam").notna()))
    return out.sort_values("date").reset_index(drop=True)
