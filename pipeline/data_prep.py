"""Model inputs for a live run: the prepared dataset (FORECAST_DATA_DIR) plus recent station rain
and forecast rain, which the dataset does not hold."""
import csv
import datetime as dt
import json
import urllib.request

import numpy as np
import pandas as pd

from common.base import next_row
from common.data import daily_frame, data_dir, monthly_frame

# Dam coordinates (same as the website's Forecast HQ) and the ThaiWater rain-station group near each dam
DAMS = {"DK": (12.890302, 101.219914, "pluakdaeng"), "NPL": (12.934647, 101.297011, "pluakdaeng"),
        "KY": (12.964319, 101.313238, "pluakdaeng"), "PS": (12.977205, 101.558297, "bothong")}
# Forecast-rain sources on Open-Meteo (the same model families the website shows)
RAIN_SOURCES = {"ECMWF": "ecmwf_ifs025", "GFS": "gfs_seamless", "ICON": "icon_seamless", "JMA": "jma_seamless"}


def monthly(site):
    """LAG table; its last row is the current (partial) month, so step 1 is the month-end Nowcast."""
    return monthly_frame(site)


def station_rain(group):
    """Daily mean rain of the ThaiWater stations of one group (data/raw/station_rain_daily.csv)."""
    path = data_dir().parent / "raw" / "station_rain_daily.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["group"] == group and r["rain_mm"] not in ("", None)]
    if not rows:
        return pd.Series(dtype=float)
    s = pd.DataFrame(rows).assign(date=lambda d: pd.to_datetime(d["date"]), rain=lambda d: d["rain_mm"].astype(float))
    return s.groupby("date")["rain"].mean()


def daily(site):
    """Daily table up to the last observed inflow, with missing rain filled from the station mean."""
    df = daily_frame(site)
    rain = station_rain(DAMS[site][2])
    if len(rain):
        fill = df["date"].map(rain)
        df["rain"] = df["rain"].fillna(fill)
    last = df["y"].last_valid_index()
    df = df.iloc[:last + 1]
    # one row per day: a missing day would shift every lag feature
    full = pd.DataFrame({"date": pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D")})
    return full.merge(df, on="date", how="left")


def forecast_rain(site, first_day, days=7):
    """{source: DataFrame(rain)} daily forecast rain (mm) at the dam from first_day on. Empty if offline."""
    lat, lon, _ = DAMS[site]
    first_day = pd.Timestamp(first_day).date()
    past = max(0, (dt.date.today() - first_day).days + 1)
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum"
           f"&timezone=Asia%2FBangkok&past_days={min(past, 92)}&forecast_days=16&models={','.join(RAIN_SOURCES.values())}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WaterIntelligenceForecast/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)["daily"]
    except Exception as e:  # noqa: BLE001
        print(f"forecast rain unavailable for {site}: {e}", flush=True)
        return {}
    dates = pd.to_datetime(d["time"])
    want = pd.date_range(first_day, periods=days, freq="D")
    out = {}
    for name, key in RAIN_SOURCES.items():
        vals = d.get(f"precipitation_sum_{key}")
        if not vals:
            continue
        s = pd.Series(vals, index=dates, dtype=float).reindex(want)
        if s.notna().sum() >= days - 1:
            out[name] = pd.DataFrame({"rain": s.fillna(0).to_numpy()})
    return out


def with_forecast_row(df, future=None):
    return next_row(df, future)


def month_ends(start, n):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(pd.Timestamp(start), periods=n, freq="ME")]


def clean(v):
    v = float(v)
    return None if np.isnan(v) else round(v, 4)
