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
    if group in QC_GROUPS:
        return qc_mean(s.pivot_table(index="date", columns="code", values="rain"))
    return s.groupby("date")["rain"].mean()


# Station groups whose daily mean leaves out gauges that read far below the others. Pluak Daeng: in Sep 2026
# บ้านเขาคันทรง (STN1653) read 0 mm all year and บ้านคลองป่าหวาย 60 mm against 488 mm at อบต.ปลวกแดง, so the
# mean the DK/KY/NPL models saw was about a third of the storm. Backtest Jun 2024-Oct 2026 (experiments/
# backtest_2026_sep_oct): better for DK/KY/NPL; worse for Bo Thong (PS), so it is not used there.
QC_GROUPS = {"pluakdaeng"}


def qc_mean(p, share=0.25, min_top=50.0):
    """Row mean of a date x station table, leaving out a station on days when its last-30-day total is under
    `share` of the highest station's 30-day total (and that total is at least `min_top` mm)."""
    p = p.reindex(pd.date_range(p.index.min(), p.index.max()))
    r30 = p.rolling(30, min_periods=20).sum()
    top = r30.max(axis=1)
    bad = r30.lt(share * top, axis=0) & top.ge(min_top).to_numpy()[:, None]
    return p.mask(bad).mean(axis=1).dropna()


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


PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"


def _archive_json(site, start, end, days):
    lat, lon, _ = DAMS[site]
    hourly = ",".join(f"precipitation_previous_day{k}" for k in range(1, days + 1))
    url = (f"{PREVIOUS_RUNS}?latitude={lat}&longitude={lon}&hourly={hourly}&timezone=Asia%2FBangkok"
           f"&start_date={start}&end_date={end}&models={','.join(RAIN_SOURCES.values())}")
    req = urllib.request.Request(url, headers={"User-Agent": "WaterIntelligenceForecast/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["hourly"]


def parse_archive(hourly, days=7):
    """Open-Meteo Previous Runs hourly JSON -> {source: DataFrame(index day, columns 1..days)}: rain (mm) of
    each day as forecast `step` days earlier, so step s matches the s-th day of a live 7-day forecast.
    A day needs at least 20 forecast hours, else it is left empty."""
    t = pd.to_datetime(hourly["time"])
    out = {}
    for name, key in RAIN_SOURCES.items():
        cols = {}
        for k in range(1, days + 1):
            vals = hourly.get(f"precipitation_previous_day{k}_{key}")
            if vals is None:
                continue
            s = pd.Series(vals, index=t, dtype=float)
            g = s.groupby(s.index.normalize())
            cols[k] = g.sum(min_count=1).where(g.count() >= 20)
        if cols:
            out[name] = pd.DataFrame(cols).sort_index()
    return out


def rain_archive(site, start, end, days=7):
    """Past daily forecast rain per source and lead (Open-Meteo Previous Runs, from about 2024).
    Fetched a year at a time; {} if the service cannot be reached."""
    parts = {}
    try:
        for a in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="365D"):
            b = min(a + pd.Timedelta(days=364), pd.Timestamp(end))
            for name, f in parse_archive(_archive_json(site, a.date(), b.date(), days), days).items():
                parts.setdefault(name, []).append(f)
    except Exception as e:  # noqa: BLE001
        print(f"forecast rain archive unavailable for {site}: {e}", flush=True)
        return {}
    return {name: pd.concat(fs) for name, fs in parts.items()}


def with_forecast_row(df, future=None):
    return next_row(df, future)


def month_ends(start, n):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(pd.Timestamp(start), periods=n, freq="ME")]


def clean(v):
    v = float(v)
    return None if np.isnan(v) else round(v, 4)
