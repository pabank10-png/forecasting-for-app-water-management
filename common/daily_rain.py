"""Complete daily catchment rain for the whole record, for models that run a daily water balance over many years.

The dam rain gauges stop in Sep 2020 and Prasae (PS) never had one, so the daily series is built the same way for
every year: the month's rain total that the monthly models use (LAG table, rain_lag0_mm: HII district rain, filled
with ThaiWater stations) spread over the days of the month with the daily pattern of ERA5 rain (Open-Meteo archive,
public, 0.25 deg, from 1994). One series without joins keeps the model calibration consistent.

Test 2014-2020 against the dam gauges (HBV calibrated on 2001-2013 with the same series it is run with):
monthly inflow MAE NPL 8.5 vs 8.2 with the gauges, DK 4.8 vs 5.2, KY 5.6 vs 5.0 MCM; raw ERA5 was worse
(NPL 10.2). ERA5 alone has daily r = 0.42 with the gauges, monthly r = 0.87.

ERA5 days are cached in data/raw/era5_rain_daily.csv beside the CSV folder; each call adds only the missing
days (ERA5 runs about 5 days behind; the last days of a partial month take the month's leftover rain evenly).
"""
import datetime as dt
import json
import urllib.request

import numpy as np
import pandas as pd

from .data import data_dir

# One ERA5 cell per rain district (DK, KY and NPL share the Pluak Daeng cell)
POINTS = {"pluakdaeng": (12.99, 101.30), "bothong": (13.05, 101.58)}
AREA = {"DK": "pluakdaeng", "KY": "pluakdaeng", "NPL": "pluakdaeng", "PS": "bothong"}
FIRST = "1994-01-01"
URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={a}&end_date={b}"
       "&daily=precipitation_sum&models=era5&timezone=Asia%2FBangkok")
_CACHE = {}


def _path():
    return data_dir().parent / "raw" / "era5_rain_daily.csv"


def _fetch(area, a, b):
    lat, lon = POINTS[area]
    req = urllib.request.Request(URL.format(lat=lat, lon=lon, a=a, b=b), headers={"User-Agent": "WaterIntelligenceForecast/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)["daily"]
    # one model in the request -> plain key; several models -> key with the model suffix
    vals = d.get("precipitation_sum", d.get("precipitation_sum_era5"))
    return pd.Series(vals, index=pd.to_datetime(d["time"]), dtype=float).dropna()


def era5(refresh=True):
    """DataFrame(index date, columns = districts) of ERA5 daily rain (mm), cached on disk."""
    if "era5" in _CACHE:
        return _CACHE["era5"]
    p = _path()
    old = pd.read_csv(p, parse_dates=["date"]).set_index("date") if p.exists() else pd.DataFrame(columns=list(POINTS))
    today = dt.date.today()
    if refresh and (old.empty or old.dropna().index.max().date() < today - dt.timedelta(days=6)):
        new = {}
        try:
            for area in POINTS:
                s = old[area].dropna() if area in old else pd.Series(dtype=float)
                a = (s.index.max() - pd.Timedelta(days=10)).date() if len(s) else dt.date.fromisoformat(FIRST)
                new[area] = _fetch(area, a, today)
        except Exception as e:  # noqa: BLE001 - keep what is cached
            print(f"ERA5 daily rain not refreshed: {type(e).__name__}: {e}", flush=True)
            new = {}
        if new:
            add = pd.DataFrame(new)
            old = add.combine_first(old) if not old.empty else add
            old.index.name = "date"
            p.parent.mkdir(parents=True, exist_ok=True)
            old.sort_index().round(2).to_csv(p)
    if old.empty:
        raise FileNotFoundError(f"ไม่มีฝน ERA5 รายวัน ({p}) และดึงจาก Open-Meteo ไม่ได้")
    _CACHE["era5"] = old.sort_index()
    return _CACHE["era5"]


def daily_rain(reservoir, monthly_rain=None, last_day=None):
    """Daily rain (mm) of a reservoir's catchment from FIRST to last_day (default: the last day of its LAG table).

    monthly_rain: Series indexed by month-end date (default: rain_lag0_mm of the reservoir's LAG rows).
    last_day: the day the partial current month's total runs to, so that total is spread over those days only."""
    if monthly_rain is None:
        from .data import monthly_frame
        m = monthly_frame(reservoir)
        monthly_rain = m.set_index("date")["rain_lag0_mm"]
    key = (reservoir, str(last_day), len(monthly_rain), float(np.nansum(monthly_rain.to_numpy(float))))
    if key in _CACHE:
        return _CACHE[key]
    e = era5()[AREA[reservoir]]
    days = pd.date_range(FIRST, last_day or monthly_rain.index.max(), freq="D")
    e = e.reindex(days)
    per = days.to_period("M")
    tot = pd.Series(monthly_rain.to_numpy(float), index=monthly_rain.index.to_period("M"))
    tot = tot[~tot.index.duplicated(keep="last")].reindex(per).to_numpy()
    # Months without a district total keep raw ERA5 times the usual district/ERA5 ratio of that calendar month
    era_m = e.groupby(per).sum(min_count=1)
    ratio = (pd.Series(tot, index=per).groupby(level=0).first() / era_m).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.groupby(ratio.index.month).median().clip(0.3, 3.0)
    tot = np.where(np.isnan(tot), e.groupby(per).transform("sum").to_numpy() * ratio.reindex(per.month).to_numpy(), tot)
    have = e.notna().to_numpy()
    era_sum = e.fillna(0).groupby(per).transform("sum").to_numpy()
    n_have = pd.Series(have.astype(int), index=days).groupby(per).transform("sum").to_numpy()
    n_all = pd.Series(1, index=days).groupby(per).transform("sum").to_numpy()
    val = np.where(have & (era_sum > 0.5), e.fillna(0).to_numpy() / np.maximum(era_sum, 1e-9), np.nan)
    # Dry ERA5 month with district rain: spread evenly; days ERA5 does not cover yet: the leftover evenly
    even = 1.0 / n_all
    val = np.where(np.isnan(val), even, val)
    if (~have).any():
        covered = n_have / n_all
        val = np.where(have, val * covered, (1 - covered) / np.maximum(n_all - n_have, 1))
    out = pd.Series(np.maximum(val * np.nan_to_num(tot), 0.0), index=days, name="rain")
    _CACHE[key] = out
    return out
