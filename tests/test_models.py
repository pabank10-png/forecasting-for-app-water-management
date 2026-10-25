"""Smoke test: every time-series model fits and returns step/p10/p50/p90 on synthetic data.

Foundation models (Chronos-2, TimesFM, Moirai) are skipped when their package
is not installed. Run: python3 tests/test_models.py   (about 2-4 minutes)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.registry import folders, load

rng = np.random.default_rng(0)
FAST = {"epochs": 2}
NOT_SERIES = {"12_rod_psu_2d", "14_radar_nowcast", "95_water_balance", "95_water_balance_daily"}
FOUNDATION = {"10_chronos2", "11_timesfm", "12_moirai", "13_chronos2_finetune", "07_chronos2_exog",
              "10_foundation_zero_shot"}


def monthly():
    n = 180
    m = np.arange(n) % 12 + 1
    rain = 150 + 120 * np.sin((m - 4) / 12 * 2 * np.pi) + rng.normal(0, 30, n)
    df = pd.DataFrame({"date": pd.date_range("2010-01-31", periods=n, freq="ME"), "month": m,
                       "rain_lag0_mm": rain, "sm_root_lag1": rng.random(n), "oni_lag1": rng.normal(0, 1, n)})
    df["y"] = np.maximum(0.2 * rain + rng.normal(0, 5, n), 0)
    df.loc[n - 1, "y"] = np.nan
    return df, None


def daily():
    n = 365 * 6
    rain = rng.gamma(0.3, 15, n)
    y = np.convolve(rain, np.exp(-np.arange(20) / 4))[:n] * 0.02
    df = pd.DataFrame({"date": pd.date_range("2010-01-01", periods=n), "y": y, "rain": rain,
                       "storage": 50.0, "outflow": 1.0})
    df.loc[n - 1, "y"] = np.nan
    return df, df.iloc[-1:].reindex(range(7))[["rain"]].fillna(5.0).reset_index(drop=True)


def hourly():
    n = 24 * 90
    rain = np.where(rng.random(n) < 0.03, rng.gamma(1.5, 5, n), 0)
    q = np.convolve(rain, np.exp(-np.arange(48) / 8))[:n] + 5
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=n, freq="h"), "rain": rain, "inflow": q,
                       "up1": 1 + q / 30, "y": 2 + np.r_[np.full(4, q[0]), q[:-4]] / 25})
    df.loc[n - 1, "y"] = np.nan
    return df, pd.DataFrame({"rain": np.zeros(48), "up1": df["up1"].iloc[-1], "inflow": q[-1]})


KW = {"task2_river_hourly/02_travel_time": {}, "task2_river_hourly/11_hec_hms_ras": {"maxiter": 3},
      "task2_river_hourly/90_error_correction": {"base": "04_arx", "n_origins": 10},
      "task2_river_hourly/92_conformal": {"base": "04_arx", "calib_hours": 240},
      "task2_river_hourly/91_rain_ensemble": {"base": "04_arx"},
      "task1_reservoir_monthly/90_conformal": {"calib_years": 2},
      "task1b_reservoir_daily/03_gr4j": {"maxiter": 2, "calib_years": 2},
      "task1b_reservoir_daily/04_hbv": {"maxiter": 2, "calib_years": 2}}
H = {"task1_reservoir_monthly": 3, "task1b_reservoir_daily": 7, "task2_river_hourly": 12}
DATA = {"task1_reservoir_monthly": monthly, "task1b_reservoir_daily": daily, "task2_river_hourly": hourly}

failed = []
for task, make in DATA.items():
    df, fut = make()
    for rel in folders(task):
        name = rel.split("/")[1]
        if name in NOT_SERIES:
            continue
        mod = load(rel)
        kw = dict(KW.get(rel, {}))
        try:
            m = mod.Model(**kw)
        except TypeError:
            m = mod.Model()
        for k, v in FAST.items():
            if hasattr(m, k):
                setattr(m, k, v)
        try:
            fc = m.fit(df).predict(df, H[task], fut)
            assert list(fc.columns) == ["step", "p10", "p50", "p90"] and len(fc) == H[task]
            assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all()
            print("ok  ", rel)
        except ImportError as e:
            if name not in FOUNDATION:
                raise
            print("skip", rel, "(", e.name, "not installed )")
        except Exception as e:
            failed.append(rel)
            print("FAIL", rel, type(e).__name__, e)
assert not failed, failed
print("all ok")
