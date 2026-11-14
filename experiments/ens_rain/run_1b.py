"""Experiment (Nov 2026): ensemble streamflow for Task 1B. For each origin (days 1/6/11/16/21/26, Feb 2023 on) a
daily inflow model forecasts days 1-7 once per ECMWF ENS member's rain (51 runs), once with the ENS mean rain,
and once with the rain that really fell. The model is refitted at the first origin of each year on data before it.
Usage: python run_1b.py MODEL SITE ENS_DIR OUT.csv.gz   (MODEL like task1b_reservoir_daily/04_hbv or a plan variant)
"""
import glob
import os
import gzip
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.daily_rain import daily_rain
from common.registry import load
from pipeline import data_prep, plan

H = 7


def history(site):
    df = data_prep.daily(site)
    df = df.iloc[:df["y"].last_valid_index() + 1].reset_index(drop=True)
    gap = df["rain"].isna()
    df.loc[gap, "rain"] = df.loc[gap, "date"].map(daily_rain(site)).to_numpy()
    return df


def ens_rain(ens_dir, site):
    """{origin: array(member, day)} of the 7 days from the origin."""
    parts = [pd.read_csv(p) for p in glob.glob(f"{ens_dir}/**/*.csv.gz", recursive=True)]
    e = pd.concat(parts)
    e = e[e["point"] == site]
    e = e[pd.to_datetime(e["date"]) >= pd.to_datetime(e["origin"])]
    out = {}
    for o, g in e.groupby("origin"):
        p = g.pivot_table(index="member", columns="date", values="mm").sort_index(axis=1)
        if p.shape[1] >= H and p.shape[0] >= 40:
            out[pd.Timestamp(o)] = p.iloc[:, :H].to_numpy(float)
    return out


def quantile_map(ens, df):
    """Bias-correct every member's daily rain by empirical quantile mapping to the rain that fell, fitted on the
    OTHER years of the test (leave-one-year-out), pooling leads and members."""
    rain = df.set_index("date")["rain"]
    pairs = {}  # year -> (ens values, observed values on the same days)
    for o, E in ens.items():
        days = pd.date_range(o, periods=H)
        ob = rain.reindex(days).to_numpy(float)
        ok = ~np.isnan(ob)
        if ok.any():
            pe, po = pairs.setdefault(o.year, ([], []))
            pe.append(E[:, ok].ravel()); po.append(ob[ok])
    q = np.linspace(0, 1, 201)
    out = {}
    for o, E in ens.items():
        others = [y for y in pairs if y != o.year]
        xe = np.concatenate([np.concatenate(pairs[y][0]) for y in others])
        xo = np.concatenate([np.concatenate(pairs[y][1]) for y in others])
        qe, qo = np.quantile(xe, q), np.quantile(xo, q)
        qe = np.maximum.accumulate(qe + np.arange(len(q)) * 1e-9)
        out[o] = np.interp(E, qe, qo, right=None) + np.maximum(E - qe[-1], 0)  # above the top: keep the excess
    return out


def main(name, site, ens_dir, out):
    mod, kw = load(plan.folder(name)), plan.kwargs(name)
    rain_model = name in plan.RAIN_MODELS
    df = history(site)
    ens = ens_rain(ens_dir, site)
    tag = ""
    if os.environ.get("BC") == "1":
        ens, tag = quantile_map(ens, df), "+qm"
    pos = {d: i for i, d in enumerate(df["date"])}
    rows, model, fit_year, t0 = [], None, None, time.time()
    for o in sorted(ens):
        i = pos.get(o)
        if i is None or i + H > len(df) or df["y"].iloc[i:i + H].isna().any():
            continue
        if model is None or o.year != fit_year:
            model, fit_year = mod.Model(**kw).fit(df.iloc[:i]), o.year
            print(o.date(), "fitted", f"{time.time() - t0:.0f}s", flush=True)
        hist = df.iloc[:i + 1].copy()
        hist.loc[hist.index[-1], "y"] = np.nan
        obs = df["rain"].iloc[i:i + H].fillna(0).to_numpy(float)
        E = ens[o]

        def fc(r):
            return model.predict(hist, H, pd.DataFrame({"rain": r}) if rain_model else None)

        f_obs, f_mean = fc(obs), fc(E.mean(axis=0))
        members = np.array([fc(r)["p50"].to_numpy(float) for r in E]) if rain_model else None
        for k in range(H):
            r = dict(site=site, model=name.split("/")[-1] + tag, origin=o.date(), lead=k + 1,
                     date=df["date"].iloc[i + k].date(), actual=df["y"].iloc[i + k], obs_rain=obs[k],
                     ens_rain_mean=E[:, k].mean(), obs_p10=f_obs["p10"].iloc[k], obs_p50=f_obs["p50"].iloc[k],
                     obs_p90=f_obs["p90"].iloc[k], mean_p10=f_mean["p10"].iloc[k], mean_p50=f_mean["p50"].iloc[k],
                     mean_p90=f_mean["p90"].iloc[k])
            if members is not None:
                r.update({f"m{j}": members[j, k] for j in range(len(members))})
            rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    print(name, site, len(rows) // H, "origins", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
