"""Experiment (Nov 2026): ensemble rain for the other uses (heavy-rain switch, Storm Trigger, chance of crossing a
storage line in 1-2 weeks, month-end nowcast). Like run_1b.py but with a longer horizon:
- days 1-14 of the forecast rain (ECMWF ENS 51 members, or WN3 mean + p10..p90 as 5 pseudo-members),
- then (to day 31) the climatological rain of that calendar day (mean of the years before the origin, +-7 days).
For each origin it writes the inflow forecast with: every member, the ensemble mean rain, the rain that fell
('obs', also for days 15-31) and climatological rain only ('clim', no forecast rain at all).
Usage: python run_uses.py MODEL SITE ENS_DIR OUT.csv.gz   env SRC=ecmwf|wn3, BC=1 for quantile mapping.
Conceptual models (GR4J, HBV) run 31 days; LightGBM 14 (its direct models are trained per day ahead).
"""
import glob
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import run_1b
from common.base import result
from common.conceptual import Conceptual
from common.registry import load
from pipeline import plan

HR = 14  # forecast-rain days


def ens_rain(ens_dir, site, min_members):
    parts = [pd.read_csv(p) for p in glob.glob(f"{ens_dir}/**/*.csv.gz", recursive=True)]
    e = pd.concat(parts)
    e = e[e["point"] == site]
    e = e[pd.to_datetime(e["date"]) >= pd.to_datetime(e["origin"])]
    out = {}
    for o, g in e.groupby("origin"):
        p = g.pivot_table(index="member", columns="date", values="mm").sort_index(axis=1)
        if p.shape[1] >= HR and p.shape[0] >= min_members:
            out[pd.Timestamp(o)] = p.iloc[:, :HR].to_numpy(float)
    return out


def clim_rain(df, o, n):
    """Mean rain of each calendar day from o on, over the years before o (window +-7 days)."""
    past = df[df["date"] < o].dropna(subset=["rain"])
    doy = past["date"].dt.dayofyear.to_numpy()
    r = past["rain"].to_numpy(float)
    out = []
    for d in pd.date_range(o, periods=n):
        k = d.dayofyear
        dist = np.minimum(np.abs(doy - k), 366 - np.abs(doy - k))
        out.append(r[dist <= 7].mean())
    return np.array(out)


def main(name, site, ens_dir, out):
    mod, kw = load(plan.folder(name)), dict(plan.kwargs(name))
    conceptual = "gr4j" in name or "hbv" in name
    H = 31 if conceptual else HR
    if not conceptual:
        kw["horizon"] = HR
    src = os.environ.get("SRC", "ecmwf")
    df = run_1b.history(site)
    run_1b.H = HR
    ens = ens_rain(ens_dir, site, 6 if src == "wn3" else 40)
    tag = "+" + src
    if os.environ.get("BC") == "1":
        ens, tag = run_1b.quantile_map(ens, df), tag + "+qm"
    pos = {d: i for i, d in enumerate(df["date"])}
    rows, model, fit_year, t0 = [], None, None, time.time()
    for o in sorted(ens):
        i = pos.get(o)
        if i is None or i + HR > len(df) or df["y"].iloc[i:i + HR].isna().any():
            continue
        if model is None or o.year != fit_year:
            model, fit_year = mod.Model(**kw).fit(df.iloc[:i]), o.year
            print(o.date(), "fitted", f"{time.time() - t0:.0f}s", flush=True)
        hist = df.iloc[:i + 1].copy()
        hist.loc[hist.index[-1], "y"] = np.nan
        n = min(H, len(df) - i)
        obs = df["rain"].iloc[i:i + n].fillna(0).to_numpy(float)
        obs = np.concatenate([obs, np.zeros(H - n)])
        cl = clim_rain(df, o, H)
        E = ens[o]
        mean_r = E.mean(axis=0)
        if src == "wn3":
            mean_r, E = E[0], E[1:]
        tail = cl[HR:]

        def full(r):
            return np.concatenate([r, tail]) if conceptual else np.asarray(r, float)

        inner = getattr(model, "m", model)
        if isinstance(inner, Conceptual):  # warm up once per origin, then only the forecast days per member
            P, Ev = inner._forcing(hist.iloc[:-1])
            _, state = inner.simulate(P, Ev, inner.par)

            def fc(r):
                q, _ = inner.simulate(np.asarray(r, float), np.full(H, inner.pet), inner.par, state)
                q = q * inner.area_fit / 1000.0
                return result(q * inner.ratio[0], q, q * inner.ratio[1])
        else:
            def fc(r):
                return model.predict(hist, H, pd.DataFrame({"rain": r}))

        f_obs, f_mean, f_clim = fc(obs if conceptual else obs[:HR]), fc(full(mean_r)), fc(cl)
        members = np.array([fc(full(r))["p50"].to_numpy(float) for r in E])
        for k in range(H):
            j = i + k
            r = dict(site=site, model=name.split("/")[-1] + tag, origin=o.date(), lead=k + 1,
                     date=(o + pd.Timedelta(days=k)).date(),
                     actual=df["y"].iloc[j] if j < len(df) else np.nan, obs_rain=obs[k],
                     fc_rain_mean=mean_r[k] if k < HR else cl[k], clim_rain=cl[k],
                     obs_p50=f_obs["p50"].iloc[k], mean_p10=f_mean["p10"].iloc[k], mean_p50=f_mean["p50"].iloc[k],
                     mean_p90=f_mean["p90"].iloc[k], clim_p50=f_clim["p50"].iloc[k])
            r.update({f"m{j2}": members[j2, k] for j2 in range(len(members))})
            if k < HR:
                r.update({f"r{j2}": E[j2, k] for j2 in range(len(E))})
            rows.append(r)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.3f")
    print(name, site, tag, len({r["origin"] for r in rows}), "origins", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
