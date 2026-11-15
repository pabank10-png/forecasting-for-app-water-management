"""Experiment (Nov 2026), Task 2 first look: does ECMWF ENS rain help forecast high river levels 1-7 days ahead?
Daily maximum level (ThaiWater waterlevel_graph, pulled from Thailand) at 3 gauges: Z.38 บ้านเขาโบสถ์ (Rayong river,
rain = Pluak Daeng group), Z.11 บ้านเขาวังไทร and RAY001 เมืองแกลง (Prasae, rain = Bo Thong group).
Event: the highest daily level of days 1-7 reaches the line (bank - 1 m, and bank). Logistic regression on today's
level, past rain and the 7-day rain, trained on every day of the other years with the rain that fell (perfect prog),
then fed: each ENS member (probability = share of members), the ENS mean, the rain that fell, and a model with no
forecast rain. Usage: python river_test.py LEVELS.csv ENS.csv.gz OUT.txt   (FORECAST_DATA_DIR set)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import run_1b
from pipeline import data_prep

LEV, ENS, OUT = sys.argv[1:4]
GAUGES = {"Z.38": ("DK", "pluakdaeng", 10.6), "Z.11": ("PS", "bothong", 9.16), "RAY001": ("PS", "bothong", 3.966)}
H = 7
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


lev = pd.read_csv(LEV, parse_dates=["date"])
ens_all = pd.read_csv(ENS)
ens_all = ens_all[pd.to_datetime(ens_all["date"]) >= pd.to_datetime(ens_all["origin"])]
rows = []
for g, (site, point, bank) in GAUGES.items():
    rain = data_prep.daily(site).set_index("date")["rain"]
    L = lev[lev["station"] == g].set_index("date")["max_m"]
    days = pd.date_range("2024-01-08", L.index.max() - pd.Timedelta(days=H))
    days = days[days <= rain.index.max() - pd.Timedelta(days=H)]
    r = rain.reindex(pd.date_range(rain.index.min(), rain.index.max()))
    feat = pd.DataFrame(index=days)
    feat["l1"] = L.reindex(days - pd.Timedelta(days=1)).to_numpy()
    feat["l3"] = pd.concat([L.reindex(days - pd.Timedelta(days=k)).reset_index(drop=True) for k in (1, 2, 3)], axis=1).max(axis=1).to_numpy()
    for k in (1, 3, 7):
        feat[f"p{k}"] = [r[d - pd.Timedelta(days=k):d - pd.Timedelta(days=1)].sum() for d in days]
    fut = np.array([r.reindex(pd.date_range(d, periods=H)).to_numpy(float) for d in days])
    feat["doy_s"], feat["doy_c"] = np.sin(2 * np.pi * days.dayofyear / 365.25), np.cos(2 * np.pi * days.dayofyear / 365.25)
    target = np.array([L.reindex(pd.date_range(d, periods=H)).max() for d in days])
    ok = feat.notna().all(axis=1).to_numpy() & ~np.isnan(fut).any(axis=1) & ~np.isnan(target)
    feat, fut, target, days = feat[ok], fut[ok], target[ok], days[ok]

    def fx(f, fr):
        x = f.copy()
        x["r7"], x["r3"], x["rmax"] = fr.sum(axis=1), fr[:, :3].sum(axis=1), fr.max(axis=1)
        return x

    X = fx(feat, fut)
    # ensemble rain for this point, on the origins
    e = ens_all[ens_all["point"] == point]
    E = {}
    for o, gg in e.groupby("origin"):
        p = gg.pivot_table(index="member", columns="date", values="mm").sort_index(axis=1)
        if p.shape[1] >= H and p.shape[0] >= 40:
            E[pd.Timestamp(o)] = p.iloc[:, :H].to_numpy(float)
    run_1b.H = H
    Eqm = run_1b.quantile_map(E, pd.DataFrame({"date": r.index, "rain": r.to_numpy()}))
    origins = [o for o in sorted(E) if o in set(days)]
    say(f"\n## {g} ({len(days)} days, {len(origins)} ENS origins); bank {bank} m; max seen {np.nanmax(target):.2f} m")
    for line_name, thr in (("bank - 1 m", bank - 1.0), ("bank", bank)):
        y = (target >= thr).astype(int)
        res = {k: [] for k in ("ens", "ens_qm", "mean", "mean_qm", "obs", "norain", "clim")}
        ev, regs = [], {k: [] for k in ("mean", "ens", "obs", "norain")}
        for yr in sorted(set(days.year)):
            tr, te = days.year != yr, [o for o in origins if o.year == yr]
            if y[tr].sum() < 3 or not te:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[tr], y[tr])
            m0 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(feat[tr], y[tr])
            rg = make_pipeline(StandardScaler(), Ridge(1.0)).fit(X[tr], target[tr])
            rg0 = make_pipeline(StandardScaler(), Ridge(1.0)).fit(feat[tr], target[tr])
            idx = [days.get_loc(o) for o in te]
            f, base = feat.iloc[idx], y[tr].mean()
            for j, o in enumerate(te):
                fj = f.iloc[[j]]
                Xm = fx(pd.concat([fj] * len(E[o])), E[o])
                Xq = fx(pd.concat([fj] * len(Eqm[o])), Eqm[o])
                res["ens"].append(m.predict_proba(Xm)[:, 1].mean())
                res["ens_qm"].append(m.predict_proba(Xq)[:, 1].mean())
                res["mean"].append(m.predict_proba(fx(fj, E[o].mean(axis=0)[None]))[0, 1])
                res["mean_qm"].append(m.predict_proba(fx(fj, Eqm[o].mean(axis=0)[None]))[0, 1])
                res["obs"].append(m.predict_proba(X.iloc[[idx[j]]])[0, 1])
                res["norain"].append(m0.predict_proba(fj)[0, 1])
                res["clim"].append(base)
                ev.append(y[idx[j]])
                if line_name == "bank":
                    regs["mean"].append(abs(rg.predict(fx(fj, E[o].mean(axis=0)[None]))[0] - target[idx[j]]))
                    regs["ens"].append(abs(np.median(rg.predict(Xm)) - target[idx[j]]))
                    regs["obs"].append(abs(rg.predict(X.iloc[[idx[j]]])[0] - target[idx[j]]))
                    regs["norain"].append(abs(rg0.predict(fj)[0] - target[idx[j]]))
        ev = np.array(ev)
        if not len(ev):
            say(f"{line_name} {thr:.2f} m: too few events to train")
            continue
        br = {k: np.mean((np.array(v) - ev) ** 2) for k, v in res.items()}
        hit = {k: (int(((np.array(v) >= 0.3) & (ev == 1)).sum()), int(((np.array(v) >= 0.3) & (ev == 0)).sum())) for k, v in res.items()}
        say(f"{line_name} ({thr:.2f} m): {int(ev.sum())} event origins of {len(ev)} (all days: {int(y.sum())} of {len(y)})")
        say("  Brier  " + "  ".join(f"{k} {v:.3f}" for k, v in br.items()))
        say("  P>=30% hits/false  " + "  ".join(f"{k} {a}/{b}" for k, (a, b) in hit.items()))
        rows.append(dict(gauge=g, line=line_name, events=int(ev.sum()), origins=len(ev), **{f"brier_{k}": v for k, v in br.items()}))
        if regs["mean"]:
            say("  7-day max level MAE (m)  " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in regs.items()))
pd.DataFrame(rows).round(4).to_csv(OUT.replace(".txt", ".csv"), index=False)
open(OUT, "w").write("\n".join(lines) + "\n")
