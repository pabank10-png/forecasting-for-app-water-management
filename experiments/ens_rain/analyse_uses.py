"""Summarise the ensemble-rain uses test (run_uses.py results in a folder).
Usage: python analyse_uses.py RES_DIR DATA_CSV_DIR OUT_PREFIX
Parts: 2a heavy-rain switch, 2b Storm Trigger, 3 storage lines in 7/14 days, 4 month-end nowcast.
"""
import glob
import sys

import numpy as np
import pandas as pd

RES, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
HYDRO = {"DK": "04_hbv", "KY": "04_hbv", "NPL": "04_hbv", "PS": "03_gr4j"}
CAP = {"THREE": 318.131, "PS": 330.0}
YELLOW = {"THREE": [220, 200, 180, 175, 170, 175, 180, 200, 210, 220, 230, 225],
          "PS": [250, 235, 220, 200, 200, 200, 200, 216, 250, 250, 250, 250]}
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def table(df, title):
    say(f"\n## {title}")
    say(df.to_string(index=False) if isinstance(df, pd.DataFrame) else str(df))


d = pd.concat([pd.read_csv(p) for p in glob.glob(f"{RES}/**/*.csv.gz", recursive=True)], ignore_index=True)
d["origin"] = pd.to_datetime(d["origin"])
d["date"] = pd.to_datetime(d["date"])
d["base_model"] = d["model"].str.split("+").str[0]
d["variant"] = d["model"].str.split("+", n=1).str[1]
mcols = [c for c in d.columns if c.startswith("m") and c[1:].isdigit()]
rcols = [c for c in d.columns if c.startswith("r") and c[1:].isdigit()]


def brier(p, ev):
    return float(np.mean((np.asarray(p, float) - np.asarray(ev, float)) ** 2))


def week(g, cols, n=7):
    return g[g["lead"] <= n].groupby("origin")[cols].sum(min_count=1)


# ---------------- 2a heavy-rain switch ----------------
rows = []
for (site, var), g in d.groupby(["site", "variant"]):
    b = g[g["base_model"] == "05_lightgbm_daily"]
    h = g[g["base_model"] == HYDRO[site]]
    if b.empty or h.empty:
        continue
    mc = [c for c in mcols if b[c].notna().any()]
    rc = [c for c in rcols if b[c].notna().any()]
    wb, wh = week(b, ["actual", "mean_p50", "obs_rain", "fc_rain_mean"] + mc + rc), week(h, ["mean_p50"] + mc)
    o = wb.index.intersection(wh.index)
    wb, wh = wb.loc[o], wh.loc[o]
    act = wb["actual"]
    r7 = wb["fc_rain_mean"]
    pr = (wb[rc] >= 100).mean(axis=1)
    base_med, hyd_med = wb[mc].median(axis=1), wh[mc].median(axis=1)
    wet = wb["obs_rain"] >= 100
    strat = {"LightGBM only": wb["mean_p50"],
             "HBV/GR4J only": wh["mean_p50"],
             "now: mean rain >=100 -> HBV/GR4J": np.where(r7 >= 100, wh["mean_p50"], wb["mean_p50"]),
             "mean rain >=70 -> HBV/GR4J": np.where(r7 >= 70, wh["mean_p50"], wb["mean_p50"])}
    for p in (0.1, 0.2, 0.3, 0.5):
        strat[f"P(rain>=100) >= {p:.0%} -> HBV/GR4J"] = np.where(pr >= p, wh["mean_p50"], wb["mean_p50"])
    strat["blend by P(rain>=100)"] = pr * wh["mean_p50"] + (1 - pr) * wb["mean_p50"]
    strat["blend by P, ensemble medians"] = pr * hyd_med + (1 - pr) * base_med
    for k, v in strat.items():
        v = pd.Series(np.asarray(v, float), index=o)
        rows.append(dict(site=site, rain=var, rule=k, mae7=np.mean(np.abs(v - act)),
                         mae7_wet=np.mean(np.abs(v - act)[wet]) if wet.any() else np.nan,
                         mae7_dry=np.mean(np.abs(v - act)[~wet]), wet_weeks=int(wet.sum()), weeks=len(o),
                         switched=int((r7 >= 100).sum()) if k.startswith("now") else
                         int((pr >= float(k.split(">= ")[1].split("%")[0]) / 100).sum()) if k.startswith("P(") else np.nan))
sw = pd.DataFrame(rows).round(2)
table(sw, "2a Heavy-rain switch: 7-day inflow MAE (MCM), LightGBM vs HBV/GR4J; wet = observed 7-day rain >= 100 mm")

# ---------------- 2b Storm Trigger (rain events) ----------------
one = d[d["base_model"] == "05_lightgbm_daily"]
rows = []
for var, g in one.groupby("variant"):
    rc = [c for c in rcols if g[c].notna().any()]
    w = g[g["lead"] <= 7].groupby(["site", "origin"])[["obs_rain", "fc_rain_mean"] + rc].sum(min_count=1)
    for name, sites in (("THREE", ["DK", "KY", "NPL"]), ("PS", ["PS"])):
        x = w.loc[w.index.get_level_values(0).isin(sites)].groupby(level=1).mean()
        for thr in (100, 150):
            ev = x["obs_rain"] >= thr
            det = x["fc_rain_mean"] >= thr
            p = (x[rc] >= thr).mean(axis=1)
            r = dict(site=name, rain=var, threshold_mm=thr, weeks=len(x), events=int(ev.sum()),
                     det_hits=int((det & ev).sum()), det_false=int((det & ~ev).sum()),
                     brier_det=brier(det, ev), brier_ens=brier(p, ev), brier_clim=brier(np.full(len(ev), ev.mean()), ev))
            for q in (0.1, 0.2, 0.3, 0.5):
                f = p >= q
                r[f"p{int(q * 100)}_hits"], r[f"p{int(q * 100)}_false"] = int((f & ev).sum()), int((f & ~ev).sum())
            rows.append(r)
            if thr == 150:
                big = x[ev | det | (p >= 0.1)].assign(p=p).round(2)
                say(f"\n{name} {var} weeks with 7-day rain >= 150 mm (observed, mean forecast or P>=10%):")
                say(big[["obs_rain", "fc_rain_mean", "p"]].to_string())
table(pd.DataFrame(rows).round(3), "2b Storm Trigger: 7-day site rain events, deterministic (mean) vs ensemble probability")

# ---------------- 3 storage lines ----------------
def daily(site):
    x = pd.read_csv(f"{DATA}/daily_{site}.csv", encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
    return x


DD = {s: daily(s) for s in ["DK", "KY", "NPL", "PS"]}
rows, checks = [], []
for (mdl, var), g in d.groupby(["base_model", "variant"]):
    mc = [c for c in mcols if g[c].notna().any()]
    for name, sites in (("THREE", ["DK", "KY", "NPL"]), ("PS", ["PS"])):
        gs = g[g["site"].isin(sites)]
        if gs["site"].nunique() < len(sites):
            continue
        cols = ["actual", "mean_p50", "clim_p50", "obs_p50"] + mc
        agg = gs[gs["lead"] <= 14].groupby(["origin", "lead"])[cols].sum(min_count=len(sites)).dropna(subset=["actual"])
        st = sum(DD[s]["storage_mcm"] for s in sites)
        out = sum(DD[s]["outflow_mcm_rid"] for s in sites)
        recs = []
        for o, a in agg.groupby(level=0):
            a = a.droplevel(0).sort_index()
            if len(a) < 14:
                continue
            days = pd.date_range(o, periods=14)
            s0 = st.get(o - pd.Timedelta(days=1), np.nan)
            q = out.reindex(days).to_numpy(float)
            sact = st.reindex(days).to_numpy(float)
            if np.isnan(s0) or np.isnan(q).any() or np.isnan(sact).any():
                continue
            path = lambda v: s0 + np.cumsum(np.asarray(v, float) - q)  # noqa: E731
            yl = np.array([YELLOW[name][dd.month - 1] for dd in days])
            rec = dict(origin=o, s0=s0, yl0=yl[0], resid=np.abs(path(a["actual"]) - sact).max())
            for n in (7, 14):
                for ev_name, f in (("above 90% cap", lambda s: (s >= 0.9 * CAP[name]).any()),
                                   ("leave yellow", lambda s: (s >= yl[:len(s)]).any()),
                                   ("enter yellow", lambda s: (s < yl[:len(s)]).any())):
                    rec[f"{ev_name}|{n}|act"] = f(sact[:n])
                    for k in ("mean_p50", "clim_p50", "obs_p50"):
                        rec[f"{ev_name}|{n}|{k}"] = f(path(a[k])[:n])
                    rec[f"{ev_name}|{n}|ens"] = np.mean([f(path(a[c])[:n]) for c in mc]) if mc else np.nan
            recs.append(rec)
        R = pd.DataFrame(recs)
        if R.empty:
            continue
        checks.append(dict(site=name, model=mdl, rain=var, origins=len(R), max_balance_residual_median=R["resid"].median()))
        for n in (7, 14):
            for ev_name, start_ok in (("above 90% cap", R["s0"] < 0.9 * CAP[name]), ("leave yellow", R["s0"] < R["yl0"]),
                                      ("enter yellow", R["s0"] >= R["yl0"])):
                Rr = R[start_ok]
                if Rr.empty:
                    continue
                ev = Rr[f"{ev_name}|{n}|act"].astype(float)
                r = dict(site=name, model=mdl, rain=var, event=ev_name, days=n, origins=len(Rr), events=int(ev.sum()),
                         brier_ens=brier(Rr[f"{ev_name}|{n}|ens"], ev),
                         brier_mean_rain=brier(Rr[f"{ev_name}|{n}|mean_p50"], ev),
                         brier_no_fc_rain=brier(Rr[f"{ev_name}|{n}|clim_p50"], ev),
                         brier_perfect_rain=brier(Rr[f"{ev_name}|{n}|obs_p50"], ev),
                         brier_base_rate=brier(np.full(len(ev), ev.mean()), ev))
                rows.append(r)
table(pd.DataFrame(checks).round(2), "3 check: storage path with real inflow and real outflow vs recorded storage (max abs diff, MCM)")
table(pd.DataFrame(rows).round(3), "3 Storage lines within 7/14 days (actual outflow assumed known); only origins that start on the other side of the line")

# ---------------- 4 month-end nowcast ----------------
rows = []
for (mdl, var), g in d.groupby(["base_model", "variant"]):
    if g["lead"].max() < 28:
        continue
    mc = [c for c in mcols if g[c].notna().any()]
    for name, sites in (("THREE", ["DK", "KY", "NPL"]), ("PS", ["PS"]), ("DK", ["DK"]), ("KY", ["KY"]), ("NPL", ["NPL"])):
        gs = g[g["site"].isin(sites)]
        if gs["site"].nunique() < len(sites):
            continue
        gs = gs[gs["date"].dt.to_period("M") == gs["origin"].dt.to_period("M")]
        cols = ["actual", "mean_p50", "mean_p10", "mean_p90", "clim_p50", "obs_p50"] + mc
        rest = gs.groupby(["origin", "site"])[cols].sum(min_count=1).groupby(level=0).sum(min_count=len(sites))
        inflow = sum(DD[s]["inflow_mcm_rid"] for s in sites)
        recs = []
        for o, r in rest.iterrows():
            me = o + pd.offsets.MonthEnd(0)
            got = inflow[o.replace(day=1):o - pd.Timedelta(days=1)]
            whole = inflow[o.replace(day=1):me]
            if whole.isna().any() or len(whole) < me.day:
                continue
            past = got.sum()
            M = r[mc].to_numpy(float) + past
            lo_r = r["mean_p10"] / r["mean_p50"] if r["mean_p50"] > 0 else 1
            hi_r = r["mean_p90"] / r["mean_p50"] if r["mean_p50"] > 0 else 1
            recs.append(dict(origin=o, day=o.day, actual=whole.sum(), clim=past + r["clim_p50"], mean=past + r["mean_p50"],
                             ens_med=np.median(M), obs=past + r["obs_p50"],
                             lo=np.percentile(M, 10) - (r["mean_p50"] - r["mean_p10"]),
                             hi=np.percentile(M, 90) + (r["mean_p90"] - r["mean_p50"]),
                             mlo=past + r["mean_p10"], mhi=past + r["mean_p90"]))
        R = pd.DataFrame(recs)
        if R.empty:
            continue
        for day_grp, Rg in (("all", R), ("day 1-11", R[R["day"] <= 11]), ("day 16-26", R[R["day"] >= 16])):
            a = Rg["actual"]
            rows.append(dict(site=name, model=mdl, rain=var, origins=day_grp, n=len(Rg),
                             mae_no_fc_rain=np.mean(np.abs(Rg["clim"] - a)), mae_mean_rain=np.mean(np.abs(Rg["mean"] - a)),
                             mae_ens_median=np.mean(np.abs(Rg["ens_med"] - a)), mae_perfect_rain=np.mean(np.abs(Rg["obs"] - a)),
                             cov_model_p10_p90=np.mean((a >= Rg["mlo"]) & (a <= Rg["mhi"])),
                             cov_ens_plus_model=np.mean((a >= Rg["lo"]) & (a <= Rg["hi"]))))
table(pd.DataFrame(rows).round(2), "4 Month-end inflow nowcast (MCM): observed so far + forecast to month end")

open(f"{OUT}.txt", "w").write("\n".join(lines) + "\n")
