"""Summarise the ensemble streamflow test (results from exp-ens-1b artifacts in res/)."""
import glob, sys
import numpy as np, pandas as pd

RES = sys.argv[1] if len(sys.argv) > 1 else "res"
d = pd.concat([pd.read_csv(p) for p in glob.glob(f"{RES}/**/*.csv.gz", recursive=True)], ignore_index=True)
mcols = [c for c in d.columns if c.startswith("m") and c[1:].isdigit()]


def crps_ens(x, y):
    x = np.sort(x[~np.isnan(x)]); n = len(x)
    if n == 0: return np.nan
    return np.mean(np.abs(x - y)) - np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * n)


def crps_q(p10, p50, p90, y):  # quantile score over the three quantiles, scaled to CRPS units
    s = 0
    for q, v in ((.1, p10), (.5, p50), (.9, p90)):
        s += 2 * ((y < v) - q) * (v - y)
    return s / 3


rows = []
for (site, model), g in d.groupby(["site", "model"]):
    y = g["actual"].to_numpy(float)
    r = dict(site=site, model=model, origins=g["origin"].nunique())
    r["mae_obs_rain"] = np.mean(np.abs(g["obs_p50"] - y))
    r["mae_mean_rain"] = np.mean(np.abs(g["mean_p50"] - y))
    r["cov_model_p10_p90"] = np.mean((y >= g["mean_p10"]) & (y <= g["mean_p90"]))
    r["crps_model"] = np.mean(crps_q(g["mean_p10"], g["mean_p50"], g["mean_p90"], y))
    if mcols and g[mcols].notna().any().any():
        M = g[mcols].to_numpy(float)
        med = np.nanmedian(M, axis=1)
        lo, hi = np.nanpercentile(M, 10, axis=1), np.nanpercentile(M, 90, axis=1)
        r["mae_ens_median"] = np.mean(np.abs(med - y))
        r["cov_ens_p10_p90"] = np.mean((y >= lo) & (y <= hi))
        r["crps_ens"] = np.mean([crps_ens(M[i], y[i]) for i in range(len(y))])
        # ensemble members + the model's own spread (each member scaled by p10/p50, p90/p50 of the mean-rain run)
        lo_r = (g["mean_p10"] / g["mean_p50"].where(g["mean_p50"] > 0)).fillna(1).to_numpy()
        hi_r = (g["mean_p90"] / g["mean_p50"].where(g["mean_p50"] > 0)).fillna(1).to_numpy()
        lo2 = np.nanpercentile(M * lo_r[:, None], 10, axis=1); hi2 = np.nanpercentile(M * hi_r[:, None], 90, axis=1)
        r["cov_ens_plus_model"] = np.mean((y >= lo2) & (y <= hi2))
        # 7-day totals
        t = g.assign(med=med).groupby("origin")
        tot = t.agg(actual=("actual", "sum"), mean=("mean_p50", "sum"), obs=("obs_p50", "sum"),
                    rain_obs=("obs_rain", "sum"), rain_ens=("ens_rain_mean", "sum"))
        Mt = g.groupby("origin")[mcols].sum(min_count=1).loc[tot.index].to_numpy(float)
        tot["ens_med"] = np.nanmedian(Mt, axis=1)
        tot["lo"], tot["hi"] = np.nanpercentile(Mt, 10, axis=1), np.nanpercentile(Mt, 90, axis=1)
        r["mae7_mean_rain"] = np.mean(np.abs(tot["mean"] - tot["actual"]))
        r["mae7_ens_median"] = np.mean(np.abs(tot["ens_med"] - tot["actual"]))
        r["mae7_obs_rain"] = np.mean(np.abs(tot["obs"] - tot["actual"]))
        r["cov7_ens"] = np.mean((tot["actual"] >= tot["lo"]) & (tot["actual"] <= tot["hi"]))
        thr = tot["actual"].quantile(0.8)
        ev = (tot["actual"] > thr).astype(float)
        pe = (Mt > thr).mean(axis=1)
        pdet = (tot["mean"] > thr).astype(float)
        base = ev.mean()
        r["brier_ens"] = np.mean((pe - ev) ** 2); r["brier_mean_rain"] = np.mean((pdet - ev) ** 2)
        r["brier_clim"] = np.mean((base - ev) ** 2)
        wet = tot["rain_obs"] >= 100
        r["wet_origins"] = int(wet.sum())
        if wet.any():
            r["mae7_wet_mean_rain"] = np.mean(np.abs(tot["mean"] - tot["actual"])[wet])
            r["mae7_wet_ens_median"] = np.mean(np.abs(tot["ens_med"] - tot["actual"])[wet])
            r["cov7_wet_ens"] = np.mean(((tot["actual"] >= tot["lo"]) & (tot["actual"] <= tot["hi"]))[wet])
    rows.append(r)
out = pd.DataFrame(rows).round(4)
out.to_csv("summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(out.to_string(index=False))

# Rain check: ENS mean 7-day rain vs the rain that fell (one model's rows carry both)
one = d[d["model"] == d["model"].iloc[0]]
t = one.groupby(["site", "origin"]).agg(obs=("obs_rain", "sum"), ens=("ens_rain_mean", "sum")).reset_index()
print("\n7-day rain, ENS mean vs observed:")
print(t.groupby("site").apply(lambda g: pd.Series({"n": len(g), "obs_mean": g.obs.mean(), "ens_mean": g.ens.mean(),
                                                    "r": g.obs.corr(g.ens), "mae": (g.obs - g.ens).abs().mean()})).round(2))
print("\nwettest observed weeks:")
print(t.sort_values("obs").tail(12).round(1).to_string(index=False))

# By lead day: MAE of mean-rain run, ensemble median, observed rain; Brier of "day inflow in top 20%"
print("\nBy lead (mean over sites, rain models):")
rm = d[d[mcols].notna().any(axis=1)].copy()
M = rm[mcols].to_numpy(float)
rm["med"] = np.nanmedian(M, axis=1)
thr = rm.groupby(["site", "model"])["actual"].transform(lambda s: s.quantile(0.8))
ev = (rm["actual"] > thr).astype(float)
rm["b_ens"] = ((M > thr.to_numpy()[:, None]).mean(axis=1) - ev) ** 2
rm["b_mean"] = ((rm["mean_p50"] > thr).astype(float) - ev) ** 2
rm["e_mean"] = (rm["mean_p50"] - rm["actual"]).abs()
rm["e_med"] = (rm["med"] - rm["actual"]).abs()
rm["e_obs"] = (rm["obs_p50"] - rm["actual"]).abs()
qm = rm["model"].str.endswith("+qm")
for tag, g in (("raw", rm[~qm]), ("qm", rm[qm])):
    t = g.groupby("lead")[["e_mean", "e_med", "e_obs", "b_mean", "b_ens"]].mean()
    t["mae_gain_%"] = 100 * (1 - t["e_med"] / t["e_mean"])
    t["brier_gain_%"] = 100 * (1 - t["b_ens"] / t["b_mean"])
    print(tag); print(t.round(4).to_string())
best = rm[~qm].groupby(["site", "model", "lead"])[["e_mean", "e_med"]].mean().reset_index()
print("\nlightgbm_storm / hbv / gr4j by lead, all sites:")
print(best[best.model.isin(["04_hbv", "03_gr4j", "05_lightgbm_storm", "05_lightgbm_daily"])].groupby(["model", "lead"])[["e_mean", "e_med"]].mean().round(4).unstack(0).to_string())
