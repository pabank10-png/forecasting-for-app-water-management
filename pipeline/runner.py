"""Run one model for every site: forecast from today and, on monthly runs, the rolling backtest."""
import time
import warnings

import numpy as np
import pandas as pd

from common import backtest
from common.metrics import coverage, mae
from common.registry import load

from . import data_prep, plan

warnings.filterwarnings("ignore")

# Backtest windows, as in the Excel model reports (2026-10-26)
M_FIRST, M_LAST, M_STEPS_SCORED = "2016-01-01", "2025-12-31", 3
D_FIRST, D_LAST, D_EVERY, D_RAIN_END = "2016-01-01", "2020-09-03", 5, "2020-09-10"


def _rows(fc, dates):
    return [{"step": int(s), "date": d, "p10": data_prep.clean(a), "p50": data_prep.clean(b), "p90": data_prep.clean(c)}
            for s, d, a, b, c in zip(fc["step"], dates, fc["p10"], fc["p50"], fc["p90"])]


def _score(bt, steps=None):
    b = bt.dropna(subset=["actual"])
    if steps:
        b = b[b["step"] <= steps]
    if not len(b):
        return None
    per = [mae(g["actual"], g["p50"]) for _, g in b.groupby("step")]
    return {"mae": mae(b["actual"], b["p50"]), "coverage": coverage(b["actual"], b["p10"], b["p90"]),
            "n": int(len(b)), "mae_steps": [round(v, 4) for v in per]}


def _bt_json(bt, steps=None):
    b = bt if not steps else bt[bt["step"] <= steps]
    return {"key": [f"{str(o)[:10]}|{s}" for o, s in zip(b["origin"], b["step"])],
            "actual": [data_prep.clean(v) for v in b["actual"]],
            "p10": [data_prep.clean(v) for v in b["p10"]], "p50": [data_prep.clean(v) for v in b["p50"]],
            "p90": [data_prep.clean(v) for v in b["p90"]]}


def monthly_site(name, site, kind):
    mod, kw, h = load(plan.folder(name)), plan.kwargs(name), plan.HORIZON[(kind, plan.T1)]
    df = data_prep.monthly(site)
    m = mod.Model(**kw).fit(df)
    fc = m.predict(df, h)
    out = {"forecast": _rows(fc, data_prep.month_ends(df["date"].iloc[-1], h))}
    if kind == "monthly":
        n_obs = df["y"].last_valid_index() + 1
        d = df["date"].astype(str).str[:10]
        origins = [i for i in df.index[:n_obs] if M_FIRST <= d[i] <= M_LAST and pd.notna(df["y"][i])]
        bt = backtest.run(lambda: mod.Model(**kw), df.iloc[:n_obs], origins, 12,
                          refit_every=plan.REFIT_EVERY.get(name, 12))
        out["score"] = _score(bt, M_STEPS_SCORED)
        out["bt"] = _bt_json(bt, M_STEPS_SCORED)
    return out


def daily_site(name, site, kind):
    mod, kw, h = load(plan.folder(name)), plan.kwargs(name), plan.HORIZON[(kind, plan.T1B)]
    df = data_prep.daily(site)
    first = df["date"].iloc[-1] + pd.Timedelta(days=1)
    rain = name in plan.RAIN_MODELS
    futures = data_prep.forecast_rain(site, first, h) if rain else {}
    cur = data_prep.with_forecast_row(df)
    m = mod.Model(**kw).fit(df)
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(first, periods=h, freq="D")]
    out = {}
    if rain and futures:
        ens = load(f"{plan.T1B}/90_rain_source_ensemble").Model(base=m)
        fc = ens.predict(cur, h, futures)  # base model is already fitted
        out["by_source"] = {s: _rows(f, dates) for s, f in ens.members.items()}
    else:
        fc = m.predict(cur, h)
        if rain:
            out["note"] = "ไม่มีฝนพยากรณ์ ใช้ฝน 0 มม."
    out["forecast"] = _rows(fc, dates)
    if kind == "monthly":
        hist, last = daily_frame_for_backtest(site, rain)
        if hist is not None:
            d = hist["date"].astype(str).str[:10]
            origins = [i for i in hist.index[(d >= D_FIRST) & (d <= last)][::D_EVERY] if pd.notna(hist["y"][i])]
            fut = ("rain",) if rain else ()
            bt = backtest.run(lambda: mod.Model(**kw), hist, origins, h, future_cols=fut, refit_every=73)
            out["score"] = _score(bt)
            out["bt"] = _bt_json(bt)
    return out


def daily_frame_for_backtest(site, rain):
    """Daily rows while the dam gauges recorded rain (to Sep 2020; the forecast rain is the observed rain).
    A reservoir without dam rain (PS) is tested on its RID record instead, by the models that need no rain."""
    from common.data import daily_frame
    df = daily_frame(site)
    if df["rain"].notna().sum() > 1500:
        return df[df["date"] <= D_RAIN_END].reset_index(drop=True), D_LAST
    if rain:
        return None, None
    df = df.iloc[:df["y"].last_valid_index() + 1].reset_index(drop=True)
    return df, str(df["date"].iloc[-8])[:10]


def run(name, kind, sites=None):
    task = name.split("/")[0]
    out = {"model": name, "task": task, "agent": plan.MODELS[name][0], "kind": kind, "sites": {}, "errors": {}}
    t0 = time.time()
    for site in sites or plan.SITES[task]:
        t = time.time()
        try:
            res = (monthly_site if task == plan.T1 else daily_site)(name, site, kind)
            res["seconds"] = round(time.time() - t, 1)
            out["sites"][site] = res
            print(f"{name} {site} ok {res['seconds']}s", flush=True)
        except Exception as e:  # noqa: BLE001 - one site failing must not stop the others
            out["errors"][site] = f"{type(e).__name__}: {e}"[:300]
            print(f"{name} {site} failed: {type(e).__name__}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out
