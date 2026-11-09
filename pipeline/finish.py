"""After the models: Error Gate, Combination + champion, month-end storage, and the summary the HQ reads."""
import numpy as np
import pandas as pd

from common import ensemble
from common.registry import load
from common.metrics import mae

from . import data_prep, plan

LAMBDA, MARGIN = 0.5, 0.02
TARGET_COVER = 0.80  # P10–P90 should hold 80% of actual values


def cover_band(score):
    """Coverage range a calibrated model lands in 95% of the time: 0.80 ± 2·√(0.8·0.2/n),
    n = backtest origins (points ÷ steps, since steps of one origin move together).
    Without a count it falls back to 0.70–0.90."""
    n, steps = score.get("n"), len(score.get("mae_steps") or []) or 1
    tol = 2 * np.sqrt(TARGET_COVER * (1 - TARGET_COVER) / (n / steps)) if n else 0.10
    return round(TARGET_COVER - tol, 3), round(TARGET_COVER + tol, 3)


def reuse(results, reused, prev_summary, kind):
    """Put the previous run's forecast back for (model, site) pairs whose inputs did not change."""
    if not prev_summary:
        return
    for name, sites in reused.items():
        task = name.split("/")[0]
        h = plan.HORIZON[(kind, task)]
        for site in sites:
            entry = (prev_summary.get("tasks") or {}).get(task, {}).get(site, {})
            m = (entry.get("models") or {}).get(name)
            if not m or not m.get("forecast"):
                continue
            r = results.setdefault(name, {"model": name, "task": task, "agent": plan.MODELS[name][0], "kind": kind,
                                          "sites": {}, "errors": {}})
            r["sites"][site] = {**{k: v for k, v in m.items() if k != "agent"}, "forecast": m["forecast"][:h],
                                "reused": True}
            if entry.get("rain7_mm") is not None:  # the forecast rain behind the reused forecast (heavy-rain switch)
                r["sites"][site]["rain7"] = entry["rain7_mm"]


def _sum_rows(rows_by_part):
    """Add P50s; P10/P90 halfway between 'parts move together' (sum of P10s) and 'parts are independent'
    (P50 ± root-sum-square of the gaps). Together alone was too wide in the backtest, independent too narrow."""
    p50 = sum(r["p50"] for r in rows_by_part)
    lo_t, hi_t = sum(r["p10"] for r in rows_by_part), sum(r["p90"] for r in rows_by_part)
    lo_i = p50 - sum(max(0.0, r["p50"] - r["p10"]) ** 2 for r in rows_by_part) ** 0.5
    hi_i = p50 + sum(max(0.0, r["p90"] - r["p50"]) ** 2 for r in rows_by_part) ** 0.5
    return max(0.0, (lo_t + lo_i) / 2), p50, (hi_t + hi_i) / 2


def _sum_bt(got, field):
    """Backtest of the parts added up, matched on 'origin|step' keys (None when a part has no backtest)."""
    if not all(g.get(field) for g in got):
        return None
    maps = [{k: i for i, k in enumerate(g[field]["key"])} for g in got]
    keys = [k for k in got[0][field]["key"] if all(k in m for m in maps)]
    bt = {"key": [], "actual": [], "p10": [], "p50": [], "p90": []}
    for k in keys:
        rows = [{q: g[field][q][m[k]] for q in ("actual", "p10", "p50", "p90")} for g, m in zip(got, maps)]
        if any(x["p50"] is None for x in rows):
            continue
        lo, mid, hi = _sum_rows(rows)
        act = None if any(x["actual"] is None for x in rows) else sum(x["actual"] for x in rows)
        for q, v in zip(("key", "actual", "p10", "p50", "p90"), (k, act, lo, mid, hi)):
            bt[q].append(v)
    return bt


def add_sums(results):
    """'<model>+sum' for THREE = the model run on DK, KY and NPL separately, then added (forecast and backtest)."""
    from .runner import _score
    for name in [n for n in results if n.startswith(plan.T1) and not n.endswith(plan.SUM)]:
        r = results[name]
        for site, parts in plan.PARTS.items():
            got = [r["sites"].get(p) for p in parts]
            if any(g is None or not g.get("forecast") for g in got):
                continue
            fc = []
            for rows in zip(*(g["forecast"] for g in got)):
                if any(x["p50"] is None for x in rows):
                    break
                lo, mid, hi = _sum_rows(rows)
                fc.append({"step": rows[0]["step"], "date": rows[0]["date"], "p10": round(lo, 4), "p50": round(mid, 4),
                           "p90": round(hi, 4)})
            entry = {"forecast": fc, "parts": list(parts)}
            bt = _sum_bt(got, "bt")
            if bt:
                df = pd.DataFrame({"step": [int(k.split("|")[1]) for k in bt["key"]], "actual": bt["actual"],
                                   "p10": bt["p10"], "p50": bt["p50"], "p90": bt["p90"]}).astype(float)
                from .runner import M_STEPS_SCORED
                entry["score"] = _score(df, M_STEPS_SCORED)
                entry["bt"] = bt
            bt12 = _sum_bt(got, "bt12")
            if bt12:
                entry["bt12"] = bt12
            results[name + plan.SUM] = {"model": name + plan.SUM, "task": r["task"], "agent": r["agent"],
                                         "kind": r["kind"], "sites": {site: entry}, "errors": {}}


def scores_from(results):
    """{task: {site: {model: score}}} from this run's backtests."""
    out = {}
    for name, r in results.items():
        for site, s in r["sites"].items():
            if s.get("score"):
                out.setdefault(r["task"], {}).setdefault(site, {})[name] = s["score"]
    return out


def gate(task, site, scores):
    """Pass = MAE below the Baseline (climatology) and P10–P90 coverage inside cover_band (about 0.73–0.87)."""
    sc = scores.get(task, {}).get(site, {})
    base = sc.get(plan.BASELINE[task], {}).get("mae")
    ev = {}
    for name, s in sc.items():
        if name == plan.BASELINE[task]:
            ev[name] = {"mae": s["mae"], "coverage": s["coverage"], "pass": None, "skill": 0.0}
            continue
        lo, hi = cover_band(s)
        ok = base is not None and s["mae"] is not None and s["mae"] < base and lo <= (s["coverage"] or 0) <= hi
        ev[name] = {"mae": s["mae"], "coverage": s["coverage"], "cover_band": [lo, hi], "pass": bool(ok),
                    "skill": None if not base or s["mae"] is None else round(1 - s["mae"] / base, 4)}
    return ev


def _blend_mae(results, task, site, weights):
    """MAE of a weighted P50 blend on the backtest points every member forecast."""
    bts = {m: results[m]["sites"][site]["bt"] for m in weights if site in results.get(m, {}).get("sites", {})
           and results[m]["sites"][site].get("bt")}
    if len(bts) != len(weights):
        return None, None
    frames = {m: pd.Series(b["p50"], index=b["key"], dtype=float) for m, b in bts.items()}
    act = pd.Series(next(iter(bts.values()))["actual"], index=next(iter(bts.values()))["key"], dtype=float)
    common = act.dropna().index
    for f in frames.values():
        common = common.intersection(f.dropna().index)
    if not len(common):
        return None, None
    blend = sum(frames[m][common] * w for m, w in weights.items())
    singles = {m: mae(act[common], f[common]) for m, f in frames.items()}
    return mae(act[common], blend), singles


def combine(task, site, results, ev, kind, previous):
    """Champion rule: blend of passing models only if it beats the best single model by >= 2%."""
    have = {m for m, r in results.items() if r["task"] == task and site in r["sites"]}
    members = [m for m, e in ev.items() if e["pass"] and m in have]
    base = plan.BASELINE[task]
    if kind != "monthly" and previous:  # daily runs keep the monthly choice
        prev = previous.get(task, {}).get(site)
        if prev:
            w = {m: v for m, v in prev["weights"].items() if m in have}
            if prev["champion"] == "ensemble" and len(w) >= 2:
                tot = sum(w.values())
                return {**prev, "weights": {m: v / tot for m, v in w.items()}, "reused": True}
            if prev["champion"] in have:
                return {**prev, "reused": True}
    if not members:
        fallback = base if base in have else next(iter(sorted(have)), None)
        return {"champion": fallback, "weights": {fallback: 1.0} if fallback else {}, "members": [],
                "reason": "ไม่มีโมเดลผ่านด่าน ใช้ Baseline" if ev else "ยังไม่มีคะแนนย้อนหลัง (รันรายเดือนก่อน)"}
    maes = {m: ev[m]["mae"] for m in members}
    w = ensemble.inverse_mae_weights(maes, LAMBDA)
    best = min(maes, key=maes.get)
    out = {"members": members, "weights": w, "best_single": best, "best_single_mae": maes[best]}
    if len(members) < 2:
        return {**out, "champion": best, "weights": {best: 1.0}}
    blend_mae, singles = _blend_mae(results, task, site, w)
    eq_mae, _ = _blend_mae(results, task, site, {m: 1 / len(members) for m in members})
    out.update(blend_mae=blend_mae, equal_mae=eq_mae)
    if blend_mae is not None and singles:
        best_same = min(singles.values())
        out["champion"] = "ensemble" if blend_mae <= best_same * (1 - MARGIN) else best
    else:
        out["champion"] = best
    if out["champion"] != "ensemble":
        out["weights"] = {best: 1.0}
    return out


def rain7(site, results):
    """7-day forecast rain (mm, mean of the sources) at a dam, as the daily rain models received it."""
    return next((r["sites"][site]["rain7"] for r in results.values()
                 if r.get("task") == plan.T1B and site in r.get("sites", {}) and r["sites"][site].get("rain7") is not None),
                None)


def heavy_rain_switch(site, results, fin):
    """Final 1-7 day forecast of a dam when its 7-day forecast rain is heavy: the hydrology model's forecast
    (plan.HEAVY_RAIN_MODEL). Returns (final, info); info is None when the switch does not apply."""
    hydro = plan.HEAVY_RAIN_MODEL.get(site)
    r7 = rain7(site, results)
    if hydro is None or r7 is None or r7 < plan.HEAVY_RAIN_MM:
        return fin, None
    rows = ((results.get(hydro) or {}).get("sites", {}).get(site) or {}).get("forecast")
    if not rows or any(r.get("p50") is None for r in rows):
        return fin, None
    return [dict(r) for r in rows], {"rain7_mm": r7, "threshold_mm": plan.HEAVY_RAIN_MM, "model": hydro}


def final_forecast(task, site, results, choice):
    if not choice.get("champion"):
        return None
    if choice["champion"] != "ensemble":
        return results[choice["champion"]]["sites"][site]["forecast"]
    fcs = {m: results[m]["sites"][site]["forecast"] for m in choice["weights"]}
    rows = next(iter(fcs.values()))
    out = []
    for i, r in enumerate(rows):
        v = {q: sum(choice["weights"][m] * (fcs[m][i][q] or 0) for m in fcs) for q in ("p10", "p50", "p90")}
        lo, mid, hi = sorted(v.values())
        out.append({"step": r["step"], "date": r["date"], "p10": round(lo, 4), "p50": round(mid, 4), "p90": round(hi, 4)})
    return out


def step_mae(bt):
    """{step: MAE of the P50} from a backtest in JSON form (keys 'origin|step')."""
    if not bt:
        return {}
    df = pd.DataFrame({"step": [int(k.split("|")[1]) for k in bt["key"]], "a": bt["actual"], "f": bt["p50"]},
                      dtype=float).dropna()
    return {int(st): float((g["f"] - g["a"]).abs().mean()) for st, g in df.groupby("step")}


def far_months(site, results, choice):
    """Months ahead where the Task 1 champion does not beat the Baseline in the 12-month backtest
    (backtest 2016-2025: no model beat it beyond ~3 months). Those months of the final forecast use the
    Baseline instead. Decided on monthly runs; daily runs carry the decision in the reused choice."""
    from .runner import M_STEPS_SCORED
    champ = choice.get("champion")
    if champ == "ensemble":
        champ = choice.get("best_single")
    bt12 = lambda m: ((results.get(m) or {}).get("sites", {}).get(site) or {}).get("bt12")
    cm = step_mae(bt12(champ)) if champ else {}
    bases = [b for b in (plan.BASELINE[plan.T1], plan.BASELINE[plan.T1] + plan.SUM) if step_mae(bt12(b))]
    if not cm or not bases or champ in bases:
        return None
    base = min(bases, key=lambda b: np.mean(list(step_mae(bt12(b)).values())))
    bm = step_mae(bt12(base))
    steps = [s for s in range(M_STEPS_SCORED + 1, 13) if s in cm and s in bm and not cm[s] < bm[s] * (1 - MARGIN)]
    return {"baseline": base, "steps": steps, "from_backtest": "2016-latest",
            "mae_champion": [round(cm.get(s), 3) if s in cm else None for s in range(1, 13)],
            "mae_baseline": [round(bm.get(s), 3) if s in bm else None for s in range(1, 13)]}


def apply_far(fin, results, site, far):
    """Swap the far months (far['steps']) of the final forecast for the Baseline's forecast."""
    if not fin or not far or not far.get("steps"):
        return fin
    base = {r["step"]: r for r in (((results.get(far["baseline"]) or {}).get("sites", {}).get(site) or {})
                                   .get("forecast") or [])}
    out = []
    for r in fin:
        b = base.get(r["step"])
        if r["step"] in far["steps"] and b and b.get("p50") is not None:
            r = {**r, **{q: b[q] for q in ("p10", "p50", "p90")}, "source": "baseline_far"}
        out.append(r)
    return out


COMPONENTS = {"THREE": ("DK", "KY", "NPL"), "PS": ("PS",)}


def storage_base(site, task_scores, champion):
    """Starting point for the month-end storage the HQ page works out (water-dashboard rule):
    end = storage today + (forecast month inflow - inflow so far this month) + (transfer - outflow) x days left,
    then + inflow + (transfer - outflow) x days for each later month, kept between 0 and capacity.
    Outflow and transfer are typed in by the user on the HQ page, so they are not here."""
    wb = load(f"{plan.T1}/95_water_balance")
    frames = [data_prep.daily(c) for c in COMPONENTS[site]]
    day = min(f.loc[f["storage"].notna(), "date"].max() for f in frames)
    month0 = day.replace(day=1)
    today = sum(float(f.loc[f["date"] == day, "storage"].iloc[0]) for f in frames)
    mtd = sum(float(f.loc[(f["date"] >= month0) & (f["date"] <= day), "y"].sum()) for f in frames)
    sc = (task_scores or {}).get(champion) or {}
    return {"date": day.strftime("%Y-%m-%d"), "storage_today": round(today, 3), "inflow_month_to_date": round(mtd, 3),
            "capacity": wb.RESERVOIRS[site]["capacity"], "dead": wb.RESERVOIRS[site]["dead"],
            "inflow_mae_steps": sc.get("mae_steps"), "inflow_mae": sc.get("mae")}


# Daily Nowcast: P10/P90 of the month total = seen + (P50 - seen) x these ratios, from the 2016-2020 backtest
# (/mnt/project-files/forecasting/experiments/daily_nowcast): ratios of actual to forecast for the unseen part.
# Rain-driven daily models (THREE parts) are tighter than the day-average fallback (PS).
NOWCAST_RATIO = {"rain": (0.5, 1.9), "average": (0.4, 3.0)}


def _day_average(frame):
    """Average inflow of each calendar day over the whole record."""
    s = frame.dropna(subset=["y"])
    return s.groupby([s["date"].dt.month, s["date"].dt.day])["y"].mean()


def daily_nowcast(site, month_end, daily_finals, rain_parts):
    """Month total for the current month from daily data (user, 2026-10-29):
    inflow already recorded this month + Task 1B daily forecast for the next days of the month
    + the past same-day average for days after the 7-day forecast.
    In the 2016-2020 backtest this beat the monthly models' month forecast whatever the days left
    (THREE MAE 1.6 vs 20 with <=7 days left, 12.7 vs 24 with 22-31 days left)."""
    end = pd.Timestamp(month_end)
    start = end.replace(day=1)
    seen = fc = avg = 0.0
    lo_ratio, hi_ratio = NOWCAST_RATIO["rain" if rain_parts else "average"]
    last_seen = None
    for c in COMPONENTS.get(site, (site,)):
        rows = daily_finals.get(c)
        if not rows:
            return None
        f = data_prep.daily(c)
        month = f[(f["date"] >= start) & (f["date"] <= end)]
        seen += float(month["y"].sum())
        covered = set(month.loc[month["y"].notna(), "date"].dt.strftime("%Y-%m-%d"))
        last_seen = max(filter(None, [last_seen, *covered]), default=None)
        for r in rows:
            if start.strftime("%Y-%m-%d") <= r["date"] <= end.strftime("%Y-%m-%d") and r["date"] not in covered:
                fc += r["p50"] or 0.0
                covered.add(r["date"])
        mean = _day_average(f)
        for d in pd.date_range(start, end):
            if d.strftime("%Y-%m-%d") not in covered:
                avg += float(mean.get((d.month, d.day), 0.0))
    p50 = seen + fc + avg
    rest = p50 - seen
    return {"date": end.strftime("%Y-%m-%d"), "p10": round(seen + rest * lo_ratio, 4), "p50": round(p50, 4),
            "p90": round(seen + rest * hi_ratio, 4), "seen": round(seen, 4), "daily_forecast": round(fc, 4),
            "day_average": round(avg, 4), "seen_until": last_seen, "days_left": int((end - pd.Timestamp(last_seen)).days)
            if last_seen else end.day}


def apply_nowcast(out):
    """Replace step 1 (current month) of every Task 1 final forecast with the Daily Nowcast."""
    t1, t1b = out["tasks"].get(plan.T1, {}), out["tasks"].get(plan.T1B, {})
    finals = {c: e.get("final") for c, e in t1b.items()}
    rain = {c for c, ch in out["choice"].get(plan.T1B, {}).items()
            if any(m in plan.RAIN_MODELS for m in (ch.get("weights") or {}))}
    for site, entry in t1.items():
        fin = entry.get("final")
        if not fin:
            continue
        try:
            nc = daily_nowcast(site, fin[0]["date"], finals, all(c in rain for c in COMPONENTS.get(site, (site,))))
        except Exception as e:  # noqa: BLE001 - the monthly forecast stays if the daily data is not usable
            entry["nowcast_error"] = f"{type(e).__name__}: {e}"[:200]
            continue
        if not nc:
            continue
        entry["nowcast"] = {**nc, "monthly_model": {q: fin[0][q] for q in ("p10", "p50", "p90")}}
        fin[0] = {**fin[0], **{q: nc[q] for q in ("p10", "p50", "p90")}, "source": "daily_nowcast"}


def apply_storm(out, results):
    """Storm Trigger: raise months 1-2 of a Task 1 final forecast to the ESP run with the heavy forecast rain
    (plan.STORM_*). Never lowers a month. Records what it did in entry["storm_trigger"]."""
    r = results.get(plan.STORM_MODEL) or {}
    for site, entry in out["tasks"].get(plan.T1, {}).items():
        got = (r.get("sites") or {}).get(site) or {}
        if got.get("storm_rain7") is not None:
            entry["storm_rain7_mm"] = got["storm_rain7"]
        storm, fin = got.get("storm"), entry.get("final")
        if not storm or not fin:
            continue
        changed = []
        for i, row in enumerate(storm[:plan.STORM_STEPS]):
            if i >= len(fin) or row.get("p50") is None or row["date"] != fin[i]["date"]:
                continue
            if row["p50"] > (fin[i].get("p50") or 0):
                changed.append({"step": i + 1, "date": row["date"], "before_p50": fin[i].get("p50"), "after_p50": row["p50"]})
                fin[i] = {**fin[i], **{q: max(row[q], fin[i].get(q) or 0) for q in ("p10", "p50", "p90")},
                          "source": "storm_trigger"}
        entry["storm_trigger"] = {"rain7_mm": got["storm_rain7"], "threshold_mm": plan.STORM_RAIN_MM,
                                  "model": plan.STORM_MODEL, "changed": changed}


def extend_year(out, previous_summary):
    """Daily runs forecast Task 1 only 4 months ahead; carry the later months (up to 12) of the newest
    monthly run onto each site's final forecast, so the HQ always shows the whole year ahead."""
    prev_t1 = ((previous_summary or {}).get("tasks") or {}).get(plan.T1, {})
    for site, entry in out["tasks"].get(plan.T1, {}).items():
        before = prev_t1.get(site) or {}
        _extend(entry.get("final"), before.get("final"))
        # each model's own line too, so the HQ tables and charts line up with the final forecast
        for name, m in (entry.get("models") or {}).items():
            _extend(m.get("forecast"), ((before.get("models") or {}).get(name) or {}).get("forecast"))


def _extend(rows, older):
    if not rows or not older:
        return
    last = rows[-1]["date"]
    extra = [r for r in older if r.get("date") and r["date"] > last][:12 - len(rows)]
    n = len(rows)
    rows.extend([{**r, "step": n + i + 1, "source": "monthly_run"} for i, r in enumerate(extra)])


def history(task, site):
    if task == plan.T1:
        df = data_prep.monthly(site).dropna(subset=["y"]).tail(24)
        return {"date": df["date"].dt.strftime("%Y-%m-%d").tolist(), "y": [data_prep.clean(v) for v in df["y"]]}
    df = data_prep.daily(site).tail(30)
    return {"date": df["date"].dt.strftime("%Y-%m-%d").tolist(), "y": [data_prep.clean(v) for v in df["y"]],
            "rain": [data_prep.clean(v) for v in df["rain"]]}


def summarise(results, kind, previous_summary):
    """Everything the HQ needs, one JSON object."""
    prev = (previous_summary or {}) if kind != "monthly" else {}
    scores = scores_from(results) if kind == "monthly" else prev.get("scores", {})
    out = {"kind": kind, "scores": scores, "choice": {}, "tasks": {},
           "scores_from": "this run" if kind == "monthly" else prev.get("scores_run")}
    for task in (plan.T1, plan.T1B):
        for site in plan.summary_sites(task):
            ev = gate(task, site, scores)
            choice = combine(task, site, results, ev, kind, prev.get("choice"))
            out["choice"].setdefault(task, {})[site] = choice
            models = {m: {"agent": r["agent"], "forecast": r["sites"][site]["forecast"],
                          **({"by_source": r["sites"][site]["by_source"]} if "by_source" in r["sites"][site] else {}),
                          **({"source_scores": r["sites"][site]["source_scores"]}
                             if "source_scores" in r["sites"][site] else {}),
                          **({"note": r["sites"][site]["note"]} if "note" in r["sites"][site] else {})}
                      for m, r in results.items() if r["task"] == task and site in r["sites"]}
            fin = final_forecast(task, site, results, choice)
            if task == plan.T1:
                if kind == "monthly" and choice.get("champion"):
                    choice["far"] = far_months(site, results, choice)
                fin = apply_far(fin, results, site, choice.get("far"))
            heavy = None
            if task == plan.T1B:
                fin, heavy = heavy_rain_switch(site, results, fin)
            entry = {"history": history(task, site), "models": models, "gate": ev, "final": fin,
                     "errors": {m: r["errors"][site] for m, r in results.items() if site in r.get("errors", {})}}
            if task == plan.T1B:
                entry["rain7_mm"] = rain7(site, results)
            if heavy:
                entry["heavy_rain"] = heavy
            if task == plan.T1 and fin and site in COMPONENTS:
                try:
                    champ = choice.get("champion")
                    sc = scores.get(task, {}).get(site, {})
                    if champ == "ensemble":  # the blend's error per step is not stored; use its best member's
                        champ = choice.get("best_single")
                    entry["storage"] = storage_base(site, sc, champ)
                except Exception as e:  # noqa: BLE001
                    entry["storage_error"] = f"{type(e).__name__}: {e}"[:200]
            out["tasks"].setdefault(task, {})[site] = entry
    apply_nowcast(out)
    apply_storm(out, results)
    if kind != "monthly":
        extend_year(out, prev)
    return out


def mvp(summary):
    """HQ character whose models carry the most weight across sites."""
    tot = {}
    for task in summary["choice"].values():
        for c in task.values():
            for m, w in (c.get("weights") or {}).items():
                a = plan.MODELS.get(m, ("base",))[0]
                tot[a] = tot.get(a, 0) + w
    return max(tot, key=tot.get) if tot else None


def values_rows(run_id, summary):
    rows = []
    for task, sites in summary["tasks"].items():
        for site, e in sites.items():
            for name, m in [*e["models"].items(), ("final", {"forecast": e["final"] or []})]:
                for r in m["forecast"]:
                    rows.append({"run_id": run_id, "task": task, "model": name.split("/")[-1], "site": site,
                                 "step": r["step"], "target_date": r["date"], "p10": r["p10"], "p50": r["p50"],
                                 "p90": r["p90"]})
    return rows


def nan_free(o):
    if isinstance(o, dict):
        return {k: nan_free(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nan_free(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) or np.isinf(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
