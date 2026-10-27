"""After the models: Error Gate, Combination + champion, month-end storage, and the summary the HQ reads."""
import numpy as np
import pandas as pd

from common import ensemble
from common.registry import load
from common.metrics import mae

from . import data_prep, plan

LAMBDA, MARGIN, MIN_COVER = 0.5, 0.02, 0.70


def reuse(results, reused, prev_summary, kind):
    """Put the previous run's forecast back for (model, site) pairs whose inputs did not change."""
    if not prev_summary:
        return
    for name, sites in reused.items():
        task = name.split("/")[0]
        h = plan.HORIZON[(kind, task)]
        for site in sites:
            m = ((prev_summary.get("tasks") or {}).get(task, {}).get(site, {}).get("models") or {}).get(name)
            if not m or not m.get("forecast"):
                continue
            r = results.setdefault(name, {"model": name, "task": task, "agent": plan.MODELS[name][0], "kind": kind,
                                          "sites": {}, "errors": {}})
            r["sites"][site] = {**{k: v for k, v in m.items() if k != "agent"}, "forecast": m["forecast"][:h],
                                "reused": True}


def _sum_rows(rows_by_part):
    """Add P50s; P10/P90 halfway between 'parts move together' (sum of P10s) and 'parts are independent'
    (P50 ± root-sum-square of the gaps). Together alone was too wide in the backtest, independent too narrow."""
    p50 = sum(r["p50"] for r in rows_by_part)
    lo_t, hi_t = sum(r["p10"] for r in rows_by_part), sum(r["p90"] for r in rows_by_part)
    lo_i = p50 - sum(max(0.0, r["p50"] - r["p10"]) ** 2 for r in rows_by_part) ** 0.5
    hi_i = p50 + sum(max(0.0, r["p90"] - r["p50"]) ** 2 for r in rows_by_part) ** 0.5
    return max(0.0, (lo_t + lo_i) / 2), p50, (hi_t + hi_i) / 2


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
            if all(g.get("bt") for g in got):
                maps = [{k: i for i, k in enumerate(g["bt"]["key"])} for g in got]
                keys = [k for k in got[0]["bt"]["key"] if all(k in m for m in maps)]
                bt = {"key": [], "actual": [], "p10": [], "p50": [], "p90": []}
                for k in keys:
                    rows = [{q: g["bt"][q][m[k]] for q in ("actual", "p10", "p50", "p90")} for g, m in zip(got, maps)]
                    if any(x["p50"] is None for x in rows):
                        continue
                    lo, mid, hi = _sum_rows(rows)
                    act = None if any(x["actual"] is None for x in rows) else sum(x["actual"] for x in rows)
                    for q, v in zip(("key", "actual", "p10", "p50", "p90"), (k, act, lo, mid, hi)):
                        bt[q].append(v)
                df = pd.DataFrame({"step": [int(k.split("|")[1]) for k in bt["key"]], "actual": bt["actual"],
                                   "p10": bt["p10"], "p50": bt["p50"], "p90": bt["p90"]}).astype(float)
                from .runner import M_STEPS_SCORED
                entry["score"] = _score(df, M_STEPS_SCORED)
                entry["bt"] = bt
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
    """Pass = MAE below the Baseline (climatology) and P10–P90 covering at least 70% of actual values."""
    sc = scores.get(task, {}).get(site, {})
    base = sc.get(plan.BASELINE[task], {}).get("mae")
    ev = {}
    for name, s in sc.items():
        if name == plan.BASELINE[task]:
            ev[name] = {"mae": s["mae"], "coverage": s["coverage"], "pass": None, "skill": 0.0}
            continue
        ok = base is not None and s["mae"] is not None and s["mae"] < base and (s["coverage"] or 0) >= MIN_COVER
        ev[name] = {"mae": s["mae"], "coverage": s["coverage"], "pass": bool(ok),
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
                          **({"note": r["sites"][site]["note"]} if "note" in r["sites"][site] else {})}
                      for m, r in results.items() if r["task"] == task and site in r["sites"]}
            fin = final_forecast(task, site, results, choice)
            entry = {"history": history(task, site), "models": models, "gate": ev, "final": fin,
                     "errors": {m: r["errors"][site] for m, r in results.items() if site in r.get("errors", {})}}
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
