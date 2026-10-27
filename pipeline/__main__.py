"""Command line used by .github/workflows/forecast.yml (and for local tests).

  python -m pipeline start  --kind daily|monthly --trigger schedule|manual [--run-id ID] [--models a,b]
  python -m pipeline step   --run-id ID --name data --status running|done|failed [--message TEXT]
  python -m pipeline model  --run-id ID --kind K --name task1_reservoir_monthly/05_lightgbm_quantile --out results
  python -m pipeline finish --run-id ID --kind K --results results
  python -m pipeline fail   --run-id ID --message TEXT

Without SUPABASE_URL / SUPABASE_BACKEND_KEY nothing is written to Supabase (local test).
"""
import argparse
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import finish, fingerprint, plan, report, runner, store  # noqa: E402


def gh_output(**kv):
    path = os.environ.get("GITHUB_OUTPUT")
    for k, v in kv.items():
        line = f"{k}={v}"
        if path:
            with open(path, "a") as f:
                f.write(line + "\n")
        print(line)


def cmd_start(a):
    run_id = a.run_id or str(uuid.uuid4())
    names = plan.models(a.kind, [m.strip() for m in a.models.split(",") if m.strip()] if a.models else None)
    url = None
    if os.environ.get("GITHUB_RUN_ID"):
        url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}" \
              f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    extra = {} if a.run_id else {"trigger": a.trigger}  # a run made by the website keeps trigger "website"
    store.create_run(run_id, kind=a.kind, **extra, status="running", stage="data",
                     started_at=store.now(), github_run_id=int(os.environ.get("GITHUB_RUN_ID", 0)) or None,
                     github_url=url, models=names)
    if store.enabled():
        rows = [{"run_id": run_id, "step": s, "agent": ag, "status": "queued", "updated_at": store.now()}
                for s, ag in plan.STAGES]
        rows += [{"run_id": run_id, "step": f"model:{m}", "agent": plan.MODELS[m][0], "status": "queued",
                  "updated_at": store.now()} for m in names]
        store.upsert("forecast_run_steps", rows, "run_id,step")
    gh_output(run_id=run_id)


def cmd_plan(a):
    """After the data refresh: which (model, site) pairs must run. Monthly runs always run everything (backtest);
    daily runs skip pairs whose inputs did not change since the last finished run and reuse its forecast."""
    row = store.select("forecast_runs", f"select=models&id=eq.{a.run_id}") if store.enabled() else []
    names = (row[0].get("models") if row else None) or plan.models(a.kind)
    prints = fingerprint.all_inputs(names)
    prev = store.latest_done() if a.kind == "daily" else None
    old = ((prev or {}).get("summary") or {}).get("fingerprints") or {}
    todo, reused = {}, {}
    for m in names:
        if a.kind == "daily" and m in plan.MONTHLY_ONLY:
            reused[m] = list(prints[m])
            continue
        sites = [s for s, h in prints[m].items() if old.get(m, {}).get(s) != h]
        if sites:
            todo[m] = sites
        if len(sites) < len(prints[m]):
            reused[m] = [s for s in prints[m] if s not in sites]
    for m in names:
        if m not in todo:
            store.step(a.run_id, f"model:{m}", "skipped", agent=plan.MODELS[m][0],
                       message="รันเฉพาะรอบรายเดือน ใช้ผลรายเดือนล่าสุด" if m in plan.MONTHLY_ONLY and a.kind == "daily"
                       else "ข้อมูลไม่เปลี่ยน ใช้ผลรอบก่อน")
    matrix = [{"model": m, "sites": ",".join(ss), "group": plan.group(m), "id": m.replace("/", "__")} for m, ss in todo.items()]
    Path(a.out).write_text(json.dumps({"fingerprints": prints, "reused": reused, "todo": todo,
                                       "from_run": (prev or {}).get("id")}))
    gh_output(matrix=json.dumps({"include": matrix}, separators=(",", ":")), has_models=str(bool(matrix)).lower(),
              message=("ข้อมูลไม่เปลี่ยนเลย ใช้ผลรอบก่อนทั้งหมด" if not matrix else
                       f"รันใหม่ {len(matrix)} โมเดล" + (f" ใช้ผลเดิม {len(names) - len(matrix)} โมเดล" if len(matrix) < len(names) else "")))


def cmd_step(a):
    store.step(a.run_id, a.name, a.status, agent=dict(plan.STAGES).get(a.name), message=a.message)
    if a.name == "data" and a.status == "done":
        store.update_run(a.run_id, stage="models")


def cmd_model(a):
    step = f"model:{a.name}"
    t0 = time.time()
    store.step(a.run_id, step, "running", agent=plan.MODELS[a.name][0])
    try:
        res = runner.run(a.name, a.kind, a.sites.split(",") if a.sites else None)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        store.step(a.run_id, step, "failed", message=f"{type(e).__name__}: {e}", seconds=time.time() - t0)
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name.replace('/', '__')}.json").write_text(json.dumps(finish.nan_free(res)))
    ok = bool(res["sites"])
    detail = {"sites": {s: {"p50": [r["p50"] for r in v["forecast"]], "score": v.get("score"),
                            "seconds": v.get("seconds")} for s, v in res["sites"].items()},
              "errors": res["errors"]}
    msg = None if not res["errors"] else "; ".join(f"{s}: {e}" for s, e in res["errors"].items())
    store.step(a.run_id, step, "done" if ok else "failed", message=msg, detail=finish.nan_free(detail),
               seconds=time.time() - t0)


def cmd_finish(a):
    results = {}
    for p in sorted(Path(a.results).glob("**/*__*.json")):  # one file per model, not summary.json
        r = json.loads(p.read_text())
        results[r["model"]] = r
    plan_info = json.loads(Path(a.plan).read_text()) if a.plan and Path(a.plan).exists() else {}
    if plan_info.get("reused") and plan_info.get("from_run"):
        finish.reuse(results, plan_info["reused"], store.run_summary(plan_info["from_run"]), a.kind)
    prev_run = store.latest_monthly_summary() if a.kind != "monthly" else None
    prev = (prev_run or {}).get("summary") or {}
    if prev_run:
        prev["scores_run"] = prev_run["id"]

    store.update_run(a.run_id, stage="evaluate")
    store.step(a.run_id, "evaluate", "running")
    summary = finish.summarise(results, a.kind, prev)
    n_pass = sum(1 for t in summary["choice"].values() for c in t.values() for _ in c.get("members", []))
    store.step(a.run_id, "evaluate", "done", message=f"{len(results)} โมเดลมีผล" +
               ("" if summary["scores"] else " · ยังไม่มีคะแนนย้อนหลัง"))

    store.update_run(a.run_id, stage="combine")
    store.step(a.run_id, "combine", "running")
    champs = {f"{t.split('_')[0]}/{s}": c.get("champion") for t, sites in summary["choice"].items()
              for s, c in sites.items()}
    store.step(a.run_id, "combine", "done", detail=champs, message=f"ผ่านด่านรวม {n_pass} ที่")

    store.update_run(a.run_id, stage="publish")
    store.step(a.run_id, "publish", "running")
    summary["mvp"] = finish.mvp(summary)
    summary["fingerprints"] = plan_info.get("fingerprints") or {}
    summary["reused"] = {"from_run": plan_info.get("from_run"), "models": plan_info.get("reused") or {}}
    summary = finish.nan_free(summary)
    Path(a.results, "summary.json").write_text(json.dumps(summary))
    if store.enabled():
        store.upsert("forecast_values", finish.values_rows(a.run_id, summary), "run_id,task,model,site,step")
    row = store.select("forecast_runs", f"select=models&id=eq.{a.run_id}") if store.enabled() else []
    expected = (row[0].get("models") if row else None) or list(results)
    failed = [m for m in expected if m not in results or not results[m]["sites"]]
    notes = [f"{len(failed)} โมเดลไม่มีผล"] if failed else []
    if plan_info and not plan_info.get("todo"):
        notes.append("ข้อมูลไม่เปลี่ยน ใช้ผลรอบก่อนทั้งหมด")
    elif plan_info.get("reused"):
        notes.append(f"ใช้ผลเดิม {sum(len(v) for v in plan_info['reused'].values())} คู่โมเดล-อ่างที่ข้อมูลไม่เปลี่ยน")
    summary["report"] = publish_report(a, results, summary)
    store.step(a.run_id, "publish", "done")
    store.update_run(a.run_id, status="done", stage="done", finished_at=store.now(), summary=summary,
                     message=" · ".join(notes) or None)


def publish_report(a, results, summary):
    """Excel pack of this run for admins to download from HQ. A failure here never fails the run."""
    try:
        e = os.environ
        url = (f"{e['GITHUB_SERVER_URL']}/{e['GITHUB_REPOSITORY']}/actions/runs/{e['GITHUB_RUN_ID']}"
               if e.get("GITHUB_RUN_ID") else None)
        data, name = report.build(results, summary, {"id": a.run_id, "kind": a.kind, "finished": report.thai_now(),
                                                     "url": url})
        Path(a.results, name).write_bytes(data)
        if store.upload_report(name, data):
            print(f"[report] {name}: {len(data) // 1024} KB", flush=True)
            return {"path": name, "kb": len(data) // 1024}
    except Exception as err:  # noqa: BLE001
        traceback.print_exc()
        print(f"[report] not published: {err}", flush=True)
    return None


def cmd_fail(a):
    store.update_run(a.run_id, status="failed", finished_at=store.now(), message=a.message[:500])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--kind", choices=("daily", "monthly"), required=True)
    s.add_argument("--trigger", default="manual")
    s.add_argument("--run-id")
    s.add_argument("--models")
    s = sub.add_parser("plan")
    s.add_argument("--run-id", required=True)
    s.add_argument("--kind", required=True)
    s.add_argument("--out", default="plan.json")
    s = sub.add_parser("step")
    s.add_argument("--run-id", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--status", required=True)
    s.add_argument("--message")
    s = sub.add_parser("model")
    s.add_argument("--run-id", required=True)
    s.add_argument("--kind", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--out", default="results")
    s.add_argument("--sites", default="")
    s = sub.add_parser("finish")
    s.add_argument("--run-id", required=True)
    s.add_argument("--kind", required=True)
    s.add_argument("--results", default="results")
    s.add_argument("--plan", default="plan.json")
    s = sub.add_parser("fail")
    s.add_argument("--run-id", required=True)
    s.add_argument("--message", default="")
    a = ap.parse_args(argv)
    {"start": cmd_start, "plan": cmd_plan, "step": cmd_step, "model": cmd_model, "finish": cmd_finish, "fail": cmd_fail}[a.cmd](a)


if __name__ == "__main__":
    main()
