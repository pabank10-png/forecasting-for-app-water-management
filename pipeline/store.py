"""Write run status and results to Supabase (REST). Without SUPABASE_URL it only prints, for local tests.

Tables (migration in the website repo, supabase/migrations/*_forecast_runs.sql):
  forecast_runs       one row per run: kind, trigger, status, timestamps, GitHub run link, summary (json for HQ)
  forecast_run_steps  one row per step of a run (data, each model, evaluate, combine, storage, publish)
  forecast_values     P10/P50/P90 per run, task, model, site and step
"""
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _cfg():
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_BACKEND_KEY")
    if not (url and key):
        return None, None
    # Accept the URL with or without the /rest/v1 suffix the Supabase dashboard sometimes shows
    url = url.strip().rstrip("/").removesuffix("/rest/v1")
    return url, key.strip()


def enabled():
    return _cfg()[0] is not None


def _request(method, path, body=None, prefer=None):
    base, key = _cfg()
    if base is None:
        return None
    headers = {"apikey": key, "Content-Type": "application/json", "Accept": "application/json"}
    if not key.startswith("sb_secret_"):
        headers["Authorization"] = f"Bearer {key}"
    if prefer:
        headers["Prefer"] = prefer
    data = None if body is None else json.dumps(body, allow_nan=False, default=str).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(f"{base}/rest/v1/{path}", data=data, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code < 500 or attempt == 3:
                raise RuntimeError(f"Supabase {method} {path.split('?')[0]} HTTP {e.code}: {msg}") from None
        except urllib.error.URLError:
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def upsert(table, rows, on_conflict):
    rows = rows if isinstance(rows, list) else [rows]
    for i in range(0, len(rows), 500):
        _request("POST", f"{table}?on_conflict={on_conflict}", rows[i:i + 500],
                 "resolution=merge-duplicates,return=minimal")


def select(table, query):
    return _request("GET", f"{table}?{query}") or []


def create_run(run_id, **fields):
    """Insert the run, or fill in the row the website already made (it holds kind, so NOT NULL is met)."""
    upsert("forecast_runs", {"id": run_id, **fields}, "id")


def update_run(run_id, **fields):
    _request("PATCH", f"forecast_runs?id=eq.{run_id}", fields, "return=minimal")


def step(run_id, name, status, agent=None, message=None, detail=None, started=None, seconds=None):
    """Record one step. status: queued, running, done, failed, skipped."""
    row = {"run_id": run_id, "step": name, "status": status, "updated_at": now()}
    if agent:
        row["agent"] = agent
    if message is not None:
        row["message"] = message[:500]
    if detail is not None:
        row["detail"] = detail
    if status == "running":
        row["started_at"] = started or now()
    if status in ("done", "failed", "skipped"):
        row["finished_at"] = now()
        if seconds is not None:
            row["seconds"] = round(seconds, 1)
    print(f"[step] {name}: {status}" + (f" ({message})" if message else ""), flush=True)
    if enabled():
        upsert("forecast_run_steps", row, "run_id,step")


def latest_monthly_summary():
    """Scores and champion of the newest finished monthly run (daily runs reuse them)."""
    if not enabled():
        return None
    rows = select("forecast_runs", "select=id,finished_at,summary&kind=eq.monthly&status=eq.done"
                                   "&order=finished_at.desc&limit=1")
    return rows[0] if rows else None


def latest_done():
    if not enabled():
        return None
    rows = select("forecast_runs", "select=id,kind,finished_at,summary&status=eq.done&order=finished_at.desc&limit=1")
    return rows[0] if rows else None


def run_summary(run_id):
    if not enabled():
        return None
    rows = select("forecast_runs", f"select=summary&id=eq.{run_id}")
    return (rows[0].get("summary") if rows else None) or None
