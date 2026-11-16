"""WeatherNext 3 (and WeatherNext 2) forecast rain at the dams for a live run (prepare step), saved for the model jobs.

The heavy-rain switch and the Storm Trigger use this 7-day rain when it is there (test Nov 2026, Feb 2024-Oct 2026:
WN3 mean rain caught the storms with fewer false switches; WN2 mean ~2% behind WN3). Order (user, 15 Nov 2026):
WN3, then WN2 (Open-Meteo ensemble mean, free), then ECMWF IFS, then the mean of the Open-Meteo sources. One BigQuery query per run: newest published run (6-hourly, ~7 h lag), hourly
IMERG-calibrated ensemble-mean rain of the 0.1 deg cells within 0.1 deg of each dam, summed per Bangkok day.
Billed ~0.1-0.3 GB. Uses the newest run that covers day 1 (from 00:00 Bangkok of the day after the last recorded inflow).
Usage: GCP_SA_KEY=... python -m pipeline.wn3_rain OUT.json   (no key: writes nothing, exit 0)
"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data_prep import DAMS  # noqa: E402

W, S, EAST, N = 101.0, 12.7, 101.8, 13.2  # box around the 4 dams
RUN_STEP_H, LAG_H, TRIES = 6, 7, 3
# WN3 is partitioned by init DAY and clustered by geography: BigQuery checks maximumBytesBilled against its estimate
# of the whole day partition (~50-260 GB) though a one-run query bills ~0.2 GB, so a 5 GB cap refused every query
# (16 Nov 2026). Same cap as the website's WN3 job; a run billing over BILLED_ALARM_GB means pruning stopped working.
MAX_BILLED_GB = 400
BILLED_ALARM_GB = 5
BKK = dt.timezone(dt.timedelta(hours=7))


def query(project, run):
    return f"""SELECT f.hours AS h, STRING_AGG(FORMAT('%.2f:%.2f:%g', ST_X(t.geography), ST_Y(t.geography), f.imerg_tp_1hr_mean), ',') AS v
  FROM `{project}.weathernext_3.weathernext_3_0_0_0p1deg` t, t.forecast f
  WHERE t.init_time = TIMESTAMP('{run:%Y-%m-%dT%H}:00:00Z')
    AND ST_INTERSECTS(t.geography, ST_GEOGFROMTEXT('POLYGON(({W} {S}, {EAST} {S}, {EAST} {N}, {W} {N}, {W} {S}))'))
    AND ST_X(t.geography) > {W} AND ST_X(t.geography) < {EAST} AND ST_Y(t.geography) > {S} AND ST_Y(t.geography) < {N}
    AND f.hours BETWEEN 1 AND 360
  GROUP BY h"""


def daily(rows, run):
    """{dam: {bangkok day: mm}}, full days only (24 hours)."""
    acc = {}
    for row in rows:
        h, packed = int(row["f"][0]["v"]), row["f"][1]["v"] or ""
        cells = [list(map(float, c.split(":"))) for c in packed.split(",") if c]
        day = (run + dt.timedelta(hours=h - 1)).astimezone(BKK).date().isoformat()  # hour h = rain of hour h-1..h
        for p, (lat, lon, _) in DAMS.items():
            near = [c[2] for c in cells if abs(c[0] - lon) <= 0.101 and abs(c[1] - lat) <= 0.101]
            if near:
                a = acc.setdefault(p, {}).setdefault(day, [0.0, 0])
                a[0] += sum(near) / len(near) * 1000
                a[1] += 1
    return {p: {d: round(max(0.0, v), 2) for d, (v, n) in sorted(days.items()) if n == 24} for p, days in acc.items()}


def first_day():
    """Earliest day a live forecast starts on: the day after the last recorded daily inflow of any dam."""
    from pipeline import data_prep
    return min(data_prep.daily(p)["date"].iloc[-1] for p in DAMS).date() + dt.timedelta(days=1)


def fetch(first, now=None):
    import requests
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    info = json.loads(os.environ["GCP_SA_KEY"])
    cred = service_account.Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/bigquery"])
    cred.refresh(Request())
    url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{info['project_id']}/queries"
    now = now or dt.datetime.now(dt.timezone.utc)
    # newest published run that starts no later than 00:00 Bangkok of the first forecast day, so day 1 is complete
    start = dt.datetime.combine(first, dt.time(0), BKK).timestamp()
    newest = int(min(now.timestamp() - LAG_H * 3600, start) // (RUN_STEP_H * 3600)) * RUN_STEP_H * 3600
    billed = 0
    for i in range(TRIES):
        run = dt.datetime.fromtimestamp(newest - i * RUN_STEP_H * 3600, dt.timezone.utc)
        body = {"query": query(info["project_id"], run), "useLegacySql": False, "timeoutMs": 120000,
                "maxResults": 10000, "maximumBytesBilled": str(MAX_BILLED_GB * 2**30)}
        r = requests.post(url, headers={"Authorization": f"Bearer {cred.token}"}, json=body, timeout=180).json()
        if "error" in r:
            raise RuntimeError(r["error"].get("message", "")[:200])
        billed += int(r.get("totalBytesBilled", 0))
        rows = r.get("rows", []) if r.get("jobComplete") else []
        if billed > BILLED_ALARM_GB * 2**30:
            print(f"::warning::WN3 query billed {billed / 2**30:.1f} GB (expected ~0.2 GB): check the partition filter", flush=True)
        if len(rows) >= 240:  # run published (at least 10 days of hours)
            return {"model": "weathernext3", "run": run.isoformat(), "billed_gb": round(billed / 2**30, 3),
                    "daily": daily(rows, run)}
        print(f"WN3 run {run:%Y-%m-%d %HZ} not complete ({len(rows)} hours), trying the one before", flush=True)
    return None


WN2_URL = ("https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}&daily=precipitation_sum"
           "&timezone=Asia%2FBangkok&past_days=3&forecast_days=15&models=google_weathernext2_ensemble")


def fetch_wn2():
    """{dam: {bangkok day: mm}}: mean of the 64 WeatherNext 2 members (Open-Meteo ensemble API, free, no key)."""
    import urllib.request
    out = {}
    for p, (lat, lon, _) in DAMS.items():
        req = urllib.request.Request(WN2_URL.format(lat=lat, lon=lon), headers={"User-Agent": "WaterIntelligenceForecast/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)["daily"]
        keys = [k for k in d if k.startswith("precipitation_sum")]
        days = {}
        for j, day in enumerate(d["time"]):
            v = [d[k][j] for k in keys if d[k][j] is not None]
            if len(v) >= 30:
                days[day] = round(max(0.0, sum(v) / len(v)), 2)
        out[p] = days
    return out


def main(out):
    """Writes {"WN3": {run, billed_gb, daily}, "WN2": {daily}} with whichever source answered; nothing if neither."""
    first = first_day()
    res = {}
    if (dt.datetime.now(BKK).date() - first).days > 3:
        print(f"inflow record ends {first - dt.timedelta(days=1)}: too old for WN3/WN2, the switch uses ECMWF")
        return
    if os.environ.get("GCP_SA_KEY", "").strip():
        try:
            wn3 = fetch(first)
            if wn3:
                res["WN3"] = wn3
                print(f"WN3 run {wn3['run']}, {wn3['billed_gb']} GB billed, days per dam "
                      f"{ {p: len(v) for p, v in wn3['daily'].items()} }")
            else:
                print("WN3 rain unavailable: no complete run")
        except Exception as e:  # noqa: BLE001 - the run goes on with WN2 / ECMWF
            print(f"WN3 rain unavailable: {str(e)[:200]}")
    else:
        print("GCP_SA_KEY not set: no WN3 rain")
    try:
        res["WN2"] = {"model": "weathernext2_ensemble_mean", "daily": fetch_wn2()}
        print(f"WN2 days per dam { {p: len(v) for p, v in res['WN2']['daily'].items()} }")
    except Exception as e:  # noqa: BLE001
        print(f"WN2 rain unavailable: {str(e)[:200]}")
    if res:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1])
