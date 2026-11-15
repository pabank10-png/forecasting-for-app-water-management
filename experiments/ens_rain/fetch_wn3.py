"""Experiment (Nov 2026): WeatherNext 3 past forecast rain at the dams from BigQuery, for the same origins as the
ECMWF ENS test (days 1/6/11/16/21/26 from Feb 2024, 00Z run of the day before). BigQuery holds no members, only the
ensemble mean and hourly percentiles of imerg_tp_1hr; daily sums of the hourly p10..p90 are written as members 1-5
(a rough range: a sum of hourly percentiles is not the percentile of the daily sum) and the mean as member 0.
Each dam = mean of the 0.1 deg cells within 0.1 deg of it. One literal init and one simple polygon per query keep
BigQuery pruning (~0.1-0.3 GB billed each).
Usage: GCP_SA_KEY=... python fetch_wn3.py OUT.csv.gz
"""
import datetime as dt
import gzip
import json
import os
import sys
import time

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

import fetch_ecmwf_ens as E

DAMS = {k: v for k, v in E.POINTS.items() if k in ("DK", "KY", "NPL", "PS")}
STATS = ["mean", "p10", "p25", "p50", "p75", "p90"]
W, S, EAST, N = 101.0, 12.7, 101.8, 13.2


def sql(project, init):
    """One row per forecast hour; every 0.1 deg cell of the small box packed as x:y:mean:p10:...:p90."""
    vals = ", ".join(f"f.imerg_tp_1hr_{s_}" for s_ in STATS)
    fmt = ":".join(["%.2f", "%.2f"] + ["%g"] * len(STATS))
    return f"""SELECT f.hours AS h, STRING_AGG(FORMAT('{fmt}', ST_X(t.geography), ST_Y(t.geography), {vals}), ',') AS v
  FROM `{project}.weathernext_3.weathernext_3_0_0_0p1deg` t, t.forecast f
  WHERE t.init_time = TIMESTAMP('{init:%Y-%m-%d}T00:00:00Z')
    AND ST_INTERSECTS(t.geography, ST_GEOGFROMTEXT('POLYGON(({W} {S}, {EAST} {S}, {EAST} {N}, {W} {N}, {W} {S}))'))
    AND ST_X(t.geography) > {W} AND ST_X(t.geography) < {EAST} AND ST_Y(t.geography) > {S} AND ST_Y(t.geography) < {N}
    AND f.hours BETWEEN 1 AND 360
  GROUP BY h"""


def main(out):
    info = json.loads(os.environ["GCP_SA_KEY"])
    cred = service_account.Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/bigquery"])
    project = info["project_id"]
    base = f"https://bigquery.googleapis.com/bigquery/v2/projects/{project}/queries"
    billed = 0
    with gzip.open(out, "wt") as f:
        f.write("origin,init,member,date,lead,point,mm\n")
        for o in E.origins():
            init = o - dt.timedelta(days=1)
            if not cred.valid:
                cred.refresh(Request())
            body = {"query": sql(project, init), "useLegacySql": False, "timeoutMs": 120000, "maxResults": 10000,
                    "maximumBytesBilled": str(900 * 2**30)}
            try:
                r = requests.post(base, headers={"Authorization": f"Bearer {cred.token}"}, json=body, timeout=180).json()
                if "error" in r:
                    print(o, "error", r["error"].get("message", "")[:200], flush=True)
                    continue
                if not r.get("jobComplete"):
                    print(o, "not complete", flush=True)
                    continue
            except Exception as e:  # noqa: BLE001
                print(o, "failed", e, flush=True)
                continue
            billed += int(r.get("totalBytesBilled", 0))
            daily = {}
            for row in r.get("rows", []):
                h, packed = int(row["f"][0]["v"]), row["f"][1]["v"] or ""
                k = (h - 1) // 24 + 1  # UTC day k of the run: hours 24(k-1)+1 .. 24k
                cells = [list(map(float, c.split(":"))) for c in packed.split(",") if c]
                for p, (lat, lon) in DAMS.items():
                    near = [c for c in cells if abs(c[0] - lon) <= 0.101 and abs(c[1] - lat) <= 0.101]
                    if not near:
                        continue
                    acc = daily.setdefault((p, k), [0.0] * len(STATS) + [0])
                    for i in range(len(STATS)):
                        acc[i] += sum(c[2 + i] for c in near) / len(near) * 1000
                    acc[-1] += 1
            for (p, k), acc in sorted(daily.items()):
                if acc[-1] < 24:
                    continue
                day = init + dt.timedelta(days=k - 1)
                for m, val in enumerate(acc[:len(STATS)]):
                    f.write(f"{o},{init},{m},{day},{k},{p},{max(0.0, val):.2f}\n")
            print(o, len(daily), "dam-days", f"{billed / 2**30:.1f} GiB billed so far", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main(sys.argv[1])
