"""Experiment (Nov 2026): WeatherNext 2 past forecast rain (ensemble mean, BigQuery weathernext_2_mean) at the dams,
same origins as the ECMWF ENS / WN3 tests (days 1/6/11/16/21/26 from Feb 2024, 00Z run of the day before).
The 6-hourly precipitation column is found from INFORMATION_SCHEMA (no bytes billed). Each dam = bilinear mix of the
4 surrounding 0.25 deg cells. Written as member 0 only (BigQuery holds no members). Usage: GCP_SA_KEY=... python fetch_wn2.py OUT.csv.gz
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
DATASET, TABLE = "weathernext_2_mean", "weathernext_2_0_0_mean"
W, S, EAST, N = 100.9, 12.6, 101.9, 13.3


def run(base, cred, sql, cap=300):
    if not cred.valid:
        cred.refresh(Request())
    body = {"query": sql, "useLegacySql": False, "timeoutMs": 120000, "maxResults": 10000,
            "maximumBytesBilled": str(cap * 2**30)}
    return requests.post(base, headers={"Authorization": f"Bearer {cred.token}"}, json=body, timeout=180).json()


def weights(lat, lon):
    y0, x0 = int(lat // 0.25) * 0.25, int(lon // 0.25) * 0.25
    ty, tx = (lat - y0) / 0.25, (lon - x0) / 0.25
    return {(round(x0 + b * 0.25, 2), round(y0 + a * 0.25, 2)): (ty if a else 1 - ty) * (tx if b else 1 - tx)
            for a in (0, 1) for b in (0, 1)}


def main(out):
    info = json.loads(os.environ["GCP_SA_KEY"])
    cred = service_account.Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/bigquery"])
    project = info["project_id"]
    base = f"https://bigquery.googleapis.com/bigquery/v2/projects/{project}/queries"
    meta = run(base, cred, f"""SELECT field_path, data_type FROM `{project}.{DATASET}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
      WHERE table_name = '{TABLE}'""")
    fields = [r["f"][0]["v"] for r in meta.get("rows", [])]
    print("fields:", fields, flush=True)
    rain = [f for f in fields if f.startswith("forecast.") and "precipitation" in f]
    if not rain:
        sys.exit(f"no precipitation field: {meta.get('error') or fields}")
    col = sorted(rain, key=lambda f: ("6hr" not in f, f))[0].split(".", 1)[1]
    print("using", col, flush=True)
    W8 = {p: weights(*ll) for p, ll in DAMS.items()}
    billed = 0
    with gzip.open(out, "wt") as f:
        f.write("origin,init,member,date,lead,point,mm\n")
        for o in E.origins():
            init = o - dt.timedelta(days=1)
            sql = f"""SELECT f.hours AS h, STRING_AGG(FORMAT('%.2f:%.2f:%g', ST_X(t.geography), ST_Y(t.geography), f.`{col}`), ',') AS v
  FROM `{project}.{DATASET}.{TABLE}` t, t.forecast f
  WHERE t.init_time = TIMESTAMP('{init:%Y-%m-%d}T00:00:00Z')
    AND ST_INTERSECTS(t.geography, ST_GEOGFROMTEXT('POLYGON(({W} {S}, {EAST} {S}, {EAST} {N}, {W} {N}, {W} {S}))'))
    AND f.hours BETWEEN 1 AND 360
  GROUP BY h"""
            try:
                r = run(base, cred, sql)
                if "error" in r or not r.get("jobComplete"):
                    print(o, "error", str(r.get("error", "not complete"))[:200], flush=True)
                    continue
            except Exception as e:  # noqa: BLE001
                print(o, "failed", e, flush=True)
                continue
            billed += int(r.get("totalBytesBilled", 0))
            daily = {}
            for row in r.get("rows", []):
                h, packed = int(row["f"][0]["v"]), row["f"][1]["v"] or ""
                k = (h - 1) // 24 + 1
                cells = {}
                for c in packed.split(","):
                    if c:
                        x, y, v = c.split(":")
                        cells[(round(float(x), 2), round(float(y), 2))] = float(v)
                for p, w in W8.items():
                    if all(c in cells for c in w):
                        acc = daily.setdefault((p, k), [0.0, 0])
                        acc[0] += sum(cells[c] * wt for c, wt in w.items()) * 1000
                        acc[1] += 1
            for (p, k), (mm, n) in sorted(daily.items()):
                if n < 4:  # four 6-hour steps make a day
                    continue
                f.write(f"{o},{init},0,{init + dt.timedelta(days=k - 1)},{k},{p},{max(0.0, mm):.2f}\n")
            print(o, len(daily), "dam-days", f"{billed / 2**30:.2f} GiB billed so far", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main(sys.argv[1])
