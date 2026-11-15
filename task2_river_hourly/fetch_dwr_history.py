"""Task 2 data: hourly water level, rain and flow of the DWR telemetry gauges in Rayong (station codes TA17*).

Includes TA170203 (Khlong Yai bridge, Choeng Noen, Mueang Rayong): the city-centre flood point
(banks 3.02 / 3.95 m MSL, bed -1.44 m MSL). Source: DWR's public hourly report
(telemetry.dwr.go.th/report/hourly), one day per request for all stations. Asked gently (one request a second).
User OK 16 Nov 2026: fetch the history once, then add one day at a time; keep it in the public repo.

    python fetch_dwr_history.py OUT.csv.gz --from 2015-01-01 [--to YYYY-MM-DD] [--append OLD.csv.gz]

With --append the old file is copied first and only days after its last day are asked for.
"""
import argparse
import csv
import datetime as dt
import gzip
import json
import time
import urllib.request

BASE = "https://telemetry.dwr.go.th/api/public/report/hour/"
KINDS = {"getWL": "wl_m_msl", "getRF": "rain_mm", "getFlow": "flow_m3s"}
PREFIX = "TA17"  # Rayong


def post(kind, day, tries=4):
    body = json.dumps({"dataLastUpdated": f"{day.isoformat()}T00:00:00+07:00"}).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(BASE + kind, data=body, method="POST", headers={
                "Content-Type": "application/json", "User-Agent": "WaterIntelligenceForecast/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r).get("value") or []
        except Exception as e:  # noqa: BLE001
            print(day, kind, "retry", type(e).__name__, flush=True)
            time.sleep(5 * (k + 1))
    return None


def day_rows(day):
    """{(code, hour): {col: value}} for the Rayong stations on one Bangkok day; None if every request failed."""
    out, ok = {}, False
    for kind, col in KINDS.items():
        v = post(kind, day)
        time.sleep(1)
        if v is None:
            continue
        ok = True
        for st in v:
            code = (st.get("station") or {}).get("stationCode") or ""
            if not code.startswith(PREFIX):
                continue
            for d in st.get("dataList") or []:
                t, x = d.get("dataLastUpdated") or "", d.get("value")
                if t[:10] == day.isoformat() and isinstance(x, (int, float)):
                    out.setdefault((code, t[:13].replace("T", " ") + ":00"), {})[col] = x
    return out if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--from", dest="start", default="2015-01-01")
    ap.add_argument("--to", dest="end", default=(dt.date.today() - dt.timedelta(days=1)).isoformat())
    ap.add_argument("--append")
    ap.add_argument("--max-minutes", type=float, default=280, help="stop cleanly after this; the next run continues")
    a = ap.parse_args()
    t0 = time.time()
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    cols = ["code", "hour", *KINDS.values()]
    old = []
    if a.append:
        with gzip.open(a.append, "rt", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        if old:
            start = max(start, dt.date.fromisoformat(max(r["hour"] for r in old)[:10]) + dt.timedelta(days=1))
    if not old:  # first fill: skip the years before DWR has any Rayong data (probe 1 Jan and 1 Jul of each year)
        y = start.year
        while y < end.year and not any(day_rows(dt.date(y, m, 1)) for m in (1, 7)):
            y += 1
        start = max(start, dt.date(y, 1, 1))
        print("first year with data", y, flush=True)
    rows, failed, empty = 0, [], 0
    with gzip.open(a.out, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(old)
        day = start
        while day <= end and (time.time() - t0) < a.max_minutes * 60:
            got = day_rows(day)
            if got is None:
                failed.append(day.isoformat())
            elif not got:
                empty += 1
            for (code, hour), vals in sorted(got.items()) if got else []:
                w.writerow({"code": code, "hour": hour, **vals})
                rows += 1
            if day.day == 1:
                print(day, "rows so far", rows, flush=True)
            day += dt.timedelta(days=1)
    print(f"old rows {len(old)}, new rows {rows}, days {start}..{day - dt.timedelta(days=1)} (asked to {end}), empty days {empty}, failed {len(failed)} {failed[:20]}",
          flush=True)


if __name__ == "__main__":
    main()
