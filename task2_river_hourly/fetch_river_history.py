"""Task 2 data: hourly river level (and discharge) history of the Rayong pilot gauges from ThaiWater.

ThaiWater's dated waterlevel_graph route returns 10-minute (or hourly) readings; this keeps one row per gauge and
Bangkok hour: last, max and mean level (m MSL), last discharge (m3/s) and the number of readings. History depth
checked Oct 2026 (forecasting/2-river-flood/history_depth.csv in the website repo): RAY001 from 2013, RAY002 2016,
Z.38/Z.11/Z.18 2020. Asked a month at a time, gently (one request a second, retries with back-off).
Public data (user OK 13 Nov 2026 for public archives). Usage:
    python fetch_river_history.py OUT.csv.gz [--from 2013-01-01] [--to YYYY-MM-DD]
"""
import argparse
import csv
import datetime as dt
import gzip
import json
import time
import urllib.request

URL = ("https://api-v3.thaiwater.net/api/v1/thaiwater30/public/waterlevel_graph"
       "?station_type=tele_waterlevel&station_id={sid}&start_date={a}&end_date={b}")
# ThaiWater station id -> (code, name, first year with data)
GAUGES = {"115": ("RAY001", "เมืองแกลง", 2013), "118": ("RAY002", "หนองบัว", 2016), "117": ("RAY004", "บ้านค่าย", 2013),
          "505011": ("Z.38", "บ้านเขาโบสถ์", 2020), "2669": ("Z.11", "บ้านเขาวังไทร", 2020),
          "2668": ("Z.18", "บ้านซำฆ้อ", 2020)}


def num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def get(sid, a, b, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(URL.format(sid=sid, a=a, b=b), headers={"User-Agent": "WaterIntelligenceForecast/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                body = json.load(r)
            return (body.get("data") or {}).get("graph_data") or []
        except Exception as e:  # noqa: BLE001
            print(sid, a, "retry", type(e).__name__, flush=True)
            time.sleep(5 * (k + 1))
    return None


def months(a, b):
    m = dt.date(a.year, a.month, 1)
    while m <= b:
        n = (m.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield max(m, a), min(n - dt.timedelta(days=1), b)
        m = n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--from", dest="start", default="2013-01-01")
    ap.add_argument("--to", dest="end", default=(dt.date.today() - dt.timedelta(days=1)).isoformat())
    a = ap.parse_args()
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    rows, failed = 0, []
    with gzip.open(a.out, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "code", "hour", "level_last", "level_max", "level_mean", "discharge_last", "n"])
        for sid, (code, name, y0) in GAUGES.items():
            n0 = rows
            for m0, m1 in months(max(start, dt.date(y0, 1, 1)), end):
                data = get(sid, m0.isoformat(), m1.isoformat())
                time.sleep(1)
                if data is None:
                    failed.append(f"{code} {m0:%Y-%m}")
                    continue
                hours = {}
                for r in data:
                    t, v, q = r.get("datetime"), num(r.get("value")), num(r.get("discharge"))
                    if not t or (v is None and q is None):
                        continue
                    hours.setdefault(t[:13], []).append((t, v, q))
                for h in sorted(hours):
                    xs = sorted(hours[h])
                    lv = [v for _, v, _ in xs if v is not None]
                    qs = [q for _, _, q in xs if q is not None]
                    w.writerow([sid, code, h + ":00", lv[-1] if lv else "", max(lv) if lv else "",
                                round(sum(lv) / len(lv), 3) if lv else "", qs[-1] if qs else "", len(xs)])
                    rows += 1
            print(code, name, rows - n0, "hours", flush=True)
    print("rows", rows, "failed months", len(failed), failed[:20], flush=True)


if __name__ == "__main__":
    main()
