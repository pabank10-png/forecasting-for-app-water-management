"""Task 2 data: hourly rain history over the Rayong river basins from the Open-Meteo archive (best match:
ERA5 / ERA5-Land / ECMWF IFS reanalysis blend; `models=era5_land` returns no precipitation there).

Gauge rain only exists hourly from Apr 2025 (DWR) and Oct 2026 (our own collector), so the long river record
(RAY001 2013, RAY002/RAY004 2016, Z.38 2020) is paired with this reanalysis rain. One column per grid point plus the basin
means used by the models. Times are Bangkok local hours; rain of hour h-1..h is stored at h.
The archive runs ~5 days behind, so a top-up re-asks the last 10 stored days.

    python fetch_rain_hourly.py OUT.csv.gz [--from 2013-01-01] [--append OLD.csv.gz]
"""
import argparse
import csv
import datetime as dt
import gzip
import json
import time
import urllib.request

URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={a}&end_date={b}"
       "&hourly=precipitation&timezone=Asia%2FBangkok")
# Upper Khlong Yai / Rayong river (Dok Krai, Nong Pla Lai, Ban Khai, city) and the Z.38 side.
POINTS = {"p1275_10120": (12.75, 101.20), "p1275_10135": (12.75, 101.35), "p1290_10120": (12.90, 101.20),
          "p1290_10135": (12.90, 101.35), "p1305_10125": (13.05, 101.25), "p1305_10140": (13.05, 101.40)}


def get(a, b, tries=6):
    """{point: {time: mm}} for all points in one request (Open-Meteo takes coordinate lists)."""
    lats = ",".join(str(p[0]) for p in POINTS.values())
    lons = ",".join(str(p[1]) for p in POINTS.values())
    for k in range(tries):
        try:
            req = urllib.request.Request(URL.format(lat=lats, lon=lons, a=a, b=b),
                                         headers={"User-Agent": "WaterIntelligenceForecast/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                body = json.load(r)
            body = body if isinstance(body, list) else [body]
            return {n: dict(zip(x["hourly"]["time"], x["hourly"]["precipitation"])) for n, x in zip(POINTS, body)}
        except Exception as e:  # noqa: BLE001
            print(a, "retry", type(e).__name__, str(e)[:80], flush=True)
            time.sleep(30 * (k + 1))
    raise RuntimeError(f"Open-Meteo archive failed {a}..{b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--from", dest="start", default="2013-01-01")
    ap.add_argument("--append")
    a = ap.parse_args()
    start, end = dt.date.fromisoformat(a.start), dt.date.today() - dt.timedelta(days=2)
    cols = ["hour", *POINTS, "basin_mean"]
    old = []
    if a.append:
        with gzip.open(a.append, "rt", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        if old:
            start = dt.date.fromisoformat(max(r["hour"] for r in old)[:10]) - dt.timedelta(days=10)
            old = [r for r in old if r["hour"][:10] < start.isoformat()]
    series = {n: {} for n in POINTS}
    y = start
    while y <= end:
        b = min(dt.date(y.year, 12, 31), end)
        for n, v in get(y.isoformat(), b.isoformat()).items():
            series[n].update(v)
        print(y.year, sum(x is not None for x in series[next(iter(POINTS))].values()), "hours with rain value", flush=True)
        time.sleep(5)
        y = b + dt.timedelta(days=1)
    hours = sorted(set().union(*[s.keys() for s in series.values()]))
    new = 0
    with gzip.open(a.out, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(old)
        for t in hours:
            vals = {n: series[n].get(t) for n in POINTS}
            got = [v for v in vals.values() if v is not None]
            if not got:
                continue
            w.writerow({"hour": t.replace("T", " "), **vals, "basin_mean": round(sum(got) / len(got), 3)})
            new += 1
    print("old rows", len(old), "new rows", new, "from", start, flush=True)


if __name__ == "__main__":
    main()
