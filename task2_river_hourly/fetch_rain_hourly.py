"""Task 2 data: hourly rain history over the Rayong river basins from ERA5-Land (Open-Meteo archive, ~0.1 deg).

Gauge rain only exists hourly from Apr 2025 (DWR) and Oct 2026 (our own collector), so the long river record
(RAY001 2013, RAY002/RAY004 2016, Z.38 2020) is paired with ERA5-Land. One column per grid point plus the basin
means used by the models. Times are Bangkok local hours; rain of hour h-1..h is stored at h.
ERA5-Land runs ~5 days behind, so a top-up re-asks the last 10 stored days.

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
       "&hourly=precipitation&models=era5_land&timezone=Asia%2FBangkok")
# Upper Khlong Yai / Rayong river (Dok Krai, Nong Pla Lai, Ban Khai, city) and the Z.38 side.
POINTS = {"p1275_10120": (12.75, 101.20), "p1275_10135": (12.75, 101.35), "p1290_10120": (12.90, 101.20),
          "p1290_10135": (12.90, 101.35), "p1305_10125": (13.05, 101.25), "p1305_10140": (13.05, 101.40)}


def get(lat, lon, a, b, tries=5):
    lats, lons = ",".join(str(p[0]) for p in [(lat, lon)]), ",".join(str(p[1]) for p in [(lat, lon)])
    for k in range(tries):
        try:
            req = urllib.request.Request(URL.format(lat=lats, lon=lons, a=a, b=b),
                                         headers={"User-Agent": "WaterIntelligenceForecast/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                h = json.load(r)["hourly"]
            return dict(zip(h["time"], h["precipitation"]))
        except Exception as e:  # noqa: BLE001
            print(lat, lon, a, "retry", type(e).__name__, flush=True)
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"ERA5-Land failed {lat},{lon} {a}..{b}")


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
    series = {}
    for name, (lat, lon) in POINTS.items():
        y = start
        while y <= end:
            b = min(dt.date(y.year, 12, 31), end)
            series[name] = {**series.get(name, {}), **get(lat, lon, y.isoformat(), b.isoformat())}
            time.sleep(2)
            y = b + dt.timedelta(days=1)
        print(name, len(series[name]), "hours", flush=True)
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
