"""Daily archive of ensemble forecast rain at the reservoirs, for a later fair test of ensemble inflow forecasts
(user, 15 Nov 2026: collect first, then check it really is better before it goes into the live forecast).

Sources, per dam, daily totals over UTC days (the gauge day), 15-16 days ahead:
  ecmwf_ens : ECMWF ENS 00Z run, 51 members (open data, Google Cloud mirror)
  wn2       : WeatherNext 2 ensemble, 64 members (Open-Meteo ensemble API, latest run)
Writes OUT.csv.gz with rows: source, init, member, date, lead, point, mm   (lead 1 = init day)
Usage: python live_archive.py OUT.csv.gz
"""
import datetime as dt
import gzip
import json
import sys
import time
import urllib.request

import fetch_ecmwf_ens as E

DAMS = {k: v for k, v in E.POINTS.items() if k in ("DK", "KY", "NPL", "PS")}
WN2 = ("https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}&daily=precipitation_sum"
       "&timezone=GMT&forecast_days=16&models=google_weathernext2_ensemble")


def ecmwf(rows):
    today = dt.datetime.now(dt.timezone.utc).date()
    for init in (today, today - dt.timedelta(days=1)):
        acc = E.run_init(init)
        if acc and len({m for _, m in acc}) >= 40 and all((s, 0) in acc for s in E.STEPS):
            break
    else:
        print("ECMWF ENS: no complete run", flush=True)
        return
    for m in sorted({m for _, m in acc}):
        prev = {p: 0.0 for p in DAMS}
        for k, s in enumerate(E.STEPS, start=1):
            cur = acc.get((s, m))
            if cur is None:
                break
            for p in DAMS:
                rows.append(("ecmwf_ens", init, m, init + dt.timedelta(days=k - 1), k, p, max(0.0, (cur[p] - prev[p]) * 1000)))
            prev = cur
    print("ECMWF ENS", init, len({m for _, m in acc}), "members", flush=True)


def wn2(rows):
    for p, (lat, lon) in DAMS.items():
        for k in range(5):
            try:
                req = urllib.request.Request(WN2.format(lat=lat, lon=lon), headers={"User-Agent": "WaterIntelligenceForecast/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    d = json.load(r)["daily"]
                break
            except Exception as e:  # noqa: BLE001
                print("WN2", p, "retry", type(e).__name__, flush=True)
                time.sleep(10 * (k + 1))
        else:
            continue
        days = [dt.date.fromisoformat(x) for x in d["time"]]
        keys = sorted(k for k in d if k.startswith("precipitation_sum"))
        for key in keys:
            m = 0 if key == "precipitation_sum" else int(key.rsplit("member", 1)[1])
            for k, (day, v) in enumerate(zip(days, d[key]), start=1):
                if v is not None:
                    rows.append(("wn2", days[0], m, day, k, p, float(v)))
        print("WN2", p, len(keys), "members", flush=True)


def main(out):
    rows = []
    for f in (ecmwf, wn2):
        try:
            f(rows)
        except Exception as e:  # noqa: BLE001 - one source failing must not lose the other
            print(f.__name__, "failed", e, flush=True)
    with gzip.open(out, "wt") as f:
        f.write("source,init,member,date,lead,point,mm\n")
        for r in rows:
            f.write(",".join(str(x) for x in r[:-1]) + f",{r[-1]:.2f}\n")
    print(len(rows), "rows", flush=True)
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1])
