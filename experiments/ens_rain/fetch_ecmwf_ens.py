"""Experiment (Nov 2026): ECMWF ENS 51-member daily rain at the reservoirs, from the free open-data archive on AWS
(s3://ecmwf-forecasts, Feb 2023 on). Only the tp (total precipitation) messages are range-downloaded.

Origins are days 1, 6, 11, 16, 21, 26 of each month; each uses the 00Z run of the day before, so day 1 of the
forecast for origin T is lead day 2 of that run (what would really be available on the morning of T).
Writes CSV rows: origin, init, member (0 = control), date (UTC day), lead (days after init), point, mm.
Usage: python fetch_ecmwf_ens.py OUT.csv.gz CHUNK NCHUNKS
"""
import concurrent.futures as cf
import datetime as dt
import gzip
import io
import json
import sys
import time
import urllib.error
import urllib.request

import eccodes
import numpy as np

BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
DIRS = ["ifs/0p25/enfo", "0p25/enfo", "0p4-beta/enfo"]
POINTS = {"DK": (12.890302, 101.219914), "NPL": (12.934647, 101.297011), "KY": (12.964319, 101.313238),
          "PS": (12.977205, 101.558297), "pluakdaeng": (12.99, 101.30), "bothong": (13.05, 101.58)}
STEPS = list(range(24, 361, 24))


def get(url, rng=None, tries=8):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 3 * k)
        except Exception:
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"failed {url} {rng}")


def origins():
    out, d = [], dt.date(2023, 2, 1)
    while d <= dt.date.today() - dt.timedelta(days=2):
        if d.day in (1, 6, 11, 16, 21, 26):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


GRID = {}


def weights(h):
    key = tuple(eccodes.codes_get(h, k) for k in ("Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
                                                   "longitudeOfFirstGridPointInDegrees", "iDirectionIncrementInDegrees",
                                                   "jDirectionIncrementInDegrees"))
    if key not in GRID:
        ni, nj, la0, lo0, di, dj = key
        w = {}
        for name, (lat, lon) in POINTS.items():
            fy = (la0 - lat) / dj
            fx = ((lon - lo0) % 360) / di
            y0, x0 = int(np.floor(fy)), int(np.floor(fx))
            ty, tx = fy - y0, fx - x0
            w[name] = [((y0 + a) * ni + (x0 + b) % ni, (ty if a else 1 - ty) * (tx if b else 1 - tx))
                       for a in (0, 1) for b in (0, 1)]
        GRID[key] = w
    return GRID[key]


def one_message(url, off, ln):
    raw = get(url, (off, off + ln - 1))
    h = eccodes.codes_new_from_message(raw)
    try:
        num = int(eccodes.codes_get(h, "number")) if eccodes.codes_get(h, "dataType") == "pf" else 0
        v = eccodes.codes_get_values(h)
        vals = {p: float(sum(v[i] * wt for i, wt in w)) for p, w in weights(h).items()}
    finally:
        eccodes.codes_release(h)
    return num, vals


def run_init(init):
    ds = init.strftime("%Y%m%d")
    for d in DIRS:
        idx = get(f"{BASE}/{ds}/00z/{d}/{ds}000000-24h-enfo-ef.index")
        if idx:
            break
    else:
        return None
    acc = {}  # (step, member) -> {point: m}
    jobs = []
    with cf.ThreadPoolExecutor(12) as ex:
        for s in STEPS:
            stem = f"{BASE}/{ds}/00z/{d}/{ds}000000-{s}h-enfo-ef"
            index = idx if s == 24 else get(stem + ".index")
            if not index:
                return None
            for line in index.decode().splitlines():
                j = json.loads(line)
                if j.get("param") == "tp":
                    jobs.append((s, ex.submit(one_message, stem + ".grib2", j["_offset"], j["_length"])))
        for s, fut in jobs:
            num, vals = fut.result()
            acc[(s, num)] = vals
    return acc


def main(out, chunk, nchunks):
    todo = origins()[int(chunk)::int(nchunks)]
    with gzip.open(out, "wt") as f:
        f.write("origin,init,member,date,lead,point,mm\n")
        for o in todo:
            init = o - dt.timedelta(days=1)
            t = time.time()
            acc = run_init(init)
            if not acc:
                print(o, "missing", flush=True)
                continue
            members = sorted({m for _, m in acc})
            for m in members:
                prev = {p: 0.0 for p in POINTS}
                for k, s in enumerate(STEPS, start=1):
                    cur = acc.get((s, m))
                    if cur is None:
                        break
                    day = init + dt.timedelta(days=k - 1)
                    for p in POINTS:
                        f.write(f"{o},{init},{m},{day},{k},{p},{max(0.0, (cur[p] - prev[p]) * 1000):.2f}\n")
                    prev = cur
            print(o, len(members), "members", f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
