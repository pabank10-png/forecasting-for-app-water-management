"""Experiment (Nov 2026): the forecast rain the live pipeline uses (simple mean of ECMWF IFS, GFS, ICON and JMA from
Open-Meteo, as runner.py takes for rain7 and the Storm Trigger), on the same origins as the ECMWF ENS / WN3 tests.
From Open-Meteo Previous Runs: lead k of origin o = day o+k-1 as forecast k days earlier (run of the day before
the origin for lead 1). Days 1-7 only. Written as member 0. Usage: python fetch_live_rain.py SITE OUT.csv.gz
"""
import datetime as dt
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import fetch_ecmwf_ens as E
from pipeline import data_prep


def main(site, out):
    origins = E.origins()
    arch = data_prep.rain_archive(site, origins[0], origins[-1] + dt.timedelta(days=7), 7)
    if not arch:
        sys.exit("no archive")
    print(site, "sources", list(arch), flush=True)
    n = 0
    with gzip.open(out, "wt") as f:
        f.write("origin,init,member,date,lead,point,mm\n")
        for o in origins:
            vals = []
            for k in range(1, 8):
                day = pd.Timestamp(o) + pd.Timedelta(days=k - 1)
                v = [a.at[day, k] for a in arch.values() if day in a.index and k in a.columns and not np.isnan(a.at[day, k])]
                vals.append(np.mean(v) if len(v) >= 2 else np.nan)
            if np.isnan(vals).any():
                continue
            init = o - dt.timedelta(days=1)
            for k, v in enumerate(vals, start=1):
                f.write(f"{o},{init},0,{o + dt.timedelta(days=k - 1)},{k},{site},{max(0.0, v):.2f}\n")
            n += 1
    print(site, n, "origins of", len(origins), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
