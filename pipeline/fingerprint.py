"""What each model would read for each site, as a hash. A daily run compares it with the last finished run and
reruns only the (model, site) pairs whose inputs changed; the others keep the previous forecast."""
import datetime as dt
import hashlib

import pandas as pd

from common.data import data_dir

from . import data_prep, plan


def _hash(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:16]


def _rows(path, key, value):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    return df[df[key] == value].to_csv(index=False).encode()


def site_inputs(task, site, rain_model, today=None):
    d = data_dir()
    if task == plan.T1:
        return _hash(_rows(d / "lag.csv", "reservoir", site))
    parts = [(d / f"daily_{site}.csv").read_bytes()]
    if rain_model:
        rain = data_prep.station_rain(data_prep.DAMS[site][2])
        # forecast rain changes with every model cycle: a rain model reruns at least once a day
        parts += [rain.to_csv().encode(), (today or dt.date.today()).isoformat()]
    return _hash(*parts)


def all_inputs(models):
    """{model: {site: hash}}"""
    out = {}
    for m in models:
        task = m.split("/")[0]
        out[m] = {s: site_inputs(task, s, m in plan.RAIN_MODELS) for s in plan.SITES[task]}
    return out
