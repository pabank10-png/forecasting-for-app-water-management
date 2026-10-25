"""Hourly river data layout used by task2 models.

df columns: date (hourly), y = water level at the target station (m),
upstream station levels (names given per model, e.g. "up1"), rain = basin
rain mm/h, optional release = reservoir release/spill from task 1B (m3/s).
Forecast rain and planned releases for the next hours go in `future`.
"""
from functools import partial

from .features import build


def river_builder(upstream=(), rain="rain", extra=()):
    lag_cols = {u: (1, 2, 3, 6) for u in upstream}
    lag_cols.update({e: (1, 3, 6) for e in extra})
    return partial(build, lags=(1, 2, 3, 6, 12, 24), lag_cols=lag_cols,
                   sums={rain: (1, 3, 6, 12, 24, 72)}, calendar="hour")


STEPS = (1, 2, 3, 6, 12, 24, 36, 48)
