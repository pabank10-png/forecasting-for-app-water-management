"""HEC-HMS style lumped basin model + rating curve, written in Python.

Rain (mm/h) -> SCS Curve Number excess (cumulative per storm; a storm ends
after 24 dry hours) -> SCS triangular unit hydrograph -> discharge (m3/s)
+ baseflow -> optional Muskingum reach -> water level by rating curve
h = h0 + (Q/a)^(1/b). CN, lag, baseflow and the rating are calibrated on
the observed level. This is the hydrology half; full HEC-RAS hydraulics
(Saint-Venant on surveyed cross-sections) needs the HEC software itself.

Forecast rain for the next hours goes in `future["rain"]`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.base import history, interval_from_errors
from common.routing import muskingum


def storm_rain(rain, dry_hours=24):
    """Cumulative rain since the start of the current storm, hour by hour."""
    r = np.nan_to_num(np.asarray(rain, float))
    wet = np.convolve(r > 0, np.ones(dry_hours), "full")[:len(r)] > 0
    start = ~wet | np.r_[True, ~wet[:-1]]
    c = np.cumsum(r)
    base = np.maximum.accumulate(np.where(start, c - r, 0))
    return c - base


def scs_excess(p_cum, cn):
    s = 25400.0 / cn - 254.0
    ia = 0.2 * s
    q = np.where(p_cum > ia, (p_cum - ia) ** 2 / (p_cum - ia + s), 0.0)
    return np.diff(q, prepend=0.0).clip(min=0)


def unit_hydrograph(area_km2, lag_h, dt=1.0):
    """SCS triangular UH: m3/s per mm of excess. tp = dt/2 + lag, tb = 2.67 tp, qp = 0.208 A / tp."""
    tp = dt / 2 + lag_h
    tb = 2.67 * tp
    t = np.arange(dt, tb + dt, dt)
    qp = 0.208 * area_km2 / tp
    return np.where(t <= tp, qp * t / tp, qp * np.maximum(tb - t, 0) / (tb - tp))


class Model:
    BOUNDS = {"cn": (40, 98), "lag": (0.5, 24), "base": (0, 50), "a": (0.5, 500), "b": (1.2, 2.8)}

    def __init__(self, area_km2=100.0, horizon=48, calib_hours=24 * 365, reach=None, maxiter=25, seed=0):
        """reach: (K, X) of a Muskingum reach between basin outlet and station, or None."""
        self.area, self.horizon, self.calib, self.reach = area_km2, horizon, calib_hours, reach
        self.maxiter, self.seed = maxiter, seed

    def _q(self, p, rain):
        cn, lag, base = p[:3]
        ex = scs_excess(storm_rain(rain), cn)
        q = np.convolve(ex, unit_hydrograph(self.area, lag))[:len(ex)] + base
        return muskingum(q, *self.reach) if self.reach else q

    def _level(self, p, rain):
        return self.h0 + (self._q(p, rain) / p[3]) ** (1 / p[4])

    def fit(self, df):
        from scipy.optimize import differential_evolution
        y = history(df)
        rain = df["rain"].to_numpy(float)[:len(y)]
        lo = max(0, len(y) - self.calib)
        y, rain = y[lo:], rain[lo:]
        self.h0 = float(np.nanmin(y)) - 0.05
        ok = ~np.isnan(y)
        warm = min(240, len(y) // 4)
        ok[:warm] = False

        def cost(p):
            return float(np.mean((self._level(p, rain)[ok] - y[ok]) ** 2))

        r = differential_evolution(cost, list(self.BOUNDS.values()), maxiter=self.maxiter, seed=self.seed,
                                   polish=False, tol=1e-6)
        self.p = r.x
        res = y - self._level(self.p, rain)
        self.errors = [res[h:] - res[:-h] for h in range(1, self.horizon + 1)]
        return self

    def params(self):
        return dict(zip(self.BOUNDS, self.p), h0=self.h0)

    def predict(self, df, horizon=48, future=None):
        y = history(df)
        n = len(y)
        rain = df["rain"].to_numpy(float)[:n]
        fut = future["rain"].to_numpy(float)[:horizon] if future is not None and "rain" in future \
            else np.zeros(horizon)
        lvl = self._level(self.p, np.concatenate([rain, fut]))
        offset = y[-1] - lvl[n - 1]  # start from the observed level
        p50 = lvl[n:n + horizon] + offset
        return interval_from_errors(p50, self.errors, floor=None)
