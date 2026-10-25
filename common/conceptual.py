"""Run a conceptual rainfall-runoff model (GR4J, HBV) as a forecaster.

rain and PET are mm/day; inflow y is MCM/day; area_km2 converts between them
(MCM/day = mm/day * km2 / 1000). If area_km2 is None it is calibrated too.
Missing rain is treated as 0 mm, so fill rain (stations / gridded) first.
P10/P90 come from past ratios actual/simulated.
"""
import numpy as np

from .base import result
from .hydro import calibrate, ratio_errors


class Conceptual:
    def __init__(self, simulate, bounds, area_km2=None, pet=4.0, warmup=365, calib_years=8, maxiter=40):
        self.simulate, self.bounds = simulate, list(bounds)
        self.area, self.pet, self.warmup = area_km2, pet, warmup
        self.calib_days, self.maxiter = calib_years * 365, maxiter

    def _forcing(self, df):
        P = np.nan_to_num(df["rain"].to_numpy(float))
        E = df["pet"].to_numpy(float) if "pet" in df else np.full(len(df), self.pet)
        return P, np.nan_to_num(E, nan=self.pet)

    def _split(self, p):
        if self.area is None:
            return p[:-1], p[-1]
        return p, self.area

    def fit(self, df):
        d = df[df["rain"].notna() & df["y"].notna()]
        d = d.iloc[-(self.calib_days + self.warmup):]
        P, E = self._forcing(d)
        obs = d["y"].to_numpy(float)
        bounds = self.bounds + ([(10.0, 3000.0)] if self.area is None else [])

        def sim(p):
            par, area = self._split(p)
            return self.simulate(P, E, par)[0] * area / 1000.0

        self.params = calibrate(sim, obs, bounds, warmup=self.warmup, maxiter=self.maxiter)
        self.par, self.area_fit = self._split(self.params)
        s = sim(self.params)
        self.ratio = ratio_errors(obs[self.warmup:], s[self.warmup:])
        return self

    def predict(self, df, horizon=7, future=None):
        hist = df.iloc[:-1]
        P, E = self._forcing(hist)
        _, state = self.simulate(P, E, self.par)
        fP = np.zeros(horizon)
        if future is not None and "rain" in future:
            v = np.nan_to_num(future["rain"].to_numpy(float))[:horizon]
            fP[:len(v)] = v
        elif not np.isnan(df["rain"].iloc[-1]):
            fP[0] = df["rain"].iloc[-1]
        fE = np.full(horizon, self.pet)
        q, _ = self.simulate(fP, fE, self.par, state)
        q = q * self.area_fit / 1000.0
        return result(q * self.ratio[0], q, q * self.ratio[1])
