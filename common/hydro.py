"""Helpers for the conceptual hydrology models: goodness of fit and calibration."""
import numpy as np


def nse(obs, sim):
    """Nash–Sutcliffe efficiency (1 = perfect, 0 = as good as the mean)."""
    o, s = np.asarray(obs, float), np.asarray(sim, float)
    ok = ~(np.isnan(o) | np.isnan(s))
    o, s = o[ok], s[ok]
    return 1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2) if len(o) > 1 else np.nan


def kge(obs, sim):
    """Kling–Gupta efficiency (1 = perfect)."""
    o, s = np.asarray(obs, float), np.asarray(sim, float)
    ok = ~(np.isnan(o) | np.isnan(s))
    o, s = o[ok], s[ok]
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def calibrate(simulate, obs, bounds, warmup=365, maxiter=60, seed=0):
    """Find parameters that maximise KGE on sqrt flows (balances floods and low flows).

    simulate(params) -> simulated series aligned with obs. Returns the best params.
    """
    from scipy.optimize import differential_evolution
    o = np.sqrt(np.maximum(np.asarray(obs, float), 0))[warmup:]

    def cost(p):
        s = np.sqrt(np.maximum(simulate(p), 0))[warmup:]
        v = kge(o, s)
        return 10.0 if np.isnan(v) else -v

    r = differential_evolution(cost, bounds, maxiter=maxiter, seed=seed, tol=1e-4, polish=False)
    return r.x


def ratio_errors(obs, sim, horizon=1):
    """Past ratios actual / simulated for wet days, used to turn a P50 into P10/P90."""
    o, s = np.asarray(obs, float), np.asarray(sim, float)
    ok = (s > 0.05) & ~np.isnan(o)
    r = o[ok] / s[ok]
    return np.quantile(r, [0.1, 0.9]) if len(r) > 20 else np.array([0.6, 1.6])
