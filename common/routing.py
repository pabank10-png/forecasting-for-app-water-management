"""River routing and stage–discharge helpers (Muskingum, rating curve)."""
import numpy as np


def muskingum_coefficients(K, X, dt):
    d = 2 * K * (1 - X) + dt
    return (dt - 2 * K * X) / d, (dt + 2 * K * X) / d, (2 * K * (1 - X) - dt) / d


def muskingum(inflow, K, X, dt=1.0, out0=None):
    """Route an inflow hydrograph (m3/s) through a reach: O2 = C0*I2 + C1*I1 + C2*O1."""
    c0, c1, c2 = muskingum_coefficients(K, X, dt)
    I = np.asarray(inflow, float)
    O = np.empty_like(I)
    O[0] = I[0] if out0 is None else out0
    for t in range(1, len(I)):
        O[t] = c0 * I[t] + c1 * I[t - 1] + c2 * O[t - 1]
    return O


def fit_muskingum(inflow, outflow, dt=1.0):
    """K (hours) and X that best reproduce observed outflow. Stable when dt/2 <= K*X... is checked by bounds."""
    from scipy.optimize import minimize
    I, O = np.asarray(inflow, float), np.asarray(outflow, float)

    def cost(p):
        K, X = p
        if K <= 0 or not 0 <= X <= 0.5 or (dt - 2 * K * X) < -1e-9 or (2 * K * (1 - X) - dt) < -1e-9:
            return 1e12
        s = muskingum(I, K, X, dt, O[0])
        ok = ~np.isnan(O)
        return float(np.mean((s[ok] - O[ok]) ** 2))

    best = min((minimize(cost, [k, x], method="Nelder-Mead") for k in (2, 6, 12, 24) for x in (0.1, 0.25)),
               key=lambda r: r.fun)
    return tuple(best.x)


def fit_rating(level, discharge):
    """Q = a * (h - h0)^b fitted to gauged pairs. Returns (a, b, h0)."""
    from scipy.optimize import curve_fit
    h, q = np.asarray(level, float), np.asarray(discharge, float)
    ok = ~(np.isnan(h) | np.isnan(q)) & (q > 0)
    h, q = h[ok], q[ok]

    def f(x, a, b, h0):
        return a * np.maximum(x - h0, 1e-6) ** b

    p, _ = curve_fit(f, h, q, p0=[max(q.mean(), 1.0), 1.6, h.min() - 0.5], maxfev=20000)
    return tuple(p)


def discharge(level, a, b, h0):
    return a * np.maximum(np.asarray(level, float) - h0, 0) ** b


def level(q, a, b, h0):
    return h0 + (np.maximum(np.asarray(q, float), 0) / a) ** (1 / b)
