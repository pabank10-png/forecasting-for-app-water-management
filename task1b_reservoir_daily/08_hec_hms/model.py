"""HEC-HMS style daily basin model (continuous), written in Python for comparison with GR4J/HBV.

Uses the HEC-HMS methods meant for continuous simulation:
loss = Deficit and Constant (soil deficit D up to dmax mm, refilled by rain, dried by PET;
once full, a constant rate f mm/day percolates and the rest runs off),
transform = SCS unit hydrograph (lag days), baseflow = linear reservoir (recession k days)
fed by a fraction g of the percolation. Calibrated by maximising KGE on sqrt(flow).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.conceptual import Conceptual

try:
    from numba import njit
except ImportError:  # works without numba, just slower
    def njit(f):
        return f

# dmax mm, f mm/day, lag days, k days, g fraction
BOUNDS = [(5.0, 600.0), (0.1, 40.0), (0.2, 6.0), (2.0, 300.0), (0.0, 1.0)]


def unit_hydrograph(lag):
    """SCS triangular unit hydrograph on a daily step: Tp = dt/2 + lag, base = 2.67 Tp."""
    tp = 0.5 + lag
    tb = 2.67 * tp

    def cum(t):  # area under the triangle up to t (total 1)
        if t <= 0:
            return 0.0
        if t <= tp:
            return t * t / (tp * tb)
        if t < tb:
            return 1 - (tb - t) ** 2 / ((tb - tp) * tb)
        return 1.0
    n = int(np.ceil(tb))
    return np.array([cum(j) - cum(j - 1) for j in range(1, n + 1)])


@njit
def _run(P, E, dmax, f, g, k, uh, D, G, q):
    n = len(P)
    Q = np.zeros(n)
    for t in range(n):
        p = P[t]
        D = min(dmax, D + E[t]) if p < E[t] else D
        p = max(0.0, p - E[t]) if p >= E[t] else 0.0
        fill = min(p, D)
        D -= fill
        p -= fill
        perc = min(f, p) if D <= 0 else 0.0
        excess = p - perc
        for j in range(len(uh)):
            q[j] += excess * uh[j]
        direct = q[0]
        q[:-1] = q[1:].copy()
        q[-1] = 0.0
        G += g * perc
        base = G / k
        G -= base
        Q[t] = direct + base
    return Q, D, G


def simulate(P, E, params, state=None):
    """Daily runoff in mm/day; returns (Q, state) so a forecast can continue from the last day."""
    dmax, f, lag, k, g = params
    uh = unit_hydrograph(lag)
    if state is None:
        state = (0.5 * dmax, 0.0, np.zeros(len(uh)))
    D, G, q = state[0], state[1], state[2].copy()
    Q, D, G = _run(np.asarray(P, float), np.asarray(E, float), dmax, f, g, k, uh, D, G, q)
    return Q, (D, G, q)


class Model(Conceptual):
    def __init__(self, area_km2=None, pet=4.0, **kw):
        super().__init__(simulate, BOUNDS, area_km2, pet, **kw)
