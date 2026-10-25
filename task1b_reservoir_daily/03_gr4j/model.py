"""GR4J (Perrin et al. 2003): 4-parameter daily rainfall-runoff model.

x1 production (soil) store capacity mm, x2 groundwater exchange mm/day,
x3 routing store capacity mm, x4 unit-hydrograph time base days.
Calibrated by maximising KGE on sqrt(flow) (see common/hydro.py).
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

BOUNDS = [(50.0, 1500.0), (-5.0, 3.0), (10.0, 500.0), (0.5, 6.0)]


def _sh1(t, x4):
    return 0.0 if t <= 0 else (1.0 if t >= x4 else (t / x4) ** 2.5)


def _sh2(t, x4):
    if t <= 0:
        return 0.0
    if t < x4:
        return 0.5 * (t / x4) ** 2.5
    if t < 2 * x4:
        return 1 - 0.5 * (2 - t / x4) ** 2.5
    return 1.0


def unit_hydrographs(x4):
    n1, n2 = int(np.ceil(x4)), int(np.ceil(2 * x4))
    uh1 = np.array([_sh1(j, x4) - _sh1(j - 1, x4) for j in range(1, n1 + 1)])
    uh2 = np.array([_sh2(j, x4) - _sh2(j - 1, x4) for j in range(1, n2 + 1)])
    return uh1, uh2


@njit
def _run(P, E, x1, x2, x3, uh1, uh2, S, R, q1, q2):
    n = len(P)
    Q = np.zeros(n)
    for t in range(n):
        if P[t] >= E[t]:
            pn, en = P[t] - E[t], 0.0
            ws = np.tanh(min(pn / x1, 13.0))
            ps = x1 * (1 - (S / x1) ** 2) * ws / (1 + S / x1 * ws)
            es = 0.0
        else:
            pn, en = 0.0, E[t] - P[t]
            ws = np.tanh(min(en / x1, 13.0))
            es = S * (2 - S / x1) * ws / (1 + (1 - S / x1) * ws)
            ps = 0.0
        S = S - es + ps
        perc = S * (1 - (1 + (4.0 / 9.0 * S / x1) ** 4) ** -0.25)
        S -= perc
        pr = perc + (pn - ps)
        for j in range(len(uh1)):
            q1[j] += 0.9 * pr * uh1[j]
        for j in range(len(uh2)):
            q2[j] += 0.1 * pr * uh2[j]
        a, b = q1[0], q2[0]
        q1[:-1] = q1[1:].copy()
        q1[-1] = 0.0
        q2[:-1] = q2[1:].copy()
        q2[-1] = 0.0
        F = x2 * (max(R, 0.0) / x3) ** 3.5
        R = max(0.0, R + a + F)
        qr = R * (1 - (1 + (R / x3) ** 4) ** -0.25)
        R -= qr
        Q[t] = qr + max(0.0, b + F)
    return Q, S, R


def simulate(P, E, params, state=None):
    """Daily runoff in mm/day; returns (Q, state) so a forecast can continue from the last day."""
    x1, x2, x3, x4 = params
    uh1, uh2 = unit_hydrographs(x4)
    if state is None:
        state = (0.5 * x1, 0.3 * x3, np.zeros(len(uh1)), np.zeros(len(uh2)))
    S, R, q1, q2 = state[0], state[1], state[2].copy(), state[3].copy()
    Q, S, R = _run(np.asarray(P, float), np.asarray(E, float), x1, x2, x3, uh1, uh2, S, R, q1, q2)
    return Q, (S, R, q1, q2)


class Model(Conceptual):
    def __init__(self, area_km2=None, pet=4.0, **kw):
        super().__init__(simulate, BOUNDS, area_km2, pet, **kw)
