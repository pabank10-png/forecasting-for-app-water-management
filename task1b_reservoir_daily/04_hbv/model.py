"""HBV (SMHI) daily model without snow: soil moisture box, upper (fast) and lower (slow)
groundwater boxes, triangular routing over MAXBAS days.

Params: FC, BETA, LP, K0, K1, K2, UZL, PERC, MAXBAS.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from common.conceptual import Conceptual

BOUNDS = [(50.0, 700.0), (1.0, 6.0), (0.3, 1.0), (0.05, 0.6), (0.01, 0.4), (0.001, 0.15),
          (0.0, 70.0), (0.0, 6.0), (1.0, 6.0)]


def triangle(maxbas):
    n = int(np.ceil(maxbas))
    t = np.arange(1, n + 1) - 0.5
    w = np.maximum(0, 1 - np.abs(2 * t / maxbas - 1))
    return w / w.sum() if w.sum() else np.ones(1)


def simulate(P, E, params, state=None):
    FC, BETA, LP, K0, K1, K2, UZL, PERC, MAXBAS = params
    w = triangle(MAXBAS)
    SM, SUZ, SLZ, buf = state if state is not None else (0.5 * FC, 0.0, 10.0, np.zeros(len(w)))
    buf = buf.copy()
    Q = np.zeros(len(P))
    for t in range(len(P)):
        rech = P[t] * (SM / FC) ** BETA
        SM += P[t] - rech
        SM = max(0.0, SM - E[t] * min(1.0, SM / (LP * FC)))
        if SM > FC:
            rech += SM - FC
            SM = FC
        SUZ += rech
        perc = min(PERC, SUZ)
        SUZ -= perc
        SLZ += perc
        q0 = K0 * max(SUZ - UZL, 0.0)
        q1 = K1 * SUZ
        q2 = K2 * SLZ
        SUZ = max(0.0, SUZ - q0 - q1)
        SLZ = max(0.0, SLZ - q2)
        buf += (q0 + q1 + q2) * w
        Q[t] = buf[0]
        buf = np.append(buf[1:], 0.0)
    return Q, (SM, SUZ, SLZ, buf)


class Model(Conceptual):
    def __init__(self, area_km2=None, pet=4.0, **kw):
        super().__init__(simulate, BOUNDS, area_km2, pet, **kw)
