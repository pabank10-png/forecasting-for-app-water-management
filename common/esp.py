"""Daily HBV and GR4J for ESP (Ensemble Streamflow Prediction) of monthly inflow, Task 1 model 14.

Same equations as the 2026-11-08 ESP experiment (forecasting/experiments/esp_task1/run2.py): seasonal PET
(mean and amplitude calibrated, peak mid-April); calibration objective = half daily KGE(sqrt Q) + half
monthly-total KGE; differential evolution. Each model returns its end state so a run can stop at the forecast
origin and continue with any rain trace.
"""
import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
from numba import njit
from scipy.optimize import differential_evolution


@njit(cache=True)
def pet(doy, pm, amp):
    return pm * (1 + amp * np.cos(2 * np.pi * (doy - 105) / 365.25))

@njit(cache=True)
def hbv(P, D, p, s):
    FC, BETA, LP, K0, K1, K2, UZL, PERC, MAXBAS, AREA, PM, AMP = p
    n = int(np.ceil(MAXBAS)); w = np.zeros(n)
    for i in range(n):
        w[i] = max(0.0, 1 - abs(2 * (i + 0.5) / MAXBAS - 1))
    w = w / w.sum() if w.sum() > 0 else np.ones(1)
    SM, SUZ, SLZ = s[0], s[1], s[2]; buf = s[3:10].copy(); Q = np.zeros(len(P))
    for t in range(len(P)):
        E = pet(D[t], PM, AMP)
        rech = P[t] * (SM / FC) ** BETA; SM += P[t] - rech
        SM = max(0.0, SM - E * min(1.0, SM / (LP * FC)))
        if SM > FC:
            rech += SM - FC; SM = FC
        SUZ += rech; perc = min(PERC, SUZ); SUZ -= perc; SLZ += perc
        q0 = K0 * max(SUZ - UZL, 0.0); q1 = K1 * SUZ; q2 = K2 * SLZ
        SUZ = max(0.0, SUZ - q0 - q1); SLZ = max(0.0, SLZ - q2)
        for i in range(len(w)): buf[i] += (q0 + q1 + q2) * w[i]
        Q[t] = buf[0] * AREA / 1000.0
        for i in range(6): buf[i] = buf[i + 1]
        buf[6] = 0.0
    o = np.zeros(10); o[0] = SM; o[1] = SUZ; o[2] = SLZ; o[3:10] = buf
    return Q, o

@njit(cache=True)
def sh1(t, x4):
    return 0.0 if t <= 0 else (1.0 if t >= x4 else (t / x4) ** 2.5)

@njit(cache=True)
def sh2(t, x4):
    if t <= 0: return 0.0
    if t < x4: return 0.5 * (t / x4) ** 2.5
    if t < 2 * x4: return 1 - 0.5 * (2 - t / x4) ** 2.5
    return 1.0

@njit(cache=True)
def gr4j(P, D, p, s):
    x1, x2, x3, x4, AREA, PM, AMP = p
    uh1 = np.zeros(7); uh2 = np.zeros(13)
    for j in range(1, 8): uh1[j - 1] = sh1(j, x4) - sh1(j - 1, x4)
    for j in range(1, 14): uh2[j - 1] = sh2(j, x4) - sh2(j - 1, x4)
    S, R = s[0], s[1]; q1 = s[2:9].copy(); q2 = s[9:22].copy(); Q = np.zeros(len(P))
    for t in range(len(P)):
        E = pet(D[t], PM, AMP)
        if P[t] >= E:
            pn = P[t] - E; ws = np.tanh(min(pn / x1, 13.0))
            ps = x1 * (1 - (S / x1) ** 2) * ws / (1 + S / x1 * ws); es = 0.0
        else:
            pn = 0.0; en = E - P[t]; ws = np.tanh(min(en / x1, 13.0))
            es = S * (2 - S / x1) * ws / (1 + (1 - S / x1) * ws); ps = 0.0
        S = S - es + ps
        perc = S * (1 - (1 + (4.0 / 9.0 * S / x1) ** 4) ** -0.25); S -= perc
        pr = perc + (pn - ps)
        for j in range(7): q1[j] += 0.9 * pr * uh1[j]
        for j in range(13): q2[j] += 0.1 * pr * uh2[j]
        a = q1[0]; b = q2[0]
        for j in range(6): q1[j] = q1[j + 1]
        q1[6] = 0.0
        for j in range(12): q2[j] = q2[j + 1]
        q2[12] = 0.0
        F = x2 * (max(R, 0.0) / x3) ** 3.5
        R = max(0.0, R + a + F); qr = R * (1 - (1 + (R / x3) ** 4) ** -0.25); R -= qr
        Q[t] = (qr + max(0.0, b + F)) * AREA / 1000.0
    o = np.zeros(22); o[0] = S; o[1] = R; o[2:9] = q1; o[9:22] = q2
    return Q, o

PETB = [(2.5, 5.5), (0.0, 0.35)]
MODELS = {
    'hbv': (hbv, [(50., 700.), (1., 6.), (.3, 1.), (.05, .6), (.01, .4), (.001, .15), (0., 70.), (0., 6.), (1., 6.), (10., 3000.)] + PETB,
            lambda p: np.r_[0.5 * p[0], 0.0, 10.0, np.zeros(7)]),
    'gr4j': (gr4j, [(50., 1500.), (-5., 3.), (10., 500.), (0.5, 6.0), (10., 3000.)] + PETB,
             lambda p: np.r_[0.5 * p[0], 0.3 * p[2], np.zeros(20)]),
}

def scale_state(name, s, r):
    s = s.copy()
    if name == 'hbv': s[1:10] *= r          # upper + lower groundwater + routing buffer
    else: s[1:22] *= r                      # routing store + unit-hydrograph queues
    return s

def kge(o, s):
    if s.std() == 0: return -10.0
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)

def calibrate(name, P, D, y, mi, warm=365):
    f, bounds, s0 = MODELS[name]
    ok = ~np.isnan(y); ok[:warm] = False
    o = np.sqrt(np.maximum(y[ok], 0))
    mo = np.bincount(mi[ok], weights=y[ok]); has = np.bincount(mi[ok]) > 25
    def cost(p):
        q, _ = f(P, D, p, s0(p))
        v = 0.5 * kge(o, np.sqrt(np.maximum(q[ok], 0))) + 0.5 * kge(mo[has], np.bincount(mi[ok], weights=q[ok])[has])
        return 10.0 if np.isnan(v) else -v
    return differential_evolution(cost, bounds, maxiter=50, seed=1, tol=1e-4, polish=False).x
