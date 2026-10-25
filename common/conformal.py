"""Conformalized Quantile Regression (CQR): widen or narrow P10–P90 so that
about 80% of past actual values fall inside it. Scores are kept per step ahead."""
import numpy as np

from .base import result


def scores(actual, p10, p90):
    """s = max(P10 - y, y - P90); negative means inside the band."""
    a, lo, hi = (np.asarray(v, float) for v in (actual, p10, p90))
    return np.maximum(lo - a, a - hi)


def qhat(s, coverage=0.8):
    s = np.asarray(s, float)
    s = s[~np.isnan(s)]
    if len(s) < 5:
        return 0.0
    k = min(1.0, np.ceil((len(s) + 1) * coverage) / len(s))
    return float(np.quantile(s, k, method="higher"))


def apply(fc, q, floor=0.0):
    """fc: model output; q: one value or one per step."""
    q = np.broadcast_to(np.asarray(q, float), (len(fc),))
    return result(fc["p10"].to_numpy() - q, fc["p50"], fc["p90"].to_numpy() + q, floor)


def from_backtest(bt, coverage=0.8):
    """bt: backtest table (step, actual, p10, p90). Returns q-hat per step."""
    out = []
    for h in sorted(bt["step"].unique()):
        b = bt[bt["step"] == h]
        out.append(qhat(scores(b["actual"], b["p10"], b["p90"]), coverage))
    return np.array(out)
