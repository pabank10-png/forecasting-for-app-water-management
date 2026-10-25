"""Combine forecasts from several models or several rain sources."""
import numpy as np

from .base import result


def inverse_mae_weights(maes, shrink=0.5):
    """w_i = (1/MAE_i) / sum(1/MAE_j), then shrunk toward equal weights: lambda*w + (1-lambda)/n."""
    names = list(maes)
    inv = np.array([1.0 / max(maes[k], 1e-9) for k in names])
    w = inv / inv.sum()
    w = shrink * w + (1 - shrink) / len(w)
    return dict(zip(names, w))


def combine(forecasts, weights=None):
    """Weighted average of P10/P50/P90 per step. forecasts: {name: model output}."""
    names = list(forecasts)
    w = weights or {k: 1 / len(names) for k in names}
    tot = sum(w[k] for k in names)
    a = sum(forecasts[k][["p10", "p50", "p90"]].to_numpy() * w[k] for k in names) / tot
    return result(a[:, 0], a[:, 1], a[:, 2])


def exceed_probability(values, weights, threshold):
    """Weighted share of members above a threshold (e.g. reservoir capacity, river bank level)."""
    return float(sum(weights[k] for k, v in values.items() if v > threshold) / sum(weights[k] for k in values))


def choose(maes, ensemble_mae, margin=0.02):
    """Champion rule: use the ensemble only if it beats the best single model by at least `margin`."""
    best = min(maes, key=maes.get)
    return "ensemble" if ensemble_mae <= maes[best] * (1 - margin) else best
