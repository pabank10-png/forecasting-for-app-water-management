"""Climatology: P10/P50/P90 of the same calendar month in past years."""
from statistics import quantiles


def fit(train):
    """train: [(month, inflow)] from full past months only."""
    by = {}
    for m, v in train:
        if v is not None:
            by.setdefault(m, []).append(v)
    return by


def predict(model, month):
    v = sorted(model.get(month, []))
    if len(v) < 3:
        return None
    q = quantiles(v, n=10, method="inclusive")
    return {"p10": q[0], "p50": quantiles(v, n=2, method="inclusive")[0], "p90": q[8]}
