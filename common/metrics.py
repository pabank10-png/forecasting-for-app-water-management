import math


def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def mae(y, p):
    pairs = [(a, b) for a, b in zip(y, p) if _ok(a) and _ok(b)]
    return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else None


def coverage(y, lo, hi):
    """Share of actual values inside [lo, hi] (P10–P90 should be about 0.8)."""
    t = [(a, l, h) for a, l, h in zip(y, lo, hi) if _ok(a) and _ok(l) and _ok(h)]
    return sum(l <= a <= h for a, l, h in t) / len(t) if t else None
