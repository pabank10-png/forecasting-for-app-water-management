def mae(y, p):
    pairs = [(a, b) for a, b in zip(y, p) if a is not None and b is not None]
    return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else None


def coverage(y, lo, hi):
    """Share of actual values inside [lo, hi] (P10–P90 should be about 0.8)."""
    t = [(a, l, h) for a, l, h in zip(y, lo, hi) if None not in (a, l, h)]
    return sum(l <= a <= h for a, l, h in t) / len(t) if t else None
