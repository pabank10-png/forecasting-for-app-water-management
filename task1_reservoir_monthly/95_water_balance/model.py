"""Month-end storage, same rule as the old water-dashboard page (runSc).

end = start + inflow * multiplier + diversion_in - outflow, clamped to 0..capacity.
Outflow and diversion are the values the user types in on the website.
"""

RESERVOIRS = {
    "THREE": {"capacity": 318.1, "dead": 31.7},
    "PS": {"capacity": 295.0, "dead": 20.0},
}


def run(start, inflows, outflows, diversions=None, multiplier=1.0, capacity=None):
    diversions = diversions or [0.0] * len(inflows)
    out, s = [], start
    for q, o, d in zip(inflows, outflows, diversions):
        s = s + q * multiplier + d - o
        s = max(0.0, s if capacity is None else min(capacity, s))
        out.append(s)
    return out
