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


def spill(start, inflows, outflows, diversions=None, multiplier=1.0, capacity=None):
    """Water above capacity each month (what run() cuts off), in MCM."""
    diversions = diversions or [0.0] * len(inflows)
    out, s = [], start
    for q, o, d in zip(inflows, outflows, diversions):
        s = max(0.0, s + q * multiplier + d - o)
        over = max(0.0, s - capacity) if capacity is not None else 0.0
        s -= over
        out.append(over)
    return out


def from_forecast(start, forecast, outflows, reservoir="THREE", diversions=None, multiplier=1.0):
    """Storage path for P10/P50/P90 inflow (a model's output) plus spill and Hard Red flags.

    forecast: DataFrame with p10/p50/p90 per month. Returns a list of dicts per month.
    """
    cap, dead = RESERVOIRS[reservoir]["capacity"], RESERVOIRS[reservoir]["dead"]
    paths = {q: run(start, list(forecast[q]), outflows, diversions, multiplier, cap) for q in ("p10", "p50", "p90")}
    spills = spill(start, list(forecast["p90"]), outflows, diversions, multiplier, cap)
    return [{"step": i + 1, "storage_p10": paths["p10"][i], "storage_p50": paths["p50"][i],
             "storage_p90": paths["p90"][i], "spill_p90": spills[i], "below_hard_red_p10": paths["p10"][i] < dead}
            for i in range(len(forecast))]
