"""Daily water balance with spillway: storage tomorrow = today + inflow - release - use/evaporation.

Water above `capacity` leaves over the spillway, up to `spillway_capacity`
(MCM/day); anything beyond that stays in the reservoir as surcharge above the
normal level, which is the case to warn about.
"""


def run(start, inflows, releases, use_evap=None, capacity=None, spillway_capacity=None):
    use_evap = use_evap or [0.0] * len(inflows)
    out, s = [], start
    for q, r, e in zip(inflows, releases, use_evap):
        s = max(0.0, s + q - r - e)
        spill = surcharge = 0.0
        if capacity is not None and s > capacity:
            over = s - capacity
            spill = over if spillway_capacity is None else min(over, spillway_capacity)
            s -= spill
            surcharge = max(0.0, s - capacity)
        out.append({"storage": s, "spill": spill, "surcharge": surcharge})
    return out


def from_forecast(start, forecast, releases, capacity, spillway_capacity=None, use_evap=None):
    """Storage and spill for each of P10/P50/P90 daily inflow (a model's output)."""
    rows = {q: run(start, list(forecast[q]), releases, use_evap, capacity, spillway_capacity)
            for q in ("p10", "p50", "p90")}
    return [{"step": i + 1, **{f"{k}_{q}": rows[q][i][k] for q in rows for k in ("storage", "spill", "surcharge")}}
            for i in range(len(forecast))]
