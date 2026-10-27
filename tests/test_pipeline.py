"""Gate, champion rule and month-end inputs of the run pipeline, on small hand-made results.

Run: python3 tests/test_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import finish, plan  # noqa: E402

T = plan.T1
BASE, A, B = plan.BASELINE[T], f"{T}/05_lightgbm_quantile", f"{T}/11_timesfm"


def res(name, p50s, actual, mae, cov):
    keys = [f"2020-01-31|{i}" for i in range(len(actual))]
    fc = [{"step": 1, "date": "2026-10-31", "p10": 1.0, "p50": 2.0, "p90": 3.0}]
    return {"model": name, "task": T, "agent": plan.MODELS[name][0], "errors": {},
            "sites": {"THREE": {"forecast": fc, "score": {"mae": mae, "coverage": cov, "mae_steps": [mae]},
                                "bt": {"key": keys, "actual": actual, "p50": p50s, "p10": p50s, "p90": p50s}}}}


actual = [10.0, 20.0, 30.0, 40.0]
results = {BASE: res(BASE, [25.0] * 4, actual, 10.0, 0.8),
           A: res(A, [12.0, 18.0, 33.0, 37.0], actual, 2.5, 0.8),   # passes
           B: res(B, [8.0, 22.0, 27.0, 43.0], actual, 2.5, 0.75)}   # passes; errors opposite to A
scores = finish.scores_from(results)
ev = finish.gate(T, "THREE", scores)
assert ev[BASE]["pass"] is None and ev[A]["pass"] and ev[B]["pass"], ev

c = finish.combine(T, "THREE", results, ev, "monthly", None)
assert c["champion"] == "ensemble", c  # A and B cancel out, the blend beats both by far
assert abs(sum(c["weights"].values()) - 1) < 1e-9

results[B]["sites"]["THREE"]["score"]["coverage"] = 0.5  # B now fails the 70% coverage rule
ev = finish.gate(T, "THREE", finish.scores_from(results))
c = finish.combine(T, "THREE", results, ev, "monthly", None)
assert c["champion"] == A and c["weights"] == {A: 1.0}, c

# a daily run keeps the monthly choice, limited to the models that ran today
prev = {T: {"THREE": {"champion": "ensemble", "weights": {A: 0.6, B: 0.4}}}}
only_a = {A: results[A]}
c = finish.combine(T, "THREE", only_a, {}, "daily", prev)
assert c["champion"] == "ensemble" or c["weights"] == {A: 1.0}
c = finish.combine(T, "THREE", {A: results[A], B: results[B]}, {}, "daily", prev)
assert c["reused"] and abs(c["weights"][A] - 0.6) < 1e-9

assert finish.nan_free({"x": float("nan"), "y": [1.0]}) == {"x": None, "y": [1.0]}
print("ok")
