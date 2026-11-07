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

assert finish.cover_band({"n": 360, "mae_steps": [1, 2, 3]}) == (0.727, 0.873)  # 120 origins
assert finish.cover_band({}) == (0.7, 0.9)
results[A]["sites"]["THREE"]["score"].update(n=120, coverage=0.95)  # too wide fails too
assert not finish.gate(T, "THREE", finish.scores_from(results))[A]["pass"]
results[A]["sites"]["THREE"]["score"]["coverage"] = 0.8
results[B]["sites"]["THREE"]["score"]["coverage"] = 0.5  # B now fails the coverage rule
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

# Daily Nowcast: 20 days seen (1 a day) + 7 forecast days (2 a day) + 4 days at the day average (3 a day)
import pandas as pd  # noqa: E402
days = pd.date_range("2026-10-01", "2026-10-20")
hist = pd.DataFrame({"date": pd.date_range("2025-10-01", "2026-10-20")})
hist["y"] = [3.0 if d < days[0] else 1.0 for d in hist["date"]]
finish.data_prep.daily = lambda c: hist
fc = [{"step": i + 1, "date": f"2026-10-{21 + i}", "p10": 1.0, "p50": 2.0, "p90": 4.0} for i in range(7)]
nc = finish.daily_nowcast("PS", "2026-10-31", {"PS": fc}, True)
assert nc["seen"] == 20 and nc["daily_forecast"] == 14 and nc["day_average"] == 12 and nc["p50"] == 46, nc
assert nc["days_left"] == 11 and nc["p10"] < 46 < nc["p90"]
assert finish.daily_nowcast("PS", "2026-10-31", {}, True) is None  # no daily forecast: monthly stays
print("ok")

# Forecast-rain archive: hourly Previous Runs JSON -> daily rain per source and lead
import numpy as np  # noqa: E402
from pipeline import data_prep, runner  # noqa: E402
hours = pd.date_range("2024-06-01", periods=48, freq="h")
js = {"time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
      "precipitation_previous_day1_ecmwf_ifs025": [1.0] * 48,
      "precipitation_previous_day2_ecmwf_ifs025": [0.5] * 30 + [None] * 18}
arc = data_prep.parse_archive(js, 2)
assert list(arc) == ["ECMWF"] and arc["ECMWF"].loc["2024-06-01", 1] == 24 and arc["ECMWF"].loc["2024-06-01", 2] == 12
assert np.isnan(arc["ECMWF"].loc["2024-06-02", 2])  # only 6 hours: left empty


class Echo:  # inflow = forecast rain, so a source's error is its rain error
    def fit(self, df):
        return self

    def predict(self, df, h, fut):
        return pd.DataFrame({"p50": fut["rain"].to_numpy(float)})


days = pd.date_range("2024-06-01", periods=200)
truth = pd.Series(np.arange(200) % 7, index=days, dtype=float)
df = pd.DataFrame({"date": days, "y": truth.to_numpy(), "rain": truth.to_numpy()})
good = pd.DataFrame({s: truth for s in range(1, 8)})
arch = {"ECMWF": good, "GFS": good + 2}
sc = runner.source_backtest(type("M", (), {"Model": Echo}), {}, df, arch, 7)
assert sc["ECMWF"]["mae"] == 0 and sc["GFS"]["mae"] == 2 and sc["blend"]["mae"] == 1 and sc["observed_rain"]["mae"] == 0, sc
print("ok archive")

# A run cancelled on the website before GitHub picked it up stops at start and is not reopened as running
import argparse  # noqa: E402
from pipeline import __main__ as cli, store  # noqa: E402
_saved = (store.enabled, store.select, store.create_run)
store.enabled = lambda: True
store.select = lambda table, query: [{"stage": "cancelled"}]
store.create_run = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reopen a cancelled run"))
try:
    cli.cmd_start(argparse.Namespace(run_id="r1", kind="daily", models=None, trigger="manual"))
    raise AssertionError("start should stop")
except SystemExit as e:
    assert "cancelled" in str(e)
finally:
    store.enabled, store.select, store.create_run = _saved
print("cancel guard ok")

# The backup schedule runs only when no run of that kind started recently; a failed check still runs
import io, contextlib  # noqa: E402,E401
_saved = (store.enabled, store.select)
store.enabled = lambda: True
try:
    for rows, want in (([{"id": "x"}], "due=false"), ([], "due=true")):
        seen = []
        store.select = lambda table, query, rows=rows: seen.append(query) or rows
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.cmd_due(argparse.Namespace(kind="daily"))
        assert want in out.getvalue(), out.getvalue()
        assert "kind=eq.daily" in seen[0] and "status=in.(running,done)" in seen[0]
    store.select = lambda table, query: (_ for _ in ()).throw(RuntimeError("down"))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli.cmd_due(argparse.Namespace(kind="monthly"))
    assert "due=true" in out.getvalue()
finally:
    store.enabled, store.select = _saved
print("due ok")


# Daily runs show the whole year: months 5-12 come from the newest monthly run
row = lambda i, d: {"step": i, "date": d, "p10": 1.0, "p50": 2.0, "p90": 3.0}
months = ["2026-11-30", "2026-12-31", "2027-01-31", "2027-02-28", "2027-03-31", "2027-04-30",
          "2027-05-31", "2027-06-30", "2027-07-31", "2027-08-31", "2027-09-30", "2027-10-31"]
out = {"tasks": {plan.T1: {"THREE": {"final": [row(i + 1, d) for i, d in enumerate(months[:4])]},
                           "PS": {"final": None}}}}
prev = {"tasks": {plan.T1: {"THREE": {"final": [row(i + 1, d) for i, d in enumerate(months)]}}}}
finish.extend_year(out, prev)
fin = out["tasks"][plan.T1]["THREE"]["final"]
assert [r["date"] for r in fin] == months and [r["step"] for r in fin] == list(range(1, 13))
assert "source" not in fin[3] and fin[4]["source"] == "monthly_run"
assert out["tasks"][plan.T1]["PS"]["final"] is None
finish.extend_year(out, {})  # no monthly run yet: unchanged
assert len(fin) == 12
print("extend year ok")
