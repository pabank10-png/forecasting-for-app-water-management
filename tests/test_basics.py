"""Quick checks that need no data files."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel / "model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


wb = load("task1_reservoir_monthly/95_water_balance")
assert wb.run(100, [50, 0], [20, 200], capacity=120) == [120, 0]
assert wb.run(10, [5], [0], diversions=[2], multiplier=2) == [22]

clim = load("task1_reservoir_monthly/01_climatology")
m = clim.fit([(1, v) for v in range(1, 11)])
p = clim.predict(m, 1)
assert p["p10"] < p["p50"] < p["p90"] and clim.predict(m, 2) is None

sn = load("task1_reservoir_monthly/02_seasonal_naive")
assert sn.predict(list(range(24))) == 12

sys.path.insert(0, str(ROOT))
from common.metrics import coverage, mae
assert mae([1, 2, None], [2, 2, 5]) == 0.5
assert coverage([1, 5], [0, 0], [2, 2]) == 0.5
print("ok")
