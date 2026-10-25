"""Quick checks that need no data files: formulas match the README examples."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.metrics import coverage, mae
from common.registry import load

wb = load("task1_reservoir_monthly/95_water_balance")
assert wb.run(100, [50, 0], [20, 200], capacity=120) == [120, 0]
assert wb.run(10, [5], [0], diversions=[2], multiplier=2) == [22]

wbd = load("task1b_reservoir_daily/95_water_balance_daily")
assert abs(wbd.run(179.27, [3.34], [2.14])[0]["storage"] - 180.47) < 1e-9

assert mae([1, 2, None], [2, 2, 5]) == 0.5
assert coverage([1, 5], [0, 0], [2, 2]) == 0.5

radar = load("task2_river_hourly/14_radar_nowcast")
assert np.allclose(radar.dbz_to_rain(np.array([40.0, 35.0])), [11.53, 5.62], atol=0.01)
assert radar.blend(8, 3, 144) == 6.0

fm = load("task2_river_hourly/12_rod_psu_2d")
f = fm.FloodMap(np.array([[4.8, 5.1, 5.5, 4.0]]), np.array([[True, False, False, False]]), 30)
assert f.area_m2(5.2) == 1800.0  # the cell at 4.0 m is lower but not connected to the river

hms = load("task2_river_hourly/11_hec_hms_ras")
assert abs(hms.scs_excess(np.array([100.0]), 80)[0] - 50.54) < 0.01
print("ok")
