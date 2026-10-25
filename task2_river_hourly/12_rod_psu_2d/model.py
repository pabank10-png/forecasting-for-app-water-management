"""Flood map step: turn a forecast water level into flooded area (not a time-series model).

Two ways, same output (depth grid + flooded area per zone such as tambon):
1. Bathtub on a DEM: depth = level - ground, kept only where it is positive
   AND connected to the river cells (scipy.ndimage.label).
2. Scenario lookup of precomputed 2D maps (PSU PaYaNaFlood / RODDSS exports,
   one depth grid per station level): pick the nearest level at or above the
   forecast. Using PSU maps needs permission from PSU first.

    fm = FloodMap(dem, river_mask, cell_m=30)
    fm.area_m2(5.20, zones=tambon_grid)        # {zone_id: m2}
    fm.depth(forecast["p90"].max())            # worst case of a forecast
"""
import numpy as np


class FloodMap:
    def __init__(self, dem, river_mask, cell_m=30.0):
        self.dem = np.asarray(dem, float)
        self.river = np.asarray(river_mask, bool)
        self.cell_area = float(cell_m) ** 2

    def depth(self, level):
        from scipy import ndimage
        d = level - self.dem
        wet = (d > 0) | self.river
        lab, _ = ndimage.label(wet)
        keep = np.unique(lab[self.river & (lab > 0)])
        conn = np.isin(lab, keep) & (d > 0)
        return np.where(conn, d, 0.0)

    def area_m2(self, level, zones=None, min_depth=0.0):
        return area_by_zone(self.depth(level), self.cell_area, zones, min_depth)


def area_by_zone(depth, cell_area, zones=None, min_depth=0.0):
    wet = np.asarray(depth) > min_depth
    if zones is None:
        return float(wet.sum() * cell_area)
    z = np.asarray(zones)
    ids, counts = np.unique(z[wet], return_counts=True)
    return {k: float(c * cell_area) for k, c in zip(ids.tolist(), counts)}


class ScenarioLookup:
    """maps: {station level (m): depth grid} from the 2D model runs."""

    def __init__(self, maps, cell_m=30.0):
        self.levels = np.array(sorted(maps))
        self.maps, self.cell_area = maps, float(cell_m) ** 2

    def depth(self, level):
        i = min(np.searchsorted(self.levels, level), len(self.levels) - 1)
        if level < self.levels[0]:
            return np.zeros_like(np.asarray(self.maps[self.levels[0]], float))
        return np.asarray(self.maps[self.levels[i]], float)

    def area_m2(self, level, zones=None, min_depth=0.0):
        return area_by_zone(self.depth(level), self.cell_area, zones, min_depth)
