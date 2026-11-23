"""Google GEM-15 bare-earth DTM (15 m) for flood depth and flow-direction work (Task 2).

GEM-15 v1 is published by Google as public 1x1 degree Cloud-Optimised GeoTIFFs in UTM
(EGM96 heights). This module reads only the window it needs straight from Google's bucket
(HTTP range requests, no key, nothing stored on our side) and writes one GeoTIFF on the
native 15 m UTM grid, so no resampling touches the heights.
Licence CC BY 4.0, attribution: "This dataset is produced by Google".
Experimental dataset: cells Google leaves empty (sea, a few voids) stay NaN.

    python -m common.gem15_dtm                       # Rayong + basins -> data/gem15/rayong_dtm_15m.tif
    python -m common.gem15_dtm --bbox 100.9 12.5 101.9 13.4 --surface dsm --out x.tif

Needs rasterio (pip install rasterio); only this step uses it.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

BASE = "https://storage.googleapis.com/climate-and-sustainability/gem-15"
# Rayong province plus the upstream parts of its reservoir/river basins in Chonburi and Chanthaburi.
RAYONG_BBOX = (100.85, 12.50, 102.00, 13.35)  # west, south, east, north
CELL = 15.0


def tile_url(lon: int, lat: int, surface: str = "dtm") -> str:
    return f"{BASE}/p{lat // 10 * 10}_p{lon // 10 * 10}/w{lon}s{lat}/{surface}/zall/mosaic.tif"


def build(bbox=RAYONG_BBOX, out="data/gem15/rayong_dtm_15m.tif", surface="dtm", epsg=32647):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import from_bounds

    west, south, east, north = bbox
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [west, east, west, east], [south, south, north, north])
    x0 = math.floor(min(xs) / CELL) * CELL
    x1 = math.ceil(max(xs) / CELL) * CELL
    y0 = math.floor(min(ys) / CELL) * CELL
    y1 = math.ceil(max(ys) / CELL) * CELL
    width, height = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
    grid = np.full((height, width), np.nan, dtype="float32")

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES"):
        for lat in range(math.floor(south), math.ceil(north)):
            for lon in range(math.floor(west), math.ceil(east)):
                url = tile_url(lon, lat, surface)
                try:
                    src = rasterio.open(f"/vsicurl/{url}")
                except rasterio.errors.RasterioIOError:
                    print(f"{surface} w{lon}s{lat}: no tile (open sea)")
                    continue
                with src:
                    if src.crs.to_epsg() != epsg:
                        raise ValueError(f"w{lon}s{lat} is EPSG:{src.crs.to_epsg()}, expected {epsg}")
                    l, b, r, t = (max(x0, src.bounds.left), max(y0, src.bounds.bottom),
                                  min(x1, src.bounds.right), min(y1, src.bounds.top))
                    if l >= r or b >= t:
                        continue
                    win = from_bounds(l, b, r, t, src.transform).round_offsets().round_lengths()
                    band = src.read(1, window=win).astype("float32")
                    band[~np.isfinite(band) | (band < -1000)] = np.nan
                    wl, wt = src.transform * (win.col_off, win.row_off)
                    c0, r0 = int(round((wl - x0) / CELL)), int(round((y1 - wt) / CELL))
                    part = grid[r0:r0 + band.shape[0], c0:c0 + band.shape[1]]
                    band = band[:part.shape[0], :part.shape[1]]
                    np.copyto(part, band, where=np.isfinite(band) & np.isnan(part))
                    print(f"{surface} w{lon}s{lat}: {band.shape[1]}x{band.shape[0]} cells")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", width=width, height=height, count=1, dtype="float32", crs=f"EPSG:{epsg}",
                   transform=from_origin(x0, y1, CELL, CELL), nodata=np.nan, compress="deflate", predictor=3,
                   tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(grid, 1)
        dst.update_tags(source="Google GEM-15 v1 " + surface.upper() + " (CC BY 4.0, This dataset is produced by Google)",
                        vertical_datum="EGM96", bbox_wgs84=",".join(map(str, bbox)))
    valid = float(np.isfinite(grid).mean())
    print(f"wrote {out} {width}x{height} @ {CELL:.0f} m, {valid:.1%} cells with data")
    return out


def load(path="data/gem15/rayong_dtm_15m.tif"):
    """Heights (NaN = no data), the affine transform and the CRS, ready for FloodMap(dem, river, cell_m=15)."""
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bbox", nargs=4, type=float, default=RAYONG_BBOX, metavar=("W", "S", "E", "N"))
    ap.add_argument("--surface", choices=["dtm", "dsm"], default="dtm")
    ap.add_argument("--out", default="data/gem15/rayong_dtm_15m.tif")
    a = ap.parse_args()
    build(tuple(a.bbox), a.out, a.surface)
