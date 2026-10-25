"""Radar rain nowcast 0-60 min (+ blend with model forecast rain to 6 h).

Steps:
1. dBZ -> rain rate: Z = 10^(dBZ/10), R = (Z/200)^(1/1.6)  (Marshall-Palmer).
2. Motion between the last two frames by phase correlation (one vector for the
   whole image: simple advection).
3. Shift the latest rain field along that motion for each lead time.
4. Basin rain = mean over the basin mask; blend with NWP rain:
   w = max(0, 1 - lead/360 min), R = w R_radar + (1 - w) R_NWP.
With pysteps installed, `nowcast(..., method="steps")` uses its Lucas-Kanade
motion + S-PROG extrapolation instead (better for growing/decaying storms).

    rain = [dbz_to_rain(f) for f in frames]                 # frames every 5-15 min
    fc = nowcast(rain, dt_min=10, leads=range(10, 61, 10))  # list of 2D fields
    basin_series(fc, mask)
"""
import numpy as np


def dbz_to_rain(dbz, min_dbz=10.0):
    dbz = np.asarray(dbz, float)
    r = (10 ** (dbz / 10) / 200.0) ** (1 / 1.6)
    return np.where(np.isnan(dbz) | (dbz < min_dbz), 0.0, r)


def motion(prev, curr):
    """Pixels moved between two frames (dy, dx) by phase correlation."""
    a, b = np.nan_to_num(prev), np.nan_to_num(curr)
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    c = np.abs(np.fft.ifft2(F / np.maximum(np.abs(F), 1e-12)))
    dy, dx = np.unravel_index(np.argmax(c), c.shape)
    ny, nx = a.shape
    dy = dy - ny if dy > ny // 2 else dy
    dx = dx - nx if dx > nx // 2 else dx
    return -float(dy), -float(dx)


def nowcast(rain_frames, dt_min=10, leads=(10, 20, 30, 40, 50, 60), method="advection"):
    if method == "steps":
        from pysteps import motion as pm, nowcasts
        R = np.array(rain_frames[-3:])
        V = pm.get_method("lucaskanade")(R)
        n = int(max(leads) // dt_min)
        out = nowcasts.get_method("sprog")(R, V, n, n_cascade_levels=6, R_thr=0.1)
        return [np.nan_to_num(out[int(l // dt_min) - 1]) for l in leads]
    from scipy import ndimage
    vy, vx = motion(rain_frames[-2], rain_frames[-1])
    last = np.nan_to_num(rain_frames[-1])
    return [ndimage.shift(last, (vy * l / dt_min, vx * l / dt_min), order=1, mode="constant", cval=0.0)
            for l in leads]


def basin_series(fields, mask):
    m = np.asarray(mask, bool)
    return np.array([float(np.asarray(f)[m].mean()) for f in fields])


def blend(radar_mm_h, nwp_mm_h, lead_min, full_nwp_at=360):
    w = np.clip(1 - np.asarray(lead_min, float) / full_nwp_at, 0, 1)
    return w * np.asarray(radar_mm_h, float) + (1 - w) * np.asarray(nwp_mm_h, float)


def csi(obs, fc, threshold=1.0):
    """Critical success index of rain/no-rain at a threshold (mm/h)."""
    o, f = np.asarray(obs) >= threshold, np.asarray(fc) >= threshold
    hits, miss, false = (o & f).sum(), (o & ~f).sum(), (~o & f).sum()
    return float(hits / max(hits + miss + false, 1))
