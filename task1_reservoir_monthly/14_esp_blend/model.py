"""ESP + regression blend: physics-based seasonal forecast averaged with the inflow/rain regression.

ESP (Ensemble Streamflow Prediction), per reservoir:
  1. Daily HBV and GR4J (common/esp.py) calibrated on the last `calib_years` years before the forecast, driven by
     the complete daily catchment rain (common/daily_rain.py: month totals spread with the ERA5 daily pattern).
  2. Run with observed rain up to the forecast origin, so the soil and groundwater stores hold today's wetness.
     HBV's groundwater and routing stores are then scaled so its last 30 days match the observed inflow
     (state updating; GR4J is left as it is, scaling broke its first month in the test).
  3. From the origin, each earlier year's rain on the same calendar days is one member (about 20-30 members).
     Member months are multiplied by the calibration-period observed/simulated ratio of that calendar month.
     HBV and GR4J are averaged member by member; the ESP forecast is the members' median.
THREE adds DK + KY + NPL member by member (DK and NPL have about 10 more years of daily record than THREE).
Regression ('climrain', as in the SEAS5 test): sqrt(inflow) anomaly of the target month on the anomaly at the
origin and standardised rain of the origin month (rain after the origin is unknown, so its anomaly is 0).
Final P50 = (ESP + regression) / 2. P10/P90 = P50 times the 5th/95th percentile of log errors of the same
blend over the last `err_years` years before the forecast, per step ahead. These errors are in-sample for the
calibration, so the 10th/90th percentiles gave too narrow a band (68-75% of months inside in the 2016-2025
backtest); the 5th/95th give about 80%.

Backtest 2002-2020 (experiments/esp_task1, 2026-11-08), MAE better than Baseline, steps 1-6:
DK 5%, KY 12%, NPL 1%, PS 1%, THREE 13%. Live, the partial current month uses the observed daily inflow so far
and the members fill the rest of the month.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import esp  # noqa: E402
from common.base import result  # noqa: E402
from common.daily_rain import daily_rain  # noqa: E402
from common.data import daily_frame, monthly_frame  # noqa: E402

PARTS = {"THREE": ("DK", "KY", "NPL")}
WET = {5, 6, 7, 8, 9, 10, 11}
_FITS = {}  # (reservoir, last observed month) -> fitted part, shared by THREE and its parts within one process
_DAILY = {}


def _daily(res):
    """Daily inflow (MCM/day) and rain for one reservoir, from the first observed inflow."""
    if res not in _DAILY:
        d = daily_frame(res).drop_duplicates("date", keep="last").set_index("date")
        m = monthly_frame(res)
        last = d["y"].last_valid_index()
        rain = daily_rain(res, m.set_index("date")["rain_lag0_mm"], last_day=last)
        days = pd.date_range(max(d["y"].first_valid_index(), rain.index[0]), last, freq="D")
        _DAILY[res] = pd.DataFrame({"y": d["y"].reindex(days), "rain": rain.reindex(days).fillna(0.0)})
    return _DAILY[res]


class Part:
    """HBV + GR4J of one reservoir, calibrated on daily data up to `end` (a month end)."""

    def __init__(self, res, end, calib_years):
        d = _daily(res)
        self.d = d
        self.P = d["rain"].to_numpy(float)
        self.D = d.index.dayofyear.to_numpy().astype(float)
        self.pos = {t: i for i, t in enumerate(d.index)}
        lo = max(d.index[0], end - pd.DateOffset(years=calib_years) + pd.Timedelta(days=1))
        tr = d.loc[lo:end]
        self.i0 = self.pos[tr.index[0]]
        mi = ((tr.index.year - tr.index[0].year) * 12 + tr.index.month - 1).to_numpy()
        self.p, self.ratio = {}, {}
        for name, (f, _, s0) in esp.MODELS.items():
            p = esp.calibrate(name, tr["rain"].to_numpy(float), tr.index.dayofyear.to_numpy().astype(float),
                              tr["y"].to_numpy(float), mi)
            q, _ = f(tr["rain"].to_numpy(float), tr.index.dayofyear.to_numpy().astype(float), p, s0(p))
            sim = pd.Series(q, tr.index).iloc[365:]
            obs = tr["y"].iloc[365:]
            ok = obs.notna()
            sm = sim[ok].groupby(sim[ok].index.month).sum()
            om = obs[ok].groupby(obs[ok].index.month).sum()
            self.p[name] = p
            self.ratio[name] = (om / sm.clip(lower=1e-6)).clip(0.3, 3.0).reindex(range(1, 13)).fillna(1.0).to_numpy()

    def states(self, upto):
        """States of both models at the end of day `upto` (HBV with state updating), run from the calibration start."""
        io = self.pos[upto]
        out = {}
        for name, (f, _, s0) in esp.MODELS.items():
            p = self.p[name]
            q, st = f(self.P[self.i0:io + 1], self.D[self.i0:io + 1], p, s0(p))
            if name == "hbv":
                obs = self.d["y"].iloc[io - 29:io + 1].to_numpy(float)
                sim = q[-30:]
                ok = ~np.isnan(obs)
                if ok.sum() >= 20 and sim[ok].sum() > 0:
                    st = esp.scale_state(name, st, float(np.clip(obs[ok].sum() / sim[ok].sum(), 0.25, 4.0)))
            out[name] = st
        return out

    def members(self, upto, months, years, rain=None):
        """{member year: monthly totals (len(months))} from the day after `upto` to the end of the last month.
        months: list of month-end Timestamps (the first may be the current, partial month).
        rain: optional Series (date -> mm) of forecast rain that replaces each member's rain on those days."""
        st = self.states(upto)
        end = months[-1]
        n = (end - upto).days
        out = {}
        for yy in years:
            shift = pd.DateOffset(years=upto.year - yy)
            a = upto - shift
            if a not in self.pos:
                continue
            ia = self.pos[a] + 1
            if ia + n > len(self.P):
                continue
            P = self.P[ia:ia + n]
            days = pd.date_range(upto + pd.Timedelta(days=1), periods=n, freq="D")
            if rain is not None:
                f = rain.reindex(days).to_numpy(float)
                P = np.where(np.isnan(f), P, f)
            D = days.dayofyear.to_numpy().astype(float)
            mon = days.to_period("M")
            tot = 0.0
            for name, (f, _, _) in esp.MODELS.items():
                q, _ = f(P, D, self.p[name], st[name])
                m = pd.Series(q, index=days).groupby(mon).sum()
                tot = tot + m.to_numpy() * self.ratio[name][m.index.month - 1]
            out[yy] = tot / len(esp.MODELS)
        return out


class Model:
    def __init__(self, horizon=12, calib_years=13, err_years=6, err_q=(0.05, 0.95), max_member_years=30):
        self.horizon, self.calib_years, self.err_years, self.err_q = horizon, calib_years, err_years, err_q
        self.max_member_years = max_member_years

    # ---------- fit ----------
    def fit(self, df):
        self.res = str(df["reservoir"].iloc[0])
        obs = df[df["y"].notna()]
        self.last = pd.Timestamp(obs["date"].iloc[-1])  # last observed month end
        self.parts = {}
        for p in PARTS.get(self.res, (self.res,)):
            key = (p, self.last, self.calib_years)
            if key not in _FITS:
                _FITS[key] = Part(p, self.last, self.calib_years)
            self.parts[p] = _FITS[key]
        self._fit_regression(df)
        self._fit_errors(df)
        return self

    def _monthly(self, df):
        q = pd.Series(np.sqrt(np.maximum(df["y"].to_numpy(float), 0)), index=pd.to_datetime(df["date"]))
        r = pd.Series(df["rain_lag0_mm"].to_numpy(float), index=q.index)
        return q, r

    def _fit_regression(self, df):
        tr = df[pd.to_datetime(df["date"]) <= self.last]
        q, r = self._monthly(tr)
        q = q.dropna()
        self.qc = q.groupby(q.index.month).mean()
        rr = r.dropna()
        self.rm, self.rs = rr.groupby(rr.index.month).mean(), rr.groupby(rr.index.month).std()
        self.coef = {}
        idx = {t: i for i, t in enumerate(q.index)}
        for h in range(1, self.horizon + 1):
            X, y = [], []
            for T in q.index:
                o = T - pd.offsets.MonthEnd(h)
                if o not in idx:
                    continue
                f = self._feat(q[o], r.get(o, np.nan), o, T)
                if np.isnan(f).any():
                    continue
                X.append(f)
                y.append(q[T] - self.qc[T.month])
            X = np.array(X)
            self.coef[h] = np.linalg.lstsq(np.c_[X, np.ones(len(X))], np.array(y), rcond=None)[0] if len(X) > 20 else None

    def _feat(self, q0, r0, o, T):
        z = (r0 - self.rm.get(o.month, np.nan)) / self.rs.get(o.month, np.nan)
        w = o.month in WET
        return np.array([q0 - self.qc.get(o.month, np.nan), z * w, z * (1 - w)], float)

    def _regression(self, q0, r0, o, h):
        T = o + pd.offsets.MonthEnd(h)
        c = self.coef.get(h)
        if c is None or np.isnan(q0):
            return self.qc[T.month] ** 2
        f = self._feat(q0, r0, o, T)
        f = np.where(np.isnan(f), 0.0, f)
        return max(self.qc[T.month] + np.r_[f, 1] @ c, 0.0) ** 2

    def _esp(self, upto, months, rain=None):
        """Median over members of the summed parts (the first month counts only the days after `upto`).
        rain: optional {part: Series(date -> mm)} forecast rain used for those days in every member."""
        tot = None
        years = None
        for name, p in self.parts.items():
            m = p.members(upto, months, self._years(p, upto), (rain or {}).get(name))
            years = set(m) if years is None else years & set(m)
            tot = m if tot is None else {y: tot[y] + m[y] for y in years}
        if not years or len(years) < 5:
            return np.full(len(months), np.nan)
        arr = np.array([tot[y] for y in sorted(years)])
        return np.median(arr, axis=0)

    def _years(self, part, upto):
        first = part.d.index[0].year + 1
        ys = [y for y in range(first, upto.year) if y != upto.year]
        return ys[-self.max_member_years:]

    def _fit_errors(self, df):
        """Log errors of the blend for origins in the last err_years years (month ends), per step."""
        q, r = self._monthly(df[pd.to_datetime(df["date"]) <= self.last])
        start = self.last - pd.DateOffset(years=self.err_years)
        origins = [o for o in q.index if start < o <= self.last - pd.offsets.MonthEnd(1)]
        errs = {h: [] for h in range(1, self.horizon + 1)}
        for o in origins:
            n = min(self.horizon, (self.last.to_period("M") - o.to_period("M")).n)
            months = [o + pd.offsets.MonthEnd(h) for h in range(1, n + 1)]
            e = self._esp(o, months)
            for h in range(1, n + 1):
                act = q.get(months[h - 1], np.nan) ** 2
                if np.isnan(act) or np.isnan(e[h - 1]):
                    continue
                p50 = 0.5 * (e[h - 1] + self._regression(q.get(o, np.nan), r.get(o, np.nan), o, h))
                errs[h].append(np.log((act + 0.1) / (p50 + 0.1)))
        self.err = {}
        for h in range(1, self.horizon + 1):
            e = np.array(errs[h] if len(errs[h]) >= 10 else sum(errs.values(), []))
            self.err[h] = np.quantile(e, self.err_q) if len(e) >= 10 else np.array([-0.9, 0.9])

    # ---------- predict ----------
    def predict(self, df, horizon=12, future=None):
        """The last row of df is the month to forecast first (y empty); step 1 = that month.
        future: optional {part: Series(date -> mm)} of forecast daily rain (Storm Trigger). With it, months 1-2 are
        the ESP alone (members' rain replaced by the forecast on those days), not averaged with the regression,
        because the regression cannot see the storm."""
        row = df.iloc[-1]
        first = pd.Timestamp(row["date"])
        o = first - pd.offsets.MonthEnd(1)  # origin = end of the month before
        months = [first + pd.offsets.MonthEnd(h) for h in range(horizon)]
        hist = df.iloc[:-1]
        q, r = self._monthly(hist)
        upto, partial = o, 0.0
        if bool(row.get("is_partial_month", False)):
            # live: run to the last day with inflow data in the current month, keep what has flowed so far
            last_day = min(p.d.index[-1] for p in self.parts.values())
            if last_day > o:
                upto = last_day
                partial = sum(float(p.d["y"].loc[o + pd.Timedelta(days=1):upto].fillna(0).sum()) for p in self.parts.values())
        if upto >= months[0]:
            upto, partial = o, 0.0
        e = self._esp(upto, months, future)
        q0, r0 = q.get(o, np.nan), r.get(o, np.nan)
        p50, lo, hi = [], [], []
        for h in range(1, horizon + 1):
            k = self.err.get(h, self.err[max(self.err)])
            if h == 1 and upto > o and not np.isnan(e[0]):
                # current month: inflow so far + ESP for the days left (the regression only knows whole months)
                rest = e[0]
                p50.append(partial + rest)
                lo.append(partial + (rest + 0.1) * np.exp(k[0]) - 0.1)
                hi.append(partial + (rest + 0.1) * np.exp(k[1]) - 0.1)
                continue
            reg = self._regression(q0, r0, o, h)
            v = reg if np.isnan(e[h - 1]) else (e[h - 1] if future and h <= 2 else 0.5 * (e[h - 1] + reg))  # too few member years: regression only
            p50.append(v)
            lo.append((v + 0.1) * np.exp(k[0]) - 0.1)
            hi.append((v + 0.1) * np.exp(k[1]) - 0.1)
        return result(lo, p50, hi)
