"""Fixed-in-January vs rolling forecasts, year by year (Task 1 monthly backtest).

fixed   : the forecast made at the start of January for every month of that year (origin = January, steps 1–12)
roll1   : the forecast made at the start of the month itself (step 1)
roll3   : the forecast made three months before (step 3)
total_est: the year total as known at the start of each month = actual so far + forecast for the rest of the year
"""
import numpy as np
import pandas as pd

MONTHS_TH = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]


def from_json(bt):
    """Backtest JSON {key: ['YYYY-MM-DD|step'], actual, p10, p50, p90} to a frame."""
    if not bt or not bt.get("key"):
        return pd.DataFrame(columns=["origin", "step", "actual", "p10", "p50", "p90"])
    origin, step = zip(*(k.split("|") for k in bt["key"]))
    return pd.DataFrame({"origin": pd.to_datetime(list(origin)), "step": [int(s) for s in step],
                         "actual": bt["actual"], "p10": bt["p10"], "p50": bt["p50"], "p90": bt["p90"]})


def tables(bt):
    """bt: frame with origin (date of the first forecast month), step, actual, p10, p50, p90.
    Returns (monthly, stats): one row per year and month, one row per year."""
    if bt is None or not len(bt):
        return pd.DataFrame(), pd.DataFrame()
    b = bt.copy()
    for c in ("actual", "p10", "p50", "p90"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["origin"] = pd.to_datetime(b["origin"]).dt.to_period("M")
    b["target"] = b["origin"] + (b["step"] - 1)
    b = b.dropna(subset=["actual"])
    act = b.drop_duplicates("target").set_index("target")["actual"]
    p50 = {(o, t): v for o, t, v in zip(b["origin"], b["target"], b["p50"])}
    rows, stats = [], []
    for y in sorted({t.year for t in act.index}):
        months = [t for t in pd.period_range(f"{y}-01", f"{y}-12", freq="M") if t in act.index]
        jan = b[b["origin"] == pd.Period(f"{y}-01", "M")].set_index("target")
        if not len(jan):
            continue
        yr = []
        for m in months:
            f = jan.loc[m] if m in jan.index else None
            rest = [p50.get((m, k)) for k in months if k >= m]
            known = sum(act[k] for k in months if k < m)
            yr.append({"ปี": y, "เดือน": m.month, "ชื่อเดือน": MONTHS_TH[m.month - 1], "ค่าจริง": act[m],
                       "ทายต้น ม.ค. P10": None if f is None else f["p10"],
                       "ทายต้น ม.ค. P50": None if f is None else f["p50"],
                       "ทายต้น ม.ค. P90": None if f is None else f["p90"],
                       "ทายต้นเดือนนั้น (1 เดือน)": p50.get((m, m)),
                       "ทาย 3 เดือนก่อน": p50.get((m - 2, m)),
                       "ยอดรวมทั้งปีที่ประเมิน ณ ต้นเดือน": None if any(v is None or v != v for v in rest)
                       else known + sum(rest)})
        rows += yr
        df = pd.DataFrame(yr)

        def mae(col):
            d = df[["ค่าจริง", col]].dropna()
            return float((d["ค่าจริง"] - d[col]).abs().mean()) if len(d) else None
        ta = float(df["ค่าจริง"].sum())
        tf = float(df["ทายต้น ม.ค. P50"].sum()) if df["ทายต้น ม.ค. P50"].notna().all() else None
        stats.append({"ปี": y, "จำนวนเดือน": len(df), "MAE ทายต้นปี": mae("ทายต้น ม.ค. P50"),
                      "MAE ทายใหม่ 1 เดือน": mae("ทายต้นเดือนนั้น (1 เดือน)"), "MAE ทายใหม่ 3 เดือน": mae("ทาย 3 เดือนก่อน"),
                      "Inflow จริงรวม": ta, "ยอดรวมที่ทายต้น ม.ค.": tf,
                      "คลาดจากจริง (%)": None if tf is None or not ta else (tf - ta) / ta * 100})
    st = pd.DataFrame(stats)
    if len(st):
        st["แบบที่แม่นกว่า"] = np.where(st["MAE ทายใหม่ 1 เดือน"] < st["MAE ทายต้นปี"], "ทายใหม่ทุกเดือน",
                                        np.where(st["MAE ทายใหม่ 1 เดือน"] > st["MAE ทายต้นปี"], "ทายต้นปี", "เท่ากัน"))
    return pd.DataFrame(rows), st


def averages(stats, last_year=2025):
    """One row of averages over the complete years (stats from tables())."""
    s = stats[stats["ปี"] <= last_year]
    if not len(s):
        return {}
    out = {k: float(s[k].mean()) for k in ("MAE ทายต้นปี", "MAE ทายใหม่ 1 เดือน", "MAE ทายใหม่ 3 เดือน")}
    out["Error ยอดรวมทั้งปีที่ทายต้นปี (%)"] = float(s["คลาดจากจริง (%)"].abs().mean())
    out["ปีที่ทายใหม่ทุกเดือนแม่นกว่า"] = int((s["แบบที่แม่นกว่า"] == "ทายใหม่ทุกเดือน").sum())
    out["จำนวนปี"] = len(s)
    return out
