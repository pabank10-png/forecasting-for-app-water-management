"""Which models a run uses, and which Forecast HQ character stands for each one.

daily   : Task 1 monthly models forecast the current month (month-end Nowcast) and the next 3 months,
          Task 1B daily models forecast inflow 1-7 days with forecast rain. Uses the scores and
          champion from the latest monthly run.
monthly : Task 1 forecasts 12 months, and every model also runs its rolling backtest
          (2016-2025 monthly, 2016-2020 daily) so the gate, weights and champion are chosen again.
"""
T1, T1B = "task1_reservoir_monthly", "task1b_reservoir_daily"

# folder -> (HQ character id, uses torch, uses Foundation Model packages)
MODELS = {
    f"{T1}/01_climatology": ("base", False, False),
    f"{T1}/02_seasonal_naive": ("base", False, False),
    f"{T1}/03_sarimax": ("stat", False, False),
    f"{T1}/04_ets": ("stat", False, False),
    f"{T1}/05_lightgbm_quantile": ("ml", False, False),
    f"{T1}/06_xgboost_quantile": ("ml", False, False),
    f"{T1}/07_nbeats": ("dec", True, False),
    f"{T1}/08_nhits": ("dec", True, False),
    f"{T1}/09_lstm": ("rnn", True, False),
    f"{T1}/10_chronos2": ("fm", True, True),
    f"{T1}/11_timesfm": ("fm", True, True),
    f"{T1}/12_moirai": ("fm", True, True),
    f"{T1}/13_chronos2_finetune": ("fm", True, True),
    f"{T1}/03_sarimax_inflow": ("stat", False, False),
    f"{T1}/07_nbeats_inflow": ("dec", True, False),
    f"{T1}/08_nhits_inflow": ("dec", True, False),
    f"{T1}/09_lstm_inflow": ("rnn", True, False),
    f"{T1}/10_chronos2_inflow": ("fm", True, True),
    f"{T1}/11_timesfm_inflow": ("fm", True, True),
    f"{T1}/12_moirai_inflow": ("fm", True, True),
    f"{T1B}/01_persistence": ("base", False, False),
    f"{T1B}/02_day_climatology": ("base", False, False),
    f"{T1B}/03_gr4j": ("hydro", False, False),
    f"{T1B}/04_hbv": ("hydro", False, False),
    f"{T1B}/05_lightgbm_daily": ("ml", False, False),
    f"{T1B}/06_lstm_seq2seq": ("rnn", True, False),
    f"{T1B}/07_chronos2_exog": ("fm", True, True),
}
# Left out on purpose: 90_conformal / 91_ensemble (the finish step does the gate, weights and champion itself),
# 95_water_balance (not a model: month-end storage is computed on the HQ page), Task 2 (no hourly river history yet).

BASELINE = {T1: f"{T1}/01_climatology", T1B: f"{T1B}/02_day_climatology"}
SITES = {T1: ("THREE", "PS"), T1B: ("DK", "KY", "NPL", "PS")}
HORIZON = {("daily", T1): 4, ("monthly", T1): 12, ("daily", T1B): 7, ("monthly", T1B): 7}
# Same code as another folder with other settings: name -> (folder, Model kwargs). Every model that reads rain,
# soil moisture, ENSO and IOD also runs on inflow alone; the backtest MAE and the gate decide each month which
# one counts (user, 2026-10-27: covariates made THREE worse in Oct 2026, but may help in other years).
VARIANTS = {
    f"{T1}/03_sarimax_inflow": (f"{T1}/03_sarimax", {"exog": ()}),
    f"{T1}/07_nbeats_inflow": (f"{T1}/07_nbeats", {"past_cols": ()}),
    f"{T1}/08_nhits_inflow": (f"{T1}/08_nhits", {"past_cols": ()}),
    f"{T1}/09_lstm_inflow": (f"{T1}/09_lstm", {"past_cols": ()}),
    f"{T1}/10_chronos2_inflow": (f"{T1}/10_chronos2", {"covariates": ()}),
    f"{T1}/11_timesfm_inflow": (f"{T1}/11_timesfm", {"covariates": ()}),
    f"{T1}/12_moirai_inflow": (f"{T1}/12_moirai", {"past_covariates": ()}),
}


def base(name):
    """The model a '+sum' entry adds up."""
    return name[:-len(SUM)] if name.endswith(SUM) else name


def folder(name):
    name = base(name)
    return VARIANTS.get(name, (name, {}))[0]


def kwargs(name):
    name = base(name)
    return VARIANTS.get(name, (name, {}))[1]


# THREE forecast a second way: each reservoir on its own, then added (DK and NPL have about 10 more years of
# history than THREE; this won for every model in the 2026-10-27 test). The parts run inside each model's job;
# the finish step adds them up as '<model>+sum' and lets the backtest pick between it and the direct forecast.
PARTS = {"THREE": ("DK", "KY", "NPL")}
SUM = "+sum"


# Trained only in the monthly run (about 13 min per fit on CPU); daily runs reuse its latest forecast
MONTHLY_ONLY = {f"{T1}/13_chronos2_finetune"}
# Backtest refit interval in months (default 12); fewer refits keep slow models inside the job time limit
REFIT_EVERY = {f"{T1}/13_chronos2_finetune": 36}
MODELS.update({m + SUM: v for m, v in list(MODELS.items()) if m.startswith(T1) and m not in MONTHLY_ONLY})
RAIN_MODELS = {f"{T1B}/03_gr4j", f"{T1B}/04_hbv", f"{T1B}/05_lightgbm_daily", f"{T1B}/06_lstm_seq2seq",
               f"{T1B}/07_chronos2_exog"}

# Steps every run records, in order (the HQ rooms follow them)
# Month-end storage is not a step: it is worked out on the HQ result page from the final inflow forecast and
# the outflow / transfer the user types in (water-dashboard rule), so the run only stores its starting point.
STAGES = (("data", "data"), ("evaluate", "err"), ("combine", "cmb"), ("publish", "pub"))


def models(kind, only=None):
    """Models that get a job of their own ('+sum' entries are built in the finish step)."""
    names = [m for m in MODELS if not m.endswith(SUM) and (not only or m in only or m.split("/")[1] in only)]
    return names


def sites(name):
    """Sites a model's job runs: Task 1 adds the THREE parts, except for the monthly-only fine-tune (too slow)."""
    task = name.split("/")[0]
    if task != T1 or name in MONTHLY_ONLY:
        return SITES[task]
    return SITES[task] + tuple(p for parts in PARTS.values() for p in parts)


def summary_sites(task):
    return SITES[task] + (tuple(p for parts in PARTS.values() for p in parts) if task == T1 else ())


def group(name):
    """Install group of a model: base, torch or foundation."""
    _, torch, fm = MODELS[name]
    return "foundation" if fm else "torch" if torch else "base"
