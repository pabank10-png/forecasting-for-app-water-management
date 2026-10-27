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
# Trained only in the monthly run (about 13 min per fit on CPU); daily runs reuse its latest forecast
MONTHLY_ONLY = {f"{T1}/13_chronos2_finetune"}
# Backtest refit interval in months (default 12); fewer refits keep slow models inside the job time limit
REFIT_EVERY = {f"{T1}/13_chronos2_finetune": 36}
RAIN_MODELS = {f"{T1B}/03_gr4j", f"{T1B}/04_hbv", f"{T1B}/05_lightgbm_daily", f"{T1B}/06_lstm_seq2seq",
               f"{T1B}/07_chronos2_exog"}

# Steps every run records, in order (the HQ rooms follow them)
# Month-end storage is not a step: it is worked out on the HQ result page from the final inflow forecast and
# the outflow / transfer the user types in (water-dashboard rule), so the run only stores its starting point.
STAGES = (("data", "data"), ("evaluate", "err"), ("combine", "cmb"), ("publish", "pub"))


def models(kind, only=None):
    names = [m for m in MODELS if not only or m in only or m.split("/")[1] in only]
    return names


def group(name):
    """Install group of a model: base, torch or foundation."""
    _, torch, fm = MODELS[name]
    return "foundation" if fm else "torch" if torch else "base"
