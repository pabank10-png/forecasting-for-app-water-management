"""Foundation models without training: Chronos-2 (default, with rain as covariate), TimesFM or Moirai.
Useful as a benchmark and for new stations with little history."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import foundation


class Model:
    def __init__(self, engine="chronos2", covariates=("rain",)):
        self.engine, self.covariates = engine, covariates

    def fit(self, df):
        return self  # zero-shot

    def predict(self, df, horizon=48, future=None):
        if self.engine == "timesfm":
            return foundation.timesfm(df, horizon)
        fn = foundation.moirai if self.engine == "moirai" else foundation.chronos2
        fc = fn(df, horizon, future, self.covariates, freq="h")
        fc[["p10", "p50", "p90"]] = fc[["p10", "p50", "p90"]].clip(lower=None)
        return fc
