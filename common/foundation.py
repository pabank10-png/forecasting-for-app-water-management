"""Pretrained time-series Foundation Models used zero-shot (no training on our data).

Weights download from Hugging Face on first use, so this needs internet and
the matching package:
    Chronos-2  pip install "chronos-forecasting>=2.0"   (amazon/chronos-2)
    TimesFM    pip install timesfm                      (google/timesfm-2.5-200m-pytorch)
    Moirai     pip install uni2ts                       (Salesforce/moirai-2.0-R-small)
The calls follow each package's README; they have not been run in this repo
yet because the weights could not be downloaded here.
"""
import numpy as np
import pandas as pd

from .base import QUANTILES, history, result

_CACHE = {}


def _timestamps(df, n, freq):
    if "date" in df.columns and df["date"].notna().sum() >= n:
        return pd.to_datetime(df["date"].iloc[:n]).reset_index(drop=True)
    return pd.Series(pd.date_range("2000-01-01", periods=n, freq=freq))


def chronos2(df, horizon, future=None, covariates=(), freq="MS", model_id="amazon/chronos-2", pipeline=None):
    """Chronos-2 forecast. covariates: columns given as past values and, when `future` has them, future values."""
    from chronos import BaseChronosPipeline
    if pipeline is None:
        pipeline = _CACHE.get(model_id) or BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")
        _CACHE[model_id] = pipeline
    y = history(df)
    n = len(y)
    ts = _timestamps(df, n, freq)
    if hasattr(pipeline, "predict_df"):
        ctx = pd.DataFrame({"id": "s", "timestamp": ts, "target": y})
        for c in covariates:
            ctx[c] = df[c].to_numpy(float)[:n]
        fut = None
        if future is not None and any(c in future for c in covariates):
            step = ts.iloc[-1] - ts.iloc[-2]
            fut = pd.DataFrame({"id": "s", "timestamp": [ts.iloc[-1] + step * (i + 1) for i in range(horizon)]})
            for c in covariates:
                if c in future:
                    fut[c] = future[c].to_numpy(float)[:horizon]
        out = pipeline.predict_df(ctx, future_df=fut, prediction_length=horizon, quantile_levels=list(QUANTILES),
                                  id_column="id", timestamp_column="timestamp", target="target")
        q = out[[str(t) for t in QUANTILES]].to_numpy(float)
    else:  # Chronos / Chronos-Bolt (univariate)
        import torch
        qs, _ = pipeline.predict_quantiles(context=torch.tensor(y, dtype=torch.float32),
                                           prediction_length=horizon, quantile_levels=list(QUANTILES))
        q = qs[0].numpy()
    return result(q[:, 0], q[:, 1], q[:, 2])


def timesfm(df, horizon, model_id="google/timesfm-2.5-200m-pytorch", max_context=1024):
    """TimesFM 2.5 forecast (univariate). Its quantile head returns mean + deciles 0.1..0.9."""
    import timesfm as tf
    key = (model_id, max_context, horizon)
    model = _CACHE.get(key)
    if model is None:
        model = tf.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        model.compile(tf.ForecastConfig(max_context=max_context, max_horizon=max(horizon, 16),
                                        normalize_inputs=True, use_continuous_quantile_head=True,
                                        fix_quantile_crossing=True))
        _CACHE[key] = model
    y = history(df)[-max_context:]
    _, quant = model.forecast(horizon=horizon, inputs=[y])
    q = np.asarray(quant)[0]  # (horizon, 10): mean, q10, q20, ..., q90
    return result(q[:, 1], q[:, 5], q[:, 9])


def moirai(df, horizon, future=None, covariates=(), freq="MS", model_id="Salesforce/moirai-2.0-R-small",
           context=200, samples=200):
    """Moirai forecast via uni2ts + GluonTS; covariates need values for the forecast steps too."""
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    y = history(df)
    n = len(y)
    ts = pd.DatetimeIndex(_timestamps(df, n, freq))
    cov = [c for c in covariates if future is not None and c in future]
    frame = pd.DataFrame({"target": y}, index=ts)
    for c in cov:
        frame[c] = df[c].to_numpy(float)[:n]
    if cov:  # GluonTS wants dynamic features to cover the forecast steps as well
        idx = pd.date_range(ts[-1], periods=horizon + 1, freq=pd.infer_freq(ts) or freq)[1:]
        tail = pd.DataFrame({c: future[c].to_numpy(float)[:horizon] for c in cov}, index=idx)
        tail["target"] = np.nan
        frame = pd.concat([frame, tail])
    module = _CACHE.get(model_id) or Moirai2Module.from_pretrained(model_id)
    _CACHE[model_id] = module
    model = Moirai2Forecast(module=module, prediction_length=horizon,
                            context_length=min(context, n), target_dim=1,
                            feat_dynamic_real_dim=len(cov), past_feat_dynamic_real_dim=0)
    # Moirai2Forecast.predict skips the GluonTS predictor, whose output format changed in gluonts 0.17
    fdr = [frame[cov].to_numpy(float)] if cov else None
    out = model.predict(past_target=[np.asarray(y, float)], feat_dynamic_real=fdr)
    levels = list(module.quantile_levels)
    q = np.column_stack([np.asarray(out)[0, levels.index(t), :horizon].ravel() for t in QUANTILES])
    return result(q[:, 0], q[:, 1], q[:, 2])
