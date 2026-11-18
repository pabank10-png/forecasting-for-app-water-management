"""Persist full monthly-origin horizons, including targets with no Actual.

Isolated report replay: never writes the live forecast database or changes champions.
Uses today's prepared history, not a vintage snapshot of what was known in 2026.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def number(value):
    return None if pd.isna(value) else float(value)


def replay(df, make_model, first='2026-01', last='2026-11'):
    """Twelve saved steps per origin; inputs end strictly before that origin."""
    df=df.sort_values('date').reset_index(drop=True)
    actual={pd.Timestamp(d).strftime('%Y-%m'):number(y) for d,y in zip(df['date'],df['y'])}
    rows=[]
    for origin in pd.period_range(first,last,freq='M'):
        start=origin.start_time
        history=df.loc[df['date']<start].copy().reset_index(drop=True)
        if history.empty or history['y'].notna().sum()<12:
            raise ValueError(f'Insufficient pre-origin history: {origin}')
        if history['date'].iloc[-1].to_period('M')!=origin-1 or pd.isna(history['y'].iloc[-1]):
            raise ValueError(f'Incomplete previous month before {origin}; refresh the dataset')
        # Target-month covariates and inflow must never enter the model, even
        # when the current dataset has those values. Lag columns are unused
        # by these two foundation-model wrappers.
        blank={c:np.nan for c in df.columns}
        blank['date']=origin.to_timestamp('M')
        context=pd.concat([history,pd.DataFrame([blank])],ignore_index=True)
        fc=make_model().fit(history.copy()).predict(context,12)
        if len(fc)!=12:
            raise ValueError(f'Incomplete forecast: {origin}: {len(fc)}')
        for step,(_,r) in enumerate(fc.iterrows(),1):
            target=str(origin+step-1)
            q={key:number(r[key]) for key in ['p10','p50','p90']}
            if any(v is None or not math.isfinite(v) for v in q.values()):
                raise ValueError(f'Non-finite forecast: {origin} step {step}')
            rows.append(dict(origin=str(start.date()),target=target,step=step,
                             training_through=str(history['date'].iloc[-1].date()),
                             actual=actual.get(target),**q))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['timesfm','chronos2'],required=True)
    parser.add_argument('--first',default='2026-01')
    parser.add_argument('--last',default='2026-11')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    from common.data import data_dir, monthly_frame
    from common.registry import load
    from pipeline import plan
    name='task1_reservoir_monthly/'+('11_timesfm_inflow' if args.model=='timesfm' else '10_chronos2')
    sites=['DK','KY','NPL'] if args.model=='timesfm' else ['PS']
    module=load(plan.folder(name))
    baseline=load('task1_reservoir_monthly/01_climatology')
    packages={}
    for package in ['torch','timesfm','chronos-forecasting','numpy','pandas']:
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    result=dict(schema_version=1,kind='regenerated monthly-origin forecasts',
                note='Current prepared history; not vintage historical website output. Historical daily Nowcast not replayed.',
                code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                data_sha=os.environ.get('REPLAY_DATA_SHA'),run_id=os.environ.get('GITHUB_RUN_ID'),
                input_sha256=hashlib.sha256((data_dir()/'lag.csv').read_bytes()).hexdigest(),
                first=args.first,last=args.last,model=name,packages=packages,sites={})
    for site in sites:
        began=time.monotonic()
        df=monthly_frame(site)
        champion=replay(df,lambda:module.Model(**plan.kwargs(name)),args.first,args.last)
        climate=replay(df,baseline.Model,args.first,args.last)
        result['sites'][site]=dict(champion=champion,climatology=climate,
                                   seconds=round(time.monotonic()-began,2))
        print(f'{site}: saved {len(champion)} Champion and {len(climate)} Climatology targets',flush=True)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
