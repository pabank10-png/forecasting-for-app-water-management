"""Controlled Chronos-2 sensitivity checks, Prasae October 2026, fixed replay inputs.

This measures response to input changes, not additive feature importance or a new model selection.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

COVS=('rain_mm','sm_root','oni','dmi')


def build_cases(frame):
    history=frame.loc[frame.date<pd.Timestamp('2026-10-01')].copy().reset_index(drop=True)
    assert history.date.iloc[-1]==pd.Timestamp('2026-09-30') and pd.notna(history.y.iloc[-1])
    blank={c:np.nan for c in frame};blank['date']=pd.Timestamp('2026-10-31')
    context=pd.concat([history,pd.DataFrame([blank])],ignore_index=True)
    cases=[]
    def add(name,covs,df=None):
        cases.append({'name':name,'covariates':tuple(covs),'context':(context if df is None else df).copy()})
    add('full',COVS)
    add('inflow_only',())
    for c in COVS:add('without_'+c,[x for x in COVS if x!=c])
    add('rain_soil_only',('rain_mm','sm_root'))
    add('climate_only',('oni','dmi'))
    old=history.loc[(history.date.dt.year<2026)&(history.date.dt.month==9)]
    norm_y=float(old.y.median())
    altered=context.copy();altered.loc[altered.date==pd.Timestamp('2026-09-30'),'y']=norm_y
    add('normal_september_inflow',COVS,altered)
    add('normal_september_inflow_only',(),altered)
    for c in COVS:
        norm=float(old[c].median())
        if np.isfinite(norm):
            altered=context.copy();altered.loc[altered.date==pd.Timestamp('2026-09-30'),c]=norm
            add('normal_september_'+c,COVS,altered)
    filled=context.copy()
    filled.loc[filled.date<'2026-10-01',['oni','dmi']]=filled.loc[filled.date<'2026-10-01',['oni','dmi']].ffill()
    add('forward_fill_missing_climate',COVS,filled)
    return cases,norm_y


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    from common.data import monthly_frame,data_dir
    from common.foundation import chronos2
    from chronos import BaseChronosPipeline
    frame=monthly_frame('PS');cases,normal=build_cases(frame)
    pipeline=BaseChronosPipeline.from_pretrained('amazon/chronos-2',device_map='cpu')
    history=cases[0]['context'].iloc[:-1]
    result={'site':'PS','origin':'2026-10-01','training_through':'2026-09-30',
            'model':'amazon/chronos-2','code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'input_sha256':hashlib.sha256((data_dir()/'lag.csv').read_bytes()).hexdigest(),
            'packages':{p:importlib.metadata.version(p) for p in ['chronos-forecasting','torch','pandas','numpy']},
            'normal_september_inflow':normal,'actual_october':float(frame.loc[frame.date==pd.Timestamp('2026-10-31'),'y'].iloc[0]),
            'recent_inputs':json.loads(history[['date','y',*COVS]].tail(12).to_json(orient='records',date_format='iso')),
            'interpretation':'Single-origin sensitivity, not additive feature attribution or evidence of general accuracy.', 'cases':[]}
    for case in cases:
        start=time.monotonic()
        fc=chronos2(case['context'],12,covariates=case['covariates'],freq='ME',pipeline=pipeline)
        r={'name':case['name'],'covariates':list(case['covariates']),
           'october':{q:float(fc[q].iloc[0]) for q in ['p10','p50','p90']},
           'seconds':round(time.monotonic()-start,3)}
        if not all(np.isfinite(v) for v in r['october'].values()):raise ValueError('Nonfinite output')
        result['cases'].append(r)
        print('Completed sensitivity case '+case['name'],flush=True)
    if abs(result['cases'][0]['october']['p50']-59.339393615722656)>0.01:
        raise ValueError('Full-input baseline did not reproduce archived October forecast')
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,allow_nan=False,ensure_ascii=False))


if __name__=='__main__':main()
