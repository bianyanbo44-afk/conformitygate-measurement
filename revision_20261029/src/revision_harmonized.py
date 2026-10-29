"""Harmonized daily-processing transfer sensitivity (Reviewer 2, Comment 5).

UK and Indian sensor days use the U.S. completeness level (at least 22 of 24 hours,
about 90%), so all three domains share one daily completeness rule. The common-feature
score additionally drops valid-hour and channel fields unavailable in the U.S. Every
upstream step is rebuilt under the same outer isolation, and the UK-trained estimators
are frozen for external scoring. Primary results are not modified.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse,json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold

from revision_data import ROOT,REV,CONFIG,F,G,save,ref_daily
from build_window_manifest import make_windows
from prepare_epa_us_extension import aggregate_daily
from revision_engine import get_predictions,build_table,fit_score,feature_columns,groupkey,META
from revision_statistics import point,bootstrap

OUT=REV/'results/harmonized'
HOURS=22
METHODS=['full','without_missing']

def sensor_daily(hourly,hours=HOURS):
    clean=hourly.dropna(subset=F).copy()
    clean['date']=clean.time.dt.floor('D')
    aggregations={c:(c,'mean') for c in F+['cf1_channel_gap','atm_channel_gap']}
    aggregations['valid_hours']=('time','nunique')
    result=clean.groupby(G+['date'],as_index=False).agg(**aggregations)
    return result[result.valid_hours.ge(hours)].reset_index(drop=True)

def build_daily():
    hourly=pd.read_csv(REV/'data/sensor_hourly.csv.gz',parse_dates=['time'])
    sensor=sensor_daily(hourly)
    raw_ref=pd.read_csv(ROOT/CONFIG['raw_data']['quant_reference'],parse_dates=['time'])
    raw_ref=raw_ref[raw_ref.measurand.eq('PM2.5') & raw_ref.version.eq('Ratified')].dropna(subset=['measurement']).copy()
    raw_ref['date']=raw_ref.time.dt.floor('D')
    qr=raw_ref.groupby(['location','date'],as_index=False).agg(reference_pm25=('measurement','mean'),reference_hours=('time','nunique'))
    qr=qr[qr.reference_hours.ge(18)].rename(columns={'location':'site'})
    qs=sensor[sensor.dataset.eq('QUANT')].merge(qr,on=['site','date'],how='left',validate='many_to_one')
    ir=ref_daily(hourly[hourly.dataset.eq('CampmierIndia')])
    ids=sensor[sensor.dataset.eq('CampmierIndia')].merge(ir,on=G+['date'],how='left',validate='one_to_one')
    raw_us=pd.read_csv(ROOT/'data/raw/epa_us/Data_DevelopmentUSPAcorrection_210408/Full24hrdataset.csv',low_memory=False)
    raw_us['Date']=pd.to_datetime(raw_us.Date).dt.normalize()
    us=aggregate_daily(raw_us,'mean_all_regulatory',require_reference=False)
    us['reference_hours']=np.nan
    return pd.concat([qs,ids,us],ignore_index=True).sort_values(G+['date']).reset_index(drop=True)

def build_windows(daily):
    manifests=[]
    for dataset,group in daily.groupby('dataset',sort=True):
        m=make_windows(group[G+['date']],CONFIG,'revision_sensor_harmonized',30,True)
        m['block_id']=[f'{dataset}_H{j:04d}' for j in range(len(m))]
        lookup={key:g.set_index('date') for key,g in group.groupby(G)}
        for idx,w in m.iterrows():
            g=lookup[(w.dataset,w.site,w.device)]
            start=pd.Timestamp(w.evaluation_start);end=pd.Timestamp(w.evaluation_end)
            paired=g[g.reference_pm25.notna()]
            m.loc[idx,'paired_valid_days']=int(paired.index.to_series().between(start,end).sum())
            pool=paired.index[(paired.index>=start-pd.Timedelta(days=14))&(paired.index<start)]
            for b in [0,1,3,7]:
                chosen=pool[-b:] if b else pool[:0]
                m.loc[idx,f'eligible_b{b}']=len(chosen)==b
                m.loc[idx,f'audit_dates_b{b}']=';'.join(d.strftime('%Y-%m-%d') for d in chosen) if len(chosen)==b else ''
        m['evaluable']=m.paired_valid_days.ge(23)
        manifests.append(m)
    return pd.concat(manifests,ignore_index=True)

def run(jobs=4):
    OUT.mkdir(parents=True,exist_ok=True);cache=OUT/'cache';cache.mkdir(exist_ok=True)
    daily=build_daily();windows=build_windows(daily)
    save(daily,OUT/'daily_harmonized.csv.gz');save(windows,OUT/'windows_harmonized.csv')
    ukwindows=windows[windows.dataset.eq('QUANT')];labels=ukwindows[ukwindows.evaluable]
    groups=groupkey(labels).to_numpy();parts=[]
    for fold,(ti,vi) in enumerate(GroupKFold(n_splits=5).split(labels,groups=groups)):
        held=set(groups[vi]);path=cache/('fold_'+'_'.join(sorted(held)).replace('|','-')+'.joblib')
        if path.exists():table=joblib.load(path)
        else:
            print(f'harmonized fold {fold+1}: rebuild upstream, held {sorted(held)}',flush=True)
            predictions=get_predictions(daily,held,jobs=jobs)
            source=daily[daily.dataset.eq('QUANT') & ~groupkey(daily).isin(held)]
            table=build_table(daily,ukwindows,predictions,source);joblib.dump(table,path,compress=3)
        usable=table[table.evaluable]
        train=usable[~groupkey(usable).isin(held)];test=usable[groupkey(usable).isin(held)]
        columns=feature_columns(table)
        for method in METHODS:
            scores,_=fit_score(train,test,method,columns)
            p=test[META].copy();p['method']=method;p['score']=scores;p['outer_fold']=fold;parts.append(p)
        print(f'harmonized fold {fold+1} completed: {len(test)} evaluated candidates',flush=True)
    source=daily[daily.dataset.eq('QUANT')]
    fullpath=cache/'full_uk.joblib';extpath=cache/'external.joblib'
    if fullpath.exists():train=joblib.load(fullpath)
    else:
        train=build_table(daily,ukwindows,get_predictions(daily,set(),jobs=jobs),source);joblib.dump(train,fullpath,compress=3)
    if extpath.exists():test=joblib.load(extpath)
    else:
        test=build_table(daily,windows[~windows.dataset.eq('QUANT')],get_predictions(daily,set(),external=True,jobs=jobs),source);joblib.dump(test,extpath,compress=3)
    train=train[train.evaluable];columns=feature_columns(train)
    for method in METHODS:
        scores,_=fit_score(train,test,method,columns)
        p=test[META].copy();p['method']=method;p['score']=scores;p['outer_fold']=-1;parts.append(p)
    scores=pd.concat(parts,ignore_index=True);scores=scores[scores.evaluable].copy()
    save(scores,OUT/'harmonized_scores.csv.gz')
    rows=[]
    for (dataset,method),g in scores.groupby(['dataset','method']):
        rows.append({'dataset':dataset,'method':method,**point(g),**bootstrap(g)})
    save(pd.DataFrame(rows),OUT/'harmonized_ranking.csv')
    flow=windows.groupby(['dataset','evaluable']).size().rename('blocks').reset_index()
    save(flow,OUT/'harmonized_flow.csv')
    print(pd.DataFrame(rows)[['dataset','method','n','blocks','prevalence','auroc','auroc_lo','auroc_hi','top20_precision']].to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=4)
    run(ap.parse_args().jobs)
