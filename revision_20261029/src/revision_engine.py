"""Reviewer-directed outer-isolated calibration and conformity experiments.

All assessment-reference labels are constructed after sensor feature summaries.
Targets of the error-regression comparators are never prediction inputs.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import sys,json,argparse,warnings
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel,delayed
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from revision_data import ROOT,REV,CONFIG,F,G,save
sys.path.insert(0,str(ROOT/'src'))
from run_baseline_feasibility import predict_group,conformity_metrics,fitted_models
from run_gate_prototype import build_estimator
from build_gate_dataset import summarize,source_support,audit_features

SEED=20261029
MODELS=['epa_correction','hist_gradient_boosting','linear','random_forest','raw_cf1','ridge']
KEY=['dataset','site','device','block_id','model','reference_budget_days','adaptation']
META=KEY+['evaluable','valid_days','conforms','slope','intercept','r2','bias','rmse','nrmse_percent','violation']
ABLATIONS=['I','E','P','IE','IP','EP','IEP','without_month','without_missing','without_audit']

def groupkey(frame):
    return frame.site.astype(str)+'|'+frame.device.astype(str)

def feature_columns(table):
    return [c for c in table if c in ['model','adaptation','reference_budget_days'] or c.startswith(('e_','p_','support_','audit_')) or c=='applied_audit_offset']

def subset_columns(columns,method):
    ident=[c for c in columns if c in ['model','adaptation','reference_budget_days']]
    env=[c for c in columns if c.startswith(('e_','support_'))]
    pred=[c for c in columns if c.startswith('p_')]
    if method in ['I','E','P','IE','IP','EP','IEP']:
        return (ident if 'I' in method else [])+(env if 'E' in method else [])+(pred if 'P' in method else [])
    if method=='without_audit':
        return [c for c in columns if not c.startswith('audit_') and c!='applied_audit_offset']
    if method=='without_month':
        return [c for c in columns if 'month' not in c]
    if method=='without_missing':
        return [c for c in columns if 'valid_hours' not in c and 'channel' not in c]
    return columns

def get_predictions(daily,held,external=False,jobs=4):
    uk=daily[daily.dataset.eq('QUANT')]
    pool=uk[~groupkey(uk).isin(held)]
    target=daily[~daily.dataset.eq('QUANT')] if external else uk
    def one(key,g):
        source=pool if external else pool[~pool.site.eq(key[1]) & ~pool.device.eq(key[2])]
        source=source.dropna(subset=F+['reference_pm25'])
        if source.empty: raise ValueError(f'No isolated calibration training rows for {key}')
        return predict_group(source,g,SEED)
    parts=Parallel(n_jobs=jobs,prefer='threads')(delayed(one)(key,g) for key,g in target.groupby(G,sort=True))
    return pd.concat(parts,ignore_index=True)

def build_table(daily,windows,predictions,source):
    """Only audit_features and label calculation may access reference_pm25."""
    output=[]
    predgroups={k:g.pivot(index='date',columns='model',values='prediction_pm25').sort_index() for k,g in predictions.groupby(G,sort=True)}
    sensors={k:g.sort_values('date').set_index('date') for k,g in daily.groupby(G,sort=True)}
    for w in windows.itertuples(index=False):
        key=(w.dataset,w.site,w.device)
        if key not in predgroups:continue
        group=sensors[key];pr=predgroups[key]
        start=pd.Timestamp(w.evaluation_start);end=pd.Timestamp(w.evaluation_end)
        block=group.loc[start:end]
        # No assessment reference field or missingness is selected here.
        observed=block[F+['valid_hours','cf1_channel_gap','atm_channel_gap']]
        common={'e_valid_days':len(observed),'e_mean_valid_hours':observed.valid_hours.mean(),
                'e_month_sin':np.sin(2*np.pi*start.month/12),'e_month_cos':np.cos(2*np.pi*start.month/12)}
        for c in F:common.update(summarize(observed[c],'e_'+c))
        common.update(summarize(observed.cf1_channel_gap,'e_channel_gap'))
        common['e_channel_available']=int(observed.cf1_channel_gap.notna().any())
        support=source_support(source,w.dataset,w.site,w.device)
        z=(observed[F]-support['mean'])/support['std']
        common['support_mean_z_l2']=np.sqrt(z.pow(2).sum(axis=1)).mean()
        common['support_max_abs_z']=z.abs().max(axis=1).max()
        common['support_outside_fraction']=((observed[F]<support['q01'])|(observed[F]>support['q99'])).to_numpy().mean()
        bp=pr.loc[start:end]
        common.update(summarize(bp.std(axis=1),'p_disagreement'))
        noepa=summarize(bp.drop(columns=['epa_correction']).std(axis=1),'noepa_disagreement')
        for model in MODELS:
            for b in [0,1,3,7]:
                if not getattr(w,f'eligible_b{b}'):continue
                text=getattr(w,f'audit_dates_b{b}')
                dates=[] if pd.isna(text) or not text else [pd.Timestamp(t) for t in str(text).split(';')]
                if any(t>=start for t in dates):raise AssertionError('Audit date entered assessment')
                audit=pd.DataFrame({'reference_pm25':group.reindex(dates).reference_pm25.to_numpy(),
                                    'prediction_pm25':pr.reindex(dates)[model].to_numpy()})
                af,offset=audit_features(audit,b)
                for adapt in (['none'] if not b else ['none','shrunk_offset']):
                    applied=offset if adapt=='shrunk_offset' else 0.
                    prediction=(bp[model]+applied).clip(lower=0)
                    feats=dict(common)
                    feats.update(summarize(prediction,'p_prediction'))
                    feats.update(af);feats['applied_audit_offset']=applied
                    # Labels are appended only after the complete predictor record exists.
                    reference=block.reference_pm25.reindex(prediction.index)
                    mask=reference.notna() & prediction.notna()
                    lab={'conforms':np.nan,'slope':np.nan,'intercept':np.nan,'r2':np.nan,'bias':np.nan,'rmse':np.nan,'nrmse_percent':np.nan,'violation':np.nan}
                    if mask.sum()>=23:
                        lab.update(conformity_metrics(reference[mask].to_numpy(),prediction[mask].to_numpy(),CONFIG['conformity_targets']))
                        lab['violation']=max(abs(lab['slope']-1)/.35-1,abs(lab['intercept'])/5-1,(.70-lab['r2'])/.70,
                                             min(lab['rmse']/7-1,lab['nrmse_percent']/30-1))
                    record={'dataset':w.dataset,'site':w.site,'device':w.device,'block_id':w.block_id,
                            'model':model,'reference_budget_days':b,'adaptation':adapt,'evaluable':mask.sum()>=23,
                            'valid_days':int(mask.sum()),**lab,**feats,**noepa}
                    output.append(record)
    return pd.DataFrame(output)

def prior_scores(train,test):
    keys=['model','reference_budget_days','adaptation']
    rates=train.groupby(keys).conforms.mean()
    return rates.reindex(pd.MultiIndex.from_frame(test[keys])).fillna(train.conforms.mean()).to_numpy(float)

def fit_score(train,test,method,columns):
    if method=='variant_prior':return prior_scores(train,test),None
    if method=='support_only':return -test.support_mean_z_l2.to_numpy(float),None
    if method=='audit_only':return -test.audit_abs_error_mean.fillna(1e6).to_numpy(float),None
    target={'predicted_rmse':'rmse','predicted_bias':'bias','predicted_margin':'violation'}.get(method)
    if target:
        pre=clone(build_estimator('hist_gradient_boosting_gate',columns,SEED).named_steps['preprocess'])
        reg=HistGradientBoostingRegressor(learning_rate=.05,max_iter=160,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.,early_stopping=False,random_state=SEED)
        estimator=Pipeline([('preprocess',pre),('regressor',reg)])
        y=train[target].abs() if target=='bias' else train[target]
        estimator.fit(train[columns],y)
        return -estimator.predict(test[columns]),estimator
    cols=subset_columns(columns,method)
    name='logistic_gate' if method=='logistic' else 'hist_gradient_boosting_gate'
    estimator=build_estimator(name,cols,SEED)
    estimator.fit(train[cols],train.conforms.astype(int))
    if not np.array_equal(estimator.classes_,[0,1]):raise AssertionError('Positive-class orientation mismatch')
    return estimator.predict_proba(test[cols])[:,1],estimator

def fixed_pipeline_choices(train,test):
    rows=[]
    for b in [0,1,3,7]:
        t=train[train.reference_budget_days.eq(b)]
        rates=t.groupby(['model','adaptation']).conforms.agg(['mean','size']).reset_index().sort_values(['mean','model','adaptation'],ascending=[False,True,True])
        if rates.empty:continue
        choice=rates.iloc[0]
        q=test[test.reference_budget_days.eq(b)].copy()
        q['score']=np.where(q.model.eq(choice.model)&q.adaptation.eq(choice.adaptation),1.,0.)
        q['method']='source_fixed';q['selected_model']=choice.model;q['selected_adaptation']=choice.adaptation
        rows.append(q[META+['method','score','selected_model','selected_adaptation']])
    return pd.concat(rows,ignore_index=True)

def run(mode='main',jobs=4):
    daily=pd.read_csv(REV/'data/daily.csv.gz',parse_dates=['date'])
    windows=pd.read_csv(REV/'data/windows.csv')
    ukwindows=windows[windows.dataset.eq('QUANT')]
    labels=ukwindows[ukwindows.evaluable]
    groups=groupkey(labels).to_numpy()
    splitter=LeaveOneGroupOut() if mode=='loco' else GroupKFold(n_splits=5)
    methods=['full'] if mode=='loco' else ['full','logistic','variant_prior','support_only','audit_only','predicted_rmse','predicted_bias','predicted_margin']+ABLATIONS
    cache=REV/'results/cache';cache.mkdir(exist_ok=True)
    parts=[];calibration_parts=[]
    for fold,(ti,vi) in enumerate(splitter.split(labels,groups=groups)):
        held=set(groups[vi]);heldstr='_'.join(sorted(held)).replace('|','-')
        cachepath=cache/f'fold_{heldstr}.joblib'
        if cachepath.exists():table=joblib.load(cachepath)
        else:
            print(f'{mode} fold {fold+1}: rebuild upstream, held {sorted(held)}',flush=True)
            predictions=get_predictions(daily,held,jobs=jobs)
            source=daily[daily.dataset.eq('QUANT') & ~groupkey(daily).isin(held)]
            table=build_table(daily,ukwindows,predictions,source)
            joblib.dump(table,cachepath,compress=3)
        usable=table[table.evaluable].copy()
        train=usable[~groupkey(usable).isin(held)].copy()
        test=usable[groupkey(usable).isin(held)].copy()
        columns=feature_columns(table)
        for method in methods:
            scores,est=fit_score(train,test,method,columns)
            p=test[META].copy();p['method']=method;p['score']=scores;p['outer_fold']=fold
            parts.append(p)
        if mode=='main':
            fixed=fixed_pipeline_choices(train,test);fixed['outer_fold']=fold;parts.append(fixed)
            p=test[META].copy();p['method']='fixed_epa';p['score']=np.where(p.model.eq('epa_correction') & p.adaptation.eq('none'),1.,0.);p['outer_fold']=fold;parts.append(p)
            # Strong no-EPA check excludes EPA candidates and their cross-model summaries.
            tr=train[~train.model.eq('epa_correction')].copy();te=test[~test.model.eq('epa_correction')].copy()
            for c in columns:
                if c.startswith('p_disagreement_'):
                    tr[c]=tr[c.replace('p_disagreement_','noepa_disagreement_')]
                    te[c]=te[c.replace('p_disagreement_','noepa_disagreement_')]
            score,_=fit_score(tr,te,'full',columns)
            p=te[META].copy();p['method']='noepa_refit';p['score']=score;p['outer_fold']=fold;parts.append(p)
        print(f'{mode} fold {fold+1} completed: {len(test)} evaluated candidates',flush=True)
        save(pd.concat(parts,ignore_index=True),REV/f'results/{mode}_uk_scores.csv.gz')
    if mode=='loco':return
    source=daily[daily.dataset.eq('QUANT')]
    fullpath=cache/'full_uk.joblib';extpath=cache/'external.joblib'
    if fullpath.exists():train=joblib.load(fullpath)
    else:
        pr=get_predictions(daily,set(),jobs=jobs)
        train=build_table(daily,ukwindows,pr,source);joblib.dump(train,fullpath,compress=3)
    if extpath.exists():test=joblib.load(extpath)
    else:
        pr=get_predictions(daily,set(),external=True,jobs=jobs)
        save(pr,REV/'data/external_predictions.csv.gz')
        test=build_table(daily,windows[~windows.dataset.eq('QUANT')],pr,source)
        joblib.dump(test,extpath,compress=3)
    train=train[train.evaluable].copy();columns=feature_columns(train)
    external_parts=[]
    (REV/'results/models').mkdir(exist_ok=True)
    for method in methods:
        scores,est=fit_score(train,test,method,columns)
        p=test[META].copy();p['method']=method;p['score']=scores;p['outer_fold']=-1;external_parts.append(p)
        if est is not None:joblib.dump(est,REV/f'results/models/{method}.joblib',compress=3)
    external_parts.append(fixed_pipeline_choices(train,test))
    p=test[META].copy();p['method']='fixed_epa';p['score']=np.where(p.model.eq('epa_correction')&p.adaptation.eq('none'),1.,0.);external_parts.append(p)
    tr=train[~train.model.eq('epa_correction')].copy();te=test[~test.model.eq('epa_correction')].copy()
    for c in columns:
        if c.startswith('p_disagreement_'):
            tr[c]=tr[c.replace('p_disagreement_','noepa_disagreement_')]
            te[c]=te[c.replace('p_disagreement_','noepa_disagreement_')]
    score,est=fit_score(tr,te,'full',columns)
    p=te[META].copy();p['method']='noepa_refit';p['score']=score;external_parts.append(p)
    joblib.dump(est,REV/'results/models/noepa_refit.joblib',compress=3)
    save(pd.concat(external_parts,ignore_index=True),REV/'results/main_external_scores.csv.gz')
    save(train,REV/'data/uk_training_table.csv.gz');save(test,REV/'data/external_table.csv.gz')
    # Demonstrate score-time operation for sensor windows without evaluable labels.
    unlabelled=joblib.load(fullpath)
    unlabelled=unlabelled[~unlabelled.evaluable].copy()
    if not unlabelled.empty:
        frozen=joblib.load(REV/'results/models/full.joblib')
        unlabelled['score']=frozen.predict_proba(unlabelled[columns])[:,1]
        save(unlabelled[META+['score']],REV/'results/unlabelled_uk_scores.csv.gz')
    # Core predictor invariance check with audit information held fixed.
    ext=daily[~daily.dataset.eq('QUANT')].copy();altered=ext.copy()
    dates=pd.to_datetime(windows[~windows.dataset.eq('QUANT')].evaluation_start)
    # Check zero-budget rows so neither label changes nor audit masking can enter predictors.
    allmasked=daily.copy();allmasked.loc[~allmasked.dataset.eq('QUANT'),'reference_pm25']=np.nan
    masked_windows=windows[~windows.dataset.eq('QUANT')].copy()
    for b in [1,3,7]:masked_windows[f'eligible_b{b}']=False
    rebuilt=build_table(allmasked,masked_windows,pd.read_csv(REV/'data/external_predictions.csv.gz',parse_dates=['date']),source)
    expected=test[test.reference_budget_days.eq(0)].sort_values(KEY).reset_index(drop=True)
    observed=rebuilt.sort_values(KEY).reset_index(drop=True)
    invariant=np.allclose(expected[columns].select_dtypes(include=[np.number]),observed[columns].select_dtypes(include=[np.number]),equal_nan=True)
    if not invariant:raise AssertionError('Assessment-reference availability changed score inputs')
    (REV/'results/integrity.json').write_text(json.dumps({'assessment_reference_mask_invariance':True,'positive_class_index_verified':True,'daily_calibration':True,'external_training':'UK only','shrinkage':{'1':.25,'3':.5,'7':.7}},indent=2))
    print('All frozen external experiments and invariance check completed',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['main','loco'],default='main');ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args();run(args.mode,args.jobs)
