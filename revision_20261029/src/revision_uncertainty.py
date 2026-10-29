"""Criterion-level moving-block uncertainty and calibration/QC diagnostics."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
import joblib,json
from joblib import Parallel,delayed
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from revision_data import ROOT,REV,G,F,CONFIG,save
from revision_engine import groupkey,SEED
from run_baseline_feasibility import predict_group,conformity_metrics
from build_gate_dataset import audit_features

def batch_metrics(r,p):
    rm=r.mean(axis=1);pm=p.mean(axis=1)
    rd=r-rm[:,None];pd=p-pm[:,None]
    vr=(rd*rd).sum(axis=1);vp=(pd*pd).sum(axis=1);cov=(rd*pd).sum(axis=1)
    slope=np.divide(cov,vr,out=np.full(len(vr),np.nan),where=vr>0)
    intercept=pm-slope*rm
    r2=np.divide(cov*cov,vr*vp,out=np.zeros(len(vr)),where=(vr*vp)>0)
    rmse=np.sqrt(((p-r)**2).mean(axis=1))
    nrmse=np.divide(100*rmse,rm,out=np.full(len(rm),np.inf),where=rm>0)
    conforms=(slope>=.65)&(slope<=1.35)&(np.abs(intercept)<=5)&(r2>=.70)&((rmse<7)|(nrmse<30))
    return {'slope':slope,'intercept':intercept,'r2':r2,'rmse':rmse,'nrmse_percent':nrmse,'conforms':conforms}

def run():
    daily=pd.read_csv(REV/'data/daily.csv.gz',parse_dates=['date'])
    windows=pd.read_csv(REV/'data/windows.csv');ukw=windows[windows.dataset.eq('QUANT')&windows.evaluable]
    grouplist=groupkey(ukw).to_numpy();folds=GroupKFold(n_splits=5)
    held_by_group={}
    for _,ix in folds.split(ukw,groups=grouplist):
        held=set(grouplist[ix])
        for g in held:held_by_group[g]=held
    uk=daily[daily.dataset.eq('QUANT')]
    predpath=REV/'data/uk_oof_daily_predictions.csv.gz'
    if predpath.exists():pred=pd.read_csv(predpath,parse_dates=['date'])
    else:
        def one(key,target):
            held=held_by_group.get(key[1]+'|'+key[2],set())
            source=uk[~groupkey(uk).isin(held)&~uk.site.eq(key[1])&~uk.device.eq(key[2])].dropna(subset=['reference_pm25'])
            return predict_group(source,target,SEED)
        groups=[(k,g) for k,g in uk.groupby(G) if k[1]+'|'+k[2] in held_by_group]
        pred=pd.concat(Parallel(n_jobs=2,prefer='threads')(delayed(one)(k,g) for k,g in groups),ignore_index=True)
        save(pred,predpath)
    external=pd.read_csv(REV/'data/external_predictions.csv.gz',parse_dates=['date'])
    pred=pd.concat([pred,external],ignore_index=True)
    pg={k:g.pivot(index='date',columns='model',values='prediction_pm25').sort_index() for k,g in pred.groupby(G)}
    dg={k:g.sort_values('date').set_index('date') for k,g in daily.groupby(G)}
    scores=pd.concat([pd.read_csv(REV/'results/main_uk_scores.csv.gz'),pd.read_csv(REV/'results/main_external_scores.csv.gz')],ignore_index=True)
    scores=scores[scores.method.eq('full')&scores.evaluable].copy()
    windows=windows.set_index('block_id')
    rows=[];scenarios=[];qcrows=[]
    rng=np.random.default_rng(SEED)
    for number,(bid,candidates) in enumerate(scores.groupby('block_id',sort=True)):
        w=windows.loc[bid];key=(w.dataset,w.site,w.device);group=dg[key];pr=pg[key]
        block=group.loc[w.evaluation_start:w.evaluation_end]
        dates=block.index[block.reference_pm25.notna()]
        r=block.loc[dates,'reference_pm25'].to_numpy();n=len(r)
        indices={}
        for L in [3,5]:
            starts=rng.integers(0,n,size=(500,int(np.ceil(n/L))))
            indices[L]=((starts[:,:,None]+np.arange(L))%n).reshape(500,-1)[:,:n]
        for c in candidates.itertuples(index=False):
            b=c.reference_budget_days;at=w[f'audit_dates_b{b}']
            ad=[] if pd.isna(at) or not at else [pd.Timestamp(x) for x in at.split(';')]
            audit=pd.DataFrame({'reference_pm25':group.reindex(ad).reference_pm25.to_numpy(),'prediction_pm25':pr.reindex(ad)[c.model].to_numpy()})
            _,off=audit_features(audit,b)
            if c.adaptation=='none':off=0.
            p=(pr.loc[dates,c.model].to_numpy()+off).clip(min=0.)
            observed=conformity_metrics(r,p,CONFIG['conformity_targets'])
            if bool(observed['conforms'])!=bool(c.conforms):raise AssertionError(f'Bootstrap prediction reconstruction label differs: {bid}/{c.model}/{b}/{c.adaptation}')
            for L,ix in indices.items():
                bm=batch_metrics(r[ix],p[ix]);rec={k:getattr(c,k) for k in ['dataset','site','device','block_id','model','reference_budget_days','adaptation','score','conforms']}
                rec['block_length']=L;rec['pass_stability']=bm['conforms'].mean();rec['valid_resamples']=int(np.isfinite(bm['slope']).sum())
                for name in ['slope','intercept','r2','rmse','nrmse_percent']:
                    rec[name+'_lo'],rec[name+'_hi']=np.nanquantile(bm[name],[.025,.975])
                rows.append(rec)
            # Conditional reference stress tests, explicitly not calibrated uncertainty budgets.
            for name,rr in [('offset_minus1',r-1),('offset_plus1',r+1),('gain_minus2pct',r*.98),('gain_plus2pct',r*1.02)]:
                m=conformity_metrics(rr,p,CONFIG['conformity_targets'])
                scenarios.append({'dataset':c.dataset,'block_id':bid,'model':c.model,'budget':b,'adaptation':c.adaptation,'scenario':name,'conforms':m['conforms'],'changed':bool(m['conforms'])!=bool(c.conforms)})
            if c.model=='epa_correction' and b==0:
                keep=~block.loc[dates].ab_qc_fail.to_numpy(bool)
                m=conformity_metrics(r[keep],p[keep],CONFIG['conformity_targets']) if keep.sum()>=23 else {}
                qcrows.append({'dataset':c.dataset,'block_id':bid,'native_days':n,'qc_days':int(keep.sum()),'native_conforms':bool(c.conforms),'qc_evaluable':keep.sum()>=23,'qc_conforms':m.get('conforms',np.nan)})
        if number%50==0:print(f'uncertainty block {number+1}/{scores.block_id.nunique()}',flush=True)
    result=pd.DataFrame(rows)
    save(result,REV/'results/criterion_uncertainty.csv.gz');save(pd.DataFrame(scenarios),REV/'results/reference_stress_tests.csv.gz');save(pd.DataFrame(qcrows),REV/'results/epa_channel_qc_sensitivity.csv')
    summary=[]
    for (dataset,L),g in result.groupby(['dataset','block_length']):
        label=g.conforms.astype(float).astype(bool)
        stable=np.where(label,g.pass_stability.ge(.9),g.pass_stability.le(.1))
        summary.append({'dataset':dataset,'block_length':L,'candidates':len(g),'mean_pass_stability':g.pass_stability.mean(),'stable_label_fraction':stable.mean(),
                        'uncertain_label_fraction':1-stable.mean(),'median_slope_ci_width':(g.slope_hi-g.slope_lo).median(),'median_intercept_ci_width':(g.intercept_hi-g.intercept_lo).median(),
                        'median_r2_ci_width':(g.r2_hi-g.r2_lo).median(),'median_rmse_ci_width':(g.rmse_hi-g.rmse_lo).median(),'median_nrmse_ci_width':(g.nrmse_percent_hi-g.nrmse_percent_lo).median()})
    save(pd.DataFrame(summary),REV/'results/uncertainty_summary.csv')
    stress=pd.DataFrame(scenarios).groupby(['dataset','scenario']).agg(candidates=('changed','size'),label_flips=('changed','sum'),flip_fraction=('changed','mean'),conformity=('conforms','mean')).reset_index()
    save(stress,REV/'results/reference_stress_summary.csv')
    # Directly diagnose standardized ridge versus OLS under each actual outer calibration pool.
    coefrows=[];diagnostics=[]
    for (dataset,site,device),target in uk.groupby(G):
        key=site+'|'+device
        if key not in held_by_group:continue
        source=uk[~groupkey(uk).isin(held_by_group[key])&~uk.site.eq(site)&~uk.device.eq(device)].dropna(subset=['reference_pm25'])
        test=target.dropna(subset=['reference_pm25'])
        X=source[F].to_numpy();y=source.reference_pm25.to_numpy()
        for name,model in [('ols',LinearRegression()),('ridge_alpha0',make_pipeline(StandardScaler(),Ridge(alpha=0))),('ridge_alpha1',make_pipeline(StandardScaler(),Ridge(alpha=1))),('ridge_alpha10',make_pipeline(StandardScaler(),Ridge(alpha=10))),('ridge_alpha100',make_pipeline(StandardScaler(),Ridge(alpha=100)))]:
            model.fit(X,y);prediction=model.predict(test[F].to_numpy()).clip(min=0)
            if name=='ols':coef=model.coef_;inter=model.intercept_
            else:
                scale=model.named_steps['standardscaler'];reg=model.named_steps['ridge']
                coef=reg.coef_/scale.scale_;inter=reg.intercept_-np.dot(coef,scale.mean_)
            for j,c in enumerate(F):coefrows.append({'site':site,'device':device,'model':name,'feature':c,'coefficient_original_units':coef[j],'intercept':inter})
            diagnostics.append({'site':site,'device':device,'model':name,'source_days':len(source),'test_days':len(test),'rmse':np.sqrt(np.mean((prediction-test.reference_pm25)**2)),'bias':np.mean(prediction-test.reference_pm25),
                                'cf1_atm_correlation':source.cf1.corr(source.atm),'standardized_condition_number':np.linalg.cond(StandardScaler().fit_transform(X))})
    save(pd.DataFrame(coefrows),REV/'results/ridge_coefficients.csv');save(pd.DataFrame(diagnostics),REV/'results/ridge_diagnostics.csv')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':run()
