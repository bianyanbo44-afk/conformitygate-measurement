"""Ranking, matched deployment policies, uncertainty and exclusion diagnostics."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.stats import spearmanr
from revision_data import ROOT,REV,G,F,save
from revision_engine import META,KEY,groupkey

SEED=20261029
NBOOT=2000
Q=[.05,.1,.2,.3]
MAIN=['full','logistic','variant_prior','support_only','audit_only','predicted_rmse','predicted_bias','predicted_margin','noepa_refit']

def metric_arrays(y,s,w=None,q=.2):
    """Weighted AUC and tie-averaged precision; equal scores receive equal review probability."""
    y=np.asarray(y,float);s=np.asarray(s,float)
    w=np.ones(len(y)) if w is None else np.asarray(w,float)
    _,si=np.unique(s,return_inverse=True)
    pos=np.bincount(si,weights=w*y);neg=np.bincount(si,weights=w*(1-y));count=pos+neg
    P=pos.sum();N=neg.sum()
    auc=np.sum(pos*(np.cumsum(neg)-neg/2))/(P*N) if P and N else np.nan
    target=max(1,np.ceil(q*w.sum()));total=0.;passed=0.
    for n,p in zip(count[::-1],pos[::-1]):
        if not n:continue
        take=min(n,target-total);passed+=take*p/n;total+=take
        if total>=target:break
    return float(auc),float(passed/total),float(P/(P+N))

def clusters(frame,site_india=False):
    if frame.dataset.iloc[0]=='QUANT':return groupkey(frame)
    if frame.dataset.iloc[0]=='EPA_US_24h':return frame.site.astype(str)
    return frame.site.astype(str) if site_india else (frame.block_id.astype(str) if 'block_id' in frame else pd.Series(frame.index.astype(str),index=frame.index))

def point(frame):
    y=frame.conforms.astype(int).to_numpy();s=frame.score.to_numpy(float)
    auc,precision,prevalence=metric_arrays(y,s)
    out={'n':len(frame),'blocks':frame.block_id.nunique(),'clusters':clusters(frame).nunique(),'prevalence':prevalence,'auroc':auc,
         'ap':average_precision_score(y,s) if len(np.unique(y))==2 else np.nan,'top20_precision':precision,'top20_lift':precision/prevalence if prevalence else np.nan}
    for q in Q:out[f'precision_{int(q*100)}']=metric_arrays(y,s,q=q)[1]
    return out

def bootstrap(frame,site_india=False):
    y=frame.conforms.astype(int).to_numpy();s=frame.score.to_numpy(float)
    _,ci=np.unique(clusters(frame,site_india),return_inverse=True);k=ci.max()+1
    rng=np.random.default_rng(SEED)
    draws=rng.multinomial(k,np.repeat(1/k,k),size=NBOOT)
    records=[]
    for d in draws:
        auc,precision,prev=metric_arrays(y,s,d[ci]);records.append([auc,precision,precision-prev,precision/prev if prev else np.nan])
    a=np.asarray(records)
    lo,hi=np.nanquantile(a,[.025,.975],axis=0)
    return {name+'_'+side:float(value) for j,name in enumerate(['auroc','top20_precision','top20_gain','top20_lift']) for side,value in [('lo',lo[j]),('hi',hi[j])]}

def select(frame):
    return frame.sort_values(['block_id','score','model','adaptation'],ascending=[True,False,True,True],kind='mergesort').drop_duplicates('block_id').copy()

def criterion_labels(frame,change=None):
    p={'slope':.35,'intercept':5.,'r2':.70,'rmse':7.,'nrmse':30.}
    if change:p.update(change)
    return ((frame.slope-1).abs().le(p['slope']) & frame.intercept.abs().le(p['intercept']) & frame.r2.ge(p['r2']) & (frame.rmse.lt(p['rmse'])|frame.nrmse_percent.lt(p['nrmse']))).astype(int)

def run():
    a=pd.read_csv(REV/'results/main_uk_scores.csv.gz',low_memory=False)
    b=pd.read_csv(REV/'results/main_external_scores.csv.gz',low_memory=False)
    frame=pd.concat([a,b],ignore_index=True)
    frame=frame[frame.evaluable].copy();frame['conforms']=frame.conforms.astype(float).astype(int)
    noepa=frame[frame.method.eq('full')&~frame.model.eq('epa_correction')].copy();noepa['method']='noepa_frozen'
    frame=pd.concat([frame,noepa],ignore_index=True)
    ranking=[];within=[];jackknife=[]
    for (dataset,method),g in frame.groupby(['dataset','method']):
        if method in ['fixed_epa','source_fixed']:continue
        row={'dataset':dataset,'method':method,'budget':'pooled',**point(g)}
        if method=='full' or (dataset=='EPA_US_24h' and method.startswith('noepa')):
            row.update(bootstrap(g))
        ranking.append(row)
        for budget,part in g.groupby('reference_budget_days'):
            ranking.append({'dataset':dataset,'method':method,'budget':str(budget),**point(part)})
        if method=='full':
            for (model,budget),part in g.groupby(['model','reference_budget_days']):within.append({'dataset':dataset,'model':model,'budget':budget,**point(part)})
            for model,part in g.groupby('model'):within.append({'dataset':dataset,'model':model,'budget':'pooled',**point(part)})
            for c in clusters(g).unique():
                part=g[clusters(g).ne(c)]
                jackknife.append({'dataset':dataset,'omitted':c,**point(part)})
    save(pd.DataFrame(ranking),REV/'results/ranking.csv');save(pd.DataFrame(within),REV/'results/within_pipeline.csv');save(pd.DataFrame(jackknife),REV/'results/cluster_deletion.csv')
    policies=[];chosen=[]
    policy_methods=MAIN+['fixed_epa','source_fixed','noepa_frozen']
    for (dataset,method,budget),g in frame[frame.method.isin(policy_methods)].groupby(['dataset','method','reference_budget_days']):
        s=select(g);chosen.append(s)
        row={'dataset':dataset,'method':method,'budget':budget,**point(s),'conforming_products':int(s.conforms.sum()),
             'prescore_reference_days':int(budget*len(s)),'software_variants_per_block':len(g)/len(s)}
        for q in Q:
            n=max(1,int(np.ceil(len(s)*q)));row[f'review_blocks_{int(q*100)}']=n;row[f'expected_verified_products_{int(q*100)}']=metric_arrays(s.conforms,s.score,q=q)[1]*n
        if method=='full':row.update(bootstrap(s))
        policies.append(row)
    selected=pd.concat(chosen,ignore_index=True)
    save(pd.DataFrame(policies),REV/'results/deployment_policies.csv');save(selected,REV/'results/selected_products.csv.gz')
    # Paired block-level product-yield contrasts, with the same blocks and budget.
    contrasts=[];review_contrasts=[]
    for (dataset,budget),g in selected.groupby(['dataset','reference_budget_days']):
        primary=g[g.method.eq('full')].set_index('block_id')
        for method in policy_methods:
            if method=='full':continue
            other=g[g.method.eq(method)].set_index('block_id')
            ix=primary.index.intersection(other.index);p=primary.loc[ix];o=other.loc[ix]
            delta=p.conforms.to_numpy()-o.conforms.to_numpy()
            _,ci=np.unique(clusters(p),return_inverse=True);k=ci.max()+1
            draws=np.random.default_rng(SEED).multinomial(k,np.repeat(1/k,k),size=NBOOT)
            numer=np.bincount(ci,weights=delta);denom=np.bincount(ci)
            values=draws@numer/(draws@denom)
            contrasts.append({'dataset':dataset,'budget':budget,'comparator':method,'blocks':len(ix),'primary_yield':p.conforms.mean(),'comparator_yield':o.conforms.mean(),
                              'difference':delta.mean(),'lo':np.quantile(values,.025),'hi':np.quantile(values,.975)})
            if budget==0 and method in ['fixed_epa','source_fixed','variant_prior']:
                for q in Q:
                    values=np.array([metric_arrays(p.conforms,p.score,d[ci],q)[1]-metric_arrays(o.conforms,o.score,d[ci],q)[1] for d in draws])
                    review_contrasts.append({'dataset':dataset,'budget':budget,'comparator':method,'q':q,'difference':metric_arrays(p.conforms,p.score,q=q)[1]-metric_arrays(o.conforms,o.score,q=q)[1],
                                            'lo':np.nanquantile(values,.025),'hi':np.nanquantile(values,.975)})
    save(pd.DataFrame(contrasts),REV/'results/paired_deployment_contrasts.csv')
    save(pd.DataFrame(review_contrasts),REV/'results/paired_review_contrasts.csv')
    auc_contrasts=[]
    for dataset,g in frame.groupby('dataset'):
        primary=g[g.method.eq('full')].sort_values(KEY).reset_index(drop=True)
        _,ci=np.unique(clusters(primary),return_inverse=True);k=ci.max()+1
        draws=np.random.default_rng(SEED).multinomial(k,np.repeat(1/k,k),size=NBOOT)
        y=primary.conforms.to_numpy(float)
        def prepared(s):
            return np.unique(s,return_inverse=True)[1]
        def auc(ix,w):
            p=np.bincount(ix,weights=w*y);n=np.bincount(ix,weights=w*(1-y));P=p.sum();N=n.sum()
            return np.sum(p*(np.cumsum(n)-n/2))/(P*N) if P and N else np.nan
        pi=prepared(primary.score)
        for method in [m for m in MAIN if m not in ['full','noepa_refit']]:
            other=g[g.method.eq(method)].sort_values(KEY).reset_index(drop=True)
            if not primary[KEY].equals(other[KEY]):raise AssertionError('Unmatched ranking comparator')
            oi=prepared(other.score)
            delta=np.array([auc(pi,d[ci])-auc(oi,d[ci]) for d in draws])
            auc_contrasts.append({'dataset':dataset,'comparator':method,'difference':point(primary)['auroc']-point(other)['auroc'],
                                  'lo':np.nanquantile(delta,.025),'hi':np.nanquantile(delta,.975)})
    save(pd.DataFrame(auc_contrasts),REV/'results/paired_auroc_contrasts.csv')
    # Label perturbation is an evaluation sensitivity, with frozen scores.
    grid=[('original',{}),('slope_tight',{'slope':.30}),('slope_loose',{'slope':.40}),('intercept_tight',{'intercept':4.5}),('intercept_loose',{'intercept':5.5}),
          ('r2_tight',{'r2':.72}),('r2_loose',{'r2':.68}),('rmse_tight',{'rmse':6.5}),('rmse_loose',{'rmse':7.5}),('nrmse_tight',{'nrmse':27.}),('nrmse_loose',{'nrmse':33.}),
          ('joint_tight',{'slope':.30,'intercept':4.5,'r2':.72,'rmse':6.5,'nrmse':27.}),('joint_loose',{'slope':.40,'intercept':5.5,'r2':.68,'rmse':7.5,'nrmse':33.})]
    perturb=[];failures=[];site_sensitivity=[]
    for dataset,g in frame[frame.method.eq('full')].groupby('dataset'):
        if dataset=='CampmierIndia':site_sensitivity.append({'dataset':dataset,**point(g),**bootstrap(g,True),'clusters':int(g.site.nunique()),'resampling_clusters':int(g.site.nunique())})
        for name,change in grid:
            altered=g.copy();altered['conforms']=criterion_labels(g,change)
            perturb.append({'dataset':dataset,'scenario':name,'changed_labels':int((altered.conforms!=g.conforms).sum()),'changed_fraction':(altered.conforms!=g.conforms).mean(),**point(altered)})
        for name,passed in [('slope',(g.slope-1).abs().le(.35)),('intercept',g.intercept.abs().le(5)),('r2',g.r2.ge(.7)),('error',g.rmse.lt(7)|g.nrmse_percent.lt(30))]:
            failures.append({'dataset':dataset,'criterion':name,'candidates':len(g),'failures':int((~passed).sum()),'failure_rate':(~passed).mean()})
    save(pd.DataFrame(perturb),REV/'results/threshold_perturbations.csv');save(pd.DataFrame(failures),REV/'results/criterion_failures.csv');save(pd.DataFrame(site_sensitivity),REV/'results/india_site_cluster_sensitivity.csv')
    india=frame[frame.method.eq('full')&frame.dataset.eq('CampmierIndia')]
    save(pd.DataFrame([{'omitted_site':s,**point(india[~india.site.eq(s)])} for s in india.site.unique()]),REV/'results/india_site_deletion.csv')
    # Included/excluded sensor distributions and reference concentration ranges.
    daily=pd.read_csv(REV/'data/daily.csv.gz',parse_dates=['date']);windows=pd.read_csv(REV/'data/windows.csv')
    summaries=[];ranges=[]
    for w in windows.itertuples(index=False):
        g=daily[daily.dataset.eq(w.dataset)&daily.site.eq(w.site)&daily.device.eq(w.device)&daily.date.between(w.evaluation_start,w.evaluation_end)]
        summaries.append({'dataset':w.dataset,'block_id':w.block_id,'evaluable':w.evaluable,**{c:g[c].mean() for c in F},'sensor_days':len(g),'paired_days':g.reference_pm25.notna().sum()})
        r=g.reference_pm25.dropna()
        ranges.append({'dataset':w.dataset,'site':w.site,'block_id':w.block_id,'evaluable':w.evaluable,'n':len(r),'reference_min':r.min(),'reference_max':r.max(),'reference_mean':r.mean(),'reference_sd':r.std(),'reference_q05':r.quantile(.05),'reference_q95':r.quantile(.95)})
    save(pd.DataFrame(summaries),REV/'results/inclusion_features.csv');save(pd.DataFrame(ranges),REV/'results/concentration_ranges.csv')
    # Logistic diagnostics: source/target discrimination and class-conditional feature shifts.
    diagnostics=[]
    train=pd.read_csv(REV/'data/uk_training_table.csv.gz');external=pd.read_csv(REV/'data/external_table.csv.gz')
    for dataset,g in pd.concat([train,external]).groupby('dataset'):
        g=g[g.evaluable]
        for col in [c for c in g if c.startswith(('e_','p_','support_','audit_'))]:
            x=pd.to_numeric(g[col],errors='coerce');y=g.conforms.astype(float)
            mask=x.notna()&y.notna()
            corr=spearmanr(x[mask],y[mask]).statistic if x[mask].nunique()>1 and y[mask].nunique()>1 else np.nan
            diagnostics.append({'dataset':dataset,'feature':col,'mean_conforming':x[y.eq(1)].mean(),'mean_nonconforming':x[y.eq(0)].mean(),'spearman':corr,'missing_fraction':x.isna().mean()})
    save(pd.DataFrame(diagnostics),REV/'results/feature_label_shift.csv')
    loco_path=REV/'results/loco_uk_scores.csv.gz'
    if loco_path.exists():
        g=pd.read_csv(loco_path);g['conforms']=g.conforms.astype(int)
        save(pd.DataFrame([{'dataset':'QUANT','method':'fully_retrained_LOCO',**point(g),**bootstrap(g)}]),REV/'results/loco_summary.csv')
    print(pd.DataFrame(ranking).query("method=='full' and budget=='pooled'").to_string(index=False),flush=True)
    print('Matched zero-budget policies',flush=True)
    print(pd.DataFrame(policies).query("budget==0 and method in ['full','fixed_epa','source_fixed','variant_prior']")[['dataset','method','n','conforming_products','prevalence','top20_precision']].to_string(index=False),flush=True)

if __name__=='__main__':run()
