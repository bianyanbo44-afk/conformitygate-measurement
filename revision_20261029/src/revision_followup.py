"""Second-round reviewer analyses on saved scores (no model refitting).

1. Measured-error rankings (Reviewer 2, Comment 1): rank candidates by the measured
   assessment RMSE or absolute bias. These need the hidden reference, so they are an
   oracle reference point, not a deployable comparator.
2. Label without R2 (Reviewer 4, Comment 3): frozen-score AUROC when the composite
   label uses slope, intercept, and error only.
3. Concentration dependence (Reviewer 4, Comment 3): R2 versus within-block reference
   variability, and frozen-score AUROC in low- and high-variability blocks.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from revision_data import REV,save
from revision_statistics import metric_arrays,clusters,SEED,NBOOT

T={'slope_min':.65,'slope_max':1.35,'intercept':5.,'r2':.70,'rmse':7.,'nrmse':30.}

def draws(frame):
    _,ci=np.unique(clusters(frame),return_inverse=True);k=ci.max()+1
    return ci,np.random.default_rng(SEED).multinomial(k,np.repeat(1/k,k),size=NBOOT)

def auc_ci(frame,y,s):
    ci,d=draws(frame)
    values=np.array([metric_arrays(y,s,w[ci])[0] for w in d])
    return metric_arrays(y,s)[0],*np.nanquantile(values,[.025,.975])

def paired_ci(frame,y,a,b):
    ci,d=draws(frame)
    values=np.array([metric_arrays(y,a,w[ci])[0]-metric_arrays(y,b,w[ci])[0] for w in d])
    return metric_arrays(y,a)[0]-metric_arrays(y,b)[0],*np.nanquantile(values,[.025,.975])

def label_without_r2(f):
    error=(f.rmse<T['rmse'])|(f.nrmse_percent<T['nrmse'])
    return (f.slope.between(T['slope_min'],T['slope_max'])&f.intercept.abs().le(T['intercept'])&error).astype(int)

def run():
    scores=pd.concat([pd.read_csv(REV/'results/main_uk_scores.csv.gz'),pd.read_csv(REV/'results/main_external_scores.csv.gz')],ignore_index=True)
    full=scores[scores.method.eq('full')&scores.evaluable].copy()
    conc=pd.read_csv(REV/'results/concentration_ranges.csv')[['block_id','reference_sd','reference_mean','reference_q05','reference_q95']]
    oracle=[];labels=[];strata=[];r2rows=[]
    for dataset,f in full.groupby('dataset'):
        f=f.sort_values(['block_id','model','reference_budget_days','adaptation']).reset_index(drop=True)
        y=f.conforms.astype(int).to_numpy();score=f.score.to_numpy(float)
        for name,s in [('measured_rmse',-f.rmse.to_numpy(float)),('measured_abs_bias',-f.bias.abs().to_numpy(float))]:
            auc,lo,hi=auc_ci(f,y,s);d,dlo,dhi=paired_ci(f,y,score,s)
            oracle.append({'dataset':dataset,'comparator':name,'n':len(f),'comparator_auroc':auc,'comparator_lo':lo,'comparator_hi':hi,
                           'score_minus_comparator':d,'difference_lo':dlo,'difference_hi':dhi})
        y2=label_without_r2(f).to_numpy()
        auc,lo,hi=auc_ci(f,y2,score)
        labels.append({'dataset':dataset,'label':'without_r2','n':len(f),'prevalence':y2.mean(),'changed_fraction':float((y2!=y).mean()),'auroc':auc,'auroc_lo':lo,'auroc_hi':hi})
        auc,lo,hi=auc_ci(f,y,score)
        labels.append({'dataset':dataset,'label':'primary','n':len(f),'prevalence':y.mean(),'changed_fraction':0.,'auroc':auc,'auroc_lo':lo,'auroc_hi':hi})
        g=f.merge(conc,on='block_id',how='left',validate='many_to_one')
        cut=g.drop_duplicates('block_id').reference_sd.median()
        for stratum,part in [('low_variability',g[g.reference_sd.le(cut)]),('high_variability',g[g.reference_sd.gt(cut)])]:
            part=part.reset_index(drop=True);yy=part.conforms.astype(int).to_numpy()
            auc,lo,hi=auc_ci(part,yy,part.score.to_numpy(float)) if 0<yy.sum()<len(yy) else (np.nan,np.nan,np.nan)
            strata.append({'dataset':dataset,'stratum':stratum,'reference_sd_cut':cut,'blocks':part.block_id.nunique(),'n':len(part),'prevalence':yy.mean(),'auroc':auc,'auroc_lo':lo,'auroc_hi':hi})
        epa=g[g.model.eq('epa_correction')&g.reference_budget_days.eq(0)]
        rho,p=spearmanr(epa.reference_sd,epa.r2)
        r2rows.append({'dataset':dataset,'blocks':len(epa),'spearman_r2_vs_reference_sd':rho,'p_value':p,'median_reference_mean':epa.reference_mean.median(),'median_reference_sd':epa.reference_sd.median()})
        save(epa[['dataset','block_id','reference_mean','reference_sd','r2','slope','conforms']],REV/f'results/followup_epa_blocks_{dataset}.csv')
    save(pd.DataFrame(oracle),REV/'results/followup_measured_error.csv')
    save(pd.DataFrame(labels),REV/'results/followup_label_without_r2.csv')
    save(pd.DataFrame(strata),REV/'results/followup_concentration_strata.csv')
    save(pd.DataFrame(r2rows),REV/'results/followup_r2_concentration.csv')
    for name in ['measured_error','label_without_r2','concentration_strata','r2_concentration']:
        print(name);print(pd.read_csv(REV/f'results/followup_{name}.csv').round(3).to_string(index=False),flush=True)

if __name__=='__main__':run()
