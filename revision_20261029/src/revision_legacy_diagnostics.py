"""Explain the submitted logistic result without reversing target-domain scores."""
import pandas as pd
import numpy as np
import joblib,json
from sklearn.metrics import roc_auc_score
from revision_data import ROOT,REV,save

def run():
    f=pd.read_csv(ROOT/'data/processed/gate_features.csv.gz')
    f=f[f.protocol.eq('primary')].copy()
    labs=pd.concat([pd.read_csv(ROOT/'data/private_labels/gate_labels_development.csv.gz'),pd.read_csv(ROOT/'data/private_labels/gate_labels_external_locked.csv.gz')])
    keys=['block_id','protocol','role','dataset','site','device','model','reference_budget_days','adaptation']
    f=f.merge(labs[keys+['conforms']],on=keys,validate='one_to_one')
    est=joblib.load(ROOT/'results/frozen_models/logistic_gate.joblib')
    cols=list(est.feature_names_in_);names=est.named_steps['preprocess'].get_feature_names_out()
    coefficients=est.named_steps['classifier'].coef_[0]
    assert np.array_equal(est.classes_,[False,True]) or np.array_equal(est.classes_,[0,1])
    contribution=[];metrics=[]
    for ds,g in f.groupby('dataset'):
        y=g.conforms.astype(int).to_numpy();score=est.predict_proba(g[cols])[:,1]
        X=est.named_steps['preprocess'].transform(g[cols])
        diff=(X[y==1].mean(axis=0)-X[y==0].mean(axis=0))*coefficients
        for name,c in zip(names,diff):contribution.append({'dataset':ds,'feature':name,'mean_positive_minus_negative_logit_contribution':c})
        metrics.append({'dataset':ds,'auroc':roc_auc_score(y,score),'positive_class_index':1,'mean_score_conforming':score[y==1].mean(),'mean_score_nonconforming':score[y==0].mean(),'mean_logit_difference':diff.sum()})
    save(pd.DataFrame(metrics),REV/'results/submitted_logistic_reproduction.csv')
    save(pd.DataFrame(contribution),REV/'results/submitted_logistic_contributions.csv')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print(pd.DataFrame(contribution).query("dataset=='CampmierIndia'").sort_values('mean_positive_minus_negative_logit_contribution').head(6).to_string(index=False),flush=True)

if __name__=='__main__':run()
