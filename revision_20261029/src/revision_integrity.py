"""Independent checks of isolation, metrics, budgets, and reference-mask invariance."""
import json,sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from revision_data import REV,ROOT,G,F,save
from revision_engine import build_table,feature_columns,KEY,groupkey
from revision_statistics import metric_arrays,point,bootstrap
from run_gate_prototype import calibrate_decisions
from revision_data import CONFIG

def run():
    daily=pd.read_csv(REV/'data/daily.csv.gz',parse_dates=['date']);windows=pd.read_csv(REV/'data/windows.csv')
    predictions=pd.concat([pd.read_csv(REV/'data/uk_oof_daily_predictions.csv.gz',parse_dates=['date']),pd.read_csv(REV/'data/external_predictions.csv.gz',parse_dates=['date'])])
    source=daily[daily.dataset.eq('QUANT')]
    checks={}
    for dataset in daily.dataset.unique():
        w=windows[windows.dataset.eq(dataset)&windows.eligible_b7&windows.evaluable].iloc[[0]]
        original=build_table(daily,w,predictions,source)
        altered=daily.copy();mask=altered.dataset.eq(dataset)&altered.site.eq(w.site.iloc[0])&altered.device.eq(w.device.iloc[0])&altered.date.between(w.evaluation_start.iloc[0],w.evaluation_end.iloc[0])
        altered.loc[mask,'reference_pm25']=np.nan
        hidden=build_table(altered,w,predictions,source)
        cols=feature_columns(original)
        pd.testing.assert_frame_equal(original[KEY+cols].reset_index(drop=True),hidden[KEY+cols].reset_index(drop=True),check_dtype=False)
        checks[f'{dataset}_all_budget_reference_blinding']=True
    rng=np.random.default_rng(9)
    for n in [17,200,1000]:
        y=rng.integers(0,2,n);s=rng.integers(0,20,n)/20;w=rng.integers(0,4,n)
        assert abs(metric_arrays(y,s,w)[0]-roc_auc_score(y,s,sample_weight=w))<1e-12
    assert metric_arrays([0,1,1,0],[1,1,0,0],q=.25)[1]==.5
    checks['weighted_auc_matches_sklearn_with_ties']=True;checks['review_budget_averages_boundary_ties']=True
    full=pd.concat([joblib.load(REV/'results/cache/full_uk.joblib'),joblib.load(REV/'results/cache/external.joblib')])
    m=full.adaptation.eq('shrunk_offset')
    expected=full.loc[m,'audit_residual_mean']*full.loc[m,'reference_budget_days']/(full.loc[m,'reference_budget_days']+3)
    assert np.allclose(full.loc[m,'applied_audit_offset'],expected)
    checks['all_shrinkage_offsets_match_formula']=True
    assert not set(feature_columns(full)).intersection({'reference_pm25','paired_valid_days','valid_days','conforms','slope','rmse','r2','intercept','bias','violation'})
    checks['reference_fields_excluded_from_predictor_schema']=True
    flow=pd.read_csv(REV/'results/data_flow.csv')
    scores=pd.concat([pd.read_csv(REV/'results/main_uk_scores.csv.gz'),pd.read_csv(REV/'results/main_external_scores.csv.gz')])
    for ds,g in scores[scores.method.eq('full')&scores.evaluable].groupby('dataset'):
        assert len(g)==flow[flow.dataset.eq(ds)&flow.evaluable].candidates.iloc[0]
    checks['candidate_denominators_match_flow']=True
    # Source-native day flow, before reference-dependent evaluation filtering.
    hourly=pd.read_csv(REV/'data/sensor_hourly.csv.gz',parse_dates=['time'])
    hflow=[]
    for ds,g in hourly.groupby('dataset'):
        clean=g.dropna(subset=F).copy();clean['date']=clean.time.dt.floor('D')
        counts=clean.groupby(G+['date']).time.nunique()
        hflow.append({'dataset':ds,'hourly_rows':len(g),'complete_sensor_hours':len(clean),'observed_sensor_days':len(counts),'valid_18hour_days':int(counts.ge(18).sum()),'sensor_groups':len(g[G].drop_duplicates())})
    save(pd.DataFrame(hflow),REV/'results/native_data_flow.csv')
    loco=pd.read_csv(REV/'results/loco_uk_scores.csv.gz');loco['conforms']=loco.conforms.astype(int)
    save(pd.DataFrame([{'dataset':'QUANT','method':'fully_retrained_LOCO',**point(loco),**bootstrap(loco)}]),REV/'results/loco_summary.csv')
    # Store logistic coefficient orientation and class-conditional score distributions.
    estimator=joblib.load(REV/'results/models/logistic.joblib')
    assert np.array_equal(estimator.classes_,[0,1])
    names=estimator.named_steps['preprocess'].get_feature_names_out()
    save(pd.DataFrame({'feature':names,'coefficient':estimator.named_steps['classifier'].coef_[0]}),REV/'results/logistic_coefficients.csv')
    log=scores[scores.method.eq('logistic')&scores.evaluable]
    save(log.groupby(['dataset','conforms']).score.agg(['size','mean','median','min','max']).reset_index(),REV/'results/logistic_score_diagnostics.csv')
    checks['logistic_positive_class_index_correct']=True
    # Secondary source-calibrated release stress test on one zero-audit product per block.
    selected=pd.read_csv(REV/'results/selected_products.csv.gz')
    selected=selected[selected.method.eq('full')&selected.reference_budget_days.eq(0)].copy()
    selected['conforms']=selected.conforms.astype(int)
    uk_selected=selected[selected.dataset.eq('QUANT')]
    thresholds=calibrate_decisions(uk_selected.conforms.to_numpy(bool),uk_selected.score.to_numpy(float),uk_selected.block_id.to_numpy(),CONFIG,20261029)
    (REV/'results/release_threshold.json').write_text(json.dumps(thresholds,indent=2,default=lambda v:float(v)))
    release=[]
    t=thresholds['release'].get('threshold',np.inf)
    for ds,g in selected.groupby('dataset'):
        released=g[g.score.ge(t)]
        release.append({'dataset':ds,'products':len(g),'threshold':t,'released':len(released),'coverage':len(released)/len(g),'false_acceptance':1-released.conforms.mean() if len(released) else np.nan})
    save(pd.DataFrame(release),REV/'results/release_stress_test.csv')
    checks['release_threshold_uses_UK_only']=True
    (REV/'results/independent_integrity_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':run()
