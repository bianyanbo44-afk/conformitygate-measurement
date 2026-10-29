"""Reference-independent daily inputs and sensor-defined deployment windows."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
REV=ROOT/'revision_20261029'
sys.path.insert(0,str(ROOT/'src'))
from prepare_datasets import build_quant_hourly,build_india_hourly
from prepare_epa_us_extension import aggregate_daily
from build_window_manifest import make_windows

F=['cf1','atm','rh','temperature_c']
G=['dataset','site','device']
CONFIG=json.loads((ROOT/'config/study_config.json').read_text())

def save(frame,path):
    frame.to_csv(path,index=False,compression={'method':'gzip','mtime':0} if str(path).endswith('.gz') else None)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sensor_daily(hourly):
    clean=hourly.dropna(subset=F).copy()
    clean['date']=clean.time.dt.floor('D')
    aggregations={c:(c,'mean') for c in F+['cf1_channel_gap','atm_channel_gap']}
    aggregations['valid_hours']=('time','nunique')
    for c in ['cf1_a','cf1_b']:
        if c in clean: aggregations[c]=(c,'mean')
    result=clean.groupby(G+['date'],as_index=False).agg(**aggregations)
    return result[result.valid_hours.ge(18)].reset_index(drop=True)

def ref_daily(hourly):
    clean=hourly.dropna(subset=['reference_pm25']).copy()
    clean['date']=clean.time.dt.floor('D')
    out=clean.groupby(G+['date'],as_index=False).agg(reference_pm25=('reference_pm25','mean'),reference_hours=('time','nunique'))
    return out[out.reference_hours.ge(18)]

def build():
    (REV/'data').mkdir(exist_ok=True)
    hourly_path=REV/'data/sensor_hourly.csv.gz'
    if hourly_path.exists():
        hourly=pd.read_csv(hourly_path,parse_dates=['time'])
    else:
        print('Reading raw QUANT and India without reference filtering',flush=True)
        q=build_quant_hourly(CONFIG,reference_join='left')
        i=build_india_hourly(CONFIG)
        hourly=pd.concat([q,i],ignore_index=True)
        save(hourly,hourly_path)
    sensor=sensor_daily(hourly)
    # UK reference means include all native reference hours, also hours without sensor records.
    raw_ref=pd.read_csv(ROOT/CONFIG['raw_data']['quant_reference'],parse_dates=['time'])
    raw_ref=raw_ref[raw_ref.measurand.eq('PM2.5') & raw_ref.version.eq('Ratified')].dropna(subset=['measurement']).copy()
    raw_ref['date']=raw_ref.time.dt.floor('D')
    qr=raw_ref.groupby(['location','date'],as_index=False).agg(reference_pm25=('measurement','mean'),reference_hours=('time','nunique'))
    qr=qr[qr.reference_hours.ge(18)].rename(columns={'location':'site'})
    qs=sensor[sensor.dataset.eq('QUANT')].merge(qr,on=['site','date'],how='left',validate='many_to_one')
    ir=ref_daily(hourly[hourly.dataset.eq('CampmierIndia')])
    ids=sensor[sensor.dataset.eq('CampmierIndia')].merge(ir,on=G+['date'],how='left',validate='one_to_one')
    uspath=ROOT/'data/raw/epa_us/Data_DevelopmentUSPAcorrection_210408/Full24hrdataset.csv'
    raw_us=pd.read_csv(uspath,low_memory=False)
    raw_us['Date']=pd.to_datetime(raw_us.Date).dt.normalize()
    us=aggregate_daily(raw_us,'mean_all_regulatory',require_reference=False)
    us['reference_hours']=np.nan
    daily=pd.concat([qs,ids,us],ignore_index=True).sort_values(G+['date']).reset_index(drop=True)
    save(daily,REV/'data/daily.csv.gz')
    manifests=[]
    for dataset,group in daily.groupby('dataset',sort=True):
        m=make_windows(group[G+['date']],CONFIG,'revision_sensor_primary',30,True)
        m['role']='development' if dataset=='QUANT' else 'external_test'
        m['block_id']=[f'{dataset}_{j:04d}' for j in range(len(m))]
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
    manifest=pd.concat(manifests,ignore_index=True)
    save(manifest,REV/'data/windows.csv')
    flow=[]
    for dataset,g in daily.groupby('dataset',sort=True):
        m=manifest[manifest.dataset.eq(dataset)]
        for evaluable,part in m.groupby('evaluable'):
            row={'dataset':dataset,'evaluable':bool(evaluable),'blocks':len(part),'groups':len(part[['site','device']].drop_duplicates()),'sites':part.site.nunique(),'devices':part.device.nunique()}
            row.update({f'eligible_b{b}':int(part[f'eligible_b{b}'].sum()) for b in [0,1,3,7]})
            row['candidates']=6*row['eligible_b0']+12*sum(row[f'eligible_b{b}'] for b in [1,3,7])
            flow.append(row)
    save(pd.DataFrame(flow),REV/'results/data_flow.csv')
    # Native A/B daily QC; no reference values enter this flag.
    daily['ab_qc_fail']=False
    available=daily[['cf1_a','cf1_b']].notna().all(axis=1)
    diff=(daily.cf1_a-daily.cf1_b).abs()
    relative=2*diff/(daily.cf1_a+daily.cf1_b).abs().replace(0,np.nan)
    daily.loc[available,'ab_qc_fail']=(diff.gt(5)&relative.gt(.61))[available]
    save(daily,REV/'data/daily.csv.gz')
    qc=daily.groupby('dataset').agg(days=('date','size'),failed=('ab_qc_fail','sum'))
    qc.to_csv(REV/'results/ab_qc_flow.csv')
    print(json.dumps(flow,indent=2),flush=True)

if __name__=='__main__':
    build()
