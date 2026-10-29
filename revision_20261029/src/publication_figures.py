"""Five main figures and one supplementary diagnostic, from saved scores only.

Contract: distinguish transferred ranking, within-family information, and
deployment utility. All observed candidates in each declared stratum are used.
183 mm width; editable Arial text; 600-dpi lossless raster companions.
"""
from pathlib import Path
import sys,os,json,hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
from revision_data import REV
from revision_statistics import metric_arrays,clusters,SEED,NBOOT

if os.environ.get('NATURE_FIGURE_AUDIT_DIRECTORY'):
    sys.path.insert(0,os.environ['NATURE_FIGURE_AUDIT_DIRECTORY'])
    from audit_panel_alignment import require_matplotlib_panel_alignment
else:
    from figure_alignment import require_matplotlib_panel_alignment

OUT=REV/'manuscript/figures';OUT.mkdir(exist_ok=True)
RES=REV/'results'
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':7,'ytick.labelsize':7,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
    'legend.frameon':False,'text.color':'#222222','axes.labelcolor':'#222222',
    'xtick.color':'#222222','ytick.color':'#222222','savefig.facecolor':'white'})
TEAL='#177E89';GOLD='#B88542';INK='#555D64';PURPLE='#80729B';PALE='#F1F6F6'
DOMAINS=[('QUANT','UK'),('CampmierIndia','India'),('EPA_US_24h','U.S.')]
PIPELINES=[('raw_cf1','Raw CF1'),('epa_correction','EPA correction'),('linear','OLS'),
           ('ridge','Ridge'),('random_forest','Random forest'),('hist_gradient_boosting','HGB regression')]
STYLE=[('full','ConformityGate',TEAL,'-',1.7),('variant_prior','Variant prior',PURPLE,'--',1.05),
       ('predicted_margin','Predicted margin',GOLD,':',1.15),('logistic','Logistic',INK,'-.',1.0)]

def source(frame,name):
    path=RES/name
    if name.endswith('.gz'):frame.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
    else:frame.to_csv(path,index=False)

def export(fig,name,axes=None):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,json_out=OUT/(name+'.alignment.json'),
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=600,facecolor='white')
    fig.savefig(OUT/(name+'.tiff'),dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def framework():
    fig,axes=plt.subplots(2,1,figsize=(7.2,5.4))
    fig.subplots_adjust(left=.035,right=.985,bottom=.035,top=.89,hspace=.29)
    fig.text(.5,.97,'ConformityGate: supervised learning and frozen transfer',ha='center',fontsize=7,fontweight='bold')
    for ax in axes:ax.set(xlim=(0,11.5),ylim=(0,4));ax.axis('off')
    def box(ax,x,y,w,h,heading,body,reference=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.04',
            facecolor='#FBF5ED' if reference else PALE,edgecolor=GOLD if reference else TEAL,lw=.75))
        ax.text(x+w/2,y+h-.25,heading,ha='center',va='center',fontweight='bold',fontsize=7)
        ax.text(x+w/2,y+h/2-.14,body,ha='center',va='center',fontsize=6.7,linespacing=1.28)
    def arrow(ax,a,b,reference=False,elbow=False):
        kw={'connectionstyle':'angle,angleA=0,angleB=90,rad=0'} if elbow else {}
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.85,
            color=GOLD if reference else INK,**kw))
    ax=axes[0]
    ax.text(0,1.065,'a',transform=ax.transAxes,fontweight='bold',fontsize=8)
    ax.text(.035,1.065,'UK learning under outer group isolation',transform=ax.transAxes,fontweight='bold',fontsize=7)
    ax.text(.15,3.75,'Training fold only: every learned stage excludes held-out groups',fontsize=6.7)
    nodes=[('Sensor blocks','Sensor-only daily inputs\n30-day completed products'),
           ('Candidate generation','Six calibration pipelines\nShared prior audit, B = 0/1/3/7'),
           ('Feature representation','I: identity; E: context\nP: output summaries; audit'),
           ('HGB classifier','Composite-label supervision\nHeld-out UK scores')]
    for x,(head,body) in zip([.15,3.05,5.95,8.85],nodes):box(ax,x,1.95,2.5,1.4,head,body)
    for x in [2.65,5.55,8.45]:arrow(ax,(x,2.65),(x+.4,2.65))
    box(ax,.15,.25,2.5,1.05,'Calibration references','UK source fitting days\nExcluded target site/device',True)
    box(ax,5.95,.25,2.5,1.05,'Conformity labels','UK assessment references\nSupervision, never predictors',True)
    arrow(ax,(2.65,.78),(4.3,1.95),True,True)
    arrow(ax,(8.45,.78),(10.1,1.95),True,True)
    ax=axes[1]
    ax.text(0,1.065,'b',transform=ax.transAxes,fontweight='bold',fontsize=8)
    ax.text(.035,1.065,'Frozen evaluation and review in India and the U.S.',transform=ax.transAxes,fontweight='bold',fontsize=7)
    ax.text(.15,3.75,'Identical feature schema; imputation, scaling and classifier fitted in the UK',fontsize=6.7)
    nodes=[('Target sensor blocks','Same daily interface\nPermitted prior audits'),
           ('Source-fitted features','UK imputation and scaling\nI/E/P and audit features'),
           ('Frozen UK classifier','No target-label tuning\nOne score per candidate'),
           ('Review prioritization','Rank candidate products\nThen verify with references')]
    for x,(head,body) in zip([.15,3.05,5.95,8.85],nodes):box(ax,x,1.95,2.5,1.4,head,body)
    for x in [2.65,5.55,8.45]:arrow(ax,(x,2.65),(x+.4,2.65))
    box(ax,.15,.25,5.4,1.05,'Target assessment references','At least 23 paired days; hidden during scoring',True)
    box(ax,6.0,.25,5.35,1.05,'Evaluation and verification','AUROC; review precision; matched yield\nDocument reference-backed final products')
    arrow(ax,(5.55,.78),(6.0,.78),True)
    arrow(ax,(10.1,1.95),(10.1,1.3))
    export(fig,'figure1_workflow')

def transfer(ranking):
    keys=[('QUANT','full','UK\n200 blocks; 11 groups'),('CampmierIndia','full','India\n17 blocks; 3 site aggregates'),
          ('EPA_US_24h','full','U.S.\n231 blocks; 40 identifiers'),
          ('EPA_US_24h','noepa_frozen','U.S.: EPA candidates removed'),
          ('EPA_US_24h','noepa_refit','U.S.: EPA candidates and features removed')]
    fig,ax=plt.subplots(figsize=(7.2,3.5));fig.subplots_adjust(left=.44,right=.77,bottom=.19,top=.88)
    rows=[]
    for i,(ds,method,label) in enumerate(keys):
        row=ranking[ranking.dataset.eq(ds)&ranking.method.eq(method)&ranking.budget.eq('pooled')].iloc[0]
        ax.errorbar(row.auroc,i,xerr=[[row.auroc-row.auroc_lo],[row.auroc_hi-row.auroc]],fmt='o',
            color=TEAL if method=='full' else GOLD,capsize=2.5,ms=4.5,lw=1.05)
        ax.text(1.045,i,f'{row.auroc:.3f} [{row.auroc_lo:.3f}, {row.auroc_hi:.3f}]',
            transform=ax.get_yaxis_transform(),va='center',fontsize=7)
        rows.append({'dataset':ds,'method':method,'n':row.n,'blocks':row.blocks,'clusters':row.clusters,
                     'auroc':row.auroc,'lo':row.auroc_lo,'hi':row.auroc_hi})
    ax.set(yticks=range(len(keys)),yticklabels=[v[2] for v in keys],ylim=(4.65,-.7),xlim=(.45,1.0),
        xticks=[.5,.6,.7,.8,.9,1.0],xlabel='Candidate-level AUROC')
    ax.tick_params(axis='y',length=0,pad=9)
    ax.text(1.045,1.04,'AUROC [95% interval]',transform=ax.transAxes,fontsize=7,fontweight='bold')
    ax.axvline(.5,color='#A7ADB1',ls='--',lw=.7)
    source(pd.DataFrame(rows),'figure2_source.csv');export(fig,'figure2_ranking')

def review():
    selected=pd.read_csv(RES/'selected_products.csv.gz');selected=selected[selected.reference_budget_days.eq(0)]
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.85),sharex=True,sharey=True)
    fig.subplots_adjust(left=.085,right=.985,bottom=.29,top=.80,wspace=.22)
    rows=[]
    for i,(ds,title) in enumerate(DOMAINS):
        ax=axes[i];n=int(selected[selected.dataset.eq(ds)&selected.method.eq('full')].block_id.nunique())
        for method,label,color,style in [('full','ConformityGate',TEAL,'-'),('fixed_epa','Fixed EPA',INK,'--'),('variant_prior','Variant prior',PURPLE,':')]:
            g=selected[selected.dataset.eq(ds)&selected.method.eq(method)]
            assert len(g)==n
            counts=np.arange(1,n+1);fractions=counts/n
            precision=np.array([metric_arrays(g.conforms,g.score,q=float(k/n-1e-12))[1] for k in counts])
            ax.step(100*fractions,100*precision,where='post',label=label,color=color,ls=style,lw=1.6 if method=='full' else 1.0)
            rows.extend({'dataset':ds,'method':method,'reviewed_deployments':int(k),'review_fraction':q,'precision':v} for k,q,v in zip(counts,fractions,precision))
            if method=='full':
                k=int(np.ceil(.2*n));ax.plot(100*k/n,100*precision[k-1],'o',color=TEAL,ms=3.5,zorder=5)
        actual20=100*np.ceil(.2*n)/n
        ax.axvline(actual20,color='#C5C9CC',lw=.65,ls=':')
        ax.set_title(f'{title}\n{n} deployments',fontweight='bold',pad=11)
        ax.set(xlim=(0,100),ylim=(0,105),xticks=[0,50,100],yticks=[0,25,50,75,100])
        ax.text(-.12,1.16,'abc'[i],transform=ax.transAxes,fontweight='bold',fontsize=8)
    axes[0].set_ylabel('Conformity precision (%)')
    fig.supxlabel('Deployments prioritized for review (%)',y=.13,fontsize=7)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.005),fontsize=7)
    source(pd.DataFrame(rows),'figure3_source.csv');export(fig,'figure3_review_policies')

def ablations(ranking):
    methods=['I','E','P','IE','IP','EP','IEP','full','without_month','without_missing']
    names=['Identity (I)','Sensor / context (E)','Prediction summaries (P)','I + E','I + P','E + P','I + E + P',
           'Full, including audit','Full, without month','Full, common features']
    values=np.array([[ranking[ranking.dataset.eq(ds)&ranking.method.eq(method)&ranking.budget.eq('pooled')].auroc.iloc[0]
                     for ds,_ in DOMAINS] for method in methods])
    fig,ax=plt.subplots(figsize=(7.2,4.0));fig.subplots_adjust(left=.34,right=.86,bottom=.10,top=.94)
    image=ax.imshow(values,vmin=0,vmax=1,cmap='cividis',aspect='auto',interpolation='nearest')
    ax.set(xticks=range(3),xticklabels=[v[1] for v in DOMAINS],yticks=range(len(methods)),yticklabels=names)
    ax.tick_params(length=0,pad=7)
    for i in range(len(methods)):
        for j in range(3):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if values[i,j]<.60 else '#222222')
    cax=fig.add_axes([.90,.22,.018,.61]);bar=fig.colorbar(image,cax=cax,ticks=[0,.5,1]);bar.set_label('AUROC',fontsize=7)
    source(pd.DataFrame([{'dataset':ds,'feature_group':method,'auroc':values[i,j]} for i,method in enumerate(methods) for j,(ds,_) in enumerate(DOMAINS)]),'figure4_source.csv')
    export(fig,'figure4_ablations',axes=[ax])

def stratum_interval(g):
    y=g.conforms.to_numpy(int);s=g.score.to_numpy(float)
    group=clusters(g);_,ci=np.unique(group,return_inverse=True);k=int(ci.max()+1)
    out={'n':len(g),'blocks':g.block_id.nunique(),'clusters':k,'positive_candidates':int(y.sum()),'negative_candidates':int((1-y).sum()),
         'positive_clusters':group[y==1].nunique(),'negative_clusters':group[y==0].nunique()}
    if np.unique(y).size<2:return {**out,'auroc':np.nan,'lo':np.nan,'hi':np.nan,'valid_bootstrap_draws':0}
    _,si=np.unique(s,return_inverse=True)
    # Bootstrap multiplicities resample the observed clusters; no synthetic observations.
    draws=np.random.default_rng(SEED).multinomial(k,np.repeat(1/k,k),size=NBOOT)
    values=[]
    for draw in draws:
        w=draw[ci];p=np.bincount(si,weights=w*y);n=np.bincount(si,weights=w*(1-y));P=p.sum();N=n.sum()
        if P and N:values.append(np.sum(p*(np.cumsum(n)-n/2))/(P*N))
    lo,hi=np.quantile(values,[.025,.975])
    return {**out,'auroc':roc_auc_score(y,s),'lo':float(lo),'hi':float(hi),'valid_bootstrap_draws':len(values)}

def within_pipeline(scores):
    g=scores[scores.method.eq('full')].copy();rows=[]
    existing=pd.read_csv(RES/'within_pipeline.csv')
    for ds,_ in DOMAINS:
        for model,_ in PIPELINES:
            part=g[g.dataset.eq(ds)&g.model.eq(model)]
            row={'dataset':ds,'model':model,**stratum_interval(part)}
            row['interval_displayed']=row['positive_clusters']>1 and row['negative_clusters']>1
            check=existing[existing.dataset.eq(ds)&existing.model.eq(model)&existing.budget.eq('pooled')].iloc[0]
            assert row['n']==check.n and np.isclose(row['auroc'],check.auroc,equal_nan=True)
            rows.append(row)
    table=pd.DataFrame(rows);source(table,'figure5_source.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.8),sharey=True)
    fig.subplots_adjust(left=.16,right=.88,bottom=.22,top=.78,wspace=.80)
    for j,(ds,title) in enumerate(DOMAINS):
        ax=axes[j];part=table[table.dataset.eq(ds)].set_index('model')
        ax.set(xlim=(-.04,1.07),ylim=(5.65,-.75),xticks=[0,.5,1],yticks=range(6))
        ax.axvline(.5,color='#A7ADB1',ls='--',lw=.65)
        for i,(model,_) in enumerate(PIPELINES):
            row=part.loc[model]
            if np.isfinite(row.auroc) and row.interval_displayed:
                ax.errorbar(row.auroc,i,xerr=[[row.auroc-row.lo],[row.hi-row.auroc]],fmt='o',color=TEAL,ms=3.8,lw=.9,capsize=2)
            elif np.isfinite(row.auroc):
                ax.plot(row.auroc,i,'o',ms=3.8,markerfacecolor='white',markeredgecolor=TEAL,markeredgewidth=1.05)
            else:ax.text(.03,i,'Undefined',ha='left',va='center',fontsize=6.7)
            ax.text(1.04,i,f'{int(row.positive_candidates):,} / {int(row.negative_candidates):,}',transform=ax.get_yaxis_transform(),va='center',fontsize=6.7)
        ax.set_title(title,fontweight='bold',pad=27)
        ax.text(.5,1.10,f'{int(part.iloc[0].blocks)} blocks',transform=ax.transAxes,ha='center',fontsize=7)
        ax.text(1.04,1.04,'n+ / n−',transform=ax.transAxes,fontweight='bold',fontsize=6.7)
        ax.text(-.15,1.24,'abc'[j],transform=ax.transAxes,fontweight='bold',fontsize=8)
        ax.tick_params(axis='y',length=0,pad=7)
        ax.set_xlabel('Within-pipeline AUROC',labelpad=7)
    axes[0].set_yticklabels([label for _,label in PIPELINES])
    fig.text(.16,.075,'Filled points: AUROC with cluster-bootstrap 95% intervals.',fontsize=6.7)
    fig.text(.16,.035,'Open points: one positive cluster; interval omitted. Counts: candidate variants.',fontsize=6.7)
    export(fig,'figure5_within_pipeline')

def discrimination_curves(scores,ranking):
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.0))
    fig.subplots_adjust(left=.085,right=.985,bottom=.16,top=.89,wspace=.25,hspace=.56)
    rows=[];metricrows=[]
    for j,(ds,title) in enumerate(DOMAINS):
        data=scores[scores.dataset.eq(ds)];primary=data[data.method.eq('full')]
        n=len(primary);prevalence=primary.conforms.mean()
        axes[0,j].set_title(f'{title}: {n:,} candidates\n{primary.block_id.nunique()} blocks',fontweight='bold',pad=11)
        for method,label,color,style,width in STYLE:
            part=data[data.method.eq(method)];assert len(part)==n
            y=part.conforms.to_numpy(int);s=part.score.to_numpy(float)
            fpr,tpr,thresholds=roc_curve(y,s,drop_intermediate=False)
            precision,recall,pr_thresholds=precision_recall_curve(y,s)
            auc=roc_auc_score(y,s);ap=average_precision_score(y,s)
            r=ranking[ranking.dataset.eq(ds)&ranking.method.eq(method)&ranking.budget.eq('pooled')].iloc[0]
            assert np.isclose(auc,r.auroc) and np.isclose(ap,r.ap)
            axes[0,j].plot(fpr,tpr,color=color,ls=style,lw=width,label=label)
            axes[1,j].step(recall,precision,where='post',color=color,ls=style,lw=width,label=label)
            rows.extend({'dataset':ds,'method':method,'curve':'ROC','x':x,'y':v,'threshold':t} for x,v,t in zip(fpr,tpr,thresholds))
            rows.extend({'dataset':ds,'method':method,'curve':'PR','x':x,'y':v,'threshold':t} for x,v,t in zip(recall,precision,np.r_[pr_thresholds,np.nan]))
            metricrows.append({'dataset':ds,'method':method,'n':n,'blocks':primary.block_id.nunique(),'prevalence':prevalence,'auroc':auc,'ap':ap})
        axes[0,j].plot([0,1],[0,1],color='#B9BEC2',ls=':',lw=.65,zorder=0)
        axes[1,j].axhline(prevalence,color='#B9BEC2',ls=':',lw=.65,zorder=0)
        axes[0,j].set_xlabel('False positive rate',labelpad=5)
        axes[1,j].set_xlabel('Recall',labelpad=5)
        for i in range(2):
            ax=axes[i,j];ax.set(xlim=(0,1.015),ylim=(0,1.035),xticks=[0,.5,1],yticks=[0,.5,1])
            ax.text(-.13,1.12,'abcdef'[i*3+j],transform=ax.transAxes,fontweight='bold',fontsize=8)
    axes[0,0].set_ylabel('True positive rate',labelpad=5)
    axes[1,0].set_ylabel('Precision',labelpad=5)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.025),fontsize=7,columnspacing=1.5)
    source(pd.DataFrame(rows),'figureS1_curve_source.csv.gz')
    source(pd.DataFrame(metricrows),'figureS1_metrics_source.csv')
    export(fig,'figureS1_discrimination')

def run():
    ranking=pd.read_csv(RES/'ranking.csv')
    scores=pd.concat([pd.read_csv(RES/'main_uk_scores.csv.gz',low_memory=False),pd.read_csv(RES/'main_external_scores.csv.gz',low_memory=False)],ignore_index=True)
    assert scores.evaluable.all() and scores.conforms.notna().all()
    scores['conforms']=scores.conforms.astype(int)
    framework();transfer(ranking);review();ablations(ranking);within_pipeline(scores);discrimination_curves(scores,ranking)
    inputs=['ranking.csv','within_pipeline.csv','main_uk_scores.csv.gz','main_external_scores.csv.gz','selected_products.csv.gz']
    manifest={'inputs':{p:hashlib.sha256((RES/p).read_bytes()).hexdigest() for p in inputs},
        'source_files':[p.name for p in sorted(RES.glob('figure*_source.csv*'))],
        'scope':'Five main figures and supplementary ROC/PR; all declared strata/candidates; no model refitting',
        'bootstrap':{'draws':NBOOT,'seed':SEED,'clusters':'UK site-device; India blocks conditional on three site aggregates; U.S. identifiers'},
        'export':{'width_mm':182.88,'formats':['PDF','SVG','PNG','TIFF'],'raster_dpi':600,'font':'Arial','body_font_pt':'6.7–7','panel_font_pt':8}}
    (REV/'qa/figure_source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Exported five main and one supplementary figure sets, with source data.',flush=True)

if __name__=='__main__':run()
