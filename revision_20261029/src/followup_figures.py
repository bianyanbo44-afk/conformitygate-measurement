"""Figures 6 and 7 for the second reviewer round, from saved result tables only.

Figure 6: why a requirement-set target ranks better than error magnitude.
Figure 7: concentration conditions, R2 dependence, and label robustness.
Style, export, and alignment rules are shared with publication_figures.py.
"""
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
from publication_figures import RES,REV,TEAL,GOLD,INK,PURPLE,DOMAINS,source
from figure_alignment import require_matplotlib_panel_alignment

OUT=REV/'manuscript/figures'
mpl.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans']})

def export(fig,name,axes):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,json_out=OUT/(name+'.alignment.json'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=600,facecolor='white')
    fig.savefig(OUT/(name+'.tiff'),dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def plain_log(axis,ticks):
    axis.set_major_locator(FixedLocator(ticks));axis.set_major_formatter(FixedFormatter([f'{v:g}' for v in ticks]))
    axis.set_minor_locator(NullLocator())

COLOR={'QUANT':TEAL,'CampmierIndia':GOLD,'EPA_US_24h':PURPLE}
OFFSET={'QUANT':-.22,'CampmierIndia':0.,'EPA_US_24h':.22}

def legend(fig,y):
    handles=[Line2D([],[],marker='o',ls='',color=COLOR[d],ms=4.2,label=l) for d,l in DOMAINS]
    fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,y),fontsize=7,columnspacing=1.8,handletextpad=.4)

def error_targets():
    failures=pd.read_csv(RES/'criterion_failures.csv')
    paired=pd.read_csv(RES/'paired_auroc_contrasts.csv')
    measured=pd.read_csv(RES/'followup_measured_error.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3))
    fig.subplots_adjust(left=.2,right=.985,bottom=.33,top=.87,wspace=.66)
    ax=axes[0];criteria=[('slope','Slope'),('intercept','Intercept'),('r2','R²'),('error','RMSE or NRMSE')]
    rows=[]
    for i,(c,label) in enumerate(criteria):
        for ds,_ in DOMAINS:
            v=100*failures[failures.dataset.eq(ds)&failures.criterion.eq(c)].failure_rate.iloc[0]
            ax.barh(i+OFFSET[ds],v,height=.21,color=COLOR[ds],edgecolor='none')
            ax.text(v+1.5,i+OFFSET[ds],f'{v:.0f}',va='center',fontsize=6.5)
            rows.append({'panel':'a','dataset':ds,'item':c,'value':v,'lo':np.nan,'hi':np.nan})
    ax.set(yticks=range(4),yticklabels=[l for _,l in criteria],ylim=(3.6,-.6),xlim=(0,100),xticks=[0,25,50,75,100],
           xlabel='Candidates failing the criterion (%)')
    ax.tick_params(axis='y',length=0)
    ax.text(-.36,1.08,'a',transform=ax.transAxes,fontweight='bold',fontsize=8)
    ax.text(-.30,1.08,'Most frequent failure is slope, not error',transform=ax.transAxes,fontweight='bold',fontsize=7)
    ax=axes[1]
    comps=[('predicted_rmse','Predicted RMSE'),('predicted_bias','Predicted |bias|'),('predicted_margin','Predicted margin'),
           ('measured_rmse','Measured RMSE*'),('measured_abs_bias','Measured |bias|*')]
    ax.axvline(0,color='#A7ADB1',ls='--',lw=.7)
    for i,(c,label) in enumerate(comps):
        for ds,_ in DOMAINS:
            if c.startswith('predicted'):
                r=paired[paired.dataset.eq(ds)&paired.comparator.eq(c)].iloc[0];d,lo,hi=r.difference,r.lo,r.hi
            else:
                r=measured[measured.dataset.eq(ds)&measured.comparator.eq(c)].iloc[0];d,lo,hi=r.score_minus_comparator,r.difference_lo,r.difference_hi
            face=COLOR[ds] if c.startswith('predicted') else 'white'
            ax.errorbar(d,i+OFFSET[ds],xerr=[[d-lo],[hi-d]],fmt='o',color=COLOR[ds],markerfacecolor=face,
                        markeredgewidth=1.0,ms=3.8,lw=.9,capsize=1.8)
            rows.append({'panel':'b','dataset':ds,'item':c,'value':d,'lo':lo,'hi':hi})
    ax.set(yticks=range(5),yticklabels=[l for _,l in comps],ylim=(4.6,-.6),xlim=(-.32,.62),xticks=[-.2,0,.2,.4,.6],
           xlabel='AUROC advantage of ConformityGate')
    ax.tick_params(axis='y',length=0)
    ax.text(-.42,1.08,'b',transform=ax.transAxes,fontweight='bold',fontsize=8)
    ax.text(-.36,1.08,'Score versus error-based rankings',transform=ax.transAxes,fontweight='bold',fontsize=7)
    legend(fig,.005)
    fig.text(.2,.135,'Bars and points: domain-specific values; intervals: paired cluster-bootstrap 95% intervals.',fontsize=6.5)
    fig.text(.2,.095,'*Measured errors require the hidden assessment reference (oracle), shown as open points.',fontsize=6.5)
    source(pd.DataFrame(rows),'figure6_source.csv');export(fig,'figure6_error_targets',axes=list(axes))

def label_robustness():
    conc=pd.read_csv(RES/'concentration_ranges.csv');conc=conc[conc.evaluable]
    labels=pd.read_csv(RES/'followup_label_without_r2.csv');strata=pd.read_csv(RES/'followup_concentration_strata.csv')
    rho=pd.read_csv(RES/'followup_r2_concentration.csv');perturb=pd.read_csv(RES/'threshold_perturbations.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.2))
    fig.subplots_adjust(left=.075,right=.985,bottom=.34,top=.86,wspace=.62)
    rows=[]
    ax=axes[0]
    for j,(ds,label) in enumerate(DOMAINS):
        v=np.sort(conc[conc.dataset.eq(ds)].reference_mean.to_numpy());assert (v>0).all()
        ax.boxplot(v,positions=[j],widths=.5,showfliers=False,patch_artist=True,
                   boxprops={'facecolor':'white','edgecolor':COLOR[ds],'lw':.8},medianprops={'color':COLOR[ds],'lw':1.1},
                   whiskerprops={'color':COLOR[ds],'lw':.8},capprops={'color':COLOR[ds],'lw':.8})
        x=j+np.linspace(-.17,.17,len(v)) if len(v)>1 else np.array([j])
        ax.plot(x,v,'o',ms=1.6,color=COLOR[ds],alpha=.55,mec='none')
        rows.append({'panel':'a','dataset':ds,'item':'median_block_reference_mean','value':float(np.median(v)),'lo':float(v.min()),'hi':float(v.max())})
    ax.set_yscale('log');plain_log(ax.yaxis,[3,10,30,100]);ax.set(xticks=range(3),xticklabels=[l for _,l in DOMAINS],xlim=(-.6,2.6),
                               ylabel='Block mean reference PM2.5 (µg/m³)')
    ax.text(-.30,1.08,'a',transform=ax.transAxes,fontweight='bold',fontsize=8)
    ax.text(-.20,1.08,'Concentration levels',transform=ax.transAxes,fontweight='bold',fontsize=7)
    ax=axes[1]
    for ds,label in DOMAINS:
        e=pd.read_csv(RES/f'followup_epa_blocks_{ds}.csv');assert (e.reference_sd>0).all()
        ax.plot(e.reference_sd,e.r2,'o',ms=2.2,color=COLOR[ds],alpha=.7,mec='none')
    ax.axhline(.70,color='#A7ADB1',ls='--',lw=.7)
    ax.set_xscale('log');plain_log(ax.xaxis,[1,3,10,30]);ax.set(ylim=(-.02,1.02),yticks=[0,.35,.7,1.0],xlabel='Within-block reference SD (µg/m³)',ylabel='R² of zero-budget EPA product')
    for ds,label in DOMAINS:
        r=rho[rho.dataset.eq(ds)].iloc[0]
        rows.append({'panel':'b','dataset':ds,'item':'spearman_r2_vs_reference_sd','value':r.spearman_r2_vs_reference_sd,'lo':np.nan,'hi':np.nan})
    ax.text(-.30,1.08,'b',transform=ax.transAxes,fontweight='bold',fontsize=8)
    ax.text(-.20,1.08,'R² follows concentration spread',transform=ax.transAxes,fontweight='bold',fontsize=7)
    ax=axes[2]
    conds=[('primary','Primary label'),('without_r2','Label without R²'),('low_variability','Low-spread blocks'),
           ('high_variability','High-spread blocks'),('perturbation','Limit perturbations')]
    ax.axvline(.5,color='#A7ADB1',ls='--',lw=.65)
    for i,(c,label) in enumerate(conds):
        for ds,_ in DOMAINS:
            y=i+OFFSET[ds]
            if c=='perturbation':
                p=perturb[perturb.dataset.eq(ds)].auroc;lo,hi=p.min(),p.max()
                ax.plot([lo,hi],[y,y],color=COLOR[ds],lw=2.2,solid_capstyle='butt');d=np.nan
            else:
                src=labels[labels.dataset.eq(ds)&labels.label.eq(c)] if c in ['primary','without_r2'] else strata[strata.dataset.eq(ds)&strata.stratum.eq(c)]
                r=src.iloc[0];d,lo,hi=r.auroc,r.auroc_lo,r.auroc_hi
                ax.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt='o',color=COLOR[ds],ms=3.6,lw=.9,capsize=1.8)
            rows.append({'panel':'c','dataset':ds,'item':c,'value':d,'lo':lo,'hi':hi})
    ax.set(yticks=range(5),yticklabels=[l for _,l in conds],ylim=(4.6,-.6),xlim=(.45,1.0),xticks=[.5,.75,1.0],xlabel='Candidate-level AUROC')
    ax.tick_params(axis='y',length=0)
    ax.text(-.62,1.08,'c',transform=ax.transAxes,fontweight='bold',fontsize=8)
    ax.text(-.52,1.08,'Ranking is label-robust',transform=ax.transAxes,fontweight='bold',fontsize=7)
    legend(fig,.005)
    fig.text(.075,.145,'a, evaluable blocks; b, Spearman ρ = '+', '.join(f'{rho[rho.dataset.eq(d)].iloc[0].spearman_r2_vs_reference_sd:.2f} ({l})' for d,l in DOMAINS)+'; dashed line, R² limit; c, cluster-bootstrap 95% intervals.',fontsize=6.5)
    fig.text(.075,.105,'Spread is the within-block reference SD split at each domain median; bars span all 13 one-at-a-time and joint limit settings.',fontsize=6.5)
    source(pd.DataFrame(rows),'figure7_source.csv');export(fig,'figure7_label_robustness',axes=list(axes))

def run():
    error_targets();label_robustness()
    inputs=['criterion_failures.csv','paired_auroc_contrasts.csv','followup_measured_error.csv','concentration_ranges.csv',
            'followup_label_without_r2.csv','followup_concentration_strata.csv','followup_r2_concentration.csv','threshold_perturbations.csv']
    (REV/'qa/followup_figure_manifest.json').write_text(json.dumps({'inputs':{p:hashlib.sha256((RES/p).read_bytes()).hexdigest() for p in inputs},
        'figures':['figure6_error_targets','figure7_label_robustness'],'model_refitting':False},indent=2),encoding='utf-8')
    print('Exported Figures 6 and 7 with source data.',flush=True)

if __name__=='__main__':run()
