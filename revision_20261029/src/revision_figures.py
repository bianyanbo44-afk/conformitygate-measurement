"""Figures: workflow, geographic ranking, operational review, and ablations.

Contract: reveal the distinction between ranking and selecting a final product.
Quantitative panels contain all fixed zero-budget deployments and all requested
feature combinations. Export at 183 mm with editable text and >=7 pt fonts.
"""
from pathlib import Path
import sys,os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from revision_data import REV
from revision_statistics import metric_arrays

if os.environ.get('NATURE_FIGURE_AUDIT_DIRECTORY'):
    sys.path.insert(0,os.environ['NATURE_FIGURE_AUDIT_DIRECTORY'])
    from audit_panel_alignment import require_matplotlib_panel_alignment
else:
    from figure_alignment import require_matplotlib_panel_alignment
OUT=REV/'manuscript/figures';OUT.mkdir(exist_ok=True)
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,
                     'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
TEAL='#287A86';GOLD='#BF8C47';INK='#444444';PURPLE='#8C6DA7'

def export(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/(name+'.alignment.json'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=600,facecolor='white')
    plt.close(fig)

def run():
    # Workflow: reference-independent scoring and explicit reference information boundaries.
    fig,ax=plt.subplots(figsize=(7.2,3.6));fig.subplots_adjust(left=.02,right=.98,bottom=.03,top=.97);ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
    def box(x,y,w,h,text,color='#F0F5F5'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.05',facecolor=color,edgecolor=TEAL,lw=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8,linespacing=1.4)
    def arrow(a,b,color=INK,style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color=color,lw=1,linestyle=style))
    box(.1,3.25,2.15,1.05,'Sensor observations\nIndependent daily QC\n30-day sensor windows')
    box(2.65,3.25,2.15,1.05,'Calibration pipelines\n+ allowed prior audit\nCandidate products')
    box(5.2,3.25,2.15,1.05,'Frozen conformity score\nSensor and output features\nNo assessment reference')
    box(7.75,3.25,2.15,1.05,'Rank the review queue\nOne-product policy\nreported separately')
    for x in [2.25,4.8,7.35]:arrow((x,3.77),(x+.4,3.77))
    box(2.65,.5,2.15,1.05,'Prior reference audit\nB = 0, 1, 3, or 7 days\nBefore assessment',color='#FAF3E9')
    box(5.2,.5,2.15,1.05,'Assessment reference\nAt least 23 paired days\nLabel and evaluation only',color='#FAF3E9')
    box(7.75,.5,2.15,1.05,'Expert verification\nReference evidence\nDocumented final product')
    arrow((3.72,1.55),(3.72,3.25),GOLD)
    arrow((6.27,1.55),(6.27,2.45),GOLD)
    ax.text(6.27,2.7,'Hidden during scoring',ha='center',fontsize=8,color=TEAL)
    arrow((8.82,3.25),(8.82,1.55))
    arrow((7.35,1.02),(7.75,1.02),GOLD,style='--')
    ax.text(5,4.7,'Completed-block quality assurance',ha='center',fontsize=10,fontweight='bold')
    export(fig,'figure1_workflow')
    ranking=pd.read_csv(REV/'results/ranking.csv')
    keys=[('QUANT','full','UK: all pipelines'),('CampmierIndia','full','India: all pipelines'),('EPA_US_24h','full','U.S.: all pipelines'),('EPA_US_24h','noepa_frozen','U.S.: EPA candidates removed'),('EPA_US_24h','noepa_refit','U.S.: EPA candidates and features removed')]
    fig,ax=plt.subplots(figsize=(7.2,2.7));fig.subplots_adjust(left=.45,right=.97,bottom=.23,top=.95)
    for i,(ds,m,label) in enumerate(keys):
        row=ranking[ranking.dataset.eq(ds)&ranking.method.eq(m)&ranking.budget.eq('pooled')].iloc[0]
        ax.errorbar(row.auroc,i,xerr=[[row.auroc-row.auroc_lo],[row.auroc_hi-row.auroc]],fmt='o',color=TEAL if m=='full' else GOLD,capsize=3,ms=5,lw=1.2)
    ax.set(yticks=range(5),yticklabels=[x[2] for x in keys],xlim=(.45,1),xlabel='Candidate-level AUROC')
    ax.invert_yaxis();ax.axvline(.5,color='#999999',ls='--',lw=.7);ax.grid(axis='x',color='#EEEEEE');ax.set_axisbelow(True)
    export(fig,'figure2_ranking')
    selected=pd.read_csv(REV/'results/selected_products.csv.gz');selected=selected[selected.reference_budget_days.eq(0)]
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.7),sharex=True,sharey=True);fig.subplots_adjust(left=.09,right=.98,bottom=.29,top=.83,wspace=.18)
    for i,(ds,title) in enumerate([('QUANT','UK (200 deployments)'),('CampmierIndia','India (17 deployments)'),('EPA_US_24h','U.S. (231 deployments)')]):
        ax=axes[i]
        for m,label,color,style in [('full','ConformityGate',TEAL,'-'),('fixed_epa','Fixed EPA',INK,'--'),('variant_prior','Variant prior',PURPLE,':')]:
            g=selected[selected.dataset.eq(ds)&selected.method.eq(m)]
            fractions=np.linspace(.05,1,40)
            precision=[100*metric_arrays(g.conforms,g.score,q=q)[1] for q in fractions]
            ax.plot(100*fractions,precision,label=label,color=color,ls=style,lw=1.6 if m=='full' else 1.)
        ax.axvline(20,color='#DDDDDD',lw=.7);ax.set_title(title,fontsize=8,pad=12);ax.set(xlim=(0,100),ylim=(0,105),xticks=[0,50,100],yticks=[0,25,50,75,100])
        ax.text(-.12,1.13,'abc'[i],transform=ax.transAxes,fontweight='bold',fontsize=9)
    axes[0].set_ylabel('Conformity precision (%)')
    fig.supxlabel('Deployments prioritized for review (%)',y=.13,fontsize=8)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.01),fontsize=8)
    export(fig,'figure3_review_policies')
    methods=['I','E','P','IE','IP','EP','IEP','full','without_month','without_missing']
    names=['Identity (I)','Sensor / environment (E)','Prediction summaries (P)','I + E','I + P','E + P','I + E + P','Full, including audit','Full, without month','Full, without missing-domain fields']
    datasets=['QUANT','CampmierIndia','EPA_US_24h']
    values=np.array([[ranking[ranking.dataset.eq(ds)&ranking.method.eq(m)&ranking.budget.eq('pooled')].auroc.iloc[0] for ds in datasets] for m in methods])
    fig,ax=plt.subplots(figsize=(7.2,4.1));fig.subplots_adjust(left=.44,right=.96,bottom=.13,top=.90)
    ax.imshow(values,vmin=.5,vmax=1,cmap='cividis',aspect='auto')
    ax.set(xticks=range(3),xticklabels=['UK','India','U.S.'],yticks=range(len(methods)),yticklabels=names)
    ax.tick_params(length=0,pad=7);ax.set_title('Geographic AUROC under the same outer isolation',fontsize=9,pad=12)
    for i in range(len(methods)):
        for j in range(3):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]<.75 else '#222222')
    export(fig,'figure4_ablations')
    print('Exported four source-data-backed figure sets',flush=True)

if __name__=='__main__':run()
