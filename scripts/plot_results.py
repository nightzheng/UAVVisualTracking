"""Re-render the two formal figures from the included frozen parent CSVs."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

COLORS={'B0':'#276a9e','B1':'#b87816'}

def render(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#222222','text.color':'#222222','figure.facecolor':'white','axes.facecolor':'white'})
    with (root/'results/baseline_parent.csv').open() as f:rows=list(csv.DictReader(f))
    parents=sorted({r['parent'] for r in rows});y=np.arange(6)
    fig,axes=plt.subplots(1,3,figsize=(14,5.8),sharey=True)
    for ax,condition in zip(axes,['C0','C2','C3']):
        ax.set_title({'C0':'C0 | distinct appearance','C2':'C2 | repeated appearance','C3':'C3 | local ROI (180 x 260)'}[condition],fontweight='bold',pad=15)
        ax.set_xlim(0,1.02);ax.set_ylim(5.6,-.6);ax.set_yticks(y,labels=[p.replace('heldout_','') for p in parents]);ax.grid(axis='x',color='#dddddd',linewidth=.6);ax.set_axisbelow(True)
        if condition!='C3':
            for i,parent in enumerate(parents):
                group={r['model']:float(r['AJ']) for r in rows if r['parent']==parent and r['condition']==condition}
                a,b=group['B0_native'],group['B1_native'];ax.plot([a,b],[i,i],color='#b9b9b9',lw=2,zorder=1)
                for model,v,offset in [('B0',a,-.16),('B1',b,.16)]:
                    ax.scatter(v,i+offset,s=47,marker='o' if model=='B0' else 's',color=COLORS[model],zorder=3)
                    ax.annotate(f'{v:.4f}',(v,i+offset),xytext=(-8,0),textcoords='offset points',
                                ha='right',va='center',fontsize=9,color=COLORS[model])
            ax.set_xlabel('Official TAP-Vid AJ (actual video domain)')
            ax.text(.5,-.21,{'C0':'B1 higher: 4/6 parents; B0 higher: 2/6','C2':'B1 AJ higher in 6/6 parents'}[condition],transform=ax.transAxes,ha='center',fontsize=10,fontweight='bold')
        else:
            ax.grid(False);ax.set_xticks([])
            ax.text(.5,.62,'NA',transform=ax.transAxes,ha='center',va='center',fontsize=34,color='#555555',fontweight='bold')
            ax.text(.5,.38,'No complete six-parent AJ\n5/12 queries have complete GT\nUnknown GT is retained',transform=ax.transAxes,ha='center',va='center',linespacing=1.7)
    axes[0].set_ylabel('Procedural parent (two fixed query opportunities each)')
    fig.suptitle('Formal B0 / B1 baseline pairs',x=.08,ha='left',fontsize=18,fontweight='bold',y=.98)
    fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=COLORS['B0'],label='B0 CoTracker3 native'),Line2D([],[],marker='s',lw=0,color=COLORS['B1'],label='B1 BootsTAPIR native')],loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
    fig.text(.08,.02,'Frozen Stage5B formal results | 6 parents, one synthetic rendering family | 48 frames at 24 Hz | native system configurations differ',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.08,right=.985,top=.82,bottom=.28,wspace=.28)
    fig.savefig(output/'baseline_pairs.png',dpi=180,metadata={'Title':'Formal frozen Stage5B parent AJ pairs','Source':'results/baseline_parent.csv'});plt.close(fig)
    with (root/'results/reliability_parent.csv').open() as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharex=True,sharey=True)
    for ax,model in zip(axes,['B0','B1']):
        ax.set_title(model+' | fixed M0 to M1',fontweight='bold',pad=15)
        group=[r for r in rows if r['epsilon']=='10' and r['scope']=='ALL_CONDITIONS' and r['model']==model+'_native']
        for i,parent in enumerate(sorted({r['parent'] for r in group})):
            pair={r['method']:r for r in group if r['parent']==parent}
            x0,y0=[float(pair['M0'][k]) for k in ('C_good','R_sel')];x1,y1=[float(pair['M1'][k]) for k in ('C_good','R_sel')]
            ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='->',color=COLORS[model],lw=1.7,shrinkA=5,shrinkB=5))
            ax.scatter(x0,y0,s=65,facecolors='white',edgecolors=COLORS[model],linewidths=1.5,zorder=3)
            ax.scatter(x1,y1,s=52,marker='s',color=COLORS[model],zorder=4)
            offset=({'p01':(5,8),'p02':(5,-14),'p03':(-26,8),'p04':(5,-17),'p05':(-26,9),'p06':(6,-14)} if model=='B0' else {'p01':(-25,8),'p02':(5,-13),'p03':(-30,9),'p04':(-25,9),'p05':(5,6),'p06':(5,9)})[parent[-3:]]
            ax.annotate(parent[-3:],(x1,y1),xytext=offset,textcoords='offset points',fontsize=10,color=COLORS[model])
        ax.set_xlim(.25,.85);ax.set_ylim(-.015,.33);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(color='#dddddd',linewidth=.6);ax.set_axisbelow(True);ax.set_xlabel('Correct coverage C_good = AC / known opportunities')
    axes[0].set_ylabel('Selective risk R_sel = AB / (AC + AB)')
    fig.suptitle('Risk and correct coverage: every heldout parent',x=.08,ha='left',fontsize=18,fontweight='bold',y=.98)
    fig.text(.08,.905,'Frozen Stage6B | 6 new procedural parents | epsilon = 10 source-image px | lower risk and higher correct coverage preferred',fontsize=10)
    fig.legend(handles=[Line2D([],[],marker='o',lw=0,markerfacecolor='white',markeredgecolor='#555555',label='M0 native acceptance'),Line2D([],[],marker='s',lw=0,color='#555555',label='M1 fixed score > 0.7')],loc='lower center',bbox_to_anchor=(.5,.12),ncol=2,frameon=False)
    fig.text(.08,.085,'B0 strict parent macro: risk 20.72% -> 23.61%; correct coverage 45.50% -> 51.59%; +25 wrong accepts.',fontsize=10)
    fig.text(.08,.05,'B1 strict parent macro: risk 5.58% -> 4.01%; correct coverage 68.30% -> 64.45%; -65 correct accepts.',fontsize=10)
    fig.text(.08,.017,'Each point averages all 6 condition/query opportunities within one parent. Arrows show paired changes, not independent samples or confidence intervals.',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.08,right=.98,top=.83,bottom=.25,wspace=.17)
    fig.savefig(output/'risk_coverage.png',dpi=180,metadata={'Title':'Formal frozen Stage6B risk and correct coverage pairs','Source':'results/reliability_parent.csv'});plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();render(a.root,a.output)
