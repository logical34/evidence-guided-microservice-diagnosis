"""Build the quantitative figures from saved observations."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parents[1]
R = P / "data/revision"
F = P / "figures"
BLUE,TEAL,RED,ORANGE,GRAY,INK='#24749B','#258775','#B24B3A','#D28A32','#CFD5DA','#263743'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':8,
 'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'hatch.linewidth':.35,
 'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.1,
 'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})


def figure(w=178,h=62,**kw):return plt.subplots(figsize=(w/25.4,h/25.4),layout='constrained',**kw)
def save(fig,stem):
    for ext in ['pdf','svg','png']:fig.savefig(F/f'{stem}.{ext}',dpi=450)
    plt.close(fig)


def composition():
    a=pd.read_csv(R/'actions.csv');rows=[]
    for method,label in [('llm_direct','Initial diagnosis'),('fixed_service_direct','After type revision'),('fixed_service','After release decision')]:
        g=a[a.method.eq(method)];j=g.action.eq('joint');s=g.action.eq('service')
        rows.append((label,[int((j&g.correct).sum()),int((j&~g.correct).sum()),int((s&g.correct).sum()),int((s&~g.correct).sum()),int((~(j|s)).sum())]))
    fig,ax=figure(w=178,h=44);axes=[ax];categories=['Correct full diagnosis','Wrong full diagnosis','Correct service only','Wrong service only','Referred'];source=[]
    for panel,(ax,data,n) in enumerate(zip(axes,[rows],[645])):
        values=np.array([v for _,v in data]);start=np.zeros(len(data))
        for k,(color,hatch,label) in enumerate(zip([BLUE,RED,TEAL,ORANGE,GRAY],['','/','','///',''],categories)):
            widths=100*values[:,k]/n;ax.barh(range(len(data)),widths,left=start,height=.63,color=color,edgecolor='white',linewidth=.35,hatch=hatch,label=label)
            for y,w,left,count in zip(range(len(data)),widths,start,values[:,k]):
                if w>=4:ax.text(left+w/2,y,str(count),ha='center',va='center',fontsize=7,color='white' if k in [0,1,2] else INK)
            start+=widths
        ax.set(yticks=range(len(data)),yticklabels=[s for s,_ in data],xlim=(0,100),xticks=[0,25,50,75,100],xlabel='All incidents (%)');ax.invert_yaxis()
        source.extend([dict(source=['RCAEval','Eadro'][panel],workflow=label,N=n,**dict(zip(categories,v))) for label,v in data])
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,columnspacing=1.4,handlelength=1.5)
    save(fig,'fig3_release_outcomes');pd.DataFrame(source).to_csv(F/'release_outcomes_source.csv',index=False)


def risk_coverage():
    a=pd.read_csv(R/'actions.csv');b=pd.read_csv(R/'budget_summary.csv')
    styles={'llm':('Initial diagnosis','#737D86','--'),'fixed_service':('Revised diagnosis',BLUE,'-')}
    fig,axis=figure(w=85,h=63);axes=[axis];export=[]
    for ax in axes:
        ax.axvline(50,color='#99A6AE',ls='--',lw=.65);ax.axhline(10,color='#99A6AE',ls='--',lw=.65)
        ax.fill_between([50,100],0,10,color='#EAF3EF',zorder=-2)
        ax.set(xlim=(0,100),ylim=(0,50),xlabel='Complete diagnosis coverage (%)',xticks=[0,25,50,75,100])
    axes[0].set_ylabel('Error among complete diagnoses (%)')
    for method,(label,color,ls) in styles.items():
        g=a[a.method.eq(method+'_direct')];g=g[g.valid&g.supported&np.isfinite(g.joint_score)]
        gg=g.assign(error=~(g.service_correct&g.type_correct)).groupby('joint_score',sort=True).agg(n=('case','size'),errors=('error','sum')).iloc[::-1]
        count=gg.n.cumsum();wrong=gg.errors.cumsum();coverage=100*count/645;risk=100*wrong/count
        axes[0].step(coverage,risk,where='post',color=color,ls=ls,lw=1.05)
        export.extend([dict(panel='ranking',method=method,threshold=float(t),C_J=float(c/100),R_J=float(r/100),J=int(n)) for t,c,r,n in zip(gg.index,coverage,risk,count)])
        if method=='llm':continue
        data=b[b.method.eq(method)].sort_values('budget');data=data[data.J.gt(0)]
        primary=data[np.isclose(data.budget,.05)].iloc[0]
        axes[0].scatter(primary.C_J*100,primary.R_J*100,s=30,marker='D',facecolor=TEAL,edgecolor='white',linewidth=.5,zorder=5)
        axes[0].annotate('Final release policy',(primary.C_J*100,primary.R_J*100),xytext=(43,19),fontsize=7,color=TEAL,arrowprops=dict(arrowstyle='-',color=TEAL,lw=.6))
    handles=[Line2D([0],[0],color=c,ls=ls,lw=1.1,label=label) for label,c,ls in styles.values()]
    axes[0].legend(handles=handles,loc='upper left',frameon=False,handlelength=2.)
    save(fig,'fig4_risk_coverage');pd.DataFrame(export).to_csv(F/'risk_coverage_source.csv',index=False)


def fault_types():
    a=pd.read_csv(R/'actions.csv');a=a[a.method.eq('fixed_service')];faults=['cpu','disk','delay','loss','mem','socket']
    values=np.full((2,3,6),np.nan);counts={};source=[]
    for i,suite in enumerate(['RE1','RE2','Pooled']):
        sourceframe=a if suite=='Pooled' else a[a.suite.eq(suite)]
        for j,fault in enumerate(faults):
            g=sourceframe if fault=='All' else sourceframe[sourceframe.true_fault.isin(['cpu','disk','delay','loss'])] if fault=='Hard' else sourceframe[sourceframe.true_fault.eq(fault)]
            n=len(g);released=g.action.eq('joint');k=int(released.sum());w=int((released&~g.correct).sum());counts[i,j]=(n,k,w)
            if n:values[0,i,j]=100*k/n
            if k:values[1,i,j]=100*w/k
            source.append(dict(suite=suite,fault=fault,N=n,J=k,wrong=w))
    fig,axes=figure(h=53,ncols=2)
    for k,(ax,data,cmap,maximum,metric) in enumerate(zip(axes,values,['YlGnBu','YlOrRd'],[100,25],['Complete coverage (%)','Complete diagnosis error (%)'])):
        cm=plt.get_cmap(cmap).copy();cm.set_bad('#ECEFF1');im=ax.imshow(np.ma.masked_invalid(data),cmap=cm,vmin=0,vmax=maximum,aspect='auto')
        ax.set(xticks=range(6),xticklabels=['CPU','disk I/O','delay','loss','memory','socket'],yticks=range(3),yticklabels=['RE1','RE2','RE1+RE2']);ax.tick_params(length=0,labelsize=6.5)
        ax.text(0,1.02,f'({chr(97+k)})',transform=ax.transAxes,va='bottom')
        for i in range(3):
            for j in range(6):
                n,r,w=counts[i,j];v=data[i,j]
                label='N/A' if not n else ('---' if not np.isfinite(v) else f'{v:.1f}\n'+(f'{r}/{n}' if k==0 else f'{w}/{r}'))
                ax.text(j,i,label,ha='center',va='center',fontsize=6.5,color='white' if np.isfinite(v) and v>.65*maximum else INK)
        cb=fig.colorbar(im,ax=ax,shrink=.9,pad=.02);cb.set_label(metric,fontsize=7);cb.ax.tick_params(labelsize=6)
    save(fig,'fig5_fault_type_limits');pd.DataFrame(source).to_csv(F/'fault_type_source.csv',index=False)


def main():
    F.mkdir(exist_ok=True)
    for fn in [composition,risk_coverage,fault_types]:fn()
    print('Three figures and their source CSV files were written to figures/.')


if __name__=='__main__':main()
