"""Typeset the audited revision and its matched controls from saved records."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1];D=P/'data';R=D/'revision';T=P/'tables'
NAMES={'deepseek':'DeepSeek','qwen':'Qwen','glm53':'GLM-5.3','kimi28':'Kimi K2.8'}
LABELS={'selected':'Joint rewrite (audit control)','global':'Global margin (audit control)','service_first':'Service-first (audit control)',
        'drop_relative':'Without relative block (matched)',
        'fixed_service':'Proposed','numeric':'No LLM prior + gate','no_relative':'All-invariant representation','anomaly':'Anomaly ranking + gate','llm':'LLM confidence gate',
        'llm_direct':'Initial LLM','selected_direct':'Joint rewrite, direct','fixed_service_direct':'Proposed, without gate'}


def pct(x):return '---' if pd.isna(x) else f'{100*x:.2f}'
def tab(name,columns,headers,rows):
    text=r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}'+columns+r'@{}}'+'\n'+r'\toprule'+'\n'+' & '.join(headers)+r' \\'+'\n'+r'\midrule'+'\n'
    text+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular*}'+'\n'
    (T/(name+'.tex')).write_text(text,encoding='utf-8')


def main():
    s=pd.read_csv(R/'summary.csv');allrows=s[s.scope.eq('ALL')].set_index('method')
    sequence=['llm_direct','llm','fixed_service_direct','fixed_service']
    rows=[]
    for method in sequence:
        z=allrows.loc[method];label=LABELS[method]
        if method=='fixed_service':label=r'\textbf{'+label+'}'
        rows.append([label,int(z.J),int(z.S),int(z.wrong),pct(z.C_J),pct(z.C_A),pct(z.R_A),pct(z.R_J)])
    tab('primary','lrrrrrrr',['Workflow','$J$','$S$','Errors',r'$C_J$ (\%)',r'$C_A$ (\%)',r'$R_A$ (\%)',r'$R_J$ (\%)'],rows)
    controls=[]
    for method in ['fixed_service','selected','global','service_first','numeric','drop_relative','no_relative','anomaly']:
        z=allrows.loc[method];controls.append([LABELS[method],int(z.J),int(z.S),int(z.joint_wrong),pct(z.C_J),pct(z.R_J),pct(z.R_A)])
    tab('audit_controls','lrrrrrr',['Workflow','$J$','$S$','Wrong full',r'$C_J$ (\%)',r'$R_J$ (\%)',r'$R_A$ (\%)'],controls)
    rows=[]
    for method,label in [('llm_direct','Initial LLM'),('fixed_service_direct','Proposed type revision'),('selected_direct','Joint rewrite (audit control)')]:
        z=allrows.loc[method];rows.append([label,pct(z.service_top1),pct(z.type_accuracy),pct(z.joint_accuracy)])
    tab('ablations','lrrr',['Forced-prediction workflow',r'Service Top@1 (\%)',r'Type accuracy (\%)',r'Joint accuracy (\%)'],rows)
    providers=pd.read_csv(R/'anchored_provider_summary.csv').set_index('model')
    ex=pd.read_csv(D/'external/summary.csv');deep=ex[ex.method.eq('hybrid_joint')&ex.scope.eq('ALL')].iloc[0]
    ee=pd.read_csv(D/'external/provider_summary.csv');ee=ee[ee.method.eq('v6_frozen')&ee.scope.eq('ALL')].set_index('model')
    rows=[]
    for name in ['deepseek','glm53','kimi28','qwen']:
        z=providers.loc[name];e=deep if name=='deepseek' else ee.loc[name]
        rows.append([NAMES[name],f'{int(z.J)}/270',pct(z.C_J),f'{int(z.joint_wrong)}/{int(z.J)}',pct(z.R_J),pct(z.R_A)])
    tab('providers','lrrrrr',['Provider',r'$J/N$',r'$C_J$ (\%)','Wrong/$J$',r'$R_J$ (\%)',r'$R_A$ (\%)'],rows)
    raw=pd.read_csv(D/'external/raw_llm_summary.csv');old=pd.read_csv(D/'external/base_summary.csv');noedge=pd.read_csv(D/'external/no_edges_summary.csv')
    seq=[('LLM direct',raw[raw.method.eq('raw_llm_direct')&raw.scope.eq('ALL')].iloc[0]),('LLM confidence',raw[raw.method.eq('raw_llm_confidence')&raw.scope.eq('ALL')].iloc[0]),('Base evidence',old[old.method.eq('hybrid_joint')&old.scope.eq('ALL')].iloc[0]),('Graph evidence',deep),('No-edge control',noedge[noedge.scope.eq('ALL')].iloc[0]),('Numeric evidence',ex[ex.method.eq('numeric_joint')&ex.scope.eq('ALL')].iloc[0])]
    tab('external','lrrrr',['Workflow','$J$','Errors',r'$C_J$ (\%)',r'$R_J$ (\%)'],[[label,int(z.J),int(z.joint_wrong),pct(z.C_J),pct(z.R_J)] for label,z in seq])
    tab('cohorts','llrrl',['Source','Systems','Cases','Fault types','Evaluation'],[['RCAEval RE1','OB, SS, TT',375,5,'Grouped retrospective cross-validation'],['RCAEval RE2','OB, SS, TT',270,6,'Grouped cross-validation; provider replay'],['Eadro author data','SN, TT',117,3,'36 development / 36 calibration / 45 test']])
    gates=json.loads((R/'selected_rules.json').read_text());gates={r['fold']:r['rule'] for r in gates if r['method']=='fixed_service' and r['budget']==.05}
    rows=[]
    for i in range(5):
        c=json.loads((D/f'retrospective/fold_{i}.json').read_text());g=gates[i]
        rows.append([i+1,len(c['train']),len(c['calibration']),len(c['test']),c['mode'],c['gamma'],g['service_threshold'],g['joint_threshold'],g['margin']])
    tab('folds','rrrrlrrrr',['Fold','Fit','Cal.','Test','Features',r'$\gamma$',r'$\tau_s$',r'$\tau_j$',r'$\tau_m$'],rows)
    cost=json.loads((D/'external/test_request_cost.json').read_text())
    tab('cost','lrrrr',['Provider','Calls','Tokens/case','Median (s)','P95 (s)'],[[NAMES[z['model']],z['attempts'],f"{z['reported_tokens']/45:,.0f}",f"{z['median_request_seconds']:.1f}",f"{z['p95_request_seconds']:.1f}"] for z in cost])
    iv=json.loads((R/'anchored_subgroup_intervals.json').read_text());iv={(x['scope'],x['fault']):x for x in iv}
    rows=[];g=pd.read_csv(R/'anchored_subgroup_detail.csv')
    def interval(z,key):
        values=z['intervals'][key]
        return '---' if values[0] is None else f'[{100*values[0]:.1f}, {100*values[1]:.1f}]'
    for z in g.to_dict('records'):
        q=iv[(z['dataset'],z['fault'])]
        rows.append([z['dataset'],z['fault'],z['n'],f"{int(z['J'])}/{int(z['n'])}",pct(z['C_J'])+' '+interval(q,'C_J'),
                     ('---' if pd.isna(z['R_J']) else f"{int(z['joint_wrong'])}/{int(z['J'])}"),pct(z['R_J'])+' '+interval(q,'R_J'),int(z['R'])])
    tab('subgroup_detail','llrlrlrr',['Dataset','Fault','$N$','$J/N$',r'$C_J$ [95\% range]','Wrong/$J$',r'$R_J$ [95\% range]','Refer'],rows)
    u=pd.read_csv(R/'utility_sensitivity.csv');u=u[np.isclose(u.budget,.05)]
    rows=[]
    for cost in [1,5,10,20]:
        for referral in [0.,.5]:
            zz=u[u.wrong_cost.eq(cost)&np.isclose(u.referral_cost,referral)].set_index('method')
            rows.append([cost,referral,*[f'{zz.loc[m,"utility"]:.3f}' for m in ['selected','fixed_service','anomaly','llm']]])
    tab('utility','rrrrrr',['Wrong cost','Referral cost','Joint','Fixed service','Anomaly rank','All refer'],rows)
    cases=json.loads((R/'anchored_cases.json').read_text())
    tab('cases','llll',['Observed outcome','Case','Initial type','Revised / released'],[[z['category'],z['case'].replace('_',r'\_'),z['initial_fault'],(z['fault'] if z['action']=='joint' else 'service only')+(' (correct)' if z['correct'] else ' (wrong)')] for z in cases])
    print('Manuscript and supplementary tables were written to tables/.')


if __name__=='__main__':main()
