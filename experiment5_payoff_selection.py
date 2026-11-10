#!/usr/bin/env python3
"""E5 rule-based, train-informed but with prespecified payoff-family rules.

python experiment5_payoff_selection.py --data-dir structure_study --objective efficiency --n 4096
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e5_contracts import FAMILIES

# These policies are defined without inspecting expanded E5 performance.
# Efficiency is the pre-registered PRIMARY endpoint; RMSE is secondary.
RULES = {
 'efficiency': {
  'terminal_digital':'CQMC+BB',
  'geometric_digital':'CQMC+GPCA',
  'arithmetic_digital':'CQMC+GPCA',
  'geometric_asian':'RQMC+PCA',
  'arithmetic_asian':'RQMC+PCA',
  'pathwise_barrier':'CQMC+BB',
  'geometric_knockout':'CQMC+GPCA',
 },
 'rmse': {
  'terminal_digital':'CQMC+BB',
  'geometric_digital':'CQMC+GPCA',
  'arithmetic_digital':'CQMC+PCA',
  'geometric_asian':'RQMC+GPCA(proxy)',
  'arithmetic_asian':'RQMC+GPCA(proxy)',
  'pathwise_barrier':'CQMC+GPCA',
  'geometric_knockout':'CQMC+PCA',
 }
}
# Important: `geo_knockout` E4 contains different RMSE winners; RMSE policy is
# a heuristic derived from E3, not a universal or provably optimal method.

FALLBACK=('RQMC+PCA','RQMC+BB','RQMC')

def select_rule(kind,allowed,objective='efficiency'):
    """Predict with contract family and analytical applicability only, never E5 losses."""
    allowed=set(allowed)
    family=FAMILIES.get(str(kind))
    if family is None:raise KeyError(f'Unknown E5 payoff family: {kind}')
    preferred=RULES[objective][family]
    selected=next((m for m in (preferred,*FALLBACK) if m in allowed),None)
    if selected is None:raise ValueError(f'No eligible method for {kind}')
    return selected, family, preferred, selected!=preferred

def load(folder,n):
    folder=Path(folder)
    s=pd.read_csv(folder/'study_summary.csv')
    app=pd.read_csv(folder/'method_applicability.csv')
    refs=pd.read_csv(folder/'study_references.csv')
    repl=pd.read_csv(folder/'study_replicates.csv')
    s=s[s.n==n].copy()
    if s.empty:raise ValueError(f'No results for N={n}')
    applicable={(str(r.case),str(r.method)):bool(r.applicable) for r in app.itertuples(index=False)}
    s=s[s.apply(lambda r:applicable.get((r.case,r.method),False),axis=1)].copy()
    s['loss_efficiency']=s.variance*s.total_time_s
    s['loss_rmse']=s.rmse
    return s,applicable,refs,repl

def build_rule_evidence(dev,objective):
    """E1-E4-only evidence. Descriptive, not an optimizer or a training algorithm."""
    kind_family={
      'digital':'terminal_digital','geo_digital':'geometric_digital',
      'arith_digital':'arithmetic_digital','geo_call':'geometric_asian',
      'arith_call':'arithmetic_asian','barrier':'pathwise_barrier',
      'geo_knockout':'geometric_knockout'}
    val='loss_'+objective
    rows=[]
    for family,method in RULES[objective].items():
        subset=dev[dev.kind.map(kind_family)==family]
        present=0;win=0;ratios=[];cases=[]
        for case,g in subset.groupby('case'):
            available=g.set_index('method')
            if method not in available.index or 'RQMC' not in available.index:continue
            a=float(available.loc[method,val]);base=float(available.loc['RQMC',val]);
            if not np.isfinite(a) or a<=0 or not np.isfinite(base) or base<=0:continue
            present+=1;cases.append(case);ratios.append(a/base)
            eligible=g[np.isfinite(g[val])&(g[val]>0)]
            if len(eligible) and eligible.sort_values([val,'method']).iloc[0].method==method:win+=1
        rows.append(dict(objective=objective,family=family,prespecified_rule=method,
            development_cases=present,development_wins=win,
            geometric_mean_loss_over_rqmc=float(np.exp(np.mean(np.log(ratios)))) if ratios else np.nan,
            development_case_names=';'.join(cases)))
    return pd.DataFrame(rows)

def produce_e5_oracle_diagnostics(data_dir, n, output_dir):
    """Produce all-family E5 oracle and price-agreement diagnostics from saved results.

    Called automatically during the E5 efficiency stage (after both selection
    objectives, as orchestrated by main.py). Never alters E1-E4 or pricing data.
    Price agreement is a diagnostic, NOT a test of estimator unbiasedness.
    """
    from matplotlib.patches import Patch
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(output_dir/'experiment5_decisions.csv')
    decisions = decisions[decisions['n'].eq(n)].copy()
    decisions['evaluable'] = pd.to_numeric(decisions['evaluable'], errors='coerce').fillna(0).astype(bool)
    if decisions.empty:
        raise ValueError(f'No E5 efficiency decisions at N={n}')
    valid = decisions[decisions['evaluable']].copy()
    # Labels grouped by payoff variant, not only the seven broad families.
    group_counts = (valid.groupby(['structural_family','kind','oracle_hindsight'])
                    .size().rename('oracle_wins').reset_index())
    group_counts.to_csv(output_dir/'e5_efficiency_oracle_wins_by_variant.csv',index=False)
    total_counts=(valid.oracle_hindsight.value_counts()
                  .rename_axis('method').reset_index(name='oracle_wins'))
    total_counts.to_csv(output_dir/'e5_efficiency_oracle_wins_overall.csv',index=False)
    by_variant = group_counts.pivot_table(index=['structural_family','kind'],
                                        columns='oracle_hindsight', values='oracle_wins',
                                        aggfunc='sum', fill_value=0)
    by_variant = by_variant.sort_index()
    columns = list(total_counts.method)
    by_variant=by_variant.reindex(columns=columns,fill_value=0)
    cmap=plt.get_cmap('tab20')
    colors={m:cmap(i/max(1,len(columns)-1)) for i,m in enumerate(columns)}
    fig,ax=plt.subplots(figsize=(13.5,max(6,0.52*len(by_variant)+2.2)))
    y=np.arange(len(by_variant)); left=np.zeros(len(by_variant))
    for m in columns:
        count=by_variant[m].to_numpy(dtype=float)
        ax.barh(y,count,left=left,label=m,color=colors[m],height=.78)
        left+=count
    labels=[f'{family.replace("_"," ")} | {kind.replace("_"," ")}' for family,kind in by_variant.index]
    ax.set_yticks(y,labels,fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Number of evaluable contracts won by method')
    ax.set_title(f'E5 hindsight efficiency winners by payoff variant (N={n:,})',fontsize=13,weight='semibold',loc='left')
    ax.grid(axis='x',alpha=.18)
    ax.set_axisbelow(True)
    for yy,count in zip(y,left):
        ax.text(count+.3, yy, str(int(count)),va='center',fontsize=8)
    ax.set_xlim(0,max(left)*1.12)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=5,fontsize=8,
               bbox_to_anchor=(.55,-.025),frameon=False)
    fig.tight_layout(rect=[0,.09,1,1])
    fig.savefig(output_dir/'e5_efficiency_oracle_wins_by_variant.png',dpi=180,bbox_inches='tight')
    plt.close(fig)

    # Fair price comparison: each observed estimate is matched to the SAME
    # contract's numerical reference. Plot 3 predefined, interpretable roles.
    summary = pd.read_csv(data_dir/'study_summary.csv')
    refs = pd.read_csv(data_dir/'study_references.csv')
    values = summary[(summary.experiment=='E5_selection') & (summary.n==n)].copy()
    values=values.merge(refs[['case','reference','reference_se']],on='case',how='left',suffixes=('','_benchmark'))
    # study_summary may already contain reference_se for each estimate.
    se_ref = 'reference_se_benchmark' if 'reference_se_benchmark' in values else 'reference_se'
    # Capture representative results per case: selected rule, fixed PCA,
    # and empirical oracle. Undefined oracles are omitted.
    records=[]
    for r in decisions.itertuples(index=False):
        methods={'Rule-selected':r.selected_method,'Fixed PCA':'RQMC+PCA'}
        if r.evaluable and r.oracle_hindsight not in ('UNDEFINED',None):
            methods['Efficiency oracle']=r.oracle_hindsight
        t=values[values.case==r.case]
        for role,method in methods.items():
            u=t[t.method==method]
            if u.empty:continue
            z=u.iloc[0]
            measured=float(z.price)
            reference=float(z.reference)
            est_se=float(z.mean_price_se)
            ref_se=float(getattr(z,se_ref))
            combined=np.sqrt(est_se**2+ref_se**2) if np.isfinite(est_se) and np.isfinite(ref_se) else np.nan
            records.append(dict(case=r.case,kind=r.kind,family=r.structural_family,
                role=role,method=method,price=measured,reference=reference,
                estimate_se=est_se,reference_se=ref_se,combined_se=combined,
                abs_error=abs(measured-reference) if np.isfinite(measured) and np.isfinite(reference) else np.nan,
                within_2_combined_se=bool(abs(measured-reference)<=2*combined) if combined>0 and np.isfinite(measured) and np.isfinite(reference) else np.nan))
    agreement=pd.DataFrame(records)
    agreement.to_csv(output_dir/'e5_price_agreement_all_families.csv',index=False)
    families=sorted(agreement.family.unique())
    fig,axs=plt.subplots(4,2,figsize=(13,17))
    axs=axs.ravel()
    plot_styles={'Rule-selected':('#0072B2','o'), 'Fixed PCA':('#E69F00','x'),
                 'Efficiency oracle':('#009E73','.')}
    for idx,family in enumerate(families):
        ax=axs[idx]
        g=agreement[agreement.family==family]
        vals=np.concatenate([g.price.to_numpy(float),g.reference.to_numpy(float)])
        vals=vals[np.isfinite(vals)]
        low=float(np.min(vals)); high=float(np.max(vals))
        extent=max(high-low,abs(high)*.02,1e-8)
        low-=extent*.05;high+=extent*.05
        for role,(color,marker) in plot_styles.items():
            t=g[g.role==role]
            if t.empty:continue
            ax.scatter(t.reference,t.price,s=14 if role!='Efficiency oracle' else 8,
                       alpha=.60 if role!='Efficiency oracle' else .4,
                       marker=marker,color=color,label=role)
        ax.plot([low,high],[low,high],ls='--',lw=1,color='gray')
        ax.set_xlim(low,high);ax.set_ylim(low,high)
        ax.set_title(family.replace('_',' ').title()+f'  (n={g.case.nunique()})',fontsize=11)
        ax.set_xlabel('Independent numerical reference price',fontsize=9)
        ax.set_ylabel('Estimated option price',fontsize=9)
        ax.grid(alpha=.14)
        # Share a precise comparison statistic without implying all differences
        # are bias: count how often the rule estimate agrees within 2 SE.
        t=g[g.role=='Rule-selected']; good=t.combined_se>0
        covered=t.loc[good,'within_2_combined_se'].map(lambda v: v is True or str(v).lower() == 'true')
        ax.text(.02,.98,f'Rule within 2 combined SE: {int(covered.sum())}/{int(good.sum())}',
                transform=ax.transAxes,va='top',fontsize=8,
                bbox=dict(facecolor='white',alpha=.78,edgecolor='none'))
    for j in range(len(families),len(axs)): axs[j].axis('off')
    handles=[plt.Line2D([],[],marker=mk,ls='',color=co,label=role)
             for role,(co,mk) in plot_styles.items()]
    fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.012),frameon=False)
    fig.suptitle('E5 price agreement across all payoff families',fontsize=16,weight='semibold',y=.995)
    fig.text(.5,.025,'Dashed line: estimate = numerical reference. Agreement does not establish absence of bias; rare-event estimates may be unresolved.',
             ha='center',fontsize=9,color='dimgray')
    fig.tight_layout(rect=[0,.042,1,.98])
    fig.savefig(output_dir/'e5_price_agreement_all_families.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('E5 oracle diagnostics saved:',output_dir)
    print('  e5_efficiency_oracle_wins_by_variant.png')
    print('  e5_price_agreement_all_families.png')


def run(a):
    data,amap,refs,repl=load(a.data_dir,a.n)
    dev=data[data.experiment!='E5_selection']
    hold=data[data.experiment=='E5_selection']
    if dev.empty or hold.empty:raise RuntimeError('Need saved E1-E4 and simulated E5 rows at matching N')
    if set(dev.case)&set(hold.case):raise AssertionError('Development/E5 case-name leakage')
    valid_kinds=set(FAMILIES)
    unknown=set(hold.kind)-valid_kinds
    if unknown:raise ValueError(f'Unexpected E5 kinds: {unknown}')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    build_rule_evidence(dev,a.objective).to_csv(out/'experiment5_rule_evidence_E1_E4.csv',index=False)
    loss='loss_'+a.objective
    decisions=[];evaluations=[];per=[];invalid_cases=[]
    for case,g in hold.groupby('case',sort=True):
        meta=g.iloc[0]
        allowed=[m for m in g.method if amap.get((case,m),False)]
        selected,family,preferred,fallback=select_rule(meta.kind,allowed,a.objective)
        # All-zero or partly zero loss occurs for rare/degenerate payoffs.
        # Never call a simulated zero RMSE/variance an exact oracle victory:
        # record the contract, retain its measured prices, and exclude it
        # explicitly from multiplicative selection-performance statistics.
        measured_losses=pd.to_numeric(g[loss],errors='coerce').to_numpy(float)
        finite=np.isfinite(measured_losses)
        has_zero=bool(np.any(finite & (measured_losses==0)))
        valid=g[np.isfinite(g[loss])&(g[loss]>0)]
        evaluable=bool(not has_zero and not valid.empty)
        reason=('zero_measured_loss' if has_zero else
                'no_positive_finite_loss' if valid.empty else 'ok')
        oracle=(valid.sort_values([loss,'method']).iloc[0].method
                if evaluable else None)
        if not evaluable:
            invalid_cases.append(dict(case=case,kind=meta.kind,objective=a.objective,
                                      reason=reason,number_of_zero_losses=int(np.sum(finite&(measured_losses==0))),
                                      min_finite_loss=float(np.min(measured_losses[finite])) if finite.any() else np.nan))
        measured=g.set_index('method')
        if 'RQMC' not in measured.index:raise ValueError(f'RQMC missing for {case}')
        # This fixed baseline is applicable to all current E5 kinds, and is stated
        # explicitly rather than determined by looking at the E5 results.
        roles={'rule_based':selected,'fixed_RQMC':'RQMC',
               'fixed_PCA':'RQMC+PCA',
               'simple_continuity': ('RQMC+PCA' if family in ('geometric_asian','arithmetic_asian') else 'CQMC+GPCA')}
        if oracle is not None:
            roles['hindsight_oracle']=oracle
        # The simple continuity-based policy must also obey analytical eligibility.
        if roles['simple_continuity'] not in allowed:
            roles['simple_continuity']='RQMC+PCA' if 'RQMC+PCA' in allowed else 'RQMC'
        decisions.append(dict(case=case,kind=meta.kind,structural_family=family,d=meta.d,
           K=meta.K,H=meta.H,sigma=meta.sigma,objective=a.objective,n=a.n,
           selected_method=selected,prespecified_rule=preferred,used_fallback=int(fallback),
           oracle_hindsight=oracle if oracle is not None else 'UNDEFINED',
           oracle_match=int(selected==oracle) if evaluable else np.nan,
           evaluable=int(evaluable),exclusion_reason=reason,
           rule_source='prespecified E1-E4 efficiency/RMSE analysis, not E5 labels'))
        base=measured.loc['RQMC']
        base_loss=float(base[loss])
        for role,method in roles.items():
            if method not in measured.index:raise ValueError(f'{case}: missing {role} {method}')
            r=measured.loc[method];value=float(r[loss])
            rel=(value/base_loss if evaluable and np.isfinite(value) and
                 value>0 and np.isfinite(base_loss) and base_loss>0 else np.nan)
            evalrow=dict(case=case,kind=meta.kind,structural_family=family,role=role,
                         method=method,n=a.n,objective=a.objective,
                         price_mean=float(r.price),rmse=float(r.rmse),reference_se=float(r.reference_se),
                         variance=float(r.variance),total_time_s=float(r.total_time_s),
                         efficiency=float(r.computational_efficiency),loss=value,
                         relative_loss=rel,evaluable=int(evaluable),exclusion_reason=reason)
            evaluations.append(evalrow)
            per.append(dict(case=case,role=role,method=method,loss=value,relative_loss=rel,
                            evaluable=int(evaluable),exclusion_reason=reason))
    pd.DataFrame(invalid_cases,columns=['case','kind','objective','reason','number_of_zero_losses',
                                       'min_finite_loss']).to_csv(out/'experiment5_unevaluable_cases.csv',index=False)
    if invalid_cases:
        print(f'WARNING: {len(invalid_cases)} holdouts excluded from {a.objective} ratios '
              '(zero, invalid, or nonpositive measured losses). '
              'See experiment5_unevaluable_cases.csv. These are NOT oracle wins.')
    pd.DataFrame(decisions).to_csv(out/'experiment5_decisions.csv',index=False)
    pd.DataFrame(evaluations).to_csv(out/'experiment5_evaluation.csv',index=False)
    pd.DataFrame(per).to_csv(out/'experiment5_relative_per_case.csv',index=False)
    agg=[]
    for role,g in pd.DataFrame(per).groupby('role'):
        v=np.asarray(g.relative_loss,float);good=v[np.isfinite(v)&(v>0)]
        agg.append(dict(role=role,cases=hold.case.nunique(),evaluable=len(good),
             geometric_mean_relative_loss=float(np.exp(np.mean(np.log(good)))) if len(good) else np.nan,
             median_relative_loss=float(np.median(good)) if len(good) else np.nan,
             cases_better_than_RQMC=int(np.sum(good<1))))
    summary=pd.DataFrame(agg).sort_values('geometric_mean_relative_loss')
    summary.to_csv(out/'experiment5_aggregate.csv',index=False)
    # Avoid unreadable per-contract charts: aggregate by objective, family, and dimension.
    decision_df=pd.DataFrame(decisions)
    eval_df=pd.DataFrame(evaluations)
    fam=[]
    for (family, role),g in eval_df.groupby(['structural_family','role']):
        x=g.relative_loss.to_numpy(float); x=x[np.isfinite(x)&(x>0)]
        fam.append(dict(family=family,role=role,contracts=len(g),evaluable=len(x),
            geometric_mean_relative_loss=float(np.exp(np.mean(np.log(x)))) if len(x) else np.nan))
    pd.DataFrame(fam).to_csv(out/'experiment5_by_family.csv',index=False)
    decision_df.groupby(['structural_family','kind'],as_index=False).agg(
        contracts=('case','nunique'),evaluable=('evaluable','sum'),
        oracle_matches=('oracle_match','sum')).to_csv(
            out/'experiment5_winner_matches_by_variant.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.5));ax.barh(summary.role,summary.geometric_mean_relative_loss)
    ax.set_xscale('log');ax.axvline(1,color='gray',ls='--');ax.set_xlabel('Geometric mean relative loss / RQMC (lower better)')
    fig.tight_layout();fig.savefig(out/'experiment5_aggregate.png',dpi=170);plt.close(fig)
    family_data=pd.DataFrame(fam)
    families=sorted(family_data.family.unique())
    roles=['rule_based','simple_continuity','fixed_PCA','hindsight_oracle']
    fig,axs=plt.subplots(len(families),1,figsize=(10,2.0*len(families)),sharex=True,squeeze=False)
    for ax, family in zip(axs[:,0],families):
        g=family_data[family_data.family==family].set_index('role')
        vals=[float(g.loc[r,'geometric_mean_relative_loss']) if r in g.index else np.nan for r in roles]
        ax.barh(roles,vals)
        ax.set_xscale('log');ax.invert_yaxis();ax.set_title(family,fontsize=10)
        ax.grid(axis='x',alpha=.2)
    axs[-1,0].set_xlabel('Geometric mean relative variance-time loss / RQMC')
    fig.tight_layout();fig.savefig(out/'experiment5_by_family.png',dpi=160);plt.close(fig)
    # Only six E5 selector figures across BOTH objective directories:
    # efficiency: aggregate, family, family winner rate, dimension sensitivity;
    # RMSE: aggregate, family. Full per-case results remain available as CSVs.
    if a.objective == 'efficiency':
        import matplotlib.ticker as ticker
        # Plot 3: exact oracle-match frequency by payoff family.
        win = (decision_df.groupby('structural_family', as_index=False)
                .agg(matches=('oracle_match','sum'),contracts=('case','nunique'),
                     evaluable=('evaluable','sum')))
        win['fraction'] = win.matches / win.evaluable.replace(0,np.nan)
        win=win.sort_values('fraction')
        fig,ax=plt.subplots(figsize=(9,4.5))
        bars=ax.barh(win.structural_family,win.fraction)
        ax.set_xlim(0,1.14);ax.set_xlabel('Exact hindsight-winner match rate')
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1))
        for bar, match, count in zip(bars,win.matches,win.evaluable):
            if np.isfinite(bar.get_width()):
                ax.text(bar.get_width()+.015,bar.get_y()+bar.get_height()/2,
                        f'{int(match)}/{int(count)}',va='center',fontsize=8)
        ax.set_title('E5: Rule selection accuracy by payoff family')
        fig.tight_layout();fig.savefig(out/'experiment5_winner_match_by_family.png',dpi=170);plt.close(fig)
        # Plot 4: how close the rule comes to the hindsight oracle by
        # dimension and family. Aggregate multiplicative regrets geometrically.
        selected=eval_df.loc[eval_df.role=='rule_based',
                             ['case','structural_family','loss']].rename(columns={'loss':'selected_loss'})
        oracle_vals=eval_df.loc[eval_df.role=='hindsight_oracle',
                                ['case','loss']].rename(columns={'loss':'oracle_loss'})
        dimmeta=decision_df[['case','d']]
        matched=selected.merge(oracle_vals,on='case').merge(dimmeta,on='case')
        matched['regret']=matched.selected_loss/matched.oracle_loss
        matched=matched[np.isfinite(matched.regret)&(matched.regret>0)]
        if not matched.empty:
            data=(matched.groupby(['structural_family','d']).regret
                        .apply(lambda z:float(np.exp(np.mean(np.log(z)))))
                        .unstack())
            data=data.reindex(index=families)
            fig,ax=plt.subplots(figsize=(9,5))
            # log10(regret): 0 is oracle-perfect, positive means inefficiency.
            values=np.log10(data.to_numpy(dtype=float))
            image=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap='viridis')
            ax.set_yticks(range(len(data.index)),data.index)
            ax.set_xticks(range(len(data.columns)),[str(int(d)) for d in data.columns])
            ax.set_xlabel('Monitoring dimension (d)')
            ax.set_title('E5: Geometric-mean regret vs oracle (by family and dimension)')
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    val=float(data.iloc[i,j])
                    if np.isfinite(val):
                        ax.text(j,i,f'{val:.2g}x',ha='center',va='center',fontsize=8,
                                color='white' if values[i,j]>.5 else 'black')
            cb=fig.colorbar(image,ax=ax);cb.set_label('log10(relative variance-time loss / oracle)')
            fig.tight_layout();fig.savefig(out/'experiment5_oracle_regret_by_dimension.png',dpi=170);plt.close(fig)
    (out/'README.txt').write_text("""E5 RULE-BASED HOLDOUT TEST\n\nRules are PREDECLARED based on E1-E4, not trained with E5.\n
All 416 holdout contracts are simulated before retrospective evaluation; performance used only after predictions.\n
New put/reversed-knockout/up-and-out variants share structural-family rules with E1-E4.\n
Computational efficiency is primary; RMSE objective is supporting/sensitivity analysis.\n
The `simple_continuity` policy is a strong benchmark: PCA for continuous options, CQMC+GPCA for discontinuous.\n
Hindsight oracle knows E5 measured performance; it is not a deployable selector.\n
A zero estimated variance is not proof of perfect pricing, and RMSE depends on reference uncertainty.\n""")
    if a.objective == 'efficiency':
        produce_e5_oracle_diagnostics(a.data_dir, a.n, out)
    print('E5 holdouts:',len(decisions),
          'evaluable:',len(decisions)-len(invalid_cases),
          'oracle matches:',int(np.nansum([x['oracle_match'] for x in decisions])))
    print(summary.to_string(index=False));print('Output:',out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='structure_study');p.add_argument('--objective',choices=['efficiency','rmse'],default='efficiency')
    p.add_argument('--n',type=int,default=4096);p.add_argument('--out',default=None)
    a=p.parse_args()
    if a.out is None:a.out=str(Path(a.data_dir)/f'experiment5_{a.objective}_{a.n}')
    run(a)
if __name__=='__main__':main()
