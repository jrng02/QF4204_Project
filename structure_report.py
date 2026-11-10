"""Create cross-payoff research tables and figures from structure_comparisons output.

Usage: python structure_report.py --data-dir structure_study
       python structure_report.py --data-dir structure_study --out structure_study/comparison_report
No resimulation is performed. Charts require matplotlib; tables require pandas.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'geo_call':'Geometric Asian call','arith_call':'Arithmetic Asian call','geo_digital':'Geometric Asian digital','digital':'European digital','geo_knockout':'Geometric KO arithmetic call','barrier':'Down-and-out call','digital_european':'European digital','arith_digital':'Arithmetic Asian digital','knockout_geometric_asian':'Geometric KO arithmetic call','standard_barrier':'Down-and-out call'}
ORDER=['MC','RQMC','RQMC+BB','RQMC+PCA','RQMC+GPCA(proxy)','OT-Cholesky','OT-PCA','OT+GPCA','CQMC','CQMC+BB','CQMC+PCA','CQMC+GPCA']



# Fixed, high-contrast method colours used consistently in all E5 comparison bars.
METHOD_COLORS = {
    'MC': '#161616', 'RQMC': '#0072B2',
    'RQMC+BB': '#E69F00', 'RQMC+PCA': '#009E73',
    'RQMC+GPCA(proxy)': '#CC79A7',
    'OT-Cholesky': '#D55E00', 'OT-PCA': '#56B4E9',
    'OT+GPCA': '#7B2CBF', 'CQMC': '#8C564B',
    'CQMC+BB': '#B9A70D', 'CQMC+PCA': '#E31A1C',
    'CQMC+GPCA': '#17BECF',
}

def plot_vrf_efficiency_comparisons(d, out, nmax):
    """Create E1-E4 VRF and efficiency comparison charts from cached study_summary.csv.

    VRF_RQMC = Var(RQMC) / Var(method).
    Efficiency(method) = 1 / (Var(method) * time(method)).
    Efficiency improvement vs RQMC = VRF_RQMC * time(RQMC)/time(method).
    All comparisons use matching (case, n); missing/not estimable values remain blank.
    """
    metric_specs = [
        ('vrf_rqmc', 'VRF vs RQMC (variance reduction)'),
        ('efficiency_improvement_vs_rqmc', 'Computational efficiency gain vs RQMC'),
        ('rmse_improvement_vs_rqmc', 'RMSE improvement vs RQMC'),
    ]
    selected=d[d.experiment.isin(['E1_continuity','E2_dimension','E3_geometry','E4_hybrid'])].copy()
    selected=selected[selected.n==nmax]
    if selected.empty:return
    rows=[]
    for exp, g in selected.groupby('experiment',sort=False):
        cases=list(dict.fromkeys(g.case))
        methods=[m for m in ORDER if m in set(g.method) and m != 'RQMC']
        # Preserve the original 14 filenames, but use horizontal dot plots,
        # not heatmaps. Each row is one payoff configuration; each column a metric.
        fig,axes=plt.subplots(len(cases),3,figsize=(20,max(4.0,2.8*len(cases))),
                              squeeze=False,layout='constrained')
        palette=plt.get_cmap('tab20')
        colors={m:palette(i%20) for i,m in enumerate(methods)}
        for i,case in enumerate(cases):
            sub=g[g.case==case].set_index('method')
            case_label=case.replace('holdout_','').replace('_',' ')
            for j,(key,title) in enumerate(metric_specs):
                ax=axes[i,j]
                vals=[]
                for k,m in enumerate(methods):
                    if m not in sub.index:continue
                    v=float(sub.loc[m,key])
                    if not np.isfinite(v) or v<=0:continue
                    ax.scatter(v,k,s=42,color=colors[m],zorder=3)
                    vals.append(v)
                ax.axvline(1,color='gray',ls='--',lw=1,alpha=.65)
                ax.set_xscale('log')
                ax.set_yticks(range(len(methods)))
                if j==0:
                    ax.set_yticklabels(methods,fontsize=8)
                else:
                    ax.set_yticklabels([])
                ax.set_ylim(len(methods)-.5,-.5)
                ax.grid(axis='x',alpha=.22)
                if i==0:ax.set_title(title,fontsize=11)
                if i==len(cases)-1:ax.set_xlabel('Fold relative to RQMC (log scale)')
                if j==0:
                    ax.set_ylabel(case_label,fontsize=9,fontweight='bold',labelpad=10)
                if vals:
                    low=min(vals+[1]);high=max(vals+[1]);
                    if high>low:ax.set_xlim(10**(np.log10(low)-.2),10**(np.log10(high)+.2))
                else:
                    ax.text(.5,.5,'Not estimable',transform=ax.transAxes,ha='center',va='center')
        fig.suptitle(f'{exp}: VRF, computational efficiency and RMSE | N={nmax:,} | higher is better',fontsize=14)
        fig.savefig(out/f'14_{exp}_vrf_efficiency_rmse.png',dpi=150,bbox_inches='tight')
        plt.close(fig)
        for case, group in g.groupby('case'):
            for _,item in group.iterrows():
                rows.append({'experiment':exp,'case':case,'n':nmax,'method':item.method,
                             'rmse':item.rmse,'variance':item.variance,'total_time_s':item.total_time_s,
                             'vrf_rqmc':item.vrf_rqmc,
                             'computational_efficiency':item.computational_efficiency,
                             'efficiency_gain_vs_rqmc':item.efficiency_improvement_vs_rqmc,
                             'rmse_gain_vs_rqmc':item.rmse_improvement_vs_rqmc})
    pd.DataFrame(rows).to_csv(out/f'14_performance_comparison_N{nmax}.csv',index=False)

    # E2: intuitive dimensional comparison by payoff and method.
    g=selected[selected.experiment=='E2_dimension']
    if not g.empty and 'd' in g:
        kinds=list(dict.fromkeys(g.kind))
        fig,axes=plt.subplots(2,len(kinds),figsize=(7*len(kinds),9),squeeze=False,layout='constrained')
        for col,kind in enumerate(kinds):
            sub=g[g.kind==kind]
            for row,(metric,ylabel) in enumerate([('vrf_rqmc','VRF vs RQMC (x)'),('efficiency_improvement_vs_rqmc','Efficiency gain vs RQMC (x)')]):
                ax=axes[row,col]
                for method in [m for m in ORDER if m in set(sub.method)]:
                    t=sub[sub.method==method].sort_values('d')
                    t=t[np.isfinite(t[metric])&(t[metric]>0)]
                    if len(t):ax.plot(t.d,t[metric],marker='o',label=method)
                ax.axhline(1,color='gray',ls='--',lw=1)
                ax.set_xscale('log',base=2);ax.set_yscale('log')
                ax.set_xticks(sorted(sub.d.unique()))
                ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.set_title(NAMES.get(kind,kind));ax.set_xlabel('Number of monitoring dates (d)')
                ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend(fontsize=7)
        fig.suptitle(f'E2: variance reduction vs time-adjusted efficiency | N={nmax:,}',fontsize=14)
        fig.savefig(out/'15_E2_dimension_vrf_efficiency.png',dpi=175,bbox_inches='tight');plt.close(fig)

    # Explicit identity check / reproducibility summary for all tested sample sizes.
    verify=d[d.experiment.isin(['E1_continuity','E2_dimension','E3_geometry','E4_hybrid'])].copy()
    verify['efficiency_gain_from_vrf_and_time']=verify['vrf_rqmc']*np.nan
    for (case,n),idx in verify.groupby(['case','n']).groups.items():
        base=verify.loc[idx];base=base[base.method=='RQMC']
        if base.empty:continue
        bt=float(base.total_time_s.iloc[0]);times=verify.loc[idx,'total_time_s'].to_numpy(float)
        verify.loc[idx,'efficiency_gain_from_vrf_and_time']=verify.loc[idx,'vrf_rqmc'].to_numpy(float)*bt/np.where(times>0,times,np.nan)
    verify[['experiment','case','n','method','vrf_rqmc','efficiency_improvement_vs_rqmc','efficiency_gain_from_vrf_and_time']].to_csv(out/'15_vrf_efficiency_identity_check.csv',index=False)

def plot_computational_efficiency_versions(d, out, nmax):
    """Additional 04--09 efficiency figures, retaining original figure styles.

    Unlike runtime-only figures, these show 1/(variance * amortized total time),
    or the dimensionless ratio to matching standard RQMC. Never replace 04--09
    or the separately produced 14_* figures.
    """
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(out/name,dpi=175,bbox_inches='tight')
        plt.close(fig)
    def valid(g,col):
        a=g.copy()
        a[col]=pd.to_numeric(a[col],errors='coerce')
        return a[np.isfinite(a[col]) & (a[col]>0)]
    eff='efficiency_improvement_vs_rqmc'
    s=d[d.n==nmax]

    # 04: same method-by-payoff scatter as the original continuity figure.
    part=s[s.experiment=='E1_continuity']
    if not part.empty:
        fig,ax=plt.subplots(figsize=(11,5))
        for kind,marker in [('geo_call','o'),('geo_digital','s'),('arith_call','^'),('arith_digital','D')]:
            g=valid(part[part.kind==kind],eff)
            if not g.empty:
                ax.scatter([ORDER.index(m) for m in g.method],g[eff],label=NAMES.get(kind,kind),marker=marker,s=55)
        ax.axhline(1,color='gray',ls='--');ax.set_yscale('log')
        ax.set_xticks(range(len(ORDER)),ORDER,rotation=45,ha='right')
        ax.set_ylabel('Computational-efficiency gain vs RQMC (×; higher is better)')
        ax.set_title(f'E1 — Continuity: computational efficiency | N={nmax:,}')
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
        finish(fig,'04_continuity_comparison_efficiency.png')

    # 05: same grouped two-panel bar arrangement as original E4 figure.
    part=s[s.experiment=='E4_hybrid']
    if not part.empty:
        cs=list(part.case.unique());ms=['RQMC','OT-PCA','OT+GPCA','CQMC+PCA','CQMC+GPCA']
        fig,axs=plt.subplots(1,2,figsize=(14,5));x=np.arange(len(cs));w=.16
        for ax,key,heading in zip(axs,[eff,'computational_efficiency'],
                                  ['Efficiency gain vs RQMC (×)','Absolute efficiency: 1/(variance × time)']):
            for j,m in enumerate(ms):
                vals=[]
                for case in cs:
                    t=part[(part.case==case)&(part.method==m)]
                    vals.append(float(t[key].iloc[0]) if len(t) else np.nan)
                vals=np.asarray(vals,float);vals[~np.isfinite(vals)|(vals<=0)]=np.nan
                ax.bar(x+(j-2)*w,vals,w,label=m)
            if key==eff:ax.axhline(1,color='gray',ls='--')
            ax.set_yscale('log');ax.set_xticks(x,[c.replace('hybrid_','') for c in cs],rotation=30,ha='right')
            ax.set_title(heading);ax.grid(axis='y',alpha=.2)
        axs[1].legend(fontsize=8,ncol=2)
        fig.suptitle(f'E4 — Hybrid-method computational efficiency | N={nmax:,}')
        finish(fig,'05_hybrid_comparison_efficiency.png')

    # 06: preserve scatter format, showing where VRF does/does not yield efficiency.
    part=s[s.experiment.isin(['E3_geometry','E4_hybrid'])]
    if not part.empty:
        fig,ax=plt.subplots(figsize=(9,6))
        for m in ORDER:
            g=part[part.method==m]
            x=g.vrf_rqmc.to_numpy(float);y=g[eff].to_numpy(float)
            mask=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
            if mask.any():ax.scatter(x[mask],y[mask],label=m,alpha=.75,s=36)
        ax.axhline(1,color='gray',ls='--');ax.axvline(1,color='gray',ls='--')
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set_xlabel('VRF vs RQMC (×; higher is better)')
        ax.set_ylabel('Computational-efficiency gain vs RQMC (×; higher is better)')
        ax.set_title(f'E3/E4 — Variance reduction versus time-adjusted efficiency | N={nmax:,}')
        ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=7)
        finish(fig,'06_vrf_efficiency_tradeoff.png')

    # 07: same grouped-bar two-panel E5 layout; each available N is reported.
    part=d[d.experiment=='E5_selection']
    if not part.empty:
        cases=list(part.case.drop_duplicates());ms=[m for m in ORDER if m in set(part.method)]
        for nval in sorted(part.n.unique()):
            block=part[part.n==nval]
            fig,axs=plt.subplots(2,1,figsize=(max(12,len(cases)*1.7),11),sharex=True)
            x=np.arange(len(cases));w=.84/max(len(ms),1)
            for ax,key,heading in zip(axs,[eff,'computational_efficiency'],
                ['Efficiency gain vs RQMC (×; higher is better)',
                 'Absolute efficiency: 1/(variance × time) (higher is better)']):
                for j,m in enumerate(ms):
                    v=[]
                    for c in cases:
                        hit=block[(block.case==c)&(block.method==m)]
                        v.append(float(hit[key].iloc[0]) if len(hit) else np.nan)
                    v=np.asarray(v,float);v[~np.isfinite(v)|(v<=0)]=np.nan
                    ax.bar(x+(j-(len(ms)-1)/2)*w,v,w,label=m,color=METHOD_COLORS.get(m))
                if key==eff:ax.axhline(1,color='gray',ls='--')
                ax.set_yscale('log');ax.set_ylabel(heading);ax.grid(axis='y',alpha=.2)
            axs[-1].set_xticks(x,[c.replace('holdout_','') for c in cases],rotation=35,ha='right')
            axs[0].legend(ncol=4,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.36))
            fig.suptitle(f'E5 — Computational efficiency across eligible methods | N={int(nval):,}',y=1.02)
            name=('07_holdout_method_comparison_efficiency.png' if int(nval)==nmax
                  else f'07_holdout_method_comparison_efficiency_N{int(nval)}.png')
            finish(fig,name)

        # 08: replicate winner-grid layout, reporting winner and fold advantage.
        ns=sorted(part.n.unique())
        fig,axs=plt.subplots(1,2,figsize=(17,max(5,len(cases)*.48+2)),sharey=True)
        palette=plt.get_cmap('tab20');colors={m:palette(i/max(1,len(ms))) for i,m in enumerate(ms)}
        for ax,key,heading in zip(axs,['computational_efficiency',eff],
            ['Highest absolute efficiency','Highest efficiency gain vs RQMC']):
            for i,case in enumerate(cases):
                for j,nval in enumerate(ns):
                    z=valid(part[(part.case==case)&(part.n==nval)],key)
                    if z.empty:continue
                    winner=z.loc[z[key].idxmax()];m=str(winner.method)
                    ax.scatter(j,i,s=165,c=[colors.get(m,'gray')],marker='s')
                    ax.text(j,i,m.replace('RQMC+','R+').replace('CQMC+','C+'),fontsize=6.5,ha='center',va='center',
                            bbox=dict(boxstyle='round,pad=.1',facecolor='white',alpha=.72,edgecolor='none'))
            ax.set_xticks(range(len(ns)),[str(int(n)) for n in ns]);ax.set_xlabel('Paths per replicate (N)')
            ax.set_title(heading);ax.grid(alpha=.16)
        axs[0].set_yticks(range(len(cases)),[c.replace('holdout_','') for c in cases]);axs[0].invert_yaxis()
        fig.suptitle('E5 — Retrospective best computational-efficiency method (not trained selector)')
        finish(fig,'08_e5_efficiency_winners_by_n.png')

    # 09: original E2 two-row line chart layout, both absolute and relative efficiency.
    part=s[s.experiment=='E2_dimension']
    if not part.empty:
        kinds=list(part.kind.unique())
        fig,axs=plt.subplots(2,len(kinds),figsize=(7*len(kinds),8.8),squeeze=False)
        for i,(key,heading) in enumerate([('computational_efficiency','1/(variance × time); higher is better'),
                                          (eff,'Efficiency gain vs RQMC (×)')]):
            for ax,kind in zip(axs[i],kinds):
                sub=part[part.kind==kind]
                for m in ['RQMC','RQMC+BB','RQMC+PCA','RQMC+GPCA(proxy)']:
                    g=valid(sub[sub.method==m],key).sort_values('d')
                    if not g.empty:ax.plot(g.d,g[key],marker='o',label=m)
                if key==eff:ax.axhline(1,color='gray',ls='--')
                ax.set_xscale('log',base=2);ax.set_yscale('log')
                ax.set_xticks(sorted(sub.d.unique()))
                ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.set_title(NAMES.get(kind,kind));ax.set_xlabel('Monitoring dimension (d)')
                ax.set_ylabel(heading);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'E2 — Computational efficiency by dimension | N={nmax:,}')
        finish(fig,'09_e2_dimension_computational_efficiency.png')

def generate_report(data_dir, out=None, e1_e4_only=False):
    data_dir=Path(data_dir)
    src=data_dir/'study_summary.csv'
    if not src.is_file(): raise FileNotFoundError(f'Missing {src}. Run structure_comparisons.py first or pass the correct --data-dir.')
    out=Path(out) if out is not None else data_dir/'comparison_report'
    out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(src)
    if e1_e4_only:
        d=d[d.experiment!='E5_selection'].copy()
    needed={'case','n','method','kind','experiment','rmse','variance','total_time_s'}
    if needed-set(d):raise ValueError('Missing study_summary columns: '+str(sorted(needed-set(d))))
    if 'computational_efficiency' not in d:
        v=d.variance.to_numpy(float)*d.total_time_s.to_numpy(float)
        d['computational_efficiency']=np.where(v>1e-25,1/v,np.nan)
    # Rebuild missing ratio columns from matched case/N baselines if necessary.
    # The runner normally supplies both VRFs and computational-efficiency ratios.
    if 'vrf_rqmc' not in d: d['vrf_rqmc']=np.nan
    if 'vrf_mc' not in d: d['vrf_mc']=np.nan
    for (case, n), idx in d.groupby(['case','n']).groups.items():
        block=d.loc[idx]
        for baseline, key in [('RQMC','vrf_rqmc'),('MC','vrf_mc')]:
            b=block[block.method==baseline]
            if b.empty:continue
            bvar=float(b.variance.iloc[0])
            vv=block.variance.to_numpy(float)
            # Near-zero variance is not meaningfully estimable; do not plot huge VRFs.
            good=np.isfinite(vv)&(vv>1e-25)&np.isfinite(bvar)&(bvar>1e-25)
            d.loc[idx,key]=np.where(good,bvar/np.maximum(vv,1e-25),np.nan)
    d['payoff']=d.kind.map(NAMES).fillna(d.kind)
    d['rmse_improvement_vs_rqmc']=np.nan
    d['efficiency_improvement_vs_rqmc']=np.nan
    for (_, _), ids in d.groupby(['case','n']).groups.items():
        g=d.loc[ids]; base=g[g.method=='RQMC']
        if base.empty:continue
        rm=base.rmse.iloc[0]; ef=base.computational_efficiency.iloc[0]
        if np.isfinite(rm) and rm>0:d.loc[ids,'rmse_improvement_vs_rqmc']=rm/d.loc[ids,'rmse']
        if np.isfinite(ef) and ef>0:d.loc[ids,'efficiency_improvement_vs_rqmc']=d.loc[ids,'computational_efficiency']/ef
    d.sort_values(['experiment','case','n','method']).to_csv(out/'tidy_all_results.csv',index=False)
    nmax=int(d.n.max()); s=d[d.n==nmax].copy()
    s.to_csv(out/f'comparison_at_{nmax}.csv',index=False)
    winners=[]
    for (exp,case,n),g in d.groupby(['experiment','case','n']):
        for metric,ascending in [('rmse',True),('computational_efficiency',False)]:
            valid=g[np.isfinite(g[metric]) & (g[metric]>0)]
            if valid.empty:continue
            best=valid.sort_values(metric,ascending=ascending).iloc[0]
            winners.append(dict(experiment=exp,case=case,n=n,metric=metric,best_method=best.method,best_value=best[metric]))
    pd.DataFrame(winners,columns=['experiment','case','n','metric','best_method','best_value']).to_csv(out/'winners_by_case_and_n.csv',index=False)
    plt.rcParams.update({'font.size':10,'figure.dpi':125,'savefig.dpi':175,'axes.spines.top':False,'axes.spines.right':False})
    def save(name):plt.tight_layout();plt.savefig(out/name,bbox_inches='tight');plt.close()
    def label(case):
        r=s[s.case==case].iloc[0]
        return f'{r.payoff} | d={int(r.d)}' if 'd' in r else str(case)
    # 1/2: heatmaps across comparable E1/E3/E4 configurations
    for key,title,filename in [('rmse_improvement_vs_rqmc','RMSE improvement over RQMC','01_rmse_heatmap.png'),('efficiency_improvement_vs_rqmc','Efficiency improvement over RQMC','02_efficiency_heatmap.png')]:
        subset=s[s.experiment.isin(['E1_continuity','E3_geometry','E4_hybrid'])]
        if subset.empty:continue
        cases=list(subset.case.drop_duplicates());methods=[m for m in ORDER if m in set(subset.method)]
        a=subset.pivot(index='case',columns='method',values=key).reindex(index=cases,columns=methods)
        x=a.to_numpy(float); log=np.full(x.shape,np.nan);good=np.isfinite(x)&(x>0);log[good]=np.log10(x[good]);lim=max(1.,float(np.nanpercentile(np.abs(log),95))) if good.any() else 1.
        fig,ax=plt.subplots(figsize=(max(11,len(methods)*1.0),max(5,len(cases)*.56+2)))
        cmap=plt.get_cmap('RdYlGn').copy();cmap.set_bad('#d9dce1')
        im=ax.imshow(log,aspect='auto',cmap=cmap,vmin=-lim,vmax=lim)
        ax.set_xticks(range(len(methods)),methods,rotation=50,ha='right');ax.set_yticks(range(len(cases)),[label(c) for c in cases]);ax.set_title(f'{title} | N={nmax:,} (green is better)')
        for i in range(len(cases)):
            for j in range(len(methods)):
                if good[i,j]:ax.text(j,i,f'{x[i,j]:.1f}×' if .1<=x[i,j]<1000 else f'{x[i,j]:.0e}×',ha='center',va='center',fontsize=6.5)
        fig.colorbar(im,ax=ax,shrink=.75,label='log10 improvement');save(filename)
    # E2: preserve original RMSE figure; add efficiency side-by-side by dimension.
    part=s[s.experiment=='E2_dimension']
    if not part.empty:
        kinds=list(part.kind.unique())
        for metrics, filename in [(['rmse'],'03_dimension_scaling.png'),
                                  (['computational_efficiency'],'03_dimension_scaling_efficiency.png'),
                                  (['rmse','computational_efficiency'],'09_e2_dimension_rmse_efficiency.png')]:
            fig,axs=plt.subplots(len(metrics),len(kinds),figsize=(7*len(kinds),4.4*len(metrics)),squeeze=False)
            for i,key in enumerate(metrics):
                for ax,kind in zip(axs[i],kinds):
                    t=part[part.kind==kind]
                    for m in ['RQMC','RQMC+BB','RQMC+PCA','RQMC+GPCA(proxy)']:
                        g=t[t.method==m].sort_values('d');g=g[np.isfinite(g[key])&(g[key]>0)]
                        if not g.empty:ax.plot(g.d,g[key],marker='o',label=m)
                    ax.set_yscale('log');ax.set_xscale('log',base=2)
                    ax.set_xticks(sorted(t.d.unique()))
                    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax.set_title(NAMES.get(kind,kind));ax.set_xlabel('Monitoring dimension (d)')
                    ax.set_ylabel('RMSE (lower is better)' if key=='rmse' else '1 / (variance × time) (higher is better)')
                    ax.grid(alpha=.2);ax.legend(fontsize=8)
            fig.suptitle(f'E2 — Dimension scaling | N={nmax:,}')
            save(filename)
    # 4 continuity: grouped relative error on matching continuous and digital pairs
    part=s[s.experiment=='E1_continuity']
    if not part.empty:
        fig,ax=plt.subplots(figsize=(11,5))
        for kind,marker in [('geo_call','o'),('geo_digital','s'),('arith_call','^'),('arith_digital','D')]:
            g=part[part.kind==kind]
            if g.empty:continue
            ms=[m for m in ORDER if m in set(g.method)];v=[float(g[g.method==m].rmse_improvement_vs_rqmc.iloc[0]) for m in ms]
            ax.scatter([ORDER.index(m) for m in ms],v,label=NAMES[kind],marker=marker,s=55)
        ax.axhline(1,color='gray',linestyle='--');ax.set_yscale('log');ax.set_xticks(range(len(ORDER)),ORDER,rotation=45,ha='right');ax.set_ylabel('RMSE improvement vs RQMC (×)');ax.set_title('E1 — Continuous vs digital payoffs');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8);save('04_continuity_comparison.png')
    # 5 hybrid: compare three chosen methods across payoff configurations
    part=s[s.experiment=='E4_hybrid']
    if not part.empty:
        fig,axs=plt.subplots(1,2,figsize=(14,5))
        cs=list(part.case.unique());ms=['RQMC','OT-PCA','OT+GPCA','CQMC+PCA','CQMC+GPCA'];x=np.arange(len(cs));w=.16
        for ax,key,title in zip(axs,['rmse_improvement_vs_rqmc','efficiency_improvement_vs_rqmc'],['RMSE gain vs RQMC','Efficiency gain vs RQMC']):
            for j,m in enumerate(ms):
                vals=[part[(part.case==c)&(part.method==m)][key].iloc[0] if not part[(part.case==c)&(part.method==m)].empty else np.nan for c in cs]
                ax.bar(x+(j-2)*w,vals,w,label=m)
            ax.axhline(1,color='gray',ls='--');ax.set_yscale('log');ax.set_xticks(x,[c.replace('hybrid_','') for c in cs],rotation=30,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
        axs[1].legend(fontsize=8,ncol=2);fig.suptitle('E4 — OT, hybrid, and smoothing');save('05_hybrid_comparison.png')
    # 6 accuracy efficiency tradeoff
    part=s[s.experiment.isin(['E3_geometry','E4_hybrid'])]
    if not part.empty:
        fig,ax=plt.subplots(figsize=(9,6))
        for m in ORDER:
            g=part[part.method==m];x=g.efficiency_improvement_vs_rqmc.to_numpy(float);y=g.rmse_improvement_vs_rqmc.to_numpy(float);v=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
            if v.any():ax.scatter(x[v],y[v],label=m,alpha=.75,s=36)
        ax.axhline(1,c='gray',ls='--');ax.axvline(1,c='gray',ls='--');ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Efficiency improvement vs RQMC (×)');ax.set_ylabel('RMSE improvement vs RQMC (×)');ax.set_title('E3/E4 — Accuracy versus computational efficiency');ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=7);save('06_accuracy_efficiency_tradeoff.png')
    # 7: E5 retrospective across ALL eligible methods, not a hand-picked subset.
    # For every sample size show both RMSE and computational efficiency; invalid
    # or missing methods remain blank. A maximum-N-only plot misses small-N winners.
    part=d[d.experiment=='E5_selection']
    if not part.empty:
        cases=list(part.case.drop_duplicates())
        available_methods=[m for m in ORDER if m in set(part.method)]
        for n_value in sorted(part.n.unique()):
            block=part[part.n==n_value]
            fig,axes=plt.subplots(2,1,figsize=(max(12,len(cases)*1.7),11),sharex=True)
            x=np.arange(len(cases));width=0.84/max(len(available_methods),1)
            for ax,key,title in zip(axes,
                ['rmse_improvement_vs_rqmc','efficiency_improvement_vs_rqmc'],
                ['RMSE improvement vs RQMC (higher is better)',
                 'Computational efficiency vs RQMC (higher is better)']):
                for j,method in enumerate(available_methods):
                    vals=[]
                    for case in cases:
                        subset=block[(block.case==case)&(block.method==method)]
                        vals.append(float(subset[key].iloc[0]) if len(subset) else np.nan)
                    vals=np.asarray(vals,dtype=float)
                    vals[~np.isfinite(vals)|(vals<=0)]=np.nan
                    ax.bar(x+(j-(len(available_methods)-1)/2)*width,
                           vals,width,label=method,color=METHOD_COLORS.get(method))
                ax.axhline(1,c='gray',ls='--',lw=1)
                ax.set_yscale('log');ax.set_ylabel(title)
                ax.grid(axis='y',alpha=.2)
            axes[-1].set_xticks(x,[c.replace('holdout_','') for c in cases],rotation=35,ha='right')
            axes[0].legend(ncol=4,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.36))
            fig.suptitle(f'E5 — All applicable methods | N={int(n_value):,} | retrospective results',y=1.02)
            filename='07_holdout_method_comparison.png' if int(n_value)==nmax else f'07_holdout_method_comparison_N{int(n_value)}.png'
            save(filename)

        # Winner-focused across sample sizes: includes every method, including OT-PCA.
        winners_e5=pd.DataFrame(winners)
        winners_e5=winners_e5[winners_e5.experiment=='E5_selection'].copy()
        if not winners_e5.empty:
            winners_e5['objective']=winners_e5.metric.map({'rmse':'Lowest RMSE','computational_efficiency':'Highest efficiency'})
            winners_e5.to_csv(out/'e5_all_winners.csv',index=False)
            fig,axes=plt.subplots(1,2,figsize=(17,max(5,len(cases)*.48+2)),sharey=True)
            palette=plt.get_cmap('tab20')
            colors={m:palette(i/max(len(available_methods),1)) for i,m in enumerate(available_methods)}
            for ax,metric,heading in zip(axes,['rmse','computational_efficiency'],['Lowest RMSE','Highest efficiency']):
                subset=winners_e5[winners_e5.metric==metric]
                for i,case in enumerate(cases):
                    for j,n_value in enumerate(sorted(part.n.unique())):
                        hit=subset[(subset.case==case)&(subset.n==n_value)]
                        if hit.empty:continue
                        m=str(hit.iloc[0].best_method)
                        ax.scatter(j,i,s=165,c=[colors.get(m,'gray')],marker='s')
                        ax.text(j,i,m.replace('RQMC+','R+').replace('CQMC+','C+').replace('OT-','OT-'),
                                fontsize=6.5,ha='center',va='center',color='black',
                                bbox=dict(boxstyle='round,pad=.1',facecolor='white',alpha=.72,edgecolor='none'))
                ax.set_xticks(range(part.n.nunique()),[str(int(n)) for n in sorted(part.n.unique())]);ax.set_xlabel('Paths per replicate (N)');ax.set_title(heading)
                ax.grid(alpha=.16)
            axes[0].set_yticks(range(len(cases)),[c.replace('holdout_','') for c in cases]);axes[0].invert_yaxis()
            fig.suptitle('E5 — Observed best method by payoff and sample size (not predictive selection)')
            save('08_e5_winners_by_n.png')

    # Sample-size stability across E1–E5, not only retrospective E5.
    winners_df=pd.DataFrame(winners)
    if not winners_df.empty:
        rows=[]
        for (exp,metric,n),group in winners_df.groupby(['experiment','metric','n']):
            for method,count in group.best_method.value_counts().items():
                rows.append(dict(experiment=exp,metric=metric,n=int(n),method=method,
                                 wins=int(count),cases_with_winner=int(group.case.nunique())))
        stable=pd.DataFrame(rows)
        stable.to_csv(out/'method_wins_by_sample_size.csv',index=False)
        exps=list(d.experiment.drop_duplicates())
        fig,axes=plt.subplots(len(exps),2,figsize=(15,max(4,3.6*len(exps))),squeeze=False)
        for i,exp in enumerate(exps):
            for j,(metric,title) in enumerate([('rmse','Lowest RMSE'),('computational_efficiency','Highest efficiency')]):
                ax=axes[i,j]; g=stable[(stable.experiment==exp)&(stable.metric==metric)]
                ns=sorted(d[d.experiment==exp].n.unique())
                methods=[m for m in ORDER if m in set(g.method)]
                bottoms=np.zeros(len(ns))
                for m in methods:
                    counts=np.array([g[(g.n==n)&(g.method==m)].wins.sum() for n in ns],float)
                    if counts.any():ax.bar([str(int(n)) for n in ns],counts,bottom=bottoms,label=m)
                    bottoms+=counts
                ax.set_title(f'{exp}: {title}');ax.set_ylabel('Number of payoffs won')
                ax.set_xlabel('Paths per independent estimate N')
                ax.legend(fontsize=7,ncol=2,loc='upper left',bbox_to_anchor=(1.0,1.0)) if j==1 else None
        fig.suptitle('Observed winners change with sample size (not trained predictions)',y=1.002)
        save('10_winner_stability_by_n.png')

    plot_computational_efficiency_versions(d, out, nmax)
    plot_vrf_efficiency_comparisons(d, out, nmax)

    # Convergence exponents from the runner; never infer missing/invalid fits.
    rate_path=data_dir/'convergence_rates.csv'
    if rate_path.exists():
        rates=pd.read_csv(rate_path)
        required={'experiment','case','method','metric','exponent','n_points','r_squared'}
        if required.issubset(rates):
            rates.to_csv(out/'convergence_rates.csv',index=False)
            valid=rates[(rates.n_points>=3)&np.isfinite(rates.exponent)&np.isfinite(rates.r_squared)].copy()
            valid.to_csv(out/'convergence_rates_valid.csv',index=False)
            for metric,number,heading in [('rmse','11','RMSE decay exponent α'),('variance','12','Variance decay exponent β')]:
                selection=valid[valid.metric==metric]
                if selection.empty:continue
                for exp in list(selection.experiment.drop_duplicates()):
                    block=selection[selection.experiment==exp]
                    case_ids=list(block.case.drop_duplicates()); methods=[m for m in ORDER if m in set(block.method)]
                    mat=block.pivot_table(index='case',columns='method',values='exponent').reindex(index=case_ids,columns=methods)
                    values=mat.to_numpy(dtype=float)
                    fig,ax=plt.subplots(figsize=(max(10,len(methods)*.85),max(3.5,len(case_ids)*.42+1.5)))
                    good=np.isfinite(values); lim=max(1.,float(np.nanpercentile(np.abs(values[good]),95))) if good.any() else 1.
                    cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#dedede')
                    im=ax.imshow(values,aspect='auto',cmap=cmap,vmin=-lim,vmax=lim)
                    ax.set_xticks(range(len(methods)),methods,rotation=50,ha='right')
                    ax.set_yticks(range(len(case_ids)),case_ids,fontsize=8)
                    ax.set_title(f'{exp} — {heading} (descriptive log-log fits)')
                    for ii in range(len(case_ids)):
                        for jj in range(len(methods)):
                            if good[ii,jj]:ax.text(jj,ii,f'{values[ii,jj]:.2f}',ha='center',va='center',fontsize=7)
                    fig.colorbar(im,ax=ax,shrink=.8,label='Fitted exponent')
                    save(f'{number}_convergence_{metric}_{exp}.png')
            fig,ax=plt.subplots(figsize=(11,5))
            for method in ORDER:
                g=valid[(valid.metric=='rmse')&(valid.method==method)]
                if not g.empty:ax.scatter([method]*len(g),g.exponent,label=method,alpha=.6)
            ax.axhline(.5,color='gray',ls='--',label='MC benchmark α=0.5')
            ax.set_xticks(range(len(ORDER)),ORDER,rotation=55,ha='right')
            ax.set_ylabel('Empirical RMSE decay exponent α');ax.set_title('13 — Convergence exponents across payoffs')
            ax.grid(axis='y',alpha=.2);save('13_convergence_overview.png')
    else:
        (out/'CONVERGENCE_NOTE.txt').write_text(
            'No convergence_rates.csv found in the study folder. '
            'Regenerate the study with the updated structure_comparisons.py '
            'and at least three sample sizes to obtain exponent charts.\n')
    # evidence-based, per-run interpretation
    lines=[f'# Cross-payoff comparisons (N={nmax:,})','',f'Inputs: {len(d)} method rows, {d.case.nunique()} payoff configurations, {d.n.nunique()} sample sizes.','', 'Figures compare methods across payoffs; a missing cell means an unavailable method.', 'Rankings are descriptive and need uncertainty assessment from independent scrambles.', 'Do not interpret the E5 retrospective graph as a trained method-selection result.', '', '## Winning methods at maximum N','']
    for exp,g in s.groupby('experiment'):
        lines.append(f'### {exp} ({g.case.nunique()} cases)')
        for metric,asc in [('rmse',True),('computational_efficiency',False)]:
            valid=g[np.isfinite(g[metric])&(g[metric]>0)]
            wins=valid.loc[valid.groupby('case')[metric].idxmin() if asc else valid.groupby('case')[metric].idxmax()]
            lines.append(f'- {metric}: '+', '.join(f'{k} ({v})' for k,v in wins.method.value_counts().items()))
        lines.append('')
    lines+=['## Limitations','','- Variance-based efficiency is not bias-adjusted; check RMSE and reference uncertainty alongside it.','- Zero or near-zero estimated variance makes efficiency unreliable or not estimable.','- Timing depends on hardware and preprocessing-reuse policy.','- E5 includes all eligible methods in RMSE AND efficiency graphs at every N; 08_e5_winners_by_n.png tracks retrospective winners.','- A predictive E5 conclusion requires the separate Experiment 5 selector on held-out cases.','']
    (out/'INTERPRETATION.md').write_text('\n'.join(lines))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',default='structure_study');p.add_argument('--out',default=None);p.add_argument('--e1-e4-only',action='store_true')
    a=p.parse_args();print('Report:',generate_report(a.data_dir,a.out,e1_e4_only=a.e1_e4_only))
