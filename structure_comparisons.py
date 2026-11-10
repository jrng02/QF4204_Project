"""Five-experiment study: payoff structure versus RQMC construction.

Examples:
    python structure_comparisons.py --quick
    python structure_comparisons.py --powers 8 10 12 --replicates 24 --out study

MC/RQMC estimates use independent seeds/scrambles. The only exact OT directions
supported are terminal and geometric-average indicators. GPCA on unsmoothed
payoffs uses a differentiable SURROGATE and is explicitly labeled proxy.
CQMC integrates the first Brownian coordinate analytically; 'CQMC+BB/PCA'
are residual Brownian constructions; CQMC+GPCA uses gradients of the actual
conditional expectation. OT+GPCA is a proposed, not published, hybrid.
"""
import argparse
import csv
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.stats import norm
from samplers import Sampler
from discontinuity_treatment import GeneralizedOT, payoff_indicator_direction
from conditional_smoothing import StructuredConditionalQMC
from benchmarks import bs_vanilla, BS_digital, KV_geo_asian_fixedstrike, _geo_log_moments


@dataclass(frozen=True)
class Case:
    name: str
    experiment: str
    kind: str
    d: int
    S0: float = 100.
    K: float = 100.
    H: float = 105.
    r: float = .05
    sigma: float = .2
    T: float = 1.
    cash: float = 10.


def cases():
    a=[]
    # E1: matched continuous/discontinuous payoffs at fixed dimension
    for k in ('geo_call','geo_digital','arith_call','arith_digital'):
        a.append(Case('continuity_'+k,'E1_continuity',k,64))
    # E2: dimension effect on continuous payoff, avoiding discontinuity confound
    for k in ('geo_call','arith_call'):
        for d in (8,16,32,64,128,256):
            a.append(Case(f'dimension_{k}_{d}','E2_dimension',k,d))
    # E3: geometry of discontinuity
    for k in ('digital','geo_digital','arith_digital','barrier','geo_knockout'):
        a.append(Case('geometry_'+k,'E3_geometry',k,32))
    # E4: incremental contributions, mixed nonlinear payout / aligned boundary
    for d,H,K in ((16,105,100),(64,105,100),(64,103,95),(64,110,105)):
        a.append(Case(f'hybrid_{d}_H{H}_K{K}','E4_hybrid','geo_knockout',d,H=H,K=K))
    # E5: frozen 416-case factorial holdout, checked against all E1-E4 cases.
    from e5_contracts import e5_cases
    a.extend(e5_cases(Case, development_cases=a))
    return a


def payoff(c, paths):
    st=paths[:,-1]
    avg=paths.mean(axis=1)
    geo=np.exp(np.log(paths).mean(axis=1))
    if c.kind=='call': return np.maximum(st-c.K,0)
    if c.kind=='digital': return c.cash*(st>=c.K)
    if c.kind=='digital_put': return c.cash*(st<c.K)
    if c.kind=='geo_call': return np.maximum(geo-c.K,0)
    if c.kind=='geo_put': return np.maximum(c.K-geo,0)
    if c.kind=='arith_call': return np.maximum(avg-c.K,0)
    if c.kind=='arith_put': return np.maximum(c.K-avg,0)
    if c.kind=='geo_digital': return c.cash*(geo>=c.K)
    if c.kind=='geo_digital_put': return c.cash*(geo<c.K)
    if c.kind=='arith_digital': return c.cash*(avg>=c.K)
    if c.kind=='arith_digital_put': return c.cash*(avg<c.K)
    if c.kind=='barrier': return np.maximum(st-c.K,0)*(paths.min(axis=1)>c.H)
    if c.kind=='barrier_up_out_call': return np.maximum(st-c.K,0)*(paths.max(axis=1)<c.H)
    if c.kind=='geo_knockout': return np.maximum(avg-c.K,0)*(geo<c.H)
    if c.kind=='geo_knockout_reverse': return np.maximum(avg-c.K,0)*(geo>c.H)
    raise ValueError(c.kind)


def factor(c, method):
    return GeneralizedOT.brownian_factor(c.d,c.T,method)


def stock(c, Z, A):
    W=Z@A.T
    times=np.arange(1,c.d+1)*c.T/c.d
    return c.S0*np.exp((c.r-.5*c.sigma**2)*times[None,:]+c.sigma*W)


def smooth_surrogate(c, Z, A):
    """Only used to CHOOSE coordinates. Pricing always uses exact payoff."""
    p=stock(c,Z,A)
    geo=np.exp(np.log(p).mean(axis=1))
    avg=p.mean(axis=1)
    if c.kind in ('geo_digital','geo_digital_put','geo_call','geo_put'): return geo
    if c.kind in ('arith_digital','arith_digital_put','arith_call','arith_put','geo_knockout','geo_knockout_reverse'): return avg
    if c.kind in ('digital','digital_put','call'): return p[:,-1]
    if c.kind in ('barrier','barrier_up_out_call'): return .5*p[:,-1]+.5*(p.min(axis=1) if c.kind=='barrier' else p.max(axis=1))
    raise ValueError(c.kind)


def gradient_matrix(fn, pilot, eps=2e-4):
    n,d=pilot.shape
    grad=np.empty((n,d))
    for j in range(d):
        plus=pilot.copy();minus=pilot.copy()
        plus[:,j]+=eps;minus[:,j]-=eps
        grad[:,j]=(fn(plus)-fn(minus))/(2*eps)
    return grad.T@grad/n


def ordered_basis(M, first=None):
    d=M.shape[0]
    vals,V=np.linalg.eigh((M+M.T)*.5)
    order=np.argsort(vals)[::-1]
    selected=[]
    if first is not None: selected.append(first/np.linalg.norm(first))
    for idx in order:
        v=V[:,idx].copy()
        for _ in range(2):
            for q in selected: v-=np.dot(q,v)*q
        s=np.linalg.norm(v)
        if s>1e-9: selected.append(v/s)
        if len(selected)==d: break
    for j in range(d):
        if len(selected)==d:break
        v=np.eye(d)[:,j].copy()
        for _ in range(2):
            for q in selected:v-=np.dot(q,v)*q
        s=np.linalg.norm(v)
        if s>1e-9:selected.append(v/s)
    U=np.column_stack(selected)
    if not np.allclose(U.T@U,np.eye(d),atol=1e-9):
        raise RuntimeError('Could not construct orthogonal basis')
    return U


def applicable(c):
    m=['MC','RQMC']
    if c.d>1:m+=['RQMC+BB','RQMC+PCA','RQMC+GPCA(proxy)']
    if c.kind in ('digital','digital_put','geo_digital','geo_digital_put','geo_knockout','geo_knockout_reverse'):
        m+=['OT-Cholesky','OT-PCA','OT+GPCA']
    if c.d>1 and c.kind in ('digital','digital_put','geo_digital','geo_digital_put','arith_digital','arith_digital_put','barrier','barrier_up_out_call','geo_knockout','geo_knockout_reverse'):
        m+=['CQMC','CQMC+BB','CQMC+PCA','CQMC+GPCA']
    return m


def fit(c, methods, pilot_n, seed):
    prep={}; costs={m:0. for m in methods}
    A0=factor(c,'cholesky')
    pilot=Sampler.rqmc(pilot_n,c.d,seed)
    if 'RQMC+GPCA(proxy)' in methods:
        tic=time.perf_counter()
        M=gradient_matrix(lambda z:smooth_surrogate(c,z,A0),pilot)
        prep['gpca']=ordered_basis(M)
        costs['RQMC+GPCA(proxy)']=time.perf_counter()-tic
    for method in ('OT-Cholesky','OT-PCA','OT+GPCA'):
        if method not in methods:continue
        tic=time.perf_counter()
        Aq=factor(c, 'pca' if method=='OT-PCA' else 'cholesky')
        ot=GeneralizedOT(Aq,payoff_indicator_direction(c.kind,c.d))
        if method=='OT+GPCA':
            # OT first coordinate FIXED; GPCA only in its orthogonal complement.
            M=gradient_matrix(lambda z:smooth_surrogate(c,z,Aq),pilot)
            P=np.eye(c.d)-np.outer(ot.u1,ot.u1)
            U=ordered_basis(P@M@P, first=ot.u1)
            prep[method]=Aq@U
        else:prep[method]=Aq@ot.U
        costs[method]=time.perf_counter()-tic
    if any(m.startswith('CQMC') for m in methods):
        tic=time.perf_counter()
        cq=StructuredConditionalQMC(c.S0,c.K,c.T,c.r,c.sigma,c.d,
                                    barrier=c.H,digital_payoff=c.cash)
        prep['cq']=cq
        common=time.perf_counter()-tic
        for method in methods:
            if method.startswith('CQMC'):costs[method]=common
        if 'CQMC+GPCA' in methods:
            tic=time.perf_counter()
            z=Sampler.rqmc(pilot_n,c.d-1,seed+739)
            M=gradient_matrix(lambda x:cq.conditional_values(x,c.kind,'standard',c.K,c.H,c.cash),z)
            prep['cq_gpca']=ordered_basis(M)
            costs['CQMC+GPCA']+=time.perf_counter()-tic
    return prep,costs


def estimate(c,method,n,seed,prep):
    sampler='mc' if method=='MC' else 'rqmc'
    if method.startswith('CQMC'):
        Z=Sampler.get_samples(sampler,n,c.d-1,seed)
        construction='standard'
        if method=='CQMC+BB':construction='bb'
        if method=='CQMC+PCA':construction='pca'
        if method=='CQMC+GPCA':Z=Z@prep['cq_gpca'].T
        return float(prep['cq'].conditional_values(Z,c.kind,construction,c.K,c.H,c.cash).mean())
    Z=Sampler.get_samples(sampler,n,c.d,seed)
    if method=='RQMC+BB': A=factor(c,'cholesky') if c.d==1 else bb_factor(c)
    elif method=='RQMC+PCA':A=factor(c,'pca')
    elif method=='RQMC+GPCA(proxy)':A=factor(c,'cholesky')@prep['gpca']
    elif method in ('OT-Cholesky','OT-PCA','OT+GPCA'): A=prep[method]
    else:A=factor(c,'cholesky')
    return float(np.exp(-c.r*c.T)*payoff(c,stock(c,Z,A)).mean())


def bb_factor(c):
    # BB for complete Brownian levels via the same general bridge algorithm
    # as used for conditional residual Brownian levels.
    cq=StructuredConditionalQMC(c.S0,c.K,c.T*(c.d+1)/c.d,c.r,c.sigma,c.d+1)
    return cq._residual_factor('bb')


def reference(c,power=16,scrambles=8,seed=776):
    if c.kind=='call':return bs_vanilla(c.S0,c.K,c.r,c.T,c.sigma),0.,'exact'
    if c.kind=='digital':return BS_digital(c.S0,c.K,c.cash,c.r,c.T,c.sigma),0.,'exact'
    if c.kind=='geo_call':
        return KV_geo_asian_fixedstrike(c.S0,c.K,c.r,c.sigma,c.T,c.d,sampling='discrete'),0.,'exact'
    if c.kind in ('digital_put','geo_digital_put'):
        if c.kind=='digital_put':
            return c.cash*np.exp(-c.r*c.T)-BS_digital(c.S0,c.K,c.cash,c.r,c.T,c.sigma),0.,'exact'
        mu,var,*_=_geo_log_moments(c.S0,c.r,c.sigma,c.T,c.d,'discrete')
        return c.cash*np.exp(-c.r*c.T)*norm.cdf((np.log(c.K)-mu)/np.sqrt(var)),0.,'exact'
    if c.kind=='geo_put':
        mu,var,*_=_geo_log_moments(c.S0,c.r,c.sigma,c.T,c.d,'discrete')
        expected_geo=np.exp(mu+.5*var)
        call=KV_geo_asian_fixedstrike(c.S0,c.K,c.r,c.sigma,c.T,c.d,sampling='discrete')
        return call+np.exp(-c.r*c.T)*(c.K-expected_geo),0.,'exact'
    if c.kind=='geo_digital':
        mu,var,*_=_geo_log_moments(c.S0,c.r,c.sigma,c.T,c.d,'discrete')
        return c.cash*np.exp(-c.r*c.T)*norm.sf((np.log(c.K)-mu)/np.sqrt(var)),0.,'exact'
    if c.kind in ('arith_call','arith_put'):
        # Independent scrambled RQMC reference; no closed form.
        values=[estimate(c,'RQMC+PCA',2**power,seed+i*17,{}) for i in range(scrambles)]
    else:
        # Independent CQMC scrambles: analytically integrate the discontinuity.
        prep,_=fit(c,['CQMC'],32,seed)
        values=[estimate(c,'CQMC',2**power,seed+i*17,prep) for i in range(scrambles)]
    return float(np.mean(values)),float(np.std(values,ddof=1)/np.sqrt(len(values))), 'numerical'


def compute_summary(prices, ref, ref_se, mc_var, rqmc_var):
    x=np.asarray(prices)
    var=float(np.var(x,ddof=1)) if len(x)>=2 else np.nan
    rmse=float(np.sqrt(np.mean((x-ref)**2)))
    def ratio(base):
        if not np.isfinite(var) or var<=1e-25:return np.nan
        if not np.isfinite(base):return np.nan
        return float(base/var)
    return dict(price=float(x.mean()), variance=var, rmse=rmse,
                bias=float(x.mean()-ref), vrf_mc=ratio(mc_var),
                vrf_rqmc=ratio(rqmc_var),reference_se=ref_se,
                vrf_status='N/E (zero/near-zero observed variance)' if var<=1e-25 else 'estimated')


def computational_efficiency(variance, time_seconds, variance_floor=1e-25):
    """Inverse variance-time product for ONE independent price estimate.

    variance is the sample variance across independently scrambled price
    estimates (not the variance of the pooled mean). time_seconds is the
    mean wall-clock time per independent estimate, plus amortized setup.
    A zero/near-zero observed sample variance cannot justify infinite
    efficiency, so return NaN (not estimable).
    """
    if not (np.isfinite(variance) and np.isfinite(time_seconds)):
        return float('nan')
    if variance <= variance_floor or time_seconds <= 0:
        return float('nan')
    return float(1.0 / (variance * time_seconds))


def efficiency_ratio(efficiency, baseline_efficiency):
    """Relative computational efficiency against a comparable baseline."""
    if (np.isfinite(efficiency) and np.isfinite(baseline_efficiency)
            and efficiency > 0 and baseline_efficiency > 0):
        return float(efficiency / baseline_efficiency)
    return float('nan')


def convergence_analysis(rows, out):
    """Empirical log-log RMSE and variance rates for fixed case/method.

    RMSE approximately C*N^(-alpha), variance approximately C*N^(-beta).
    Descriptive regression only: noisy RQMC data, numerical-reference error,
    and potentially zero variance can make fitted slopes unreliable.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grouped = {}
    for r in rows:
        grouped.setdefault((r['experiment'], r['case'], r['method']), []).append(r)
    results = []
    for (exp, case, method), group in sorted(grouped.items()):
        group = sorted(group, key=lambda z:z['n'])
        for key, name in [('rmse', 'rmse'), ('variance', 'variance')]:
            valid = [r for r in group if np.isfinite(r[key]) and r[key] > 1e-25 and r['n'] > 0]
            if len(valid) < 3:
                results.append(dict(experiment=exp,case=case,method=method,metric=name,
                                    n_points=len(valid),exponent=np.nan,r_squared=np.nan,
                                    n_min=min((r['n'] for r in valid),default=np.nan),
                                    n_max=max((r['n'] for r in valid),default=np.nan),
                                    status='insufficient positive points'))
                continue
            x = np.log(np.array([r['n'] for r in valid], float))
            y = np.log(np.array([r[key] for r in valid], float))
            slope, intercept = np.polyfit(x, y, 1)
            fit = slope*x+intercept
            denom = np.sum((y-y.mean())**2)
            r2 = 1-np.sum((y-fit)**2)/denom if denom>0 else np.nan
            results.append(dict(experiment=exp,case=case,method=method,metric=name,
                                n_points=len(valid),exponent=float(-slope),r_squared=float(r2),
                                n_min=valid[0]['n'],n_max=valid[-1]['n'],status='descriptive fit'))
    write_csv(out/'convergence_rates.csv',results)
    # Plot method-level distributions of fitted RMSE exponents, separated by experiment.
    good = [r for r in results if r['metric']=='rmse' and np.isfinite(r['exponent'])]
    if good:
        experiments=sorted(set(r['experiment'] for r in good))
        fig,axs=plt.subplots(len(experiments),1,figsize=(12,3.2*len(experiments)),squeeze=False)
        for ax,exp in zip(axs[:,0],experiments):
            subset=[r for r in good if r['experiment']==exp]
            methods=sorted(set(r['method'] for r in subset))
            for i,m in enumerate(methods):
                vals=[r['exponent'] for r in subset if r['method']==m]
                ax.scatter([i]*len(vals),vals,alpha=.65,s=25)
                if vals:ax.plot(i,np.median(vals),'k_',markersize=14)
            ax.set_xticks(range(len(methods)),methods,rotation=40,ha='right',fontsize=8)
            ax.set_ylabel('Fitted RMSE exponent α')
            ax.set_title(exp+' (RMSE ≈ C × N^(-α))')
            ax.axhline(.5,ls='--',color='gray',linewidth=1)
            ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(out/'convergence_rates.png',dpi=160);plt.close(fig)
    return results


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def draw_plots(df,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for c in sorted(set(r['case'] for r in df)):
        rows=[r for r in df if r['case']==c and r['n']==max(x['n'] for x in df if x['case']==c)]
        rows=sorted(rows,key=lambda x:x['rmse'])
        fig,axes=plt.subplots(1,4,figsize=(21,5.2))
        methods=[r['method'] for r in rows];xx=np.arange(len(rows))
        for ax,key,title in zip(axes,['rmse','vrf_rqmc','vrf_mc','computational_efficiency'],['RMSE (lower is better)','VRF vs RQMC','VRF vs MC','Efficiency 1/(variance × time)']):
            vals=np.array([r[key] for r in rows],float)
            shown=np.where(np.isfinite(vals)&(vals>0),vals,np.nan)
            ax.bar(xx,np.nan_to_num(shown,nan=0),color='#527da5')
            ax.set_yscale('log');ax.set_title(title)
            ax.set_xticks(xx);ax.set_xticklabels(methods,rotation=65,ha='right',fontsize=7)
            for i,value in enumerate(shown):
                if np.isnan(value):ax.annotate('N/E', (i,1), rotation=90,fontsize=7,ha='center')
        fig.suptitle(c+f' | N={rows[0]["n"]} | N/E = not estimable',fontsize=12)
        fig.tight_layout();fig.savefig(out/(c+'.png'),dpi=140);plt.close(fig)


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    choices=cases()
    if getattr(args, 'experiments', None):
        choices=[c for c in choices if c.experiment in args.experiments]
    if args.quick:
        names={'continuity_geo_call','continuity_geo_digital',
               'dimension_arith_call_16','geometry_arith_digital',
               'geometry_barrier','geometry_geo_knockout',
               'hybrid_16_H105_K100','holdout_geo_knockout_32_K95_H108_V0.25',
               'holdout_geo_digital_16_K90_H105_V0.3'}
        choices=[c for c in choices if c.name in names]
        args.powers=[8];args.replicates=3;args.reference_power=11;args.reference_scrambles=4;args.pilot=32
    else:
        choices=[c for c in choices if not args.cases or c.name in args.cases]
    if not choices:raise ValueError('No cases selected')
    if args.reuse<1:raise ValueError('--reuse must be a positive integer')
    if args.replicates<2:raise ValueError('At least two independent replicates are required to estimate variance')
    allrows=[]; reps=[]; refrows=[]; applicability=[]
    for ic,c in enumerate(choices):
        methods=applicable(c)
        for m in ['MC','RQMC','RQMC+BB','RQMC+PCA','RQMC+GPCA(proxy)',
                  'OT-Cholesky','OT-PCA','OT+GPCA','CQMC','CQMC+BB','CQMC+PCA','CQMC+GPCA']:
            applicability.append(dict(case=c.name,kind=c.kind,method=m,
                                      applicable=int(m in methods)))
        ref,refse,refkind=reference(c,args.reference_power,args.reference_scrambles,args.seed+1000*ic)
        refrows.append(dict(case=c.name,kind=c.kind,reference=ref,reference_se=refse,type=refkind))
        prep,costs=fit(c,methods,args.pilot,args.seed+ic*101)
        for n in sorted({2**p for p in args.powers}):
            estimates={}; elapsed={}
            for method in methods:
                t0=time.perf_counter()
                estimates[method]=[estimate(c,method,n,args.seed+ic*100003+i*139,prep)
                                   for i in range(args.replicates)]
                elapsed[method]=(time.perf_counter()-t0)/args.replicates
            mcvar=float(np.var(estimates['MC'],ddof=1))
            rqvar=float(np.var(estimates['RQMC'],ddof=1))
            for m in methods:
                metrics=compute_summary(estimates[m],ref,refse,mcvar,rqvar)
                total_time=elapsed[m]+costs[m]/args.reuse
                efficiency=computational_efficiency(metrics['variance'],total_time)
                allrows.append(dict(case=c.name,experiment=c.experiment,kind=c.kind,d=c.d,
                                    K=c.K,H=c.H,sigma=c.sigma,method=m,n=n,
                                    replicates=args.replicates,**metrics,
                                    price_sd=(float(np.sqrt(metrics['variance'])) if metrics['variance']>=0 else np.nan),
                                     mean_price_se=(float(np.sqrt(metrics['variance']/args.replicates)) if metrics['variance']>=0 else np.nan),
                                     prep_s=costs[m],pricing_s=elapsed[m],
                                    total_time_s=total_time,
                                    computational_efficiency=efficiency,
                                    efficiency_status=('estimated' if np.isfinite(efficiency) else 'N/E'),
                                    reuse_count=args.reuse))
                for j,price in enumerate(estimates[m]):
                    reps.append(dict(case=c.name,method=m,n=n,replicate=j,price=price))
        print(f'{ic+1}/{len(choices)} {c.name} complete',flush=True)
    # Calculate relative efficiency only within a matched (case, sample size).
    by_baseline={(r['case'],r['n'],r['method']):r['computational_efficiency'] for r in allrows}
    for row in allrows:
        row['efficiency_ratio_mc']=efficiency_ratio(
            row['computational_efficiency'],by_baseline.get((row['case'],row['n'],'MC'),float('nan')))
        row['efficiency_ratio_rqmc']=efficiency_ratio(
            row['computational_efficiency'],by_baseline.get((row['case'],row['n'],'RQMC'),float('nan')))
    write_csv(out/'study_summary.csv',allrows)
    convergence_analysis(allrows,out)
    write_csv(out/'study_replicates.csv',reps)
    write_csv(out/'study_references.csv',refrows)
    write_csv(out/'method_applicability.csv',applicability)
    # Rule-based E5 evaluation is performed only by experiment5_payoff_selection.py.
    draw_plots(allrows,out)
    (out/'study_settings.json').write_text(json.dumps(vars(args),indent=2))
    if not args.no_comparison_report:
        from structure_report import generate_report
        report_dir=generate_report(out)
        print('Cross-payoff report:',report_dir)
    print('Outputs:',out)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--powers',type=int,nargs='+',default=[8,10,12,14])
    p.add_argument('--replicates',type=int,default=24)
    p.add_argument('--pilot',type=int,default=256)
    p.add_argument('--reference-power',type=int,default=16)
    p.add_argument('--reference-scrambles',type=int,default=8)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--reuse',type=int,default=1)
    p.add_argument('--cases',nargs='*')
    p.add_argument('--experiments',nargs='+',choices=['E1_continuity','E2_dimension','E3_geometry','E4_hybrid','E5_selection'],help='Run only the specified experiments')
    p.add_argument('--out',default='structure_study')
    p.add_argument('--no-comparison-report',action='store_true',help='Skip cross-payoff CSVs and figures')
    return p.parse_args()

if __name__=='__main__':run(parse_args())
