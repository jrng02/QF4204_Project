#!/usr/bin/env python3
"""Rule-based E5 sensitivity to amortized preprocessing cost (no trained selector)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from experiment5_payoff_selection import select_rule

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='structure_study');p.add_argument('--out',default=None)
    p.add_argument('--n',type=int,default=4096);p.add_argument('--reuse-counts',type=int,nargs='+',default=[1,10,100,1000])
    a=p.parse_args()
    inp=Path(a.data_dir);out=Path(a.out) if a.out else inp/f'experiment5_extended_{a.n}'
    out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(inp/'study_summary.csv');df=df[(df.experiment=='E5_selection')&(df.n==a.n)].copy()
    app=pd.read_csv(inp/'method_applicability.csv');allowed={(r.case,r.method) for r in app.itertuples() if bool(r.applicable)}
    rows=[]
    for reuse in a.reuse_counts:
        df['variance_time']=df.variance*(df.pricing_s+df.prep_s/reuse)
        for case,g in df.groupby('case'):
            ok=[m for m in g.method if (case,m) in allowed]
            m,_,_,_=select_rule(g.iloc[0].kind,ok,'efficiency')
            z=g.set_index('method');base=float(z.loc['RQMC','variance_time'])
            valid=g[np.isfinite(g.variance_time)&(g.variance_time>0)]
            if valid.empty:continue
            oracle=valid.sort_values('variance_time').iloc[0].method
            for role,chosen in [('rule_based',m),('simple_continuity','RQMC+PCA' if g.iloc[0].kind in ('geo_call','arith_call','geo_put','arith_put') else 'CQMC+GPCA'),('fixed_PCA','RQMC+PCA'),('RQMC','RQMC'),('oracle',oracle)]:
                if chosen not in z.index:chosen='RQMC'
                val=float(z.loc[chosen,'variance_time'])
                rows.append(dict(reuse=reuse,case=case,role=role,method=chosen,
                    relative_variance_time=val/base if base>0 else np.nan))
    ans=pd.DataFrame(rows);ans.to_csv(out/'preprocessing_reuse_per_case.csv',index=False)
    summary=[]
    for (reuse,role),g in ans.groupby(['reuse','role']):
        values=np.asarray(g.relative_variance_time,float);values=values[np.isfinite(values)&(values>0)]
        summary.append(dict(reuse=reuse,role=role,n_cases=len(values),
            geometric_mean_relative_loss=float(np.exp(np.mean(np.log(values)))) if len(values) else np.nan))
    pd.DataFrame(summary).to_csv(out/'preprocessing_reuse_aggregate.csv',index=False)
    print('E5 rule-based preprocessing reuse results:',out)
if __name__=='__main__':main()
