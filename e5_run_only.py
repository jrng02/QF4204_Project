#!/usr/bin/env python3
"""Incremental, resumable E5-only simulator. Never resimulates E1-E4.

A FULL FACTORIAL DESIGN CONTAINS 416 CONTRACTS, NOT 672:
  (5 non-barrier families x 2 variants x 4 dimensions x 4 strikes) = 160
  (2 barrier families x 2 variants x 4 dimensions x 4 strikes x 4 barriers) = 256

Examples:
  python e5_run_only.py --max-new 8         # first batch (8 contracts)
  python e5_run_only.py --max-new 8         # next batch; resume automatically
  python e5_run_only.py                     # continue remaining contracts
  python e5_run_only.py --quick             # smoke-test the first 3 cases

Each completed contract is checkpointed to four CSVs. Existing E1-E4 CSV records
are never modified, and completed E5 contracts are skipped on resumption.
"""
from pathlib import Path
import argparse, csv, hashlib, json
from time import perf_counter
import numpy as np
import pandas as pd
from structure_comparisons import (cases, applicable, fit, estimate, reference,
    compute_summary, computational_efficiency, efficiency_ratio)

METHODS=('MC','RQMC','RQMC+BB','RQMC+PCA','RQMC+GPCA(proxy)',
         'OT-Cholesky','OT-PCA','OT+GPCA','CQMC','CQMC+BB','CQMC+PCA','CQMC+GPCA')
FILES=('study_summary.csv','study_replicates.csv','study_references.csv','method_applicability.csv')


def load_existing(folder):
    missing=[name for name in FILES if not (folder/name).is_file()]
    if missing: raise FileNotFoundError('Missing cached E1-E4 files: '+', '.join(missing))
    return {name:pd.read_csv(folder/name) for name in FILES}


def replace_csv_case(filename, case_name, new_rows, valid_holdout_names):
    """Preserve existing non-target CSV lines literally; replace one E5 case atomically.

    Stale E5 cases from the previous 8/28-case design are also dropped.
    """
    with filename.open('r',newline='') as f:
        first=f.readline()
        names=next(csv.reader([first]));position=names.index('case')
        experiment_pos=names.index('experiment') if 'experiment' in names else None
        kept=[]
        for line in f:
            if not line.strip():continue
            parts=next(csv.reader([line]));name=parts[position]
            if name==case_name:continue
            if experiment_pos is not None:
                if parts[experiment_pos]=='E5_selection' and name not in valid_holdout_names:continue
            elif name.startswith('holdout_') and name not in valid_holdout_names:
                continue
            kept.append(line)
    temp=filename.with_suffix(filename.suffix+'.tmp')
    with temp.open('w',newline='') as f:
        f.write(first)
        f.writelines(kept)
        if kept and not kept[-1].endswith(('\n','\r')):f.write('\n')
        writer=csv.DictWriter(f,fieldnames=names,extrasaction='ignore',lineterminator='\n')
        for item in new_rows:writer.writerow(item)
    temp.replace(filename)


def complete_cases(data, requested_powers, reps):
    summary=data['study_summary.csv'];rep=data['study_replicates.csv']
    refs=data['study_references.csv'];app=data['method_applicability.csv']
    sizes={2**p for p in requested_powers}
    present=set()
    for c in [c for c in cases() if c.experiment=='E5_selection']:
        methods=set(app.loc[(app.case==c.name)&(app.applicable==1),'method'])
        if not methods or c.name not in set(refs.case): continue
        s=summary[(summary.case==c.name)&summary.n.isin(sizes)]
        r=rep[(rep.case==c.name)&rep.n.isin(sizes)]
        summary_ok=all(set(s[s.n==n].method)==methods for n in sizes)
        repl_ok=all(all((r[(r.n==n)&(r.method==m)].replicate.nunique()==reps) for m in methods) for n in sizes)
        if summary_ok and repl_ok: present.add(c.name)
    return present


def run(a):
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
    all_cases=[c for c in cases() if c.experiment=='E5_selection']
    assert len(all_cases)==416
    valid={c.name for c in all_cases}
    if a.replicates<2 or a.reuse<1:raise ValueError('replicates >=2 and reuse >=1 required')
    if a.quick:
        all_cases=all_cases[:3] if not a.cases else [c for c in all_cases if c.name in a.cases]
        a.powers=[8];a.replicates=3;a.pilot=24;a.reference_power=10;a.reference_scrambles=3
    elif a.cases:
        all_cases=[c for c in all_cases if c.name in a.cases]
        unknown=set(a.cases)-{c.name for c in all_cases}
        if unknown:raise ValueError('Unknown E5 cases: '+str(sorted(unknown)))
    cached=load_existing(folder)
    if a.reset_e5:
        # Opt-in reset: rebuild E5 without altering E1-E4 rows.
        for fname in FILES:
            src=folder/fname
            with src.open('r',newline='') as f:
                first=f.readline();columns=next(csv.reader([first]))
                case_i=columns.index('case');exp_i=columns.index('experiment') if 'experiment' in columns else None
                lines=[]
                for line in f:
                    if not line.strip():continue
                    parts=next(csv.reader([line]));name=parts[case_i]
                    old=(parts[exp_i]=='E5_selection') if exp_i is not None else name.startswith('holdout_')
                    if not old:lines.append(line)
            dst=src.with_suffix(src.suffix+'.tmp')
            with dst.open('w',newline='') as f:f.write(first);f.writelines(lines)
            dst.replace(src)
        cached=load_existing(folder)
    done=complete_cases(cached,a.powers,a.replicates)
    pending=[c for c in all_cases if c.name not in done]
    if a.max_new is not None:
        if a.max_new<1:raise ValueError('--max-new must be positive')
        pending=pending[:a.max_new]
    print(f'E5 grid: {len(valid)} total, {len(done)} completed for N={list(2**q for q in a.powers)}, {len(pending)} in this batch',flush=True)
    for index,c in enumerate(pending,1):
        seed=a.seed+int.from_bytes(hashlib.sha256(c.name.encode()).digest()[:4],'big')%1000000
        allowed=applicable(c)
        ref,refse,refkind=reference(c,a.reference_power,a.reference_scrambles,seed+991)
        refrows=[dict(case=c.name,kind=c.kind,reference=ref,reference_se=refse,type=refkind)]
        applicability=[dict(case=c.name,kind=c.kind,method=m,applicable=int(m in allowed)) for m in METHODS]
        prep,cost=fit(c,allowed,a.pilot,seed)
        summary=[];replicates=[]
        for n in sorted({2**q for q in a.powers}):
            vals={};elapsed={}
            for m in allowed:
                start=perf_counter()
                vals[m]=[estimate(c,m,n,seed+i*139,prep) for i in range(a.replicates)]
                elapsed[m]=(perf_counter()-start)/a.replicates
            mcvar=np.var(vals['MC'],ddof=1);rqvar=np.var(vals['RQMC'],ddof=1)
            for m in allowed:
                metrics=compute_summary(vals[m],ref,refse,mcvar,rqvar)
                total=elapsed[m]+cost[m]/a.reuse
                efficiency=computational_efficiency(metrics['variance'],total)
                summary.append(dict(case=c.name,experiment='E5_selection',kind=c.kind,d=c.d,
                    K=c.K,H=c.H,sigma=c.sigma,method=m,n=n,replicates=a.replicates,
                    **metrics,price_sd=float(np.sqrt(metrics['variance'])),
                    mean_price_se=float(np.sqrt(metrics['variance']/a.replicates)),
                    prep_s=cost[m],pricing_s=elapsed[m],total_time_s=total,
                    computational_efficiency=efficiency,
                    efficiency_status='estimated' if np.isfinite(efficiency) else 'N/E',reuse_count=a.reuse))
                for j,price in enumerate(vals[m]):
                    replicates.append(dict(case=c.name,method=m,n=n,replicate=j,price=price))
        lookup={(r['n'],r['method']):r['computational_efficiency'] for r in summary}
        for r in summary:
            r['efficiency_ratio_mc']=efficiency_ratio(r['computational_efficiency'],lookup.get((r['n'],'MC'),np.nan))
            r['efficiency_ratio_rqmc']=efficiency_ratio(r['computational_efficiency'],lookup.get((r['n'],'RQMC'),np.nan))
        for name, rows in [('study_references.csv',refrows),
                           ('method_applicability.csv',applicability),
                           ('study_replicates.csv',replicates),
                           ('study_summary.csv',summary)]:
            replace_csv_case(folder/name,c.name,rows,valid)
        print(f'E5 batch {index}/{len(pending)} complete: {c.name}',flush=True)
    with (folder/'e5_run_settings.json').open('w') as f:json.dump(vars(a),f,indent=2)
    s=pd.read_csv(folder/'study_summary.csv',usecols=['case','experiment','n'])
    for n in sorted({2**q for q in a.powers}):
        count=s.loc[(s.experiment=='E5_selection')&(s.n==n),'case'].nunique()
        print(f'E5 checkpoint N={n}: {count}/416 cases',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out',default='structure_study')
    p.add_argument('--powers',nargs='+',type=int,default=[12])
    p.add_argument('--replicates',type=int,default=24)
    p.add_argument('--pilot',type=int,default=256)
    p.add_argument('--reference-power',type=int,default=16)
    p.add_argument('--reference-scrambles',type=int,default=8)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--reuse',type=int,default=1)
    p.add_argument('--cases',nargs='*')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--max-new',type=int,default=None,help='Limit new cases per invocation; rerun to resume')
    p.add_argument('--reset-e5',action='store_true',help='Delete E5 rows, never E1-E4')
    run(p.parse_args())
if __name__=='__main__':main()
