#!/usr/bin/env python3
"""Complete E1–E5 study: pricing, figures, rule-based E5 selection, extensions.

  python main.py                  # FRESH E1–E4, then 416 E5 cases, reports and selection
  python main.py --existing       # Rebuild reports / selection using complete saved data
  python main.py --reuse-e1-e4    # Keep saved E1–E4, simulate NEW E5, all reports
  python main.py --quick          # Short ALL-STAGES test in structure_study_quick

E5 is the rule-based construction; there is no kNN selector.
Note: A fresh full run is computationally expensive and overwrites output CSVs.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ('E1_continuity','E2_dimension','E3_geometry','E4_hybrid')


def call(script, *args):
    cmd = [sys.executable, str(ROOT / script), *map(str, args)]
    print('\nRUN:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def shared_sample_size(out, requested, require_all=True):
    import pandas as pd
    from structure_comparisons import cases
    d = pd.read_csv(out / 'study_summary.csv', usecols=['case','experiment','n'])
    present = set(d.experiment.dropna())
    missing = set(EXPERIMENTS + ('E5_selection',)) - present
    if missing:
        raise RuntimeError('Missing experiment results: ' + ', '.join(sorted(missing)))
    e5 = d[d.experiment.eq('E5_selection')]
    expected = {c.name for c in cases() if c.experiment == 'E5_selection'}
    found = set(e5.case)
    unexpected = found - expected
    if unexpected:
        raise RuntimeError('Unrecognized E5 cases in saved data: ' + ', '.join(sorted(unexpected)))
    if require_all and expected - found:
        raise RuntimeError(f'Only {len(found)}/{len(expected)} E5 cases available. Run python main.py --reuse-e1-e4 to simulate the remaining holdouts.')
    available = set(e5.n.astype(int)) & set(d[d.experiment.ne('E5_selection')].n.astype(int))
    if not available:
        raise RuntimeError('No shared sample size between E1–E4 and E5. Add --powers including the desired size to your simulation.')
    if requested not in available:
        raise RuntimeError(f'Requested N={requested} is unavailable; shared sample sizes: {sorted(available)}. Set --n accordingly.')
    return requested, len(found)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--existing', action='store_true', help='No simulations; regenerate reports and evaluate existing E5 holdouts')
    mode.add_argument('--reuse-e1-e4', action='store_true', help='Use saved E1–E4; only simulate the new 416 E5 holdouts')
    p.add_argument('--quick', action='store_true', help='Fast E1–E5 pipeline smoke test; writes to structure_study_quick by default')
    p.add_argument('--allow-partial', action='store_true', help='Allow a partial E5 dataset during report-only smoke testing')
    p.add_argument('--out', default=None, help='Output data directory; default structure_study (quick: structure_study_quick)')
    p.add_argument('--n', type=int, default=None, help='N for E5 selection report (default 4096; quick: 256)')
    p.add_argument('--powers', nargs='+', type=int, default=None, help='Log2 sample sizes for fresh E1–E4 and E5 runs; default 8 10 12 14')
    p.add_argument('--replicates', type=int, default=24)
    p.add_argument('--pilot', type=int, default=256)
    p.add_argument('--reference-power', type=int, default=16)
    p.add_argument('--reference-scrambles', type=int, default=8)
    p.add_argument('--e5-max-new',type=int,default=None,help='Simulate at most this many new E5 cases then exit; rerun to resume')
    p.add_argument('--e5-powers',type=int,nargs='+',default=[12],help='E5 sample powers; defaults to 12 (N=4096) to control factorial-grid cost')
    p.add_argument('--reuse-counts', nargs='+', type=int, default=[1,10,100,1000])
    a = p.parse_args()
    if a.quick and a.existing:
        p.error('--quick and --existing cannot be combined; use --out structure_study_quick --existing instead')
    if a.quick and a.reuse_e1_e4:
        p.error('--quick and --reuse-e1-e4 cannot be combined')
    out = Path(a.out or ('structure_study_quick' if a.quick else 'structure_study')).expanduser()
    if not out.is_absolute(): out = ROOT / out
    powers = a.powers or [8,10,12,14]
    n = a.n if a.n is not None else (256 if a.quick else 4096)
    if not a.existing:
        if not a.reuse_e1_e4:
            # Keep the original E1-E4 experiment definitions, saved replicate
            # structure, convergence computation, and case-level figures.
            opts = ['--out',out,'--experiments',*EXPERIMENTS,
                    '--powers',*powers,'--replicates',a.replicates,
                    '--pilot',a.pilot,'--reference-power',a.reference_power,
                    '--reference-scrambles',a.reference_scrambles,
                    '--no-comparison-report']
            if a.quick: opts.insert(0,'--quick')
            call('structure_comparisons.py',*opts)
        elif not all((out / filename).exists() for filename in
                     ('study_summary.csv','study_replicates.csv','study_references.csv','method_applicability.csv')):
            p.error(f'Cannot reuse E1–E4: cached CSVs missing in {out}')
        # Add E5 as a second phase. Preserve the E1-E4 records exactly when
        # --reuse-e1-e4 is used; on a fresh run E1–E4 are newly computed.
        e5powers = [8] if a.quick else a.e5_powers
        e5args = ['--out',out,'--powers',*e5powers,'--replicates',a.replicates,
                  '--pilot',a.pilot,'--reference-power',a.reference_power,
                  '--reference-scrambles',a.reference_scrambles]
        if a.quick: e5args.insert(0,'--quick')
        if a.e5_max_new is not None: e5args.extend(['--max-new',a.e5_max_new])
        call('e5_run_only.py',*e5args)
    elif not (out / 'study_summary.csv').exists():
        p.error(f'No cached summary file found at {out}')

    # Restore the complete descriptive report covering E1, E2, E3, E4 AND E5.
    call('structure_report.py','--data-dir',out,'--e1-e4-only')
    if a.e5_max_new is not None:
        from structure_comparisons import cases
        import pandas as pd
        df=pd.read_csv(out/'study_summary.csv',usecols=['case','experiment','n'])
        seen=set(df.loc[(df.experiment=='E5_selection')&(df.n==n),'case'])
        total=sum(c.experiment=='E5_selection' for c in cases())
        if len(seen)<total:
            print(f'E5 checkpoint saved: {len(seen)}/{total} contracts at N={n}. Rerun same command to resume.')
            print('E1–E4 comparison report completed. E5 evaluation waits for the full grid.')
            return
    n, count = shared_sample_size(out,n,require_all=not (a.quick or a.allow_partial))
    print(f'\nRule-based E5 selection: {count} holdouts at N={n:,}',flush=True)
    for objective in ('rmse','efficiency'):
        call('experiment5_payoff_selection.py','--data-dir',out,'--objective',objective,
             '--n',n,'--out',out / f'experiment5_{objective}_{n}')
    call('experiment5_extensions.py','--data-dir',out,'--n',n,
         '--reuse-counts',*a.reuse_counts,'--out',out / f'experiment5_extended_{n}')
    # Legacy structural_rules.py is purely descriptive and does not fit any model.
    call('structural_rules.py')
    print('\nALL E1–E5 STAGES COMPLETED. Results:',out,flush=True)


if __name__ == '__main__':
    main()
