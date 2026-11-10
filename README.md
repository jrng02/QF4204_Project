# Payoff-aware RQMC for exotic derivative pricing

**Project:** E1–E5 comparison of discontinuity treatments and effective-dimension reduction for exotic options.  
**Documentation verified against:** `QMC_Final_Convergence_E5(8).zip` (the bundled `main.py`, supporting scripts, settings, and results).  
**Primary entry point:** `main.py`.

## Quick start

From the project directory:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest test_structure_comparisons.py test_e5_rule_based.py -q
python3 main.py --quick
```

The quick run writes to `structure_study_quick/`. It is a **software smoke test**, not research-grade evidence. For the already-packaged results, regenerate reports without pricing simulations:

```bash
python3 main.py --existing
```

To **keep the packaged E1–E4 measurements** while continuing/rerunning E5 as needed:

```bash
python3 main.py --reuse-e1-e4
```

To recompute E1–E4 and E5 from scratch (potentially costly, with changed timing and results):

```bash
python3 main.py
```

### Run modes, precisely

| Command | E1–E4 pricing | E5 pricing | Reports and E5 evaluations |
|---|---|---|---|
| `python3 main.py` | Recomputed | Run/continued | Regenerated |
| `python3 main.py --reuse-e1-e4` | Existing four CSVs reused | Run/continued | Regenerated |
| `python3 main.py --existing` | Existing results only | Existing results only | Regenerated |
| `python3 main.py --quick` | Small smoke-test run | Small smoke-test run | Quick reports |

`--existing` requires saved data, including E5 cases; unless `--allow-partial` is supplied, the reporting phase checks that all expected E5 contracts are present. `--reuse-e1-e4` requires `study_summary.csv`, `study_replicates.csv`, `study_references.csv` and `method_applicability.csv` in the output directory. A full run can take significant compute time; the existing results should be backed up before rerunning.

For checkpoints in smaller E5 batches, use:

```bash
python3 main.py --reuse-e1-e4 --e5-max-new 8
# Repeat until all 416 E5 contracts have completed
python3 main.py --existing
```

Alternatively, `python3 e5_run_only.py --max-new 8` advances E5 directly. This script checkpoints E5 to the four core CSVs, preserving E1–E4 rows. **Do not pass `--reset-e5` to `e5_run_only.py` unless you deliberately want to discard previously completed E5 measurements.** Do not mix results from different sample budgets or replications in one dataset.

## What is studied?

| Experiment | Question | Design in this project |
|---|---|---|
| **E1 — Payoff continuity** | How do dimension-reduction and discontinuity-treatment constructions perform for continuous versus digital Asian payoffs? | Geometric and arithmetic Asian calls versus digital counterparts; fixed `d=64` |
| **E2 — Dimension scaling** | How do BB, PCA and GPCA compare as dimension increases? | Continuous arithmetic and geometric Asian calls; `d = 8, 16, 32, 64, 128, 256` |
| **E3 — Discontinuity structure** | Which eligible OT/CQMC and dimension-reduction combinations work for different types of discontinuities? | Terminal digital, geometric Asian digital, arithmetic Asian digital, down-and-out call, and geometric-average knockout |
| **E4 — Hybrid payoff** | How do discontinuity treatment and dimension reduction interact? | Arithmetic-average call payoff with geometric-average knockout; four `(d,K,H)` configurations |
| **E5 — Payoff-aware selection** | Can simple rules derived from E1–E4 outperform fixed methods on held-out contracts? | 416-contract factorial holdout, evaluated against fixed baselines and a hindsight oracle |

### Methods and terminology

- **MC:** standard Monte Carlo; **RQMC:** randomized Sobol-based quasi-Monte Carlo.
- **BB:** Brownian Bridge; **PCA:** principal-component Brownian path construction; **GPCA:** gradient-based dimension reduction.
- **OT-Cholesky / OT-PCA:** orthogonal discontinuity alignment starting from different valid Brownian covariance factors; **OT+GPCA:** an experimental hybrid.
- **CQMC:** conditional quasi-Monte Carlo / analytical conditioning; **CQMC+BB/PCA/GPCA:** conditional smoothing combined with a residual-path dimension-reduction construction.
- **RQMC+GPCA(proxy)** is explicitly a surrogate-gradient proxy, not an exact implementation of the published conditional-smoothed GPCA method. The legacy `DiscontinuityILT` is an approximate LT proxy and is not included in the main comparison matrix.
- Exact one-direction OT is not claimed for arithmetic Asian digital or conventional pathwise barrier payoffs. Barrier conditions in the experiments are monitored **discretely**.

## E5 design and selection

The 416 holdout contracts comprise **160 non-barrier** cases (5 payoff families × 2 variants × 4 dimensions × 4 strikes) and **256 barrier-dependent** cases (2 families × 2 variants × 4 dimensions × 4 strikes × 4 barriers). E5 uses `d = 16, 32, 64, 128`, `K = 90, 95, 105, 110`, and fixed `sigma = 0.25`. Variant-specific barrier levels are defined in `e5_contracts.py`; the upper geometric-average knockout uses `H = 102, 104, 108, 112`. The full grid is supplied in `E5_HOLDOUT_DESIGN_416.csv`.

The E5 selector in `experiment5_payoff_selection.py` is a **prespecified rule-based approach**, not a fitted kNN or learned classifier. The rules are based on E1–E4 results and do not train on E5 outcomes:

| Payoff structure | Selected efficiency method |
|---|---|
| Continuous arithmetic/geometric Asian | RQMC+PCA |
| Terminal digital | CQMC+BB |
| Geometric Asian digital | CQMC+GPCA |
| Arithmetic Asian digital | CQMC+GPCA |
| Pathwise barrier | CQMC+BB |
| Geometric-average knockout | CQMC+GPCA |

The hindsight oracle selects the most efficient **eligible** method after observing E5 results. It is an evaluation benchmark, not a deployable prediction rule.

## Numerical setup and metrics

Default E1–E4 sample sizes are `N=256, 1024, 4096, 16384` (powers 8, 10, 12, 14). Default E5 uses `N=4096` (power 12) to contain the cost of 416 contracts. Both use 24 randomized replications per method by default, pilot size 256, reference power 16 and 8 reference scrambles. `--powers`, `--e5-powers`, `--replicates`, `--pilot`, `--reference-power`, and `--reference-scrambles` configure these choices; see `python3 main.py --help`.

The primary comparison is **computational efficiency**:

```text
sample variance s_P^2 = variance of independently scrambled price estimates
T = mean pricing time per estimate + preprocessing time / reuse_count
Efficiency = 1 / (s_P^2 * T)
Variance–time loss = s_P^2 * T
```

RMSE is a **secondary accuracy diagnostic** relative to an exact or numerical reference price; it is not the objective optimized by the E5 efficiency rule. VRF is a variance ratio against MC or RQMC, as specified by the relevant output column. Report reference-price standard errors when interpreting small RMSE differences. Empirical zero/near-zero variance can lead to **N/E (not estimable)** efficiency/VRF values; do not interpret them as zero pricing error or unlimited genuine improvement. Preprocessing reuse is an amortized timing model, not a controlled equal-wall-clock benchmark.

## Output locations and how to read them

All paths are relative to the project directory; `--out` can change the output root.

| File/folder | Contents |
|---|---|
| `structure_study/study_summary.csv` | Main E1–E5 method-by-case-by-sample-size results: price, variance, RMSE, VRF, timing, efficiency |
| `structure_study/study_replicates.csv` | Individual randomized price estimates |
| `structure_study/study_references.csv` | Reference prices and estimated reference uncertainty |
| `structure_study/method_applicability.csv` | Eligible/ineligible methods per contract |
| `structure_study/study_settings.json` | Settings for the E1–E4 runner |
| `structure_study/e5_run_settings.json` | Settings saved by the E5 runner |
| `structure_study/comparison_report/comparison_at_16384.csv` | Key **E1–E4** comparison rows used in paper tables at `N=16384` |
| `structure_study/comparison_report/tidy_all_results.csv` | Long-form comparative results |
| `structure_study/comparison_report/INTERPRETATION.md` | **Automatically generated** interpretation of the report results; not a second setup README |
| `structure_study/experiment5_efficiency_4096/` | E5 efficiency-rule decisions, aggregate loss, agreement, comparisons and figures |
| `structure_study/experiment5_rmse_4096/` | E5 RMSE-objective diagnostic evaluations (not the paper's primary selection criterion) |
| `structure_study/experiment5_extended_4096/` | Preprocessing-reuse sensitivity analysis |

Useful E5 files include `experiment5_efficiency_4096/experiment5_aggregate.csv`, `experiment5_decisions.csv`, `experiment5_evaluation.csv`, `experiment5_relative_per_case.csv`, `experiment5_winner_matches_by_variant.csv`, and `experiment5_unevaluable_cases.csv`. The last file documents contracts excluded from finite efficiency comparisons.

### Bundled E5 result snapshot

In ZIP (8), `study_summary.csv` contains **416 distinct E5 contracts**. In the current `experiment5_efficiency_4096/experiment5_aggregate.csv`, **404 are evaluable** for finite relative efficiency loss. The geometric-mean relative loss against standard RQMC is:

| Strategy | Geometric-mean relative variance–time loss |
|---|---:|
| Standard RQMC | 1.000000 |
| Fixed RQMC+PCA | 0.062862 |
| Simple continuity rule | 0.001055 |
| Structure-guided rule | 0.000680 |
| Hindsight oracle | 0.000374 |

Lower loss is better. These are **results packaged with ZIP (8)**, not guaranteed outputs from a fresh run: timing and numerical estimates can change when rerun or when the settings differ.

## How the workflow fits together

1. `structure_comparisons.py` creates E1–E4 data, unless the corresponding stage is skipped.
2. `e5_run_only.py` processes/checkpoints the 416 E5 contracts.
3. `structure_report.py --e1-e4-only` creates the cross-payoff report restricted to E1–E4; this intentionally does not regenerate the optional E5 convergence PNGs from a separate full-report invocation.
4. `experiment5_payoff_selection.py` creates E5 evaluations separately for `efficiency` and `rmse` objectives.
5. `experiment5_extensions.py` generates preprocessing-reuse analyses.
6. `structural_rules.py` runs legacy/descriptive rule output; it is not fitting a machine-learning selector.

The previously bundled `results.py`, `visualization.py` and `experiment5_payoff_selection_zero_rmse_fix.py` are **not invoked** by the current `main.py` pipeline. Keep tests (`test_structure_comparisons.py`, `test_e5_rule_based.py`) for reproducibility even though they are not run automatically by `main.py`.

## Reproducibility and interpretation cautions

- Preserve the **same version of the E1–E4 results** used to choose the E5 rules; timing-based efficiencies from different ZIP versions should not be silently mixed.
- E5 comprises 416 **contract configurations across seven payoff families**; this is not the same as 416 independent payoff types.
- A subset of rare or degenerate contracts has non-estimable efficiency ratios. E5's 404/416 finite-efficiency denominator must be stated when reporting aggregate performance.
- Fixed RQMC baselines and the hindsight oracle use the same held-out outcomes for evaluation. The oracle is retrospective and gives an optimistic bound, not a feasible prospective selector.
- A small observed difference from a numerical reference may be within the reference's own uncertainty.
- Output directories may contain figures left over from earlier runs. A current run does **not** necessarily delete old figures, and the presence of a PNG alone does not prove it was regenerated by `main.py`.

## Literature context

The implementation and experiments discuss discontinuity alignment and conditional smoothing informed by Wang and Tan (2013) and Xiao and Wang (2018). The applied GPCA proxy and OT+GPCA experimental hybrid should **not** be presented as exact reproductions of every published algorithm.

---

**Documentation policy:** This `README.md` replaces the overlapping root-level `RUN_ME_FIRST.md`, `INTEGRATION_README.md`, `README_RULE_BASED_E5.md` and outdated `STUDY_README.md`. Keep `structure_study/comparison_report/INTERPRETATION.md` because it is a generated analysis artifact with a different purpose.
