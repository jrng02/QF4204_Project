E5 RULE-BASED HOLDOUT TEST

Rules are PREDECLARED based on E1-E4, not trained with E5.

All 416 holdout contracts are simulated before retrospective evaluation; performance used only after predictions.

New put/reversed-knockout/up-and-out variants share structural-family rules with E1-E4.

Computational efficiency is primary; RMSE objective is supporting/sensitivity analysis.

The `simple_continuity` policy is a strong benchmark: PCA for continuous options, CQMC+GPCA for discontinuous.

Hindsight oracle knows E5 measured performance; it is not a deployable selector.

A zero estimated variance is not proof of perfect pricing, and RMSE depends on reference uncertainty.
