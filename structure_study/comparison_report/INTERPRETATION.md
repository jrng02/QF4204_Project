# Cross-payoff comparisons (N=16,384)

Inputs: 772 method rows, 25 payoff configurations, 4 sample sizes.

Figures compare methods across payoffs; a missing cell means an unavailable method.
Rankings are descriptive and need uncertainty assessment from independent scrambles.
Do not interpret the E5 retrospective graph as a trained method-selection result.

## Winning methods at maximum N

### E1_continuity (4 cases)
- rmse: RQMC+GPCA(proxy) (2), CQMC+GPCA (2)
- computational_efficiency: RQMC+PCA (2), CQMC+GPCA (2)

### E2_dimension (12 cases)
- rmse: RQMC+GPCA(proxy) (9), RQMC+PCA (3)
- computational_efficiency: RQMC+PCA (9), RQMC+GPCA(proxy) (3)

### E3_geometry (5 cases)
- rmse: CQMC+PCA (2), CQMC+GPCA (2), CQMC+BB (1)
- computational_efficiency: CQMC+GPCA (3), CQMC+BB (2)

### E4_hybrid (4 cases)
- rmse: CQMC+GPCA (1), OT+GPCA (1), CQMC+PCA (1), OT-Cholesky (1)
- computational_efficiency: CQMC+GPCA (3), CQMC+PCA (1)

## Limitations

- Variance-based efficiency is not bias-adjusted; check RMSE and reference uncertainty alongside it.
- Zero or near-zero estimated variance makes efficiency unreliable or not estimable.
- Timing depends on hardware and preprocessing-reuse policy.
- E5 includes all eligible methods in RMSE AND efficiency graphs at every N; 08_e5_winners_by_n.png tracks retrospective winners.
- A predictive E5 conclusion requires the separate Experiment 5 selector on held-out cases.
