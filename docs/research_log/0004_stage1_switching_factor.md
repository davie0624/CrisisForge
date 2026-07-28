# Research Log 0004 — Stage 1 Switching Dynamic Factor Model

**Completed at (UTC):** `2026-07-28T01:46:45.547334+00:00`  
**Experiment:** `stage1_switching_factor_validation_v1`  
**Status:** completed with zero implementation failures  
**Evaluation split:** validation only; post-2019 rows governed-excluded

## Objective and estimator

Stage 1 implements the first structured asset generator:

\[
y_t=\log(1+r_t)
=\alpha(z_t)+B(z_t)f_t+D(z_t)\epsilon_t,
\qquad r_t=\operatorname{expm1}(y_t).
\]

The multi-stage estimator uses a four-factor train-only PCA representation, a
four-state sticky Gaussian HMM fitted by MAP/EM, regime-specific VAR(1) factor
dynamics, and a state-dependent observation mapping. It is accurately described
as a **MAP/empirical-Bayes baseline**, not as a full Bayesian posterior model.

Parameters are estimated once through the training cutoff. At each validation
origin, newly observed data update only the filtered state probability and current
factor. PCA, HMM, factor dynamics, and mapping parameters are not refitted.

## Registered configuration and provenance

Authoritative files:

- `configs/stage1_model.yaml`, SHA-256
  `f0f07ad5add4b43e6cef2d01c5eb0ddfd21c4d4e9fe092a81349dbcad7c5ff1d`;
- `configs/stage1_evaluation.yaml`, SHA-256
  `c8acfa1774a5dda8be9b6b3f5a2c2ea6377a0b6cc44cc0dbcffeba3f87f98b7a`;
- Phase 0 manifest SHA-256
  `f051ec35236a481858b67c5b1e7136f1698036832f427efba521dbf3fcd36d70`;
- model-matrix SHA-256
  `2a57dd7e8b43dc0d0444d4035cd3d45f1fef340e4a9faaea61fd9ac694cfe699`;
- Git commit
  `b6891133bdb6b96e1e23c6bea3bd033ea9685c7c`, clean worktree.

Key settings:

| Item | Registered value |
|---|---:|
| Return transform | `log1p` |
| Factors / HMM states | 4 / 4 |
| HMM initializations / maximum iterations | 12 / 300 |
| Sticky / transition pseudocount | 8.0 / 0.5 |
| Factor dynamics | state-specific ridge VAR(1), ridge \(10^{-4}\) |
| Residual-correlation shrinkage | 0.20 |
| Forecast horizon / stride | 20 / 20 observations |
| Scenarios per origin | 1,000 |
| Validation origins | 74 |
| Risk confidence | 95% |
| Estimation policy | fixed train parameters; sequential filtering |

## Fit diagnostics

The four factors explain `0.859949350394612` of train-return variance. Train
reconstruction RMSE is `0.005220481665907866`; maximum absolute reconstruction
error is `0.07026187159427963`.

The selected HMM initialization is 5 of 12. EM converged in 37 iterations with
log likelihood `-20129.444563068246`. State occupancies and ex-post market
profiles are:

| State | Occupancy | Mean per interval | Volatility per interval |
|---:|---:|---:|---:|
| 0 | 0.052324273381272074 | -0.001316767635757479 | 0.026391408374454646 |
| 1 | 0.2798314457770426 | -0.000214947501032292 | 0.010011165332063575 |
| 2 | 0.26165677335477755 | 0.0006731685955202115 | 0.008398786454201414 |
| 3 | 0.4061875074869277 | 0.0007117135315459657 | 0.00547551144158634 |

Transition diagonal probabilities are
`[0.857934152146799, 0.9394179220078734, 0.9866839352244431,
0.9718434955977109]`. All state-specific factor VARs are stable; spectral radii
are `[0.23847555961753927, 0.07419311751400483, 0.061409793110852234,
0.08825921478305979]`. No pooled fallback state was required.

State-specific mapping RMSEs are
`[0.010173051314476772, 0.006778549841522621, 0.0032248653035109164,
0.0035060133495632443]`. Residual correlation matrices remained positive
definite after shrinkage, with minimum eigenvalues approximately 0.20.

## Validation metrics

| Metric | Result |
|---|---:|
| Mean energy score | 0.09431537953403446 |
| Mean variogram score | 0.5886385781509599 |
| Joint VaR–ES score | -0.9491237156405643 |
| VaR violations | 1 / 74 (0.013513513513513514) |
| Kupiec \(p\)-value | 0.08936748664034849 |
| Christoffersen independence \(p\)-value | 0.8676302264382525 |
| Conditional-coverage \(p\)-value | 0.23299124388316328 |
| Mean co-crash Brier score | 0.0026815270270270273 |
| Realized co-crashes | 0 / 74 |
| Mean filtered-state entropy | 0.4787696132105992 |
| Fit / evaluation time | 472.7507475420134 / 26.142408874991816 seconds |

## Negative results and claim boundary

- Stage 1 does **not** beat the strongest conventional baselines on energy or
  variogram score. The paired evidence is recorded in Log 0005.
- The joint VaR–ES score is close to the better Stage 0 values, but paired
  intervals cross zero for every baseline comparison.
- One VaR breach is below the nominal 5% rate and is consistent with conservative
  forecasts; coverage-test non-rejection is not proof of calibration.
- No realized co-crash occurs, so the Brier score cannot establish crisis-event
  discrimination.
- State numbers are latent statistical labels ordered by the first-factor mean.
  They have no automatic economic or causal names.
- The run propagates a filtered state belief, not posterior uncertainty over all
  HMM, factor, and mapping parameters.

## Open issues

1. Separate model-class effects from the different estimation policies used by
   rolling Stage 0 and frozen-parameter Stage 1.
2. Evaluate sensitivity to factor dimension, state count, and state-label
   instability while keeping post-2019 rows governed-excluded.
3. Develop a positive-event design before making any co-crash forecasting claim.
4. Quantify how state-belief and mapping uncertainty propagate into asset risk and
   portfolio decisions.

## Durable artifacts

- `artifacts/stage1_switching_factor/cumulative_scenarios.npz`
  (`295c3fd5ce6c825dd642f5de7080a3a9b91e5c4164c3be99218ec7edb70659af`);
- `artifacts/stage1_switching_factor/rolling_results.csv`;
- `artifacts/stage1_switching_factor/summary.json`;
- `artifacts/stage1_switching_factor/diagnostics.json`;
- `artifacts/stage1_switching_factor/state_profiles.csv`;
- `artifacts/stage1_switching_factor/train_filtered_probabilities.parquet`;
- `artifacts/stage1_switching_factor/train_smoothed_probabilities.parquet`;
- `artifacts/stage1_switching_factor/failures.json`; and
- `artifacts/stage1_switching_factor/run_receipt.json`.
