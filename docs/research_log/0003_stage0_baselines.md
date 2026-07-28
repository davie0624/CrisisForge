# Research Log 0003 — Stage 0 Public Baselines

Date: 2026-07-28  
Experiment: `stage0_public_baselines_v1`  
Status: `passed`  
Evaluation split: validation only

## 1. Purpose and claim boundary

Stage 0 establishes conventional scenario-generation baselines before any regime,
factor, diffusion, tail-calibration, mapping, or portfolio-optimization module is
allowed to make a comparative claim.

This run is a **validation-only rolling-origin comparison**. It is intended to:

- confirm that all baseline generators run under one common interface;
- create minimum distributional and tail-risk performance references;
- identify trade-offs that later models must beat; and
- expose metric failures before the research design is frozen.

It is not a final model selection. The test evaluation branch was not executed, no
test-origin metric appears in the artifacts, and the test split remains sealed for
the later frozen-design evaluation.

## 2. Frozen configuration

The authoritative configuration is `configs/stage0_baselines.yaml`.

| Item | Value |
|---|---:|
| Forecast horizon | 20 target-calendar observations |
| Origin stride | 20 observations |
| Rolling training lookback | 1,500 observations |
| Scenarios per model and origin | 1,000 |
| Random seed | 20260728 |
| VaR / ES confidence level | 95% |
| Co-crash marginal threshold quantile | 5% |
| Co-crash minimum asset fraction | \(1/3\) |
| Asset count | 15 |
| Portfolio used for VaR / ES | Equal weight, \(1/15\) per asset |

The first origin is 2013-12-31 with a forecast window ending 2014-01-30. The last
origin is 2019-11-01 with a window ending 2019-12-03. Every forecast path begins
strictly after its origin and must end by the validation cutoff.

There are 74 origins. With seven models, the design contains:

\[
74 \times 7 = 518
\]

model-origin evaluations.

At each origin, a model is refit on at most the latest 1,500 asset-return rows,
including the origin and excluding its future 20-observation response path. Later
validation origins may use validation returns that were already observable by that
origin; this is sequential rolling-origin evaluation rather than one static
train-only fit.

## 3. Fixed train-only co-crash definition

Co-crash thresholds are deliberately insulated from validation outcomes.

1. Use only target returns dated on or before the initial train cutoff,
   2013-12-31.
2. Form every overlapping 20-observation cumulative simple-return outcome in that
   initial training sample.
3. For each of the 15 assets, freeze its 5th-percentile cumulative-return
   threshold.
4. At a validation origin, define a realized co-crash when at least one third of
   assets—at least 5 of 15—finish below their frozen marginal thresholds.
5. Use the same threshold vector for all 74 origins and all seven models.

The thresholds are not re-estimated on validation outcomes and do not vary by
model. This makes predicted probabilities comparable and prevents threshold
selection from leaking validation crises into the event definition.

## 4. Baseline models

| Model ID | Implemented baseline |
|---|---|
| `historical_iid` | I.i.d. resampling of historical return rows |
| `moving_block_20` | Moving-block bootstrap with block length 20 |
| `filtered_historical_ewma` | EWMA-filtered historical simulation with decay 0.94 |
| `gaussian_shrinkage` | Multivariate Gaussian with 0.05 covariance shrinkage |
| `student_t_elliptical` | Elliptical multivariate Student-\(t\) with 0.05 covariance shrinkage |
| `student_t_copula` | Student-\(t\) copula with 6 degrees of freedom and 0.05 shrinkage |
| `var1_residual_bootstrap` | Ridge-stabilized VAR(1) with residual bootstrap; ridge \(10^{-6}\) |

Every model receives the same rolling training rows, forecast horizon, scenario
count, realized path, equal-weight portfolio, and frozen co-crash thresholds.
Seeds are deterministic by model number and origin number.

## 5. Evaluation metrics

| Metric | Interpretation |
|---|---|
| Mean energy score | Multivariate distributional accuracy; lower is better |
| Mean variogram score | Cross-asset dependence accuracy; lower is better |
| Joint VaR–ES score | Strictly consistent 95% VaR/ES loss score; lower is better |
| VaR violation rate | Should be close to the nominal 5% |
| Kupiec \(p\)-value | Unconditional VaR coverage diagnostic |
| Christoffersen independence \(p\)-value | Violation clustering diagnostic |
| Christoffersen conditional-coverage \(p\)-value | Joint coverage diagnostic |
| Co-crash Brier score | Squared probability error; informative only when the evaluation sample contains adequate positive and negative events |

The \(p\)-values are diagnostic, not evidence that a model is true. With only 74
origins, these tests have limited power.

## 6. Validation results

The summary file is ordered by mean energy score.

| Energy rank | Model | Energy | Variogram | Joint VaR–ES | VaR violations | CC \(p\) | Co-crash Brier |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `filtered_historical_ewma` | 0.089739 | 0.542024 | -0.947758 | 2/74 (2.70%) | 0.5791 | 0.001656 |
| 2 | `moving_block_20` | 0.089811 | **0.532226** | -0.949032 | 4/74 (5.41%) | 0.7831 | 0.000902 |
| 3 | `var1_residual_bootstrap` | 0.091236 | 0.563301 | -0.948535 | 2/74 (2.70%) | 0.5791 | 0.001106 |
| 4 | `student_t_copula` | 0.091478 | 0.584204 | -0.947724 | 2/74 (2.70%) | 0.5791 | 0.001367 |
| 5 | `student_t_elliptical` | 0.091630 | 0.578555 | **-0.949064** | 3/74 (4.05%) | 0.8163 | 0.001454 |
| 6 | `historical_iid` | 0.091732 | 0.571600 | -0.946065 | 2/74 (2.70%) | 0.5791 | 0.001497 |
| 7 | `gaussian_shrinkage` | 0.091803 | 0.582730 | -0.948546 | 3/74 (4.05%) | 0.8163 | 0.001216 |

Important trade-offs:

- `filtered_historical_ewma` has the lowest mean energy score, but its advantage
  over `moving_block_20` is only about 0.000073. No paired uncertainty interval or
  significance test has yet established that this difference is durable.
- `moving_block_20` has the best variogram score, is almost tied on energy, and has
  the violation rate closest to the nominal 5%.
- `student_t_elliptical` has the lowest joint VaR–ES score, with
  `moving_block_20` almost tied. It does not lead the multivariate distributional
  scores.
- Models with 2/74 violations are below the nominal 5% rate and may be conservative;
  their non-rejection by coverage tests does not establish superior calibration.
- `mean_realized_loss` is `-0.006284932951818759` for every model because all
  methods are scored against the same realized equal-weight portfolio path. It is
  a property of the evaluation sample, not a model-ranking statistic.

No single baseline dominates all objectives. The current evidence supports
retaining at least EWMA-filtered historical simulation, moving-block bootstrap,
and elliptical Student-\(t\) as distinct benchmark families rather than declaring
one universal winner.

## 7. Critical co-crash prevalence audit

The realized co-crash count is:

\[
0/74 = 0\%.
\]

This is too low for the current Brier comparison to evaluate crisis-event
detection. With every outcome equal to zero, each model's Brier score reduces to
the average squared predicted probability. A lower score primarily rewards a model
for assigning smaller probabilities; it does not demonstrate:

- ability to detect an actual co-crash;
- recall or discrimination between crisis and non-crisis windows;
- calibration conditional on a positive event; or
- realistic crisis severity.

The apparent best Brier score for `moving_block_20` must therefore **not** be
described as superior co-crash forecasting. Mean predicted co-crash probabilities
were:

| Model | Mean predicted probability | Realized prevalence |
|---|---:|---:|
| `filtered_historical_ewma` | 2.819% | 0% |
| `historical_iid` | 2.735% | 0% |
| `student_t_copula` | 2.574% | 0% |
| `student_t_elliptical` | 2.449% | 0% |
| `var1_residual_bootstrap` | 2.370% | 0% |
| `moving_block_20` | 2.299% | 0% |
| `gaussian_shrinkage` | 2.239% | 0% |

### Required follow-up before a co-crash claim

Keep the test split sealed and perform a separately registered,
validation-only sensitivity analysis:

- vary the training-frozen marginal threshold and minimum asset fraction;
- examine alternative horizons;
- use denser validation origins, with dependence-aware uncertainty because those
  outcomes will overlap;
- report event prevalence before reporting Brier or calibration rankings;
- inspect pre-specified validation stress windows separately from full-period
  averages; and
- if positive validation events remain too scarce, retain co-crash as an
  information-limited metric and use controlled semi-synthetic crisis experiments
  for event-recovery diagnostics.

Threshold or event-definition changes must be declared sensitivity analyses, not
post-hoc replacements selected to manufacture positive events.

## 8. Execution and failure record

The receipt records:

| Field | Value |
|---|---|
| Status | `passed` |
| Successful model-origins | 518 |
| Failed model-origins | 0 |
| Models | 7 |
| Origins | 74 |
| Elapsed time | 6.4966 seconds |
| Config SHA-256 | `602dfe0c1a897d2fa05b66439f26d31cf421040005de3cb908985ffe385f9c3e` |
| Phase 0 manifest SHA-256 used by the run | `5991a8f204c9f40d9116e45531fd76ca13abfd0f32f5439aa1e5ba62e594791c` |

`artifacts/stage0_baselines/failures.json` is an empty list. Zero implementation
failures means the common runner completed; it does not mean every statistical
model is adequate.

Durable outputs:

- `artifacts/stage0_baselines/rolling_results.csv`: 518 model-origin records;
- `artifacts/stage0_baselines/summary.csv`: seven model summaries;
- `artifacts/stage0_baselines/failures.json`: failure ledger; and
- `artifacts/stage0_baselines/run_receipt.json`: timestamped execution receipt.

## 9. Decision

Stage 0 passes as an implementation and validation-screening milestone.

It does **not** select a final CrisisForge model. The observed metric differences
are small, objectives disagree, co-crash outcomes contain no positive event, and no
paired uncertainty comparison has yet been reported. These validation results may
guide the next preregistered experiments and baseline shortlist only.

The test split remains unopened for model comparison. It should be evaluated once,
after the regime/factor/diffusion/tail design and all selection rules are frozen.
