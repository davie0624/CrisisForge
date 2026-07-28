# Research Proposal

## CrisisForge: Decision-Focused Market Simulation under Regime Shifts

### Factor-Structured Temporal Diffusion, Asset-Level Tail Risk, and Robust Portfolio Control

**Version:** 0.3
**Date:** 2026-07-28
**Audience:** quantitative finance, financial econometrics, risk management, and
machine-learning researchers

## Technical summary

CrisisForge asks whether a posterior-aware, regime-switching latent-factor scenario
generator can improve out-of-sample multi-asset tail-risk forecasts and portfolio
decisions. It does not try to predict the date of the next crisis. Its principal
estimand is the conditional future path distribution

\[
\mathcal P_t^H=P(r_{t+1:t+H}\mid\mathcal F_t),
\]

where \(t\) indexes a common target-panel interval endpoint, \(r_t\) is the vector
of decimal **simple** returns over that endpoint interval, and \(\mathcal F_t\)
contains only information available after the close at endpoint \(t\). The public
targets are neither excess returns nor log returns. A forecast made after close
\(t\) begins with \(r_{t+1}\); it never predicts the already observed interval
\(r_t\).

The project combines:

1. a Bayesian switching dynamic factor state-space model;
2. a non-autoregressive, one-shot diffusion model for nonlinear factor-path
   residuals;
3. a regime-dependent stochastic observation mapping,
   \[
   r_t=\alpha_{z_t}+B_{z_t}f_t+D_{z_t}\epsilon_t;
   \]
4. tail-conditioned generation, peaks-over-threshold calibration, and sequential
   VaR calibration;
5. asset-level VaR, Expected Shortfall (ES), and co-crash scoring;
6. empirical CVaR and tractable Wasserstein-DRO portfolio allocation; and
7. a separate structural counterfactual extension validated first in a
   semi-synthetic market with known causal ground truth.

The study has three deliberately separate tracks:

- **Predictive:** honest rolling forecasts issued after close \(t\) for intervals
  beginning at \(t+1\).
- **Stress:** user-conditioned severe but model-consistent scenarios; not assigned
  calibrated forecast probabilities.
- **Counterfactual:** abduction–action–prediction in a known time-unrolled
  structural causal model. Real-market results are called model-based structural
  interventions unless an external identification strategy is available.

## Motivation and research gap

Historical simulation, multivariate Student-\(t\), copulas, VAR/DCC-GARCH, and block
bootstrap remain strong and interpretable risk baselines. Modern diffusion models
can represent nonlinear, multimodal joint distributions but face three acute
financial-data problems: high dimension with short histories, changing dependence
across regimes, and too few crisis observations.

Recent diffusion-factor work argues that low-dimensional structure can mitigate
the high-dimensional small-sample problem, but the most direct finance-specific
paper remains a preprint. Accepted time-series diffusion work supports
non-autoregressive multi-horizon generation, while accepted regime-switching factor
research supports state-dependent loadings and volatility. These literatures have
not yet produced a single, carefully validated system that:

- propagates regime and parameter uncertainty through asset reconstruction;
- evaluates tail risk at asset and portfolio level;
- compares statistical realism with realized decision value;
- distinguishes predictive scenarios from causal counterfactuals; and
- quantifies the trade-off between central fit, tail calibration, and portfolio
  robustness.

CrisisForge targets this gap. The novelty claim is conditional on empirical results
and will be reduced if a component does not beat strong baselines.

## Research questions

### RQ1 — Factor structure

Does factor-space generation improve held-out multivariate proper scores and
estimation stability relative to direct asset-level generation at the same model
capacity?

### RQ2 — Regime uncertainty

Does posterior-predictive regime conditioning improve crisis-window distribution,
VaR–ES, and co-crash scores relative to unconditional conditioning and a
hard-label diagnostic ablation?

### RQ3 — Observation mapping

Do regime-dependent \(B_z\), \(D_z\), and residual correlation reproduce
correlation breaks and lower-tail dependence better than a fixed mapping?

### RQ4 — Tail calibration

How much central-distribution fit must be surrendered to improve VaR–ES calibration?
The result is evaluated as a Pareto frontier, not a single winner-takes-all metric.

### RQ5 — Decision value

Do models that best match the data also produce the best realized out-of-sample
CVaR, drawdown, turnover, and transaction-cost outcomes?

### RQ6 — Uncertainty propagation

Does posterior predictive propagation improve interval coverage and portfolio
weight stability relative to plug-in posterior means?

### RQ7 — Counterfactual validity

In a known structural market, does the counterfactual module recover intervention
response curves, tail effects, and individual paths better than conditional-only
generation?

## Pre-registered hypotheses

- **H1:** Factor diffusion improves held-out energy/variogram scores and reduces
  run-to-run variance relative to direct asset diffusion.
- **H2:** Soft posterior regime conditioning improves crisis-window energy,
  joint VaR–ES, and co-crash scores relative to unconditional and hard-decoded
  diagnostic variants.
- **H3:** Regime-dependent \(B_z,D_z\) improves held-out tail covariance and
  co-crash calibration relative to fixed \(B,D\).
- **H4:** Tail calibration improves joint VaR–ES scores and ESR diagnostics but may
  worsen central-distribution scores.
- **H5:** Generator ranking by distributional score differs from ranking by
  realized portfolio decision loss.
- **H6:** Posterior predictive propagation reduces risk under-coverage and weight
  instability compared with plug-in estimation.
- **H7:** In the semi-synthetic structural market, the counterfactual module
  improves ATE, impulse-response, and counterfactual distribution error relative
  to a conditional generator.

## Data

### Public research track

Phase 0 uses a retrospective `research_core` panel with 15 return targets:

- 12 value-weighted U.S. industry research portfolios from the Kenneth French Data
  Library; and
- 2-, 5-, and 10-year Treasury duration-convexity return proxies derived from
  official daily par yields.

The industry portfolios are research constructs, not tradable ETFs. The Treasury
series are transparent return proxies, not observed bond total returns. This
wording is enforced throughout the report.

The target calendar is formed from endpoints shared by the industry and Treasury
sources. Each row compounds every source's simple returns over the common holding
interval \((e_{t-1},e_t]\). A row is therefore a **common-endpoint interval**, not
an assertion that every market supplied exactly one identical exchange session.
The interval audit records calendar duration and source-observation counts.

The public predictive context contains ten variables:

- U.S. Treasury 2-, 5-, and 10-year par-yield levels and the 10y–2y slope;
- New York Fed EFFR; and
- five backward-looking features derived from the 12 industry returns: the
  equal-weighted industry-market return, 20-observation realized volatility,
  20-observation downside semivolatility, 20-observation cross-industry
  dispersion, and 60-observation market drawdown.

Treasury and EFFR context values enter only after their catalogued one-model-session
availability lag. Return-derived context through endpoint \(t\) is available only
for the after-close \(t\rightarrow t+1\) forecast. OFR stress components are stored
as external validation labels and excluded from the model matrix, regime
estimation, and diffusion context to avoid circularity.

The clean public model panel contains **6,465 common-endpoint rows**, 15 simple-return
targets, and 10 context variables from 2000-07-05 through 2026-05-29.

Cboe website histories are not part of the public research core. A downloadable
VIX, SKEW, or VVIX file is not treated as permission for public ML training or model
release. Option-index histories enter only a separately licensed extension through
Cboe DataShop, OptionMetrics, or another source whose terms permit the intended use.

FRED/ALFRED are not used. Their current Terms of Use prohibit using FRED Services
or Content to develop or train machine-learning, deep-learning, or generative-AI
software. All configured series are therefore obtained from their original
publishers.

### Publication-grade track

Preferred paid replacements are CRSP/WRDS, Bloomberg, LSEG, or FactSet for
point-in-time security and ETF total returns; OptionMetrics or Cboe DataShop for
option-implied information; ICE for full credit/OAS and MOVE histories; CME
DataMine for roll-adjusted futures; and Haver for point-in-time macro data.

The exact variables, transformations, assumed availability lags, licenses, and
substitutes are defined in `docs/data_dictionary.md` and
`configs/data_catalog.yaml`.

## Method

### Stage 0 — Research-grade data and conventional baselines

- Local raw snapshots, source URLs, retrieval time, and hashes; restricted raw data
  remain uncommitted and unredistributed.
- Common-endpoint interval aggregation, an interval audit, availability-aware
  backward-as-of context alignment, and no backward fill.
- Chronological nested splits and later rolling-origin folds.
- Historical simulation, block bootstrap, multivariate Student-\(t\), Student-\(t\)
  copula, VAR, DCC-GARCH, and HMM-WGAN comparator.
- Verified VaR, ES, co-crash, and portfolio-loss engine before using diffusion.

### Stage 1 — Bayesian switching dynamic factor core

Estimate a joint switching state-space model for regime, factors, loadings, and
idiosyncratic scale/correlation. The initial search is \(K\in\{2,3\}\),
\(q\in\{3,4,5\}\), \(H=20\) common-endpoint intervals. Semantic names are assigned
only after state diagnostics. States with occupancy below 5% trigger a lower-\(K\)
model.

Only filtered probabilities are permitted in forecasting. Smoothed probabilities
are restricted to retrospective plots and diagnostics. Hard-decoded regimes are
diagnostic ablations only; production forecasts propagate the soft filtered
posterior and draw full posterior-predictive regime paths. OFR labels may assess
state interpretation after estimation but never determine or order states.

### Stage 2 — One-shot conditional factor diffusion

The linear state-space model supplies a conditional factor-path location and scale.
Diffusion learns the entire standardized nonlinear residual path
\(H\times q\) in one shot. Context contains historical filtered factors, filtered
regime probabilities, vintage-safe market features, and predictive regime
probabilities. Future realized regimes never enter the context.

Ablations:

1. direct asset diffusion;
2. unconditional factor diffusion;
3. hard-decoded-regime factor diffusion as a diagnostic negative control;
4. soft posterior regime-conditioned factor diffusion;
5. posterior-predictive versus plug-in reconstruction.

### Stage 3 — Tail layer

- Tail-conditioned mixture with inverse-probability correction.
- Training-only threshold and reference portfolio.
- Regime-hierarchical POT-GPD for standardized portfolio loss.
- Horizon-specific sequential VaR calibration.
- Joint VaR–ES scores and ESR diagnostics.
- Central-fit versus tail-fit Pareto frontier.

### Stage 4 — Asset mapping and decision layer

Each simulated scenario draws a regime path, factor path, posterior parameters, and
residuals before applying the regime-specific observation equation. The resulting
asset paths remain decimal simple returns. They are geometrically compounded over
the \(H\)-interval horizon before feeding VaR, ES, co-crash, empirical CVaR, and
Wasserstein-DRO. Regime-specific \(B_z,D_z,R_z\), or their Cholesky factors are
never posterior-probability averaged.

The initial decision is a single-period, \(H\)-interval buy-and-hold allocation with
full-investment, long-only or bounded-short, turnover, position, and ridge
constraints. Dynamic multistage allocation is deferred because it requires a
scenario tree and non-anticipativity.

### Stage 5 — Structural counterfactual extension

Construct a time-unrolled regime-switching structural causal model with known
innovations. Evaluate total and controlled interventions using
abduction–action–prediction. Compare against DoFlow, a conditional-only generator,
and simpler structural baselines. The confirmatory causal evidence is restricted to
the semi-synthetic market where factual and counterfactual ground truth are known.
The public financial panel is not treated as identifying a monetary-policy effect.
A real-data module is opened only after a separately registered, defensible
treatment and identification strategy exist; otherwise its outputs are only
model-based structural interventions or stress scenarios.

## Evaluation

### Predictive distribution

- energy score and variogram score;
- sliced Wasserstein distance and MMD as diagnostics;
- mean, variance, skewness, kurtosis;
- return and squared-return ACF;
- dynamic correlation and lower-tail dependence;
- drawdown depth/duration and explosive-path rate;
- condition-adherence scores for regime and stress controls.

### Tail risk

- pinball loss;
- Kupiec unconditional-coverage and Christoffersen independence diagnostics;
- Fissler–Ziegel joint VaR–ES score;
- ESR backtesting;
- co-crash Brier/log score and calibration curve;
- Monte Carlo uncertainty for all reported risk measures.

### Decision value

- realized out-of-sample ES;
- maximum drawdown and crisis loss;
- return, Sharpe, and Sortino as secondary metrics;
- turnover, transaction cost, concentration, and recovery time;
- worst-regime performance;
- paired block-bootstrap confidence intervals.

### Counterfactual validity

- average/individual treatment-effect error;
- intervention response-curve error;
- counterfactual distribution distance;
- \(\Delta ES_\alpha\) error;
- DAG misspecification and support sensitivity.

## Leakage and reproducibility controls

- All forecasts use only \(\mathcal F_t\).
- The default origin is after close at common endpoint \(t\); targets begin at
  \(t+1\). Same-interval prediction is prohibited.
- Common-endpoint rows compound source returns over \((e_{t-1},e_t]\); they are not
  silently interpreted as identical one-session returns.
- Time splits are chronological; overlapping \(H\)-interval labels are purged with an
  embargo of at least \(H\).
- Scalers, imputers, PCA/loadings, regime model, EVT threshold, and calibration are
  re-estimated inside each training fold.
- Thresholds, reference portfolio, and co-crash definitions are fixed from training
  data.
- No backward fill; macro values become usable only after their release/assumed
  availability date.
- The raw panel remains decimal simple returns without a risk-free subtraction. If
  a model uses \(y_t=\log(1+r_t)\), that is an explicit within-fold model-space
  transform; learned normalization is fit on training data only, and scenarios are
  mapped back with \(\exp(y)-1\) before risk or decision evaluation.
- Cboe/OptionMetrics features are absent from the public core and may appear only
  in a license-tagged extension.
- OFR labels never enter preprocessing, regime estimation, diffusion context, or
  hyperparameter selection.
- Hyperparameters are selected on validation only.
- Common forecast origins and random seeds are used across models.
- Overlapping-origin inference uses HAC or stationary block bootstrap.
- Negative and failed results are retained in versioned experiment records.

## Error propagation and robustness

The project measures how factor, regime, residual, and mapping error affect the
asset distribution and CVaR objective. Sensitivity experiments perturb:

- regime transition probabilities;
- \(B_z,D_z\) and residual correlation;
- factor-path error;
- posterior draws versus posterior means;
- Gaussian versus Student-\(t\) residuals;
- \(K,q,H\), tail threshold, and Wasserstein radius;
- fixed versus regime-dependent mapping; and
- true/estimated factors and residuals in module-swap experiments.

Strong convex regularization is included in the optimizer because stability of the
objective does not by itself imply stability of portfolio weights.

## Stop/go gates

- Reduce \(K\) if a state has occupancy below 5% or labels are unstable.
- Stop before diffusion if the switching factor model does not improve held-out
  reconstruction or tail covariance over fixed PCA/factor baselines.
- Retain diffusion as a negative result if it does not beat block bootstrap or
  Student-\(t\) on proper scores.
- Do not use “tail-aware” in the main conclusion unless held-out joint VaR–ES/ESR
  evidence improves.
- Report optimizer's curse or over-robustness if DRO helps simulated loss but harms
  realized out-of-sample loss after turnover.
- Keep the causal chapter semi-synthetic if real-data identification is inadequate.

## Expected contributions

Subject to the gates above, CrisisForge aims to contribute:

1. a posterior-aware interface between switching dynamic factor models and
   one-shot temporal diffusion;
2. a regime-dependent stochastic factor-to-asset mapping with explicit residual
   dependence;
3. a joint evaluation of central fit, tail calibration, and realized decision
   quality;
4. an uncertainty-aware treatment of Wasserstein robustness around model-generated
   scenarios; and
5. a clean separation between forecast, stress, and structural counterfactual
   experiments.

## Deliverables

- reproducible Python package and pinned environment;
- source catalog, snapshots, manifests, and data-quality reports;
- executed notebooks for reader-facing diagnostics;
- conventional baselines and all staged model implementations;
- experiment registry, ablations, negative results, and model cards;
- 40–60 page research report and technical appendix;
- core figures and interactive stress-testing application;
- 8–12 minute presentation and two-minute interview explanation;
- README, LinkedIn visual summary, and publication-ready caveat language.

## Claim boundary

The intended defensible conclusion is:

> On a retrospective panel of 12 industry research portfolios and three synthetic
> Treasury return proxies, a posterior-aware, regime-switching latent-factor
> scenario generator is evaluated through asset-level tail-risk forecasts and
> realized out-of-sample decisions; a structural counterfactual extension is
> validated separately in a semi-synthetic market.

The project will not claim:

> An AI system that predicts crises which have never happened.

Public-core results are not investability or live-trading evidence, and
conditional/stress scenarios from the observed panel are not labeled causal
counterfactuals.
