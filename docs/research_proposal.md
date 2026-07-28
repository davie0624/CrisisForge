# Research Proposal

## CrisisForge: Decision-Focused Market Simulation under Regime Shifts

### Factor-Structured Temporal Diffusion, Asset-Level Tail Risk, and Robust Portfolio Control

**Version:** 0.3

**Date:** 2026-07-28

**Status:** implemented public-core research system with post-2019 rows
governed-excluded from model development and evaluation

**Audience:** quantitative finance, financial econometrics, risk management, and
machine-learning researchers

## Abstract

CrisisForge studies whether structured scenario generators add value beyond strong
conventional risk models. The system combines a low-dimensional market
representation, latent regimes, one-shot temporal diffusion, a stochastic
factor-to-asset observation mapping, asset-level tail-risk measurement, and
portfolio decisions. A separate time-unrolled structural causal model tests
counterfactual machinery where ground truth is known.

The project does **not** predict the date of the next crisis, identify a real-world
monetary-policy effect, or claim that generative AI beats conventional risk models.
Its principal predictive estimand is

\[
\mathcal P_t^H=P(r_{t+1:t+H}\mid\mathcal F_t),
\]

where \(r_t\) is a vector of decimal simple returns over a common-endpoint holding
interval and \(\mathcal F_t\) contains only information available after close at
origin \(t\). The first forecast target is \(r_{t+1}\).

The completed validation evidence is deliberately mixed. Conventional
time-dependence-preserving baselines lead the switching-factor model on energy and
variogram scores. Joint VaR–ES performance is statistically inconclusive. A small
diffusion engineering pilot runs end to end but is not powered for model ranking.
Historical CVaR has lower observed validation ES and drawdown than Stage 1
scenario-based CVaR; tested Wasserstein radii barely change the solution. These
decision rankings are descriptive because no paired sampling interval was
computed. The
counterfactual engine is numerically correct in a known semi-synthetic SCM and
materially sensitive to structural misspecification. These negative results are a
central research output, not a failed marketing claim.

## Target architecture and implemented evidence

The long-run research design is broader than the public-core evidence. The
distinction below is binding.

| Component | Target design | Implemented public core | Evidence status |
|---|---|---|---|
| Regime model | Bayesian switching state-space posterior | Sticky Gaussian HMM, multi-start MAP-like EM | Complete baseline |
| Factor model | Joint dynamic latent-factor posterior | Train-only standardized PCA plus regime-weighted VAR(1) | Complete baseline |
| Observation map | Posterior draws of \(B_z,D_z,R_z\) | Regime-weighted linear map with shrunk Gaussian residual covariance | Complete baseline |
| Diffusion | Large one-shot conditional factor-path generator | Small conditional DDPM pilot, 12 steps, 3+2 epochs | Engineering pilot |
| Tail layer | Tail mixture, EVT, sequential calibration | Importance-weighted diffusion fine-tuning integrated; POT/GPD and rolling conformal utilities tested only | Partial |
| Risk engine | VaR, ES, co-crash, proper scores | Asset-level implementations and rolling validation | Complete, co-crash event-scarce |
| Decision layer | CVaR and calibrated Wasserstein DRO | Empirical CVaR and four fixed Wasserstein-radius sensitivities | Complete validation study |
| Counterfactual | Structural intervention with known truth, later identified real-data study | Time-unrolled semi-synthetic SCM with AAP and misspecification tests | Complete semi-synthetic study |

No MCMC, variational Bayes, full parameter posterior, Student-\(t\) observation
residual, regime-hierarchical EVT fit, ESR test, decision-focused end-to-end
training, or real-market causal identification is claimed.

## Motivation

Historical simulation, block bootstrap, multivariate Student-\(t\), copulas, and
VAR residual simulation remain difficult risk baselines to beat. Deep generative
models offer nonlinear path distributions but face three limitations in financial
applications:

1. the asset dimension is large relative to the number of independent crisis
   observations;
2. dependence and volatility change across latent market states; and
3. good average distribution fit may not improve the left tail or downstream
   decisions.

CrisisForge therefore treats generative modeling as a falsifiable extension to a
financially structured baseline. It asks where complexity helps, where it does not,
and how errors propagate from latent states and factor paths to asset returns and
portfolio weights.

The verified primary literature and the claim each source supports are recorded in
`docs/literature_review.md`. Preprints are marked separately from peer-reviewed
work.

## Research questions

### RQ1 — Structured versus conventional scenarios

Does the switching-factor architecture improve multivariate proper scores or
joint tail scores relative to rolling conventional generators?

### RQ2 — Regime information

What statistical states emerge from training data, how persistent are they, and
does filtered state uncertainty support stable scenario generation without using
future information?

### RQ3 — Factor-to-asset reconstruction

Can a regime-dependent observation map in transformed-return space,

\[
y_t=\log(1+r_t)
=\alpha_{z_t}+B_{z_t}f_t+D_{z_t}\epsilon_t,
\qquad r_t=\operatorname{expm1}(y_t),
\]

reconstruct a high-dimensional asset panel while preserving idiosyncratic
variation and residual dependence?

### RQ4 — One-shot temporal diffusion

Can a conditional DDPM generate an entire \(H\times q\) factor path in one shot,
and what trade-off between multivariate distribution scores and the joint
VaR–ES score appears after importance-weighted fine-tuning?

### RQ5 — Tail-risk evidence

How well do scenario generators estimate 95% VaR and ES, and is the validation
sample informative enough to evaluate co-crash probabilities?

### RQ6 — Decision value

Do statistically sophisticated scenarios improve realized validation ES,
drawdown, turnover, and net return relative to equal-weight and historical-CVaR
decisions?

### RQ7 — Counterfactual validity

Under a known time-unrolled SCM, does abduction–action–prediction recover paired
counterfactual paths, and how quickly does performance deteriorate under
misspecified transmission equations?

## Registered hypotheses and current disposition

| Hypothesis | Testable statement | Current disposition |
|---|---|---|
| H1 | Factor/regime structure improves joint distribution scores | Not supported: leading conventional baselines have lower energy and variogram scores |
| H2 | Regime conditioning improves tail-risk scores | Not isolated by the current ablation set |
| H3 | Regime-dependent mapping improves asset-tail dependence | Implemented, but fixed-map ablation remains outstanding |
| H4 | Tail emphasis changes multivariate distribution and joint tail-risk scores differently | Descriptive pilot is mixed: energy/variogram improve slightly while joint VaR–ES worsens |
| H5 | Statistical and decision rankings differ | Suggestive, not isolated: Stage 1's marginally lower joint VaR–ES point score does not translate into lower observed ES/drawdown than a separate historical-CVaR baseline; uncertainty and differing scenario/update policies prevent a clean ranking claim |
| H6 | Full uncertainty propagation stabilizes risk and weights | Not tested; the current estimator is plug-in/MAP |
| H7 | Counterfactual recovery is sensitive to structural correctness | Supported in the known SCM; not evidence of real-market identification |

These dispositions apply only to the validation evidence described below.
Post-2019 rows remain governed-excluded from model development and evaluation.

## Data

### Public research panel

The model matrix contains 6,465 common-endpoint rows from 2000-07-05 through
2026-05-29:

- 12 value-weighted U.S. industry research portfolios from the Kenneth French Data
  Library;
- three Treasury duration-convexity return proxies derived from official U.S.
  Treasury daily par yields; and
- ten availability-lagged market and macro context variables from retained current
  public-source snapshots.

The 15 targets are decimal simple returns. Industry portfolios are retrospective
research constructs, not tradable ETFs. Treasury targets are transparent proxies,
not observed bond total-return indices. The public core therefore supports
methodological research, not live-trading or investability claims.

The ten context columns comprise:

- 2-, 5-, and 10-year Treasury par-yield levels and the 10y–2y slope;
- New York Fed effective federal funds rate; and
- five backward-looking features from industry returns: equal-weighted market
  return, 20-observation realized volatility, downside semivolatility,
  cross-industry dispersion, and 60-observation drawdown.

Treasury and EFFR context values enter after their registered one-model-session
availability lag. OFR Financial Stress Index components are retained only as
external validation labels and never enter estimation or hyperparameter selection.
FRED/ALFRED and unlicensed option-index histories are excluded from the ML research
core. Exact sources, transformations, lags, terms, and paid substitutes appear in
`docs/data_dictionary.md` and `configs/data_catalog.yaml`.

These public sources are not a point-in-time vintage database. Revision-specific
claims require a separately versioned source that preserves historical releases.

### Chronological split

| Split | Rows | Boundary |
|---|---:|---|
| Train | 3,367 | through 2013-12-31 |
| Validation | 1,499 | 2014-01-02 through 2019-12-31 |
| Sealed test | 1,599 | from 2020-01-02 |

All reported scores and decisions are validation-era evidence. Stage 0–5
evaluation runners scan the model matrix only through 2019-12-31, so no estimator
fit, checkpoint, model score, or portfolio decision uses post-2019 rows. Phase 0
necessarily constructs and quality-checks the full matrix, including the 1,599
post-2019 rows. “Sealed” means governed exclusion from model development and
evaluation, not that the values are cryptographically hidden from Phase 0.

## Canonical experimental stages

### Phase 0 — Data pipeline and governance

Download public snapshots, hash raw and derived files, align source calendars,
compound source returns over common intervals, apply availability lags, build the
model matrix, and run 23 quality gates.

### Stage 0 — Conventional scenario baselines

Evaluate seven generators on 74 non-overlapping \(H=20\) validation origins with
1,000 scenarios per origin:

- historical IID;
- moving-block bootstrap;
- filtered historical simulation with EWMA volatility;
- Gaussian shrinkage;
- multivariate Student-\(t\);
- Student-\(t\) copula; and
- VAR(1) residual bootstrap.

### Stage 1 — MAP/empirical-Bayes switching factor and observation map

Fit four train-only PCA factors; estimate a four-state sticky Gaussian HMM by
multi-start MAP-like EM; fit regime-weighted stable VAR(1) factor dynamics; and fit
regime-weighted Gaussian observation mappings to \(y_t=\log(1+r_t)\). Training
smoothed probabilities are used to fit the downstream regime components. Forecast
origins use filtered probabilities only, and simulated observations are converted
back to simple returns with \(\operatorname{expm1}\).

For each scenario, draw a future state path, factor path, and correlated
asset-specific residuals. Apply the exact state-specific parameters; do not average
loadings or Cholesky factors across state probabilities.

### Stage 2 — One-shot diffusion engineering pilot

Construct 60-observation histories and \(H=20\) future factor tensors. A compact
conditional temporal DDPM denoises the full future factor path at once. Context
contains factor and macro history plus the origin filtered state probabilities.
The reported pilot has 658 training windows, 37 tuning windows, four
late-validation reporting origins, 12 diffusion steps, 64 scenarios per origin,
three base epochs, and two tail-weighted fine-tuning epochs.

Future state paths for asset reconstruction are currently sampled independently
from the generated factor path conditional on the same origin belief. Joint dynamic
consistency is therefore not guaranteed.

### Stage 3 — Paired validation comparison

Compare Stage 1 with every Stage 0 generator on identical origins using a circular
moving-block bootstrap with block length four and 10,000 draws. Intervals are
exploratory and not multiplicity-adjusted. Stage 0 uses rolling 1,500-row refits,
whereas Stage 1 freezes train parameters and only filters forward; comparisons are
not causal estimates of model-class effects.

### Stage 4 — Embedded tail and mapping functionality

No standalone Stage 4 experiment exists. Importance-weighted diffusion fine-tuning
is exercised in Stage 2. Asset mapping is exercised in Stages 1 and 2. POT/GPD and
rolling conformal utilities are unit-tested but not integrated into the reported
experiment. This stage label is retained to make the target architecture explicit
without inventing evidence.

### Stage 5 — Portfolio decisions

Solve long-only, fully invested empirical-CVaR and 1-Wasserstein robust-CVaR
problems at the same 74 origins. The position cap is 20%, L1 turnover cap is 40%,
and the linear transaction-cost rate is 10 basis points per unit of L1 turnover.
Four fixed Wasserstein radii are reported as separate validation sensitivities;
none is selected as final.

### Stage 6 — Structural counterfactual extension

Use a time-unrolled regime-switching SCM with within-period order

\[
\text{policy}\rightarrow\text{yield}\rightarrow\text{liquidity}
\rightarrow\text{credit}\rightarrow\text{equity}\rightarrow\text{volatility}
\]

and lagged equity-to-policy feedback. Abduction, action, and prediction reuse the
same exogenous noise. Oracle recovery and two structural misspecifications are
evaluated over 5,000 paired \(H=20\) paths. The experiment has known
semi-synthetic truth and no observed-market causal estimand.

## Evaluation

Primary generator metrics are energy score, variogram score, and a joint VaR–ES
score. Risk diagnostics include empirical VaR violations, Kupiec unconditional
coverage, Christoffersen conditional coverage, ES, and co-crash Brier score.
Energy and variogram scores use the 15-asset geometrically compounded \(H=20\)
return vector. VaR, ES, joint VaR–ES, and violation rates use a frozen
equal-weight portfolio with \(w_i=1/15\).

At every evaluated origin—74 Stage 0/Stage 1 origins and four Stage 2
origins—the realized co-crash label is zero; generated scenario sets may still
assign nonzero probabilities. The Brier score therefore cannot assess event
discrimination in the current sample.

Decision metrics are cumulative net return, realized 95% VaR and ES, maximum
drawdown, turnover, and transaction cost. Counterfactual metrics are terminal and
cumulative paired effects, tail effect, path RMSE, and error under known structural
misspecification. For the Stage 6 tail effect, lower cumulative equity outcomes
are losses (\(s_{\mathrm{equity}}=-1\)), while higher cumulative volatility is a
stress loss (\(s_{\mathrm{volatility}}=+1\)).

## Current validation findings

1. Filtered historical simulation has the best mean energy score
   (\(0.089739\)); moving block has the best variogram score (\(0.532226\)); and,
   among Stage 0 models, Student-\(t\) elliptical has the lowest joint VaR–ES
   point score (\(-0.949064\)).
2. Stage 1 records energy \(0.094315\), variogram \(0.588639\), and joint VaR–ES
   \(-0.949124\), marginally lower than the Stage 0 point estimate. Its paired
   difference from Student-\(t\) elliptical is \(-0.000060\), with a 95% interval
   \([-0.007544,0.006818]\), so no reliable advantage is established. Its single
   VaR violation out of 74 is conservative, not evidence of superior calibration.
3. Paired intervals show Stage 1 is worse than the leading conventional baselines
   on energy and variogram scores. Every joint VaR–ES interval crosses zero.
4. Tail-weighting in the four-origin diffusion pilot slightly improves energy and
   variogram scores but worsens the joint VaR–ES score. The sample is too small for
   inference.
5. Historical CVaR has observed validation ES \(0.017669\) and maximum drawdown
   \(0.034111\), versus \(0.020738\) and \(0.047824\) for Stage 1 scenario CVaR.
   Wasserstein sensitivities are almost indistinguishable from empirical Stage 1
   CVaR. These decision differences are descriptive; no paired sampling interval
   was computed.
6. The semi-synthetic counterfactual oracle recovers truth to numerical precision.
   Misspecified structural equations produce path RMSE from \(0.0731\) to
   \(0.2782\), demonstrating structural sensitivity.

## Leakage, provenance, and reproducibility controls

- chronological train/validation/test boundaries;
- validation-bounded Parquet scans in experiment runners;
- no backward fill and explicit availability lags;
- train-only scalers, PCA, HMM, thresholds, and standardizers;
- filtered probabilities only at forecast origins;
- identical registered validation origins for paired models;
- fixed seeds and declarative YAML configurations;
- run receipts bound to the Phase 0 manifest, source commit, inputs, and outputs;
- preservation of failures and negative results; and
- no regeneration of the bound Phase 0 manifest after dependent experiments.

## Stop/go gates and retained failures

The intended confirmatory gate—advance to diffusion only if the structured
switching-factor model clearly improves conventional baselines—was not met. A
strictly non-confirmatory Stage 2 pilot nevertheless proceeded to validate the
one-shot diffusion plumbing, checkpointing, tail-weighting mechanism, and asset
mapping. It is not promoted to a ranking result.

Future diffusion work should stop unless it:

1. expands to the complete registered validation-origin set;
2. adds direct, unconditional, hard-state, and fixed-map ablations;
3. preserves joint consistency between factor and state paths;
4. quantifies Monte Carlo uncertainty; and
5. improves a predeclared tail or decision metric without hiding deterioration in
   central fit.

## Planned extensions

The following remain research proposals rather than completed results:

- a coherent Bayesian switching state-space model with parameter posterior draws;
- posterior predictive propagation of \(B_z,D_z,R_z\) and state uncertainty;
- Student-\(t\) observation residuals and fixed-versus-regime mapping ablations;
- tail-mixture training with inverse-probability correction;
- regime-hierarchical POT/GPD and sequential conformal calibration;
- ESR tests, calibration curves, MMD/SWD, ACF, leverage, and tail-dependence
  diagnostics;
- validation-calibrated Wasserstein radius selection;
- decision-focused training; and
- a separately identified real-data causal design.

## Deliverables

The repository retains Python source, configurations, tests, data and experiment
receipts, model artifacts, eight report figures in PNG/SVG, a research report,
technical documentation, and LinkedIn material. The release package is designed
to add the final presentation and release manifest after all source-facing
artifacts are frozen. Every completed experiment is registered. The post-2019
test set remains governed-excluded until a separate signed test-evaluation
decision is made.

## Claim ledger

Defensible:

> CrisisForge is a reproducible public-core study of regime-structured and
> generative financial scenarios, evaluated through asset-level risk and portfolio
> decisions. Its validation evidence favors strong conventional baselines on
> several central-distribution and decision metrics, while a known-SCM extension
> verifies counterfactual implementation and exposes structural sensitivity.

Not defensible:

> CrisisForge predicts unseen crises, proves that generative AI beats traditional
> risk models, identifies the causal effect of monetary policy in real markets, or
> demonstrates a deployable trading advantage.
