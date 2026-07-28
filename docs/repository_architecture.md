# CrisisForge Repository Architecture

Version: `0.3.0`
Reference experiment commit: `b6891133bdb6b96e1e23c6bea3bd033ea9685c7c`

## 1. Architectural contract

CrisisForge is a Python-first, multi-stage quantitative research repository. The
implemented training strategy is staged rather than end-to-end so that failures can
be attributed to data, state estimation, factor representation, scenario generation,
asset mapping, risk measurement, or portfolio optimization.

Every reported empirical result must resolve to:

1. the Python code and frozen YAML configuration that produced it;
2. the exact Phase 0 manifest and model-matrix hash it used;
3. the chronological split, forecast origins, horizon, and information set;
4. the Git commit and dirty-worktree flag;
5. output files and SHA-256 hashes;
6. the experiment stage and comparator; and
7. a predictive, conditional-stress, decision, or structural-counterfactual claim
   boundary.

The reference Phase 0 manifest is:

```text
f051ec35236a481858b67c5b1e7136f1698036832f427efba521dbf3fcd36d70
```

The validation-era Stage 0–6 receipts were produced from a clean worktree at the
reference commit. The post-2019 chronological test split remains sealed.

## 2. Implemented research graph

```text
Phase 0: public data and point-in-time governance
                         │
                         ▼
Stage 0: rolling conventional scenario baselines
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
Stage 1: PCA factor core       asset-level proper/tail scores
  -> sticky Gaussian HMM
  -> regime VAR(1)
  -> observation mapping
             │
             ├───────────────┐
             ▼               ▼
Stage 2: one-shot       Stage 3: paired Stage 1
temporal DDPM pilot     versus Stage 0 comparison
             │
             ▼
Stage 5: CVaR and Wasserstein-DRO decisions

Stage 4 utilities:
  tail importance weighting is used in Stage 2;
  factor-to-asset mapping is used in Stage 1 and Stage 2;
  POT/GPD and rolling conformal utilities are tested but are not integrated
  into a standalone Stage 4 runner.

Stage 6:
  separately validated time-unrolled semi-synthetic structural causal model
```

The numbering reflects the research plan, not a claim that every number has an
independent executable. In particular, Stage 4 has no standalone run.

## 3. Repository map

```text
CrisisForge/
├── README.md
├── LICENSE
├── CITATION.cff
├── Makefile
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── data_catalog.yaml
│   ├── pipeline.yaml
│   ├── stage0_baselines.yaml
│   ├── stage1_model.yaml
│   ├── stage1_evaluation.yaml
│   ├── stage2_diffusion.yaml
│   ├── stage2_evaluation.yaml
│   ├── stage3_comparison.yaml
│   ├── portfolio.yaml
│   ├── stage5_decision.yaml
│   ├── counterfactual.yaml
│   └── stage6_counterfactual_evaluation.yaml
├── data/
│   ├── raw/                     # normalized public snapshots + metadata
│   ├── interim/                 # target, timing, and source-alignment audits
│   └── processed/               # model matrix and chronological splits
├── artifacts/
│   ├── phase0/
│   ├── stage0_baselines/
│   ├── stage1_switching_factor/
│   ├── stage2_public_core_pilot/
│   ├── stage3_comparison/
│   ├── stage5_decisions/
│   └── stage6_counterfactual/
├── docs/
│   ├── research_proposal.md
│   ├── mathematical_formulation.md
│   ├── literature_review.md
│   ├── data_dictionary.md
│   ├── repository_architecture.md
│   └── research_log/
├── experiments/
│   ├── registry.csv
│   └── failures/
├── reports/
│   ├── figure_contracts.md
│   └── figures/
├── scripts/
│   ├── run_phase0.py
│   └── make_figures.py
├── src/crisisforge/
│   ├── config.py
│   ├── data/
│   │   ├── providers.py
│   │   ├── pipeline.py
│   │   └── validation.py
│   ├── baselines/scenarios.py
│   ├── regimes/switching.py
│   ├── factors/dynamic.py
│   ├── diffusion/temporal.py
│   ├── tail/calibration.py
│   ├── risk/metrics.py
│   ├── portfolio/cvar.py
│   ├── counterfactual/scm.py
│   └── evaluation/
│       ├── provenance.py
│       ├── rolling.py
│       ├── stage1.py
│       ├── stage2.py
│       ├── comparison.py
│       ├── decision.py
│       └── counterfactual.py
└── tests/
```

Raw, interim, processed, model, and evaluation artifacts are local research
evidence and are ignored by Git. Source, configuration, tests, documentation, and
the durable experiment registry are versioned.

## 4. Python module responsibilities

| Module | Implemented responsibility |
|---|---|
| `crisisforge.config` | Project-root discovery and YAML loading, including installed wheel entry points |
| `crisisforge.data.providers` | Direct-origin public downloads, normalization, metadata, cache verification, and source allowlisting |
| `crisisforge.data.pipeline` | Atomic Phase 0 refresh/offline orchestration, target construction, availability alignment, quality gates, splits, manifest, and receipt |
| `crisisforge.data.validation` | Data invariants, hashes, profiles, and deterministic manifest support |
| `crisisforge.baselines.scenarios` | Historical, moving-block, filtered historical, Gaussian, Student-\(t\), Student-\(t\) copula, and VAR-residual scenario generators |
| `crisisforge.regimes.switching` | Sticky regularized Gaussian HMM fitted by MAP/empirical-Bayes EM, filtering, smoothing, and state-path simulation |
| `crisisforge.factors.dynamic` | Train-only standardized PCA, deterministic factor orientation, regime-specific VAR(1), and state-dependent observation mapping |
| `crisisforge.diffusion.temporal` | One-shot conditioned temporal residual DDPM over complete future factor tensors |
| `crisisforge.tail.calibration` | Tail-severity importance weights, POT/GPD diagnostics, and rolling conformal quantile adjustment utilities |
| `crisisforge.risk.metrics` | Path aggregation, portfolio losses, VaR, exact empirical ES, joint VaR–ES score, co-crash probability, coverage tests, Brier, energy, and variogram scores |
| `crisisforge.portfolio.cvar` | Constrained empirical CVaR and strong-duality Wasserstein-DRO linear programs |
| `crisisforge.counterfactual.scm` | Known time-unrolled semi-synthetic SCM and paired abduction–action–prediction simulation |
| `crisisforge.evaluation.provenance` | Validation-only Parquet scan bounds, Phase 0/input/receipt binding, Git state, and output hashes |
| `crisisforge.evaluation.rolling` | Stage 0 common-origin rolling baseline evaluation |
| `crisisforge.evaluation.stage1` | Stage 1 fit, sequential validation filtering, mapped scenario archive, diagnostics, and risk scores |
| `crisisforge.evaluation.stage2` | Stage 2 train/tune/report window separation, checkpointing, mapping, and four-origin pilot evaluation |
| `crisisforge.evaluation.comparison` | Stage 3 paired circular moving-block bootstrap intervals |
| `crisisforge.evaluation.decision` | Stage 5 validation portfolio decisions, turnover, costs, realized loss, and radius sensitivities |
| `crisisforge.evaluation.counterfactual` | Stage 6 oracle and misspecification evaluation |

## 5. Phase 0 data and provenance

### 5.1 Public panel

The public `research_core` panel contains:

- 12 value-weighted Kenneth French U.S. industry research portfolios;
- three synthetic Treasury duration-return proxies derived from official U.S.
  Treasury par yields;
- Treasury curve and New York Fed EFFR context;
- five after-close, backward-looking target-panel risk features; and
- OFR Financial Stress Index components as validation-only labels.

The model matrix has 6,465 complete rows, 15 return targets, and 10 model-context
variables from 2000-07-05 through 2026-05-29:

| Split | Rows | Role |
|---|---:|---|
| Train | 3,367 | Fitted preprocessing and model estimation |
| Validation | 1,499 | Rolling evaluation, tuning/reporting segments, comparison, and decision sensitivity |
| Test | 1,599 | Governed holdout; excluded from completed model evaluation |

Phase 0 passed all 23 computed gates. Its only warning is that the French daily
source ends 60 calendar days before the requested 2026-07-28 endpoint.

### 5.2 Common-endpoint targets

The target calendar is the intersection of valid French-industry dates and valid
Treasury-yield dates. Industry returns are compounded over each
\((e_{t-1},e_t]\) interval. Treasury yield changes use the same endpoints, actual
calendar time for carry, and configured duration-convexity approximations.

`data/interim/target_interval_audit.parquet` records the interval endpoints,
duration, source observation counts, and endpoint assertions. A row is a common
holding interval, not necessarily one identical exchange session.

### 5.3 Availability contract

Source lags are mapped onto the actual ordered model calendar and backward-as-of
joined only when `available_date <= model_date`. There is no backward fill.
Return-derived features at endpoint \(t\) may condition only a response path that
starts at \(t+1\).

Stage runners read the model matrix through an explicit validation-end filter at
the Parquet scan boundary. They do not load test rows and discard them later.

### 5.4 Manifest and receipts

`artifacts/phase0/manifest.json` deterministically hashes governed files. It excludes
its own receipt and the mutable `experiments/registry.csv` to avoid circular
lineage. Each stage receipt binds the exact Phase 0 manifest, model matrix,
configuration, relevant upstream receipt/artifact, Git state, and generated output
hashes.

## 6. Stage 0: conventional baselines

Stage 0 evaluates seven scenario generators on 74 identical validation forecast
origins, a 20-row horizon, a 20-row stride, 1,000 scenarios, and a rolling
1,500-observation fit window:

- i.i.d. historical simulation;
- 20-observation moving-block bootstrap;
- filtered historical simulation with EWMA decay 0.94;
- Gaussian shrinkage;
- elliptical Student-\(t\);
- Student-\(t\) copula; and
- VAR(1) residual bootstrap.

All 518 model-origin evaluations completed.

| Validation criterion | Best Stage 0 result |
|---|---|
| Energy score | Filtered historical EWMA: 0.0897386 |
| Variogram score | Moving block: 0.5322258 |
| Joint VaR–ES score | Elliptical Student-\(t\): -0.9490638 |
| VaR violation rate closest to 5% | Moving block: 5.405% |

No single baseline won every criterion.

## 7. Stage 1: switching factor and asset mapping

Stage 1 is a staged **MAP/empirical-Bayes** estimator, not a full Bayesian posterior
sampler:

1. convert simple target returns with `log1p`;
2. center and scale using training data only;
3. estimate four deterministic-sign PCA factors;
4. fit a four-state sticky Gaussian HMM with regularized MAP/empirical-Bayes EM;
5. estimate regime-specific stabilized VAR(1) factor dynamics; and
6. reconstruct transformed asset returns through

\[
y_t=\log(1+r_t)
=
\alpha(z_t)
+B(z_t)f_t
+D(z_t)\epsilon_t,
\qquad
r_t=\operatorname{expm1}(y_t),
\]

where the observation residual uses a state-specific scale and shrunk Gaussian
correlation structure.

Parameters are estimated once through the training era. At each of 74 validation
origins, available validation history updates only the filtered state belief and
current factor. The model then simulates complete regime paths, factor paths, and
state-specific asset mappings. It never averages \(B\), \(D\), or residual
correlation matrices across states as a shortcut.

Diagnostics:

- four factors explain 85.995% of standardized training variance;
- reconstruction RMSE is 0.00522;
- state occupancies are 5.23%, 27.98%, 26.17%, and 40.62%;
- mean filtered state entropy is 0.4788; and
- state numbers have no automatic economic names.

Validation results:

| Metric | Stage 1 |
|---|---:|
| Energy score | 0.0943154 |
| Variogram score | 0.5886386 |
| Joint VaR–ES score | -0.9491237 |
| VaR violations | 1/74 = 1.351% |

The Stage 1 confirmatory advancement gate failed on central distributional scores:
simple Stage 0 baselines achieved lower energy and variogram scores. The model was
retained as valid negative evidence and as an engineering dependency for the
non-confirmatory diffusion and decision pilots.

## 8. Stage 2: one-shot temporal diffusion pilot

Stage 2 directly learns standardized future \(20\times4\) factor-path tensors from:

- 60 rows of factor and macro history available through origin \(t\); and
- the filtered four-state probability vector at origin \(t\).

It uses one denoising process for the complete future tensor and has no
autoregressive horizon loop. The public-core pilot intentionally overrides the
larger registered architecture with 12 diffusion steps, 32 hidden channels, two
residual blocks, three base epochs, two tail-weighted epochs, and 64 scenarios per
origin.

The Stage 2 runner:

- forms 658 training windows;
- uses 37 early-validation windows for checkpoint tuning;
- reports on four disjoint late-validation origins;
- saves both checkpoints, training history, train-only standardizers, window
  boundaries, scenario archives, diagnostics, failures, and a receipt; and
- maps generated factor paths back to assets using the Stage 1 observation system.

Future HMM regime paths are sampled separately from the factor DDPM conditional on
the same origin belief. Their joint dynamic consistency is not guaranteed. Stage 1
parameter uncertainty is not propagated as a full posterior.

| Pilot variant | Energy | Variogram | Joint VaR–ES |
|---|---:|---:|---:|
| Base | 0.0920488 | 0.7072089 | -0.9349023 |
| Tail-weighted | 0.0916800 | 0.6935241 | -0.9333682 |

With only four reporting origins, these values validate implementation and leakage
controls only. `superiority_claim_permitted` is false.

## 9. Stage 3: paired validation comparison

Stage 3 computes exploratory 95% paired intervals using a circular moving-block
bootstrap over the 74 common origins, block length 4, and 10,000 replications.
Differences are Stage 1 minus Stage 0; lower scores are better.

Key results:

| Comparator | Metric | Mean difference | 95% interval |
|---|---|---:|---:|
| Filtered historical EWMA | Energy | +0.004577 | [0.001670, 0.007471] |
| Moving block | Energy | +0.004504 | [0.002662, 0.006352] |
| Filtered historical EWMA | Variogram | +0.046614 | [0.009220, 0.084898] |
| Moving block | Variogram | +0.056413 | [0.020279, 0.095740] |
| Filtered historical EWMA | Joint VaR–ES | -0.001366 | [-0.008359, 0.004221] |
| Moving block | Joint VaR–ES | -0.000091 | [-0.008662, 0.007442] |

The Stage 1 energy and variogram intervals against the two strongest simple
baselines favor Stage 0. All joint VaR–ES intervals cross zero. Intervals are not
multiplicity-adjusted.

Stage 0 refits a rolling window while Stage 1 freezes training-era parameters and
filters sequentially. The comparison therefore combines model-class and
estimation-policy differences; it does not identify a pure model-class effect.

## 10. Stage 4: embedded tail and mapping utilities

There is no standalone Stage 4 executable or receipt.

Implemented and integrated:

- regime-dependent factor-to-asset mapping in Stage 1 and Stage 2; and
- train-only tail-severity importance weights in the Stage 2 fine-tuning pilot.

Implemented and unit-tested, but not integrated into the current empirical runner:

- POT/GPD threshold diagnostics and fitting; and
- rolling conformal quantile adjustment.

Accordingly, the current evidence does not claim EVT or conformal improvement.

## 11. Stage 5: portfolio decisions

Stage 5 uses 74 validation origins and the Stage 1 cumulative scenario archive.
The canonical portfolio is long-only, fully invested, capped at 20% per asset, and
constrained to 0.40 L1 turnover. Transaction cost is an additive linear return
approximation of 0.001 times L1 turnover.

Strategies:

- equal weight;
- historical empirical CVaR;
- Stage 1 empirical CVaR; and
- Stage 1 one-Wasserstein DRO at radii 0.0001, 0.00025, 0.0005, and 0.001.

The Wasserstein inner problem is represented through its strong-duality convex
penalty and solved with SciPy HiGHS. The radius grid is validation-only sensitivity
analysis; no final radius is selected.

| Strategy | Cumulative net return | Realized ES | Maximum drawdown | Total L1 turnover |
|---|---:|---:|---:|---:|
| Equal weight | 0.55635 | 0.04547 | 0.08769 | 1.4476 |
| Historical CVaR | 0.44725 | **0.01767** | **0.03411** | 3.4955 |
| Stage 1 CVaR | 0.30831 | 0.02074 | 0.04782 | 12.4945 |
| Stage 1 WDRO, 0.001 | 0.30744 | 0.02074 | 0.04782 | 12.4787 |

Historical CVaR delivered lower realized validation ES and drawdown than Stage 1
scenario CVaR. The registered positive Wasserstein radii produced nearly unchanged
decisions, so this grid supplies no evidence of a meaningful robustness benefit.

## 12. Stage 6: structural counterfactual extension

Stage 6 is separate from the observed-market predictive pipeline. It uses a known
time-unrolled semi-synthetic structural causal model with within-slice order:

```text
policy -> yield -> liquidity -> credit -> equity -> volatility
```

and lagged `equity[t-1] -> policy[t]` feedback. Time unrolling prevents a directed
cycle within one slice while retaining delayed financial feedback.

The evaluation generates 5,000 paired 20-step paths with identical exogenous noise,
then tests:

- oracle abduction–action–prediction recovery;
- a reduced yield-policy coefficient; and
- removal of lagged equity-policy feedback.

The maximum oracle error is \(4.08\times10^{-17}\). Misspecified mean-effect path
RMSE ranges from 0.0731 to 0.2782. The controlled direct effect is zero by design
when the yield, liquidity, and credit mediator schedules are held identical.

These are implementation and specification-sensitivity results under known
semi-synthetic truth. They do not identify policy effects in observed markets.

## 13. Exact reproducibility commands

Reference runtime: Python 3.11 with `uv.lock`.

### 13.1 Environment, tests, and lint

```bash
uv sync --extra dev --extra models --extra reporting
uv run pytest
uv run ruff check src tests scripts
```

The repository does not hard-code a permanent passing-test count. Cite the fresh
integrated total printed by the final release run.

### 13.2 Data

Use the retained validated snapshots:

```bash
uv run crisisforge-phase0 --offline
```

Intentionally retrieve current public snapshots:

```bash
uv run crisisforge-phase0 --refresh
```

`--offline` and `--refresh` are mutually exclusive. Refreshing can create a new
sample and manifest and must be treated as a new evidence release.

### 13.3 Model and evaluation sequence

```bash
uv run crisisforge-stage0
uv run crisisforge-stage1
uv run crisisforge-compare
uv run crisisforge-stage2
uv run crisisforge-stage5
uv run crisisforge-counterfactual
uv run python scripts/make_figures.py
```

Default entry points resolve the canonical configurations listed in Section 3.
Every model-stage command accepts `--project-root` and `--config` when an explicit
root or registered alternative configuration is required.

Stage 1 dominates reference runtime at roughly eight minutes on the recorded Apple
Silicon system. Stage 2 is a deliberately small CPU pilot rather than the full
capacity specified in `configs/stage2_diffusion.yaml`.

## 14. Artifact contract

| Evidence | Required files |
|---|---|
| Phase 0 | Quality JSON/Markdown, deterministic manifest, timestamped receipt |
| Stage 0 | Rolling results, model summary, empty-or-populated failure record, receipt |
| Stage 1 | Scenario archive, rolling results, diagnostics, state profiles, filtered/smoothed probabilities, failure record, receipt |
| Stage 2 | Base/tail checkpoints, scenario archives, training history, window audit, standardizers, diagnostics, summary, failure record, receipt |
| Stage 3 | Paired intervals, summary, receipt |
| Stage 5 | Decision results, weights, summary, failure record, receipt |
| Stage 6 | Effect summary, mean paths, summary, receipt |
| Reporting | Figure contracts, PNG/SVG figures, figure manifest |

All completed Stage 0, 1, 2, and 5 sequences report zero failed model/decision
origins. A zero failure count is an execution fact, not evidence of model quality.

## 15. Gate and negative-result policy

The preregistered confirmatory logic is:

1. compare against strong conventional methods on identical validation origins;
2. advance a superiority claim only when proper, tail, or decision evidence supports
   it; and
3. retain valid failures to improve as negative results.

Stage 1 failed its confirmatory central-distribution gate. Stage 2 and Stage 5 were
still run as explicitly non-confirmatory engineering and decision diagnostics.
They cannot be used to retroactively claim that the failed gate passed.

The current negative findings are:

- Stage 1 is worse than the strongest simple baselines on paired energy and
  variogram evidence;
- joint VaR–ES differences are inconclusive;
- the Stage 2 pilot is too small for ranking;
- historical CVaR outperforms Stage 1 scenario CVaR on validation ES and drawdown;
- the registered Wasserstein radii add almost no decision change; and
- zero realized co-crash events make co-crash Brier comparisons uninformative.

Failed, invalidated, and negative-result runs remain in the experiment registry or
failure records. They are never deleted because their result is unattractive.

## 16. Claim boundaries

The repository does not claim:

- full Bayesian inference or MCMC for Stage 1;
- an economically observed ground truth for latent state numbers;
- jointly coherent future diffusion factor and HMM regime paths in Stage 2;
- EVT or conformal improvement in the current empirical runner;
- diffusion superiority from four reporting origins;
- test-set performance from validation results;
- robustness to the true distribution from a Wasserstein ball around generated
  scenarios; or
- real-world causal identification from the semi-synthetic Stage 6 SCM.

Any future real-market counterfactual output must be labeled a
**model-based structural intervention** unless an independently defensible causal
identification strategy is registered and validated.
