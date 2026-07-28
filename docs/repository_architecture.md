# CrisisForge Repository Architecture

Version: Phase 0 v0.3.0

## 1. Architectural contract

CrisisForge is a Python-first, multi-stage quantitative research repository. Its
default training strategy is a staged pipeline with uncertainty and audit evidence
passed between modules. It is not an opaque end-to-end system.

Every reported result must resolve to:

1. the exact Python code and configuration that produced it;
2. the validated raw snapshots and common-endpoint intervals it used;
3. the forecast-origin, source-availability, and split assumptions;
4. the experiment stage and benchmark family;
5. the first module that introduced an improvement or failure; and
6. whether the claim is predictive, stress-conditional, decision-focused, or
   structurally counterfactual.

Selective joint fine-tuning may be registered later as an ablation. It does not
replace module-level diagnostics or stage gates.

## 2. Current repository map

The following map distinguishes implemented code from intentionally empty package
boundaries for later research stages.

```text
CrisisForge/
├── .gitignore
├── .python-version
├── README.md
├── LICENSE
├── CITATION.cff
├── Makefile
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── data_catalog.yaml
│   └── pipeline.yaml
├── data/
│   ├── raw/
│   │   ├── returns/
│   │   │   ├── french_12_industries.csv
│   │   │   └── french_12_industries.meta.json
│   │   └── macro/
│   │       ├── us_treasury_yield_curve.csv
│   │       ├── us_treasury_yield_curve.meta.json
│   │       ├── nyfed_effr.csv
│   │       ├── nyfed_effr.meta.json
│   │       ├── ofr_fsi.csv
│   │       └── ofr_fsi.meta.json
│   ├── interim/
│   │   ├── target_returns.parquet
│   │   ├── target_interval_audit.parquet
│   │   ├── macro_features.parquet
│   │   ├── validation_labels.parquet
│   │   ├── optional_diagnostics.parquet
│   │   └── availability_alignment_audit.parquet
│   └── processed/
│       ├── model_matrix.parquet
│       └── splits/
│           ├── train.parquet
│           ├── validation.parquet
│           └── test.parquet
├── artifacts/
│   └── phase0/
│       ├── quality_report.json
│       ├── quality_report.md
│       ├── manifest.json
│       └── run_receipt.json
├── docs/
│   ├── research_proposal.md
│   ├── mathematical_formulation.md
│   ├── literature_review.md
│   ├── data_dictionary.md
│   ├── repository_architecture.md
│   └── research_log/
│       ├── 0001_phase0.md
│       └── 0002_phase0_independent_audit.md
├── experiments/
│   ├── README.md
│   ├── registry.csv
│   └── failures/
│       └── README.md
├── notebooks/
│   └── README.md
├── reports/
│   └── README.md
├── slides/
│   └── README.md
├── linkedin/
│   └── README.md
├── scripts/
│   └── run_phase0.py
├── src/
│   └── crisisforge/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── providers.py
│       │   ├── pipeline.py
│       │   └── validation.py
│       ├── baselines/
│       │   ├── __init__.py
│       │   └── scenarios.py
│       ├── risk/
│       │   ├── __init__.py
│       │   └── metrics.py
│       ├── regimes/
│       │   └── __init__.py
│       ├── factors/
│       │   └── __init__.py
│       ├── diffusion/
│       │   └── __init__.py
│       ├── mapping/
│       │   └── __init__.py
│       ├── tail/
│       │   └── __init__.py
│       ├── portfolio/
│       │   └── __init__.py
│       ├── counterfactual/
│       │   └── __init__.py
│       └── evaluation/
│           └── __init__.py
└── tests/
    ├── test_phase0.py
    ├── test_snapshot_integrity.py
    ├── test_baselines.py
    └── test_risk_metrics.py
```

Raw, interim, processed, and generated artifact files are local research evidence
and are ignored by Git except for keep files. Source code, configuration,
definitions, tests, decision logs, and experiment registry records are versioned.
The public raw cache contains no Cboe data.

## 3. Implemented responsibilities

| Path | Current responsibility |
|---|---|
| `configs/data_catalog.yaml` | Public panel definition, exact source endpoints, fields, units, roles, actual model-session lags, staleness, and derived-feature specifications |
| `configs/pipeline.yaml` | `America/New_York` clock, `after_close` decision timestamp, seed, requested sample, chronological split boundaries, quality thresholds, and output paths |
| `src/crisisforge/config.py` | YAML loading and project-root resolution |
| `src/crisisforge/data/providers.py` | Direct-origin downloads and parsing, normalized snapshot writes, provenance metadata, cached-snapshot integrity validation |
| `src/crisisforge/data/pipeline.py` | Atomic refresh orchestration, common-endpoint target construction, availability-aware context alignment, return-derived features, splits, quality gates, reports, manifest, and receipt |
| `src/crisisforge/data/validation.py` | Data invariants, return review flags, frame profiles, split assertions, file hashing, deterministic manifest |
| `src/crisisforge/baselines/scenarios.py` | Historical, moving-block bootstrap, Gaussian shrinkage, and Student-\(t\) shrinkage scenario generators |
| `src/crisisforge/risk/metrics.py` | Path aggregation, portfolio losses, empirical VaR, exact empirical ES, co-crash probability, Kupiec coverage test, and Brier score |
| `scripts/run_phase0.py` | Thin Python entry point for Phase 0 |
| `tests/test_phase0.py` | Transform, actual-calendar as-of, common-interval, Treasury proxy, derived-feature, and split tests |
| `tests/test_snapshot_integrity.py` | Corrupted snapshot hash and row-count rejection tests |
| `tests/test_baselines.py` | Scenario shape, finiteness, and block-contiguity tests |
| `tests/test_risk_metrics.py` | Path return, loss, VaR, ES, co-crash, Kupiec, and Brier definition tests |
| `experiments/registry.csv` | Durable experiment index; currently initialized with its schema |
| `experiments/failures/` | Required home for failed, invalidated, and negative-result records |

Provider adapters normalize source syntax only. Economic role, admissibility,
timing, and claim boundaries belong in the catalog and data dictionary.

## 4. Reproducible commands

Python 3.11 is the reference runtime and `uv.lock` is the dependency lock.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests scripts
uv run crisisforge-phase0 --refresh
```

Equivalent Make targets:

```bash
make setup
make test
make lint
make phase0-refresh
```

After a successful refresh has created the raw cache:

```bash
uv run crisisforge-phase0 --offline
```

`--refresh` and `--offline` are mutually exclusive. A normal run without either
flag may use a valid cache and network access as needed; the receipt distinguishes
`refresh`, `online-cache`, and `offline`.

Later model stages use:

```bash
uv sync --extra dev --extra models --extra notebooks
```

Reusable logic belongs in `src/crisisforge/`. Notebooks are executed, reader-facing
companions and must not become the only implementation of a reported method.

## 5. Phase 0 execution graph

```text
configs/data_catalog.yaml + configs/pipeline.yaml
                         │
                         ▼
         direct-origin source adapters
                         │
               refresh? ─┴──────────── cached/offline?
                  │                          │
                  ▼                          ▼
       data/.raw-staging-*          validate data/raw cache
                  │              identity/range/hash/schema
                  ▼                          │
       validate complete allowlist ◀─────────┘
                         │
                         ▼
        common-endpoint target calendar
            industry ∩ Treasury dates
                         │
          ┌──────────────┴───────────────┐
          ▼                              ▼
 compound industry returns      elapsed-time Treasury proxies
 over (previous, current]       on the same endpoints
          └──────────────┬───────────────┘
                         ▼
             15 target-return columns
                         │
                         ├── target_interval_audit.parquet
                         │
                         ▼
   actual-model-session backward-as-of alignment
       Treasury + EFFR + OFR validation labels
                         │
                         ├── availability_alignment_audit.parquet
                         │
                         ▼
   five after-close return-derived market features
                         │
                         ▼
        15 targets + 10 model-context features
                         │
          drop missing targets/context; no imputation
                         │
                         ▼
       strict train / validation / test split
                         │
                         ▼
           23 computed quality gates
                         │
                 fail? ───┴─── pass?
                  │             │
                  ▼             ▼
       retain prior release    activate raw candidate
                                with rollback backup
                                      │
                                      ▼
                         back up interim / processed /
                              artifacts/phase0
                                      │
                                      ▼
                          publish data + quality +
                           manifest + run receipt
                                      │
                           exception? ─┴── success?
                              │              │
                              ▼              ▼
                       roll back all      finalize all
                        four trees          backups
```

The public model context is Treasury levels and slope, NY Fed EFFR, and five
return-derived market features. OFR FSI is validation-only. Cboe VIX, SKEW, and
VVIX are absent from the public raw cache and may appear only in a licensed
extension.

## 6. Raw snapshot integrity and atomic refresh

### 6.1 Snapshot write contract

Every raw source is normalized into a CSV plus sibling `.meta.json`. The writer:

1. writes the CSV to a per-file temporary path;
2. computes its local SHA-256;
3. writes metadata to a temporary path; and
4. promotes each file with `os.replace`.

Metadata preserve provider, source URL(s), UTC retrieval timestamp, remote payload
SHA-256, local CSV SHA-256, row count, ordered schema, and first/last dates.

### 6.2 Cache read contract

A cache hit is admissible only after the reader verifies:

- required metadata keys;
- source ID, provider, landing page, and catalog schema identity;
- local file hash;
- exact schema and column order;
- row count;
- parseable dates;
- unique, increasing dates; and
- exact first and last dates;
- cached requested-range coverage; and
- filtering to the active requested range.

Corruption is an error, not a signal to continue with an unverifiable cache.
The raw cache is also checked against an exact catalog allowlist; unknown leftovers
and missing configured snapshot pairs are rejected.

### 6.3 Whole-release transaction

A refresh builds the complete new source set under a temporary
`data/.raw-staging-*` directory. The candidate is not activated until downloads,
cache integrity, transformations, aligned contexts, splits, and all 23 gates pass.
The prior raw tree remains available as a rollback snapshot.

Publication then backs up `data/interim`, `data/processed`, and
`artifacts/phase0` before writing any new evidence. An exception during Parquet,
quality, manifest, or receipt persistence restores those three trees and
`data/raw` exactly. Successful completion finalizes all backups. Tests inject both
Parquet-write and manifest-write failures, compare the full protected-tree hashes,
and reverify the prior manifest after rollback.

The `.gitignore` explicitly covers raw staging, failed, and backup siblings plus
interim and processed backup siblings. If the process is killed before cleanup,
those temporary trees cannot become accidental Git add candidates.

## 7. Common-endpoint and availability architecture

### 7.1 Target calendar

The target index is the intersection of valid French-industry dates and valid
Treasury-yield dates. Industry returns are compounded over every
\((e_{t-1},e_t]\) interval using cumulative log growth. Treasury yield changes use
the same endpoints and actual elapsed calendar time for carry.

`target_interval_audit.parquet` records interval start, calendar duration, endpoint
presence, source observation counts, and the combined endpoint assertion. A row is
a common holding interval, not necessarily one identical exchange session.

### 7.2 Model-session availability

Availability lags are measured on the actual ordered target calendar:

1. locate the source date's insertion position in target dates;
2. add `availability_lag_model_sessions`;
3. assign that target date as `available_date`; and
4. backward-as-of join only where `available_date <= model_date`.

Phase 0 uses one model session for Treasury context and EFFR, and two for OFR FSI.
Staleness is measured from source date in calendar days. There is no backward fill.

Five market features are derived from target returns at the after-close endpoint
\(t\). Their source and availability date is \(t\), and their forecasting contract
requires the response path to begin at \(t+1\).

Realized volatility and downside semivolatility use squared log returns divided by
the actual elapsed calendar years across each 20-observation window. The model does
not annualize an irregular common-endpoint row as though every row were one
ordinary trading day.

## 8. Phase 0 artifact contract

The audited run contains:

| Artifact | Shape / role |
|---|---|
| `data/interim/target_returns.parquet` | 6,593 × 15 target panel |
| `data/interim/target_interval_audit.parquet` | 6,593 × 7 common-interval evidence |
| `data/interim/macro_features.parquet` | 6,593 × 10 public model features |
| `data/interim/validation_labels.parquet` | 6,593 × 4 OFR labels |
| `data/interim/optional_diagnostics.parquet` | 6,593 × 0 public-core placeholder |
| `data/interim/availability_alignment_audit.parquet` | 6,593 × 48 timing evidence |
| `data/processed/model_matrix.parquet` | 6,465 × 25 finite clean matrix |
| `data/processed/splits/train.parquet` | 3,367 × 25 |
| `data/processed/splits/validation.parquet` | 1,499 × 25 |
| `data/processed/splits/test.parquet` | 1,599 × 25 |
| `artifacts/phase0/quality_report.json` | Machine-readable profiles, checks, warnings, limitations |
| `artifacts/phase0/quality_report.md` | Human-readable quality handoff |
| `artifacts/phase0/manifest.json` | Deterministic governed-file hashes |
| `artifacts/phase0/run_receipt.json` | Timestamped execution and environment handoff |

The clean matrix starts on 2000-07-05 because all 15 targets and 10 contexts are
jointly available only from that endpoint onward. Phase 0 performs no scaling,
imputation, PCA, regime estimation, or other fitted preprocessing.

## 9. Computed gates and status

Phase 0 computes, rather than merely documents, these 23 gates:

| Gate | Assertion |
|---|---|
| `raw_cache_matches_catalog_allowlist` | Raw snapshot pairs match the public-core catalog exactly |
| `minimum_asset_coverage` | Every target meets the configured 0.97 minimum |
| `target_index_unique` | No duplicate target endpoint |
| `target_index_increasing` | Target endpoints are increasing |
| `common_interval_endpoints_aligned` | Every non-initial interval has positive duration and all source endpoints |
| `target_source_internal_gaps_within_limit` | Every target source stays within its maximum internal calendar gap |
| `target_source_start_delays_within_limit` | Target sources begin within the configured delay from requested start |
| `relative_target_calendar_density` | Target-source overlap density meets its configured minimum |
| `model_context_source_coverage` | Every aligned model-context source meets 0.97 usable coverage |
| `model_context_source_internal_gaps_within_limit` | Every aligned model-context source stays within its maximum gap |
| `validation_source_coverage` | Every aligned validation-only source meets 0.97 usable coverage |
| `validation_source_internal_gaps_within_limit` | Every aligned validation-only source stays within its maximum gap |
| `model_index_relative_density` | The final complete-case model index retains at least 0.97 of target endpoints |
| `model_index_internal_gap_within_limit` | The final model index has no excessive internal calendar gap |
| `common_interval_calendar_gap_within_limit` | Common target intervals stay within the configured calendar-day maximum |
| `interval_source_observation_count_gap_within_limit` | Per-interval source observation-count disparity stays within its limit |
| `model_matrix_finite` | Every clean numeric value is finite |
| `model_index_unique` | No duplicate clean-model endpoint |
| `source_dates_not_after_model_date` | No aligned source date exceeds its model date |
| `availability_dates_not_after_model_date` | No aligned availability date exceeds its model date |
| `strict_chronological_splits` | Train, validation, and test are ordered and disjoint |
| `validation_only_features_excluded` | No `validation__*` field enters the model matrix |
| `source_freshness_within_failure_limit` | No source exceeds the 180-day failure threshold |

The audited run passed every gate and had one freshness warning for the French
snapshot. Status is:

- `failed` when any gate fails;
- `passed_with_warnings` when all gates pass but warnings exist; or
- `passed` when gates pass without warnings.

After writing quality, manifest, and receipt evidence, a failed run raises
`DataQualityError`.

## 10. Manifest and receipt separation

### 10.1 Deterministic manifest

`manifest.json` deliberately has no creation timestamp. For identical governed
files and deterministic serialization it contains:

- `manifest_schema_version: "1.0"`;
- `pipeline_version: "phase0-v0.3.0"`;
- catalog and pipeline configuration paths; and
- sorted repository-relative paths, byte sizes, and SHA-256 hashes.

Manifest inputs include:

- every raw CSV and metadata file;
- every Phase 0 interim and processed output;
- quality report JSON and Markdown;
- configuration, documentation, experiment/failure scaffolds, scripts, source
  code, and tests; and
- `.gitignore`, `.python-version`, `README.md`, `LICENSE`, `CITATION.cff`,
  `Makefile`, `pyproject.toml`, and `uv.lock`.

The manifest does not include its own timestamp, the run receipt, or
`experiments/registry.csv`. The registry is a mutable downstream-run index whose
rows refer back to Phase 0 hashes; excluding it prevents a circular dependency.
Each completed stage instead stores the exact Phase 0 hash in its immutable local
run receipt.

### 10.2 Timestamped receipt

`run_receipt.json` separately records:

- UTC creation time and run mode;
- panel, status, warnings, and failed gates;
- model and split dimensions and dates;
- Python, platform, selected package versions, and Git commit when available; and
- manifest path, SHA-256, and file count.

The v0.3.0 audit checkpoint recorded 73 manifest files. That count and its hash are
not hard-coded research constants: later integration can add governed files, and
any edit requires a new Phase 0 run before the new tree is cited.

## 11. Implemented baseline and risk interfaces

### 11.1 Scenario generator contract

The current baseline classes implement:

```python
generator.fit(returns)
paths = generator.sample(
    num_scenarios=N,
    horizon=H,
    rng=np.random.default_rng(seed),
)
```

Input returns have shape `(time, assets)` and outputs have shape
`(scenarios, horizon, assets)`.

Implemented generators:

- i.i.d. historical simulation;
- moving-block bootstrap with contiguous blocks;
- multivariate Gaussian with covariance shrinkage; and
- elliptical multivariate Student-\(t\) with covariance shrinkage and either fixed
  or kurtosis-implied degrees of freedom.

These are foundational baselines, not yet a claim-complete Stage 1 evaluation. A
future release must add common origin/horizon orchestration and further configured
benchmarks before comparing diffusion.

### 11.2 Risk metric contract

The current risk layer implements:

- simple or log path aggregation;
- portfolio loss from cumulative asset return;
- empirical VaR;
- exact integral of the right-continuous empirical loss quantile for ES;
- co-crash probability;
- Kupiec unconditional coverage; and
- Brier score.

All functions validate shape and finite input. Later asset-level backtests must
reuse these definitions or register and test an explicitly different estimator.

## 12. Planned model package boundaries

Current empty `__init__.py` packages reserve stable Python namespaces. Planned
responsibilities are:

```text
src/crisisforge/
├── regimes/
│   ├── switching_state_space.py
│   ├── filtering.py
│   └── diagnostics.py
├── factors/
│   ├── dynamic_factor.py
│   ├── identification.py
│   └── posterior.py
├── diffusion/
│   ├── schedules.py
│   ├── temporal_unet.py
│   ├── conditioning.py
│   ├── trainer.py
│   └── sampler.py
├── mapping/
│   ├── observation.py
│   ├── residuals.py
│   └── posterior_predictive.py
├── tail/
│   ├── importance_sampling.py
│   ├── evt.py
│   ├── calibration.py
│   └── dependence.py
├── portfolio/
│   ├── cvar.py
│   ├── wasserstein_dro.py
│   ├── constraints.py
│   └── backtest.py
├── counterfactual/
│   ├── scm.py
│   ├── simulator.py
│   ├── estimands.py
│   └── validation.py
└── evaluation/
    ├── distributional.py
    ├── tail.py
    ├── decision.py
    ├── counterfactual.py
    └── uncertainty.py
```

The switching-regime, factor, and observation modules form a coherent
state-space system. The one-shot diffusion module models complete future
factor-path tensors and must not quietly duplicate or contradict the factor
transition. Asset scenarios must be reconstructed through the regime-aware
observation equation:

\[
r_{t+h}
=
\alpha_{z_{t+h}}
+
B_{z_{t+h}}f_{t+h}
+
D_{z_{t+h}}\epsilon_{t+h}.
\]

Posterior predictive generation samples regime paths and applies the corresponding
loadings and idiosyncratic scale at each horizon. Averaging loadings or Cholesky
factors across regimes is not an accepted shortcut unless registered as an
explicit ablation.

## 13. Research stages and gates

### Phase 0 — Data foundation

Implemented deliverables:

- direct-origin raw snapshots and metadata;
- common-endpoint target and interval audit;
- availability-aware public context and audit;
- fixed initial splits;
- quality report, deterministic manifest, and timestamped receipt; and
- source/legal decision log.

Gate: all 23 computed Phase 0 assertions must pass. A rerun from the same validated
cache and governed tree should produce the same deterministic manifest; the
timestamped receipt is expected to differ.

### Stage 1 — Conventional scenario baselines

Build on the implemented historical, moving-block, Gaussian, and Student-\(t\)
generators. Add the preregistered benchmark set and evaluate every method on
identical forecast origins, horizons, seeds, and portfolio definitions.

Gate: a sophisticated method that fails may be debugged or retained as a negative
result, but not silently omitted.

### Stage 2 — Bayesian switching dynamic factor core

Estimate filtered regime probabilities, latent factors, regime-dependent loadings,
idiosyncratic scales, and residual dependence. Pass soft state probabilities
between stages rather than hard-label certainty.

Gate:

- address factor rotation and regime label switching;
- report regime occupancy and stability;
- compare posterior-draw and plug-in-mean predictions; and
- stop before diffusion if held-out reconstruction and tail covariance do not
  justify the factor core.

### Stage 3 — Regime-conditioned one-shot temporal diffusion

Start with \(H=20\). Generate a complete \(H\times q\) future factor tensor in one
denoising process, conditioned only on information available at the origin.

Gate:

- no future realized state enters conditioning;
- expand to \(H=60\) only after stability at \(H=20\);
- compare against moving-block and Student-\(t\) baselines using held-out proper
  scores; and
- retain a valid failure to beat baselines as a negative result.

### Stage 4 — Tail calibration and factor-to-asset mapping

Add training-fold tail sampling/weighting, EVT and calibration, then reconstruct
asset returns with regime-dependent \(B(z)\), \(D(z)\), and idiosyncratic residuals.

Gate:

- report central-fit versus tail-fit trade-offs as a Pareto frontier;
- require held-out VaR/ES and co-crash evidence before using `tail-aware` in a main
  conclusion; and
- run fixed versus regime-dependent mapping and residual-distribution ablations.

### Stage 5 — Asset-level risk and portfolio decisions

Compute asset-level VaR, ES, co-crash, and backtests; then solve CVaR and
Wasserstein-DRO allocations outside the diffusion training loop. Use the relevant
strong-duality convex reformulation rather than a daily nested inner maximization.

Gate:

- evaluate on realized out-of-sample losses after turnover and transaction costs;
- calibrate ambiguity radius on validation data;
- test weight stability and optimizer sensitivity; and
- do not call robustness around generated samples robustness to the true
  distribution without a defensible bound.

### Stage 6 — Structural counterfactual extension

Use a time-unrolled structural causal model and
abduction–action–prediction. Validate first in a semi-synthetic market with known
causal ground truth.

Gate:

- predictive, conditional-stress, and causal estimands stay separate;
- DAG and support misspecification sensitivity is reported;
- total and controlled effects are labeled separately; and
- real-market output remains a model-based structural intervention unless a
  defensible identification strategy is registered.

## 14. Experiment registry and failure policy

Every run receives a stable identifier:

```text
YYYYMMDD-HHMMSS_<stage>_<short-name>_s<seed>
```

Planned run directory:

```text
experiments/runs/<run_id>/
├── config.yaml
├── status.json
├── environment.json
├── data_manifest.json
├── metrics.json
├── notes.md
├── failure.md
└── artifact_index.json
```

Each registry row or run record must capture:

- run ID, stage, parent run, experiment family, and hypothesis;
- primary metric and comparison run;
- Git commit and dirty-worktree flag;
- Python, dependency lock hash, hardware, and seeds;
- Phase 0 manifest hash and frozen config paths;
- train/validation/test origins, horizon, purge, and embargo;
- model, optimizer, scenario count, and runtime;
- metric estimates and uncertainty intervals;
- artifact and checkpoint hashes; and
- outcome: `passed`, `failed`, `invalidated`, or `negative_result`.

`failed` means an implementation or numerical procedure failed.
`invalidated` means the evidence is not comparable, for example because of leakage
or corrupted input. `negative_result` means a valid experiment did not beat its
registered benchmark.

All three remain in the registry. No run is deleted because its result is
unattractive.

## 15. Test strategy

### Current unit and integrity tests

- At the final Phase 0 v0.3.0 audit checkpoint, 59 tests passed and the Ruff lint
  suite passed. The count may increase after later-stage integration; the final
  release must report the fresh integrated total rather than treating 59 as a
  permanent constant.
- macro transforms and return definitions;
- actual-model-session availability lag;
- common-endpoint compounding and interval counts;
- elapsed-time-compatible Treasury proxy behavior;
- elapsed-time volatility and backward-looking market features;
- chronological split separation;
- snapshot hash, identity, schema, row-count, order, date-range, and requested-range
  rejection;
- raw allowlist, target/context coverage, internal-gap, density, and model-index
  gates;
- invalid candidate rejection and full raw/interim/processed/artifact transaction
  rollback under injected persistence failures;
- deterministic refresh/offline manifest equality and crash-leftover Git ignores;
- scenario output shape, finiteness, and block continuity; and
- asset-level risk metric definitions.

### Required next tests

- regime probability normalization and label stability;
- factor and asset tensor shapes;
- one-shot diffusion conditioning and path stability;
- VaR/ES/co-crash calibration fixtures;
- CVaR and Wasserstein-DRO analytic toy problems; and
- structural abduction–action–prediction on known equations.

Real copyrighted or licensed observations must not become committed unit-test
fixtures. Use small synthetic fixtures for regression tests.

## 16. Reporting and release policy

Minimum durable artifacts by stage:

| Stage | Required evidence |
|---|---|
| Phase 0 | Snapshots, interval/availability audits, quality report, manifest, receipt |
| Baselines | Common-origin scenario receipt, distributional and tail score tables |
| Regime/factor | Posterior summaries, occupancy, label stability, reconstruction |
| Diffusion | Training curves, checkpoint hashes, sampled paths, stability diagnostics |
| Tail/mapping | EVT fits, calibration curves, \(B,D\) and residual diagnostics |
| Risk/decision | VaR/ES backtests, co-crash calibration, weights, costs, realized losses |
| Counterfactual | SCM, support checks, paired factual/counterfactual paths, ground-truth error |
| Final | Report, appendix, figures, executed notebooks, slides, README, and LinkedIn material |

Every table or figure states panel, sample, horizon, forecast origin, model version,
and uncertainty method. Captions retain “French research portfolio” and “synthetic
Treasury return proxy” labels where relevant.

Before publication:

- rerun tests and lint;
- run Phase 0 after any governed-doc/code/config edit;
- confirm every computed gate passes;
- archive the matching manifest and timestamped receipt;
- resolve every cited metric to a registered experiment;
- disclose negative results and failed alternatives;
- keep Cboe inputs outside the public core unless a licensed extension is active;
- avoid investability claims from `research_core`; and
- distinguish conditional stress generation from identified causal evidence.
