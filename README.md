# CrisisForge

**Decision-Focused Market Simulation under Regime Shifts**

CrisisForge is a Python research repository for generating multi-asset stress
scenarios, reconstructing them at the asset level, measuring tail risk, and testing
whether statistically sophisticated scenarios improve realized portfolio decisions.

The implemented system is deliberately multi-stage and auditable:

```text
availability-lagged current public-source snapshots
  -> conventional rolling scenario baselines
  -> MAP / empirical-Bayes switching dynamic factor model
  -> regime-aware factor-to-asset observation mapping
  -> one-shot conditional temporal diffusion engineering pilot
  -> asset-level VaR / Expected Shortfall / co-crash evaluation
  -> empirical CVaR and Wasserstein-DRO decisions
  -> separately validated semi-synthetic structural counterfactual extension
```

All model code is Python and saved under `src/crisisforge/`; command-line entry
points, configurations, tests, run receipts, hashes, and generated evidence are
retained in the repository tree.

## Evidence status

Version `0.3.0` contains a complete validation-era research core. Phase 0
constructs and quality-checks the complete matrix, including the chronological
test split beginning after 2019-12-31. “Sealed” means those rows are
governed-excluded from estimator fitting, checkpoint selection, model scoring,
and portfolio decisions; it does not mean Phase 0 cryptographically hides their
values. No Stage 0, 1, 2, 3, or 5 result is a test-set claim. Results below are
validation evidence produced from clean Git commit
`b6891133bdb6b96e1e23c6bea3bd033ea9685c7c` and Phase 0 manifest:

```text
f051ec35236a481858b67c5b1e7136f1698036832f427efba521dbf3fcd36d70
```

Stage 6 v2 was rerun separately from clean implementation commit
`23a4c9ad20683e5e94f7154472073a96d18f90be`.

The current evidence does **not** show that the complex generator beats simpler
benchmarks. Preserving that negative result is part of the research design.

The validated reader report is also published as a
[private interactive report](https://crisisforge-validation-2026.early-boot-9195.chatgpt.site).

## Results at a glance

| Evidence | Registered validation result | Interpretation |
|---|---:|---|
| Phase 0 data | 6,465 complete rows; 15 return targets; 10 context variables; 23/23 gates passed | Public retrospective research panel; one source-freshness warning |
| Best Stage 0 energy score | 0.08974, filtered historical EWMA | Lowest validation energy score among seven baselines |
| Best Stage 0 variogram score | 0.53223, 20-observation moving block | Lowest validation variogram score among seven baselines |
| Stage 1 switching factor | Energy 0.09432; variogram 0.58864; joint VaR–ES score -0.94912 | MAP/empirical-Bayes model did not dominate simple baselines |
| Stage 3 paired evidence | Stage 1 minus moving block energy +0.00450, 95% interval [0.00266, 0.00635] | Lower is better; Stage 1 was worse on this validation comparison |
| Stage 3 tail-score evidence | All paired joint VaR–ES intervals crossed zero | Tail-score differences were inconclusive |
| Stage 2 diffusion pilot | 4 reporting origins; tail-weighted energy 0.09168 versus base 0.09205 | Engineering/leakage-control pilot only; no superiority claim permitted |
| Stage 5 realized ES | Historical CVaR 0.01767; Stage 1 CVaR 0.02074 | Historical scenarios have lower observed validation tail loss; no paired interval was computed |
| Stage 5 maximum drawdown | Historical CVaR 0.03411; Stage 1 CVaR 0.04782 | Historical CVaR has the smaller observed validation drawdown |
| Wasserstein-DRO sensitivity | Radii 0.0001–0.001 produced nearly unchanged Stage 1 decisions | No final radius selected; little decision benefit in the registered grid |
| Stage 6 counterfactual | Oracle error \(4.08\times10^{-17}\); misspecified path RMSE 0.0731–0.2782 | Implementation recovered known semi-synthetic truth and exposed specification sensitivity |

The Stage 0 and Stage 1 comparison combines different estimation policies: Stage 0
refits a rolling 1,500-observation window, while Stage 1 estimates parameters on the
training era and updates only filtered state beliefs. The paired intervals therefore
do not isolate a causal model-class effect.

At every evaluated origin—74 Stage 0/Stage 1 origins and four Stage 2
origins—the realized co-crash label was zero; generated scenario sets may still
assign nonzero probabilities. Co-crash Brier scores are retained for completeness
but cannot assess crisis-event discrimination in this sample.

## Implemented stage map

| Stage | Implemented scope | Main evidence |
|---|---|---|
| Phase 0 | Public data acquisition, availability alignment, common-endpoint targets, chronological splits, quality and provenance | `artifacts/phase0/` |
| Stage 0 | Seven rolling conventional scenario baselines and asset-level proper/tail scores | `artifacts/stage0_baselines/` |
| Stage 1 | Four-factor PCA representation, four-state Gaussian HMM fitted by MAP/regularized EM, regime-specific VAR(1), and a state-dependent map in \(y_t=\log(1+r_t)\) space followed by \(r_t=\operatorname{expm1}(y_t)\) | `artifacts/stage1_switching_factor/` |
| Stage 2 | Small one-shot conditional temporal DDPM pilot with base and tail-weighted checkpoints | `artifacts/stage2_public_core_pilot/` |
| Stage 3 | Paired circular moving-block bootstrap intervals for Stage 1 versus Stage 0 | `artifacts/stage3_comparison/` |
| Stage 4 | Tail weighting and observation mapping are integrated upstream; POT/GPD and rolling conformal utilities are implemented and tested but not integrated into an empirical runner | No standalone Stage 4 run |
| Stage 5 | Equal-weight, historical CVaR, Stage 1 CVaR, and four Wasserstein-DRO sensitivities | `artifacts/stage5_decisions/` |
| Stage 6 | Time-unrolled semi-synthetic SCM, paired interventions, outcome-specific tail-loss signs, oracle recovery, and misspecification tests | `artifacts/stage6_counterfactual/` |

Stage 1 failed its confirmatory central-distribution advancement gate; Stage 2 and
Stage 5 proceeded only as explicitly non-confirmatory engineering and decision
diagnostics. Stage 1 is not a full Bayesian posterior sampler and does not use MCMC.
Stage 2 does not propagate a full parameter posterior, and its independently
simulated future regime path is not guaranteed to be jointly dynamically consistent
with the generated factor path. These limitations are recorded in configuration
and receipts.

## Reproduce the project

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/).

From the repository root:

```bash
uv sync --extra dev --extra models --extra reporting
uv run pytest
uv run ruff check src tests scripts
```

Rebuild Phase 0 from the retained, integrity-checked public snapshots:

```bash
uv run crisisforge-phase0 --offline
```

Use `uv run crisisforge-phase0 --refresh` only when intentionally downloading new
source snapshots. A refreshed panel can change the manifest and is a new evidence
release.

Run the validation pipeline in dependency order:

```bash
uv run crisisforge-stage0
uv run crisisforge-stage1
uv run crisisforge-compare
uv run crisisforge-stage2
uv run crisisforge-stage5
uv run crisisforge-counterfactual
uv run python scripts/make_figures.py
uv run python scripts/make_linkedin_assets.py
uv run python scripts/build_report_artifact.py
```

Approximate reference runtime is dominated by Stage 1 (about eight minutes on the
recorded Apple Silicon system). Stage 2 is intentionally a very small CPU pilot.
Every stage writes an immutable-style local receipt that binds input/config hashes,
Git state, output hashes, and the sealed-test declaration where applicable.

## Data foundation

The default public track uses:

- Kenneth French's value-weighted 12-industry research portfolios;
- official U.S. Treasury daily par yields, converted into three transparent
  duration-convexity Treasury return proxies;
- New York Fed effective federal funds rate data;
- OFR Financial Stress Index components as validation-only labels; and
- five strictly backward-looking risk features derived from the target panel.

The resulting common-endpoint matrix spans 2000-07-05 through 2026-05-29:

- train: 3,367 rows;
- validation: 1,499 rows; and
- governed post-2019 holdout: 1,599 rows.

This is a retrospective research panel, not an investable total-return universe or
live trading feed. The French source ended 60 calendar days before the requested
2026-07-28 endpoint, which is disclosed as the sole Phase 0 warning.
The retained public histories are current snapshots that can be revised; they are
not a point-in-time vintage database. Availability lags prevent same-run
look-ahead, but revision-specific claims require separately versioned historical
releases.

FRED/ALFRED are deliberately excluded because the current FRED Terms of Use
prohibit using FRED Services or Content to develop or train machine-learning,
deep-learning, or generative-AI software. The public pipeline downloads from
original publishers. A publication-grade applied extension should substitute
licensed point-in-time total returns from CRSP/WRDS, Bloomberg, LSEG, FactSet, or
an equivalent archive.

## Repository guide

- [Research proposal](docs/research_proposal.md)
- [Mathematical formulation](docs/mathematical_formulation.md)
- [Data dictionary](docs/data_dictionary.md)
- [Literature review](docs/literature_review.md)
- [Repository architecture and reproducibility contract](docs/repository_architecture.md)
- [Experiment registry](experiments/registry.csv)
- [Figure contracts](reports/figure_contracts.md)
- [Research report](reports/crisisforge_research_report.md)
- [PDF research report](reports/crisisforge_research_report.pdf)
- [Research presentation](slides/crisisforge_research_presentation.pptx)
- [LinkedIn post](linkedin/linkedin_post.md)
- [Interview pitch](linkedin/interview_pitch.md)
- [Research log](docs/research_log/)

Generated data and model artifacts are retained locally but ignored by Git. Source
code, configuration, tests, documentation, and the experiment registry are
versioned. Run receipts and cryptographic hashes make local evidence auditable.

After committing a release candidate, run the fail-closed release checks and create
the self-contained archive:

```bash
uv run python scripts/run_release_qa.py
uv run python scripts/build_release_manifest.py
uv run python scripts/build_release_bundle.py
```

## Claim boundaries

CrisisForge does not claim:

- to predict the date or cause of the next financial crisis;
- that a statistically inferred regime is an observed economic ground truth;
- that Stage 1 is fully Bayesian;
- that the diffusion pilot beats conventional baselines;
- that validation performance is test performance;
- that robustness around generated samples is robustness to the true
  data-generating process; or
- that the semi-synthetic structural experiment identifies real-world policy
  effects.

Real-market counterfactual outputs, if added later, must be described as
**model-based structural interventions** unless a separate defensible identification
design is registered.
