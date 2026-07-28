# CrisisForge

**Decision-Focused Market Simulation under Regime Shifts**

CrisisForge is a research project for generating multi-asset market paths, measuring
asset-level tail risk, and testing whether statistically realistic scenarios lead to
better out-of-sample portfolio decisions.

The core architecture is:

```text
point-in-time market information
  -> Bayesian switching dynamic factor state-space model
  -> posterior regime paths and latent factors
  -> regime-conditioned one-shot temporal diffusion
  -> r_t = alpha(z_t) + B(z_t) f_t + D(z_t) epsilon_t
  -> VaR / Expected Shortfall / co-crash risk engine
  -> empirical CVaR and Wasserstein-DRO allocation
  -> separately validated structural counterfactual extension
```

The counterfactual extension is validated first in a semi-synthetic market with a
known structural causal model. Real-market outputs are described as model-based
structural interventions unless a separate identification design is available.

## Current status

Version `0.3.0` completes the audited Phase 0 foundation:

- a public `research_core` panel with 12 value-weighted U.S. industry portfolios
  and three explicitly labeled Treasury duration-return proxies;
- model context from U.S. Treasury and New York Fed publications plus five
  backward-looking risk features derived from the target return panel;
- OFR stress components retained as validation-only labels, not circular predictors;
- raw snapshot provenance, timestamps, URLs, SHA-256 hashes, schema, and date checks;
- actual-model-session availability alignment at an after-close decision timestamp;
- common-endpoint return aggregation with a source-by-source interval audit;
- 6,465 complete observations, 15 return targets, and 10 context variables;
- strict chronological train/validation/test splits;
- automated data-quality, leakage, and reproducibility checks.

The only current data warning is that the French daily source ends on 2026-05-29.
This release is therefore a retrospective research build, not a live signal feed.

Model stages are intentionally gated. A more complex model advances only if it
beats strong conventional baselines on held-out predictive, tail-risk, or decision
metrics.

## Reproduce Phase 0

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest
uv run crisisforge-phase0 --refresh
```

To rerun without network access after raw snapshots have been cached:

```bash
uv run crisisforge-phase0 --offline
```

Important outputs:

- `data/interim/target_returns.parquet`
- `data/interim/target_interval_audit.parquet`
- `data/processed/model_matrix.parquet`
- `data/processed/splits/{train,validation,test}.parquet`
- `artifacts/phase0/quality_report.md`
- `artifacts/phase0/manifest.json`
- `artifacts/phase0/run_receipt.json`

Raw data, snapshot metadata, and generated artifacts are retained locally and are
not committed. The download code, catalog, validation rules, and governance policy
are versioned; each run also produces a local cryptographic manifest and receipt
that can be archived with a research release.

## Research documents

- [Research proposal](docs/research_proposal.md)
- [Mathematical formulation](docs/mathematical_formulation.md)
- [Data dictionary](docs/data_dictionary.md)
- [Literature review](docs/literature_review.md)
- [Repository architecture](docs/repository_architecture.md)
- [Research log](docs/research_log/0001_phase0.md)
- [Independent Phase 0 audit](docs/research_log/0002_phase0_independent_audit.md)

## Data policy

The default public track is a **retrospective research panel**, not an investable
ETF portfolio. It downloads the value-weighted daily section of Kenneth French's
12-industry file at runtime and derives three transparent duration-convexity return
proxies from official U.S. Treasury par yields. Cboe and French raw data are never
committed or redistributed. Cboe VIX, SKEW, and VVIX histories are not part of the
public core; licensed Cboe DataShop or OptionMetrics data can be added in an
applied extension.

FRED/ALFRED are deliberately excluded. The current FRED Terms of Use prohibit using
FRED Services or Content to develop or train machine-learning, deep-learning, or
generative-AI software. CrisisForge instead downloads each variable from its
original publisher.

A publication-grade applied run should replace the research targets with licensed,
point-in-time total returns from CRSP/WRDS, Bloomberg, LSEG, FactSet, or an
equivalent archive. The provider interface and data dictionary identify these
replacements.

## Research claims this project does not make

- It does not claim to predict the date or cause of the next financial crisis.
- A latent regime is not a labeled ground truth.
- A conditional scenario is not automatically a causal counterfactual.
- Robustness to generated scenarios is not automatically robustness to the real
  data-generating process.
- A model is not considered better solely because its synthetic paths look realistic.

See the proposal and mathematical formulation for estimands, hypotheses, validation
rules, and stop/go gates.
