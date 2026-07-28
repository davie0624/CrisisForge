# CrisisForge figure contracts

These contracts freeze each figure's question, data source, encoding, and
interpretive boundary before rendering. Every figure is produced by
`scripts/make_figures.py`; PNG and SVG versions are retained.

## Figure 0 — Implemented research architecture

- **Question:** How do the completed public-core modules connect, and which
  components are pilots or separate validation tracks?
- **Dataset:** implemented package modules, registered experiment stages, and
  run receipts.
- **Unit/grain:** one box per completed module or experiment family.
- **Encodings:** arrows = information flow; blue = completed core, amber =
  engineering pilot/partial tail layer, gray = separate comparator or
  semi-synthetic track.
- **Boundary:** the diagram describes implementation, not full Bayesian posterior
  inference or real-market causal identification.

## Figure 1 — Chronological research split

- **Question:** What information periods are used for training, validation, and
  the sealed test set?
- **Dataset:** `configs/pipeline.yaml` and
  `data/processed/model_matrix.parquet`.
- **Unit/grain:** calendar date; one continuous span per split.
- **Encodings:** horizontal position = date; color = split.
- **Boundary:** the post-2019 test span is shown only from registered dates and
  row counts. No test values or results are visualized.

## Figure 2 — Validation distribution scores

- **Question:** How do Stage 0 generators and the Stage 1 switching-factor model
  compare on the same 74 validation origins?
- **Dataset:** Stage 0 and Stage 1 summaries.
- **Unit/grain:** mean score per model across 74 non-overlapping 20-observation
  origins.
- **Encodings:** horizontal position = score; row = model; blue = Stage 1,
  gray = conventional baseline.
- **Boundary:** lower is better. Differences combine model class and estimation
  policy because Stage 0 rolls its window and Stage 1 freezes train parameters.

## Figure 3 — Paired validation intervals

- **Question:** Is the Stage 1 minus Stage 0 score difference directionally
  stable across validation origins?
- **Dataset:** `artifacts/stage3_comparison/paired_metric_intervals.csv`.
- **Unit/grain:** paired mean difference and circular moving-block bootstrap
  95% interval per baseline and metric.
- **Encodings:** point = mean difference; line = interval; vertical line = zero.
- **Boundary:** exploratory, not multiplicity-adjusted, and not a model-class
  treatment effect.

## Figure 4 — Latent-state profiles

- **Question:** What statistical profiles did the four unlabeled training states
  exhibit?
- **Dataset:** Stage 1 state profiles.
- **Unit/grain:** one state; train-only soft-probability-weighted summaries.
- **Encodings:** bar length = occupancy; scatter x/y = weighted mean/volatility;
  point size = occupancy.
- **Boundary:** state numbers have no automatic economic names.

## Figure 5 — Portfolio decision outcomes

- **Question:** What realized validation trade-off did each portfolio rule make
  between cumulative return, Expected Shortfall, and drawdown?
- **Dataset:** Stage 5 summary.
- **Unit/grain:** one validation strategy across 74 non-overlapping holding
  blocks.
- **Encodings:** x = realized ES; y = cumulative net return; point size =
  maximum drawdown.
- **Boundary:** all Wasserstein radii are separate validation sensitivities; none
  is selected as final.

## Figure 6 — Structural misspecification sensitivity

- **Question:** How does policy-shock effect recovery change when the known
  semi-synthetic SCM is misspecified?
- **Dataset:** Stage 6 mean effect paths.
- **Unit/grain:** mean paired potential-outcome difference at each horizon step.
- **Encodings:** x = step; y = standardized mean effect; color/line style =
  structural model.
- **Boundary:** known-ground-truth semi-synthetic validation only; no observed
  market causal effect is claimed.

## Figure 7 — Diffusion engineering pilot

- **Question:** Did base and tail-weighted training run stably, and what
  descriptive scores resulted?
- **Dataset:** Stage 2 training history and summary.
- **Unit/grain:** epoch-level denoising loss and four late-validation origins.
- **Encodings:** line = loss by epoch; grouped points = model variant.
- **Boundary:** four reporting origins, small checkpoints, no superiority claim.
