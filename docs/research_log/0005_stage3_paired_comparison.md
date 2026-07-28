# Research Log 0005 — Stage 3 Paired Validation Comparison

**Completed at (UTC):** `2026-07-28T01:46:52.763112+00:00`  
**Experiment:** `stage3_paired_validation_comparison_v1`  
**Status:** completed  
**Evaluation split:** validation only; post-2019 rows governed-excluded

## Objective and registered design

This experiment quantifies uncertainty in the difference between Stage 1 and each
Stage 0 generator at the same 74 rolling origins. For each lower-is-better metric,
the reported contrast is:

\[
\Delta=\text{Stage 1 score}-\text{Stage 0 score}.
\]

Positive values favor Stage 0; negative values favor Stage 1. The registered
circular moving-block bootstrap uses 10,000 replications, origin block length 4,
95% intervals, and seed 20260728. The 21 intervals are exploratory and receive no
multiplicity adjustment.

Configuration: `configs/stage3_comparison.yaml`, current clean-commit SHA-256
`4b25606776a45e3124ca221f976bb0b808d2bc88c0f17a6a199c76f31f5093ca`.
The run binds:

- Stage 0 detail SHA-256
  `35c6624013f94deae1b45dc231ef0b3c575851d75d6a6934e80beaf5b2d978df`;
- Stage 0 receipt SHA-256
  `07dc81db391cce736bc31d2ea169c1f772f463660e59d4bc06c6dd92b58519e9`;
- Stage 1 detail SHA-256
  `21e29d7b72b49a70ad4768c514cfe94ef334838159bc5ba707895f0674dbbd2e`;
- Stage 1 receipt SHA-256
  `c79e82d3b017ea4e048e8f637ce77b27855b972bc19ba1f9f0317f91d0efda14`;
- Git commit
  `b6891133bdb6b96e1e23c6bea3bd033ea9685c7c`, clean worktree.

Both upstream experiments bind Phase 0 manifest
`f051ec35236a481858b67c5b1e7136f1698036832f427efba521dbf3fcd36d70`.

## Exact paired intervals

| Baseline | Metric | Mean \(\Delta\) | 95% interval |
|---|---|---:|---:|
| `filtered_historical_ewma` | Energy | 0.004576743999417447 | [0.0016695457693681843, 0.007470881247590758] |
| `filtered_historical_ewma` | Variogram | 0.04661433443982349 | [0.009219895128067146, 0.08489798563593896] |
| `filtered_historical_ewma` | Joint VaR–ES | -0.001366158899028379 | [-0.00835912830866583, 0.004220794284372057] |
| `gaussian_shrinkage` | Energy | 0.0025123620248652286 | [-0.0010763668932178266, 0.005693439360695826] |
| `gaussian_shrinkage` | Variogram | 0.005908684853413853 | [-0.0685834584175895, 0.06641477233101861] |
| `gaussian_shrinkage` | Joint VaR–ES | -0.0005776004870214406 | [-0.008226812052870482, 0.006482357172587568] |
| `historical_iid` | Energy | 0.002583244623425525 | [-0.000716834669086169, 0.0055533135822941025] |
| `historical_iid` | Variogram | 0.01703845012287621 | [-0.04609089674331048, 0.07084839725340908] |
| `historical_iid` | Joint VaR–ES | -0.0030587306292663165 | [-0.01133697344268756, 0.0042835980979717934] |
| `moving_block_20` | Energy | 0.004504007429055828 | [0.0026615260799422807, 0.006351852136252796] |
| `moving_block_20` | Variogram | 0.05641273655515987 | [0.02027872704957251, 0.09573999018335864] |
| `moving_block_20` | Joint VaR–ES | -0.00009141543690447216 | [-0.008661805756586169, 0.007442411268175069] |
| `student_t_copula` | Energy | 0.002837318946037644 | [-0.0007848209989739127, 0.005981494705243305] |
| `student_t_copula` | Variogram | 0.004434835843946697 | [-0.07070585063079908, 0.06444568494589202] |
| `student_t_copula` | Joint VaR–ES | -0.0013998599885772234 | [-0.010616539515624709, 0.006401984419216807] |
| `student_t_elliptical` | Energy | 0.0026850497733599413 | [-0.0008976145380407469, 0.005833879212569745] |
| `student_t_elliptical` | Variogram | 0.010083419055751547 | [-0.0618113072810096, 0.06834602703330604] |
| `student_t_elliptical` | Joint VaR–ES | -0.00005989932842787938 | [-0.0075442779367235445, 0.006817538422409248] |
| `var1_residual_bootstrap` | Energy | 0.003079122283354016 | [0.00004665090821961871, 0.0058655406016530935] |
| `var1_residual_bootstrap` | Variogram | 0.025337968119872985 | [-0.02888206707672967, 0.07326224936421219] |
| `var1_residual_bootstrap` | Joint VaR–ES | -0.0005884117183377456 | [-0.010165538575506554, 0.006818396086398051] |

## Negative results and interpretation

- Stage 1 is worse than EWMA-filtered historical simulation and moving-block
  bootstrap on both energy and variogram scores: all four intervals are above
  zero.
- Stage 1 is also weakly worse than the VAR residual bootstrap on energy; its
  interval lower endpoint is `0.00004665090821961871`.
- Every joint VaR–ES interval crosses zero. There is no paired evidence that
  Stage 1 improves the tail-risk score.
- None of the 21 registered intervals directionally favors Stage 1.
- These are exploratory validation intervals, not confirmatory \(p\)-values.
  Multiplicity is not adjusted.

Most importantly, Stage 0 refits a rolling 1,500-observation window while Stage 1
freezes parameters through 2013 and only filters state beliefs. The contrasts
therefore combine model-class and estimation-policy differences. They do not
identify a causal effect of “using a switching factor model.”

## Open issues

1. Run a preregistered equal-estimation-policy comparison if model-class
   attribution is required.
2. Justify or sensitivity-test the origin block length without selecting it to
   improve conclusions.
3. Address multiplicity before any confirmatory claim over many models and
   metrics.
4. Keep post-2019 rows governed-excluded for one final frozen-design evaluation.

## Durable artifacts

- `artifacts/stage3_comparison/paired_metric_intervals.csv`, SHA-256
  `ac2dfc8e7182e3a33edfe0012b8875ebd8f64997986db97a419dab6c1ece1b95`;
- `artifacts/stage3_comparison/summary.json`, SHA-256
  `dcaa340da0ba641dab304e6671bf3a4a575794c63ea3912e5842a6566a74dab2`;
- `artifacts/stage3_comparison/run_receipt.json`.
