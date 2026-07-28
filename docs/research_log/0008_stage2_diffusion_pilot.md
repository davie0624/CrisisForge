# Research Log 0008 — Stage 2 One-Shot Temporal Diffusion Pilot

**Completed at (UTC):** `2026-07-28T01:53:37.277829+00:00`  
**Experiment:** `stage2_public_core_pilot_v1`  
**Status:** completed with zero implementation failures  
**Evaluation split:** late-validation reporting segment; post-2019 rows
governed-excluded

> The log number follows actual run completion, not conceptual stage number.
> Its final receipt was written after the Stage 3, 5, and 6 receipts, so it is
> listed after Log 0007 even though “Stage 2” is earlier in the conceptual model
> sequence.

## Objective and architecture

This public-core pilot verifies the leakage controls and saved-artifact path for a
genuine one-shot conditional temporal DDPM. It generates the full
`20 x 4` future factor tensor in one denoising process; it does not roll forward
one day at a time. Conditioning uses a 60-observation factor and macro history
ending at the origin plus the four filtered HMM state probabilities. Future
context is forbidden.

Generated factor paths are mapped back to 15 asset returns with the separately
fitted Stage 1 state-dependent observation equation. Two checkpoints are reported:
base score matching and two-epoch tail-importance fine-tuning.

## Registered configuration and provenance

- Pilot evaluation config: `configs/stage2_evaluation.yaml`, SHA-256
  `0e46ddff91f694b4210d0db7a8f46d181a08a8bba96a79d36c8b6401ea1e043d`;
- architecture config: `configs/stage2_diffusion.yaml`, SHA-256
  `1a5ce2003eab4ae6dac8746f4d0451a1ea5ebab96330c9c343b72953474d51d2`;
- Stage 1 model config SHA-256
  `f0f07ad5add4b43e6cef2d01c5eb0ddfd21c4d4e9fe092a81349dbcad7c5ff1d`;
- Phase 0 manifest SHA-256
  `f051ec35236a481858b67c5b1e7136f1698036832f427efba521dbf3fcd36d70`;
- model-matrix SHA-256
  `2a57dd7e8b43dc0d0444d4035cd3d45f1fef340e4a9faaea61fd9ac694cfe699`;
- Git commit
  `b6891133bdb6b96e1e23c6bea3bd033ea9685c7c`, clean worktree.

Effective pilot capacity is 12 diffusion steps, 32 hidden channels, 32-dimensional
time embedding, and two residual blocks. Training uses CPU, two threads, batch
size 64, three base epochs, two tail epochs, and 64 scenarios per reporting
origin.

There are 658 training windows, 37 disjoint early-validation tuning windows, and
4 late-validation reporting origins. Tail weights use 64 train-only tail windows,
severity threshold `3.6821891979706627`, mean weight 1.0, minimum weight
`0.7715423960456017`, and maximum weight `6.172339168364814`.
They use a two-sided standardized path-severity score, a training 90th-percentile
threshold, a capped linear excess weight with strength 3 and cap 8, global
training mean-one normalization, and minibatch renormalization during the
weighted denoising loss.

## Training record

| Stage | Epoch | Train loss | Validation denoising loss |
|---|---:|---:|---:|
| Base | 1 | 1.103238806173794 | 1.0234493017196655 |
| Base | 2 | 1.0517170849542126 | 0.9941958785057068 |
| Base | 3 | 1.023874874897641 | 0.9730676412582397 |
| Tail weighted | 1 | 1.0139242827348796 | 0.9864274859428406 |
| Tail weighted | 2 | 0.9976300838145804 | 0.9770685434341431 |

Diffusion training took `0.9935835840005893` seconds and evaluation
`0.654361875000177` seconds. The Stage 1 fit required by this independent pilot
run took `369.157285082998` seconds.

## Late-validation pilot metrics

| Variant | Energy | Variogram | Joint VaR–ES | VaR breaches | Co-crash Brier |
|---|---:|---:|---:|---:|---:|
| Base | 0.0920487515864791 | 0.7072089236556681 | **-0.934902319945545** | 0 / 4 | 0.00579833984375 |
| Tail weighted | **0.09167996787113185** | **0.6935241484921238** | -0.933368213596135 | 0 / 4 | 0.00579833984375 |

No generated factor was clipped in either variant. At all four reporting origins,
the realized co-crash label and VaR-breach indicator are zero; generated scenario
sets may still assign nonzero co-crash probabilities.

## Negative results and claim boundary

- Tail weighting slightly improves energy and variogram scores but worsens the
  joint VaR–ES score. This is a trade-off between multivariate distribution scores
  and the joint VaR–ES score, not evidence of calibrated tail improvement.
- Four reporting origins, 64 scenarios per origin, few epochs, and 12 diffusion
  steps are inadequate for model ranking or superiority claims.
- The Stage 2 segment differs from the 74-origin Stage 0/1 evaluation. Its metric
  levels must not be presented as a direct leaderboard comparison.
- Zero realized breach and co-crash labels provide no reliable calibration or
  discrimination evidence.
- Future HMM regimes are drawn independently of the diffusion factor paths
  conditional on the same origin belief. Their joint dynamic consistency is not
  guaranteed.
- Stage 1 parameters are MAP/empirical-Bayes. No full posterior uncertainty is
  propagated.
- This is a conditional factor generator, not a causal model and not a direct
  asset-return generator.

## Open issues

1. Increase origins, scenarios, epochs, and diffusion steps under a separately
   budgeted registered experiment.
2. Couple future regime and factor dynamics rather than sampling them
   independently.
3. Add the registered ablations—unconditional, hard-regime negative control, and
   soft-regime conditioning—before substantive architecture claims.
4. Evaluate POT/GPD and temporal conformal components with enough exceedances and
   realized outcomes; they are specified but this pilot is not powered to validate
   them.
5. Keep post-2019 rows governed-excluded until all tuning and selection rules are
   frozen.

## Durable artifacts

- `artifacts/stage2_public_core_pilot/checkpoints/base.pt`, SHA-256
  `6f6beed66f4be549e3c8a4e8dec9ac1dbef99c357926a8305aa89239c380ddc2`;
- `artifacts/stage2_public_core_pilot/checkpoints/tail_weighted.pt`, SHA-256
  `554072aaef4727eff51dbafbdb2bf4e7d8e96d617009b5a63125bb25d00456b9`;
- `artifacts/stage2_public_core_pilot/cumulative_asset_scenarios_base.npz`;
- `artifacts/stage2_public_core_pilot/cumulative_asset_scenarios_tail_weighted.npz`;
- `artifacts/stage2_public_core_pilot/train_only_standardizers.npz`;
- `artifacts/stage2_public_core_pilot/window_boundaries.parquet`;
- `artifacts/stage2_public_core_pilot/training_history.csv`;
- `artifacts/stage2_public_core_pilot/rolling_results.csv`;
- `artifacts/stage2_public_core_pilot/summary.json`;
- `artifacts/stage2_public_core_pilot/diagnostics.json`;
- `artifacts/stage2_public_core_pilot/failures.json`; and
- `artifacts/stage2_public_core_pilot/run_receipt.json`.
