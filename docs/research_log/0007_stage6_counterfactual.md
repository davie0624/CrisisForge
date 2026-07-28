# Research Log 0007 — Stage 6 Structural Counterfactual Validation

**Experiment:** `stage6_semisynthetic_counterfactual_v2`  
**Status:** completed  
**Scope:** semi-synthetic environment with known ground truth only  
**Release provenance:** final clean-run timestamp and receipt pending

## Objective and structural design

Stage 6 validates the counterfactual machinery in a time-unrolled structural
causal model where the data-generating equations and paired exogenous innovations
are known. The within-time order is:

`policy -> yield -> liquidity -> credit -> equity -> volatility`

with the lagged feedback `equity[t-1] -> policy[t]`. Time unrolling retains this
feedback without creating a directed cycle within a time slice.

The intervention sets policy to 0.50 instead of 0.0 for the first 5 of 20 steps.
Five thousand paired paths share exogenous innovations between reference and
treated worlds. The 95% tail effect uses outcome-specific path-loss signs:
`-1` for equity, so lower cumulative equity outcomes are losses, and `+1` for
volatility, so higher cumulative volatility is a stress loss.
A controlled direct-effect experiment fixes yield, liquidity, and credit
mediators to identical schedules in both worlds.

## Registered configuration and provenance

- Evaluation config: `configs/stage6_counterfactual_evaluation.yaml`, SHA-256
  `28a85925a16d29daf23a59a2a6273a8a430353bdbde422821a0f6a691c382d18`;
- SCM config: `configs/counterfactual.yaml`, SHA-256
  `dd270e90abf6f884d241b675a4b82fb1bb2180a17a89c82c44bd41bc6dd57b2d`;
- SCM implementation SHA-256
  `e5804b40597dc153454483f0f52dbb83a68ffe6e7f72c77b03fff00a100e7ad1`.

The final release run will bind the clean source commit and completion timestamp.
This log does not promote an interim dirty-run receipt as release provenance.

No Phase 0 manifest is an input to this stage because the validation environment
is fully semi-synthetic. This separation is intentional and prevents known-ground-
truth counterfactual validation from being misrepresented as an empirical-market
result.

The misspecification controls halve policy-to-yield transmission and separately
remove lagged equity feedback. Full structural parameters are serialized in the
run receipt.

## Counterfactual recovery and sensitivity

For outcome \(j\), define path loss
\(L_j=s_j\sum_{h=1}^{20}X_{j,h}\). The tail effect is
\(\operatorname{ES}_{0.95}(L_j^{\mathrm{treated}})
-\operatorname{ES}_{0.95}(L_j^{\mathrm{reference}})\). Path RMSE compares the
estimated and known mean intervention-effect paths, while tail-effect error is
the absolute error in that 20-step path-loss ES effect.

| Model | Outcome | Loss sign | Terminal ATE | Cumulative ATE | Tail effect | Path RMSE | Tail-effect error |
|---|---|---:|---:|---:|---:|---:|---:|
| Known ground truth | Equity | -1 | 0.05953309941664323 | -4.12901597471562 | 1.758898358191768 | — | — |
| Oracle abduction–action–prediction | Equity | -1 | 0.05953309941664323 | -4.12901597471562 | 1.758898358191768 | 4.084527897230527e-17 | 0 |
| Reduced yield transmission | Equity | -1 | 0.0339459490623773 | -2.848332299644991 | 1.156993406694287 | 0.18845593535911762 | 0.601904951497481 |
| No lagged equity feedback | Equity | -1 | -0.09331905578839479 | -8.420777255550501 | 4.973135535260013 | 0.27818363577033606 | 3.214237177068245 |
| Known ground truth | Volatility | +1 | -0.017993557948262368 | 1.4909058769164785 | 1.3190540264213162 | — | — |
| Oracle abduction–action–prediction | Volatility | +1 | -0.017993557948262368 | 1.4909058769164785 | 1.3190540264213162 | 2.110382192020081e-17 | 0 |
| Reduced yield transmission | Volatility | +1 | -0.009869416320940223 | 0.8621819225753903 | 0.7724356083804835 | 0.07306495415146443 | 0.5466184180408327 |
| No lagged equity feedback | Volatility | +1 | 0.03026786729736134 | 2.9462866410356723 | 4.754501438728273 | 0.09312170331830869 | 3.4354474123069565 |

The maximum oracle error is `4.084527897230527e-17`, consistent with numerical
recovery under the known SCM. Misspecified-model path RMSE ranges from
`0.07306495415146443` to `0.27818363577033606`. The controlled direct effect is
exactly zero for both outcomes because the registered mediator schedule blocks all
paths from policy through yield, liquidity, and credit in this SCM.

## Negative results and claim boundary

- Removing lagged feedback reverses the sign of terminal equity and volatility
  effects, demonstrating severe structural sensitivity.
- Even halving one transmission coefficient yields material path and tail-effect
  errors. Exact oracle recovery therefore validates implementation only under the
  correct known equations.
- The exogenous Markov regime process is stipulated, not learned from observed
  markets.
- This experiment does **not** identify a Federal Reserve or real-market policy
  effect. Any observed-data extension must be called a
  **model-based structural intervention**, not definitive causal evidence.
- The exact zero controlled effect follows from the registered graph and
  mediator intervention; it is not a general empirical conclusion.

## Open issues

1. Expand sensitivity analysis over graph structure, latent confounding, regime
   endogeneity, and innovation misspecification.
2. Add semi-synthetic calibrations anchored to empirical marginal dynamics while
   retaining known counterfactual truth.
3. Specify identification assumptions and external instruments or natural
   experiments before any observed-market causal application.

## Durable artifacts

- `artifacts/stage6_counterfactual/effect_summary.csv`, SHA-256
  `03006374442fc215afad63ddfd6719220a30dd9f7eaba61b39e3e276b9bed2e9`;
- `artifacts/stage6_counterfactual/mean_effect_paths.csv`, SHA-256
  `d0d486f6e6a487e435fc2478c09a2c84fa60635c5c151954376dcaa7007686e0`;
- `artifacts/stage6_counterfactual/summary.json`, SHA-256
  `cbdc24b3333f6635c51b4a5d22ec9671baa5174d7f856c053ad0cf2e24b19f78`;
- `artifacts/stage6_counterfactual/run_receipt.json` will be rebound by the final
  clean release run.
