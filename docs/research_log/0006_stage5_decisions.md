# Research Log 0006 — Stage 5 CVaR and Wasserstein-DRO Decisions

**Completed at (UTC):** `2026-07-28T01:47:07.689457+00:00`  
**Experiment:** `stage5_validation_portfolio_decisions_v1`  
**Status:** completed; 518 of 518 decisions succeeded  
**Evaluation split:** validation only; post-2019 rows governed-excluded

## Objective and decision contract

Stage 5 tests whether Stage 1 scenarios support better realized portfolio
decisions, rather than assuming that scenario similarity automatically implies
economic value. Seven strategies are evaluated across the same 74 non-overlapping
20-observation validation blocks:

- equal weight;
- historical empirical CVaR;
- Stage 1 empirical CVaR; and
- four Stage 1 one-Wasserstein robust-CVaR sensitivities.

All optimized portfolios are long-only and fully invested with weights in
`[0, 0.20]`, maximum L1 turnover 0.40, and 95% CVaR. The proportional cost is
`0.001 * sum(abs(w_new - w_previous))`. This is an additive linear block-return
approximation, not multiplicative wealth accounting or a market-impact model.
The solver is SciPy HiGHS after strong-duality reduction to a single convex
program.

## Registered configuration and provenance

- Decision config: `configs/stage5_decision.yaml`, SHA-256
  `b7a908ee6e8271c9f330dbca9cf0a55a0aeb3cb3ccbb12f6e130b22cff3cbd5c`;
- portfolio config: `configs/portfolio.yaml`, SHA-256
  `a78f05bcd97d3b4cc177802c0de71eba1dafbf3c4486efbc4e01ecdc3d202d13`;
- Phase 0 manifest SHA-256
  `f051ec35236a481858b67c5b1e7136f1698036832f427efba521dbf3fcd36d70`;
- Stage 1 scenario archive SHA-256
  `295c3fd5ce6c825dd642f5de7080a3a9b91e5c4164c3be99218ec7edb70659af`;
- Stage 1 receipt SHA-256
  `c79e82d3b017ea4e048e8f637ce77b27855b972bc19ba1f9f0317f91d0efda14`;
- Git commit
  `b6891133bdb6b96e1e23c6bea3bd033ea9685c7c`, clean worktree.

The four positive radii—0.0001, 0.00025, 0.0005, and 0.001 in decimal-return
units—are separate exploratory validation sensitivities. No final radius is
selected.

## Validation results

| Strategy | Radius | Mean net return | Cumulative net return | Realized ES | Max drawdown | Total L1 turnover |
|---|---:|---:|---:|---:|---:|---:|
| Equal weight | 0 | 0.006265370841427425 | 0.5563549429142904 | 0.045470175224325594 | 0.08769364145806102 | 1.4475961689587804 |
| Historical empirical CVaR | 0 | 0.0050677501467866525 | 0.44724987796180926 | **0.017668815040963712** | **0.03411115422403499** | 3.4955198344470326 |
| Stage 1 empirical CVaR | 0 | 0.003697956691681514 | 0.3083145382678618 | 0.020738244914858133 | 0.04782388552956096 | 12.49453661334439 |
| Stage 1 WDRO | 0.0001 | 0.003697956691681514 | 0.3083145382678618 | 0.020738244914858133 | 0.04782388552956096 | 12.49453661334439 |
| Stage 1 WDRO | 0.00025 | 0.0036979566916815147 | 0.3083145382678618 | 0.020738244914858133 | 0.04782388552956096 | 12.49453661334439 |
| Stage 1 WDRO | 0.0005 | 0.0036889774366312997 | 0.30743619005848677 | 0.02073824491485813 | 0.04782388552956118 | 12.478738277901348 |
| Stage 1 WDRO | 0.001 | 0.0036889774366312997 | 0.30743619005848677 | 0.02073824491485813 | 0.04782388552956118 | 12.478738277901348 |

For reference, realized 95% VaR is `0.04143225975073266` for equal weight,
`0.01655889033474192` for historical CVaR, and `0.014307569418155601`
for Stage 1 empirical CVaR. Total estimated transaction costs are
`0.0014475961689587808`, `0.0034955198344470326`, and
`0.012494536613344391`, respectively.

## Negative results and claim boundary

- Historical empirical CVaR has lower observed realized expected shortfall and
  maximum drawdown than the Stage 1 scenario strategy in validation. This ranking
  is descriptive: it uses 74 non-overlapping block outcomes, and no paired
  sampling interval or multiplicity-adjusted test was computed.
- The Stage 1 decision has far higher turnover and cost than historical CVaR.
- Radii 0.0001 and 0.00025 produce exactly the Stage 1 empirical decision
  sequence. Radii 0.0005 and 0.001 produce nearly identical outcomes. The
  registered radius grid therefore adds virtually no useful decision separation
  under the current constraints.
- Equal weight has the highest cumulative return, but also the worst realized ES
  and drawdown. There is no universal winner across return and risk.
- No radius is selected, no test result exists, and no live-investment,
  investability, or cost-realism claim is permitted.

## Open issues

1. Diagnose why the robust penalty is largely inactive—position limits, turnover
   limits, scenario geometry, and radius scale may each contribute.
2. Add dependence-aware uncertainty around strategy differences.
3. Evaluate richer transaction-cost and market-impact models before applied
   portfolio claims.
4. Freeze a radius-selection rule before any post-2019 test evaluation.

## Durable artifacts

- `artifacts/stage5_decisions/decision_results.csv`;
- `artifacts/stage5_decisions/weights.csv`;
- `artifacts/stage5_decisions/summary.csv`, SHA-256
  `a0dfb0f3485947b95f39d46414527c1c490c767ff7e8f4b54ba093ff4aa43376`;
- `artifacts/stage5_decisions/failures.json`; and
- `artifacts/stage5_decisions/run_receipt.json`.
