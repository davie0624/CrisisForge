# CrisisForge Interview Pitches

## Two-minute version

CrisisForge asks whether a structured generative model can produce financial
scenarios that improve tail-risk decisions, rather than merely looking realistic.

I built the project as a staged Python research system. The public panel contains
12 industry research portfolios, three Treasury return proxies, and ten
availability-aware context variables. I froze a chronological train and validation
period; the data pipeline quality-checks the full panel, while all model stages
exclude post-2019 rows from estimation, tuning, scoring, and decisions.

The main model first compresses the 15 assets into four train-only PCA factors. A
sticky Gaussian HMM estimates soft market-state probabilities, state-weighted
VAR models generate factor paths, and a state-dependent stochastic observation
equation maps factors back to asset returns with correlated idiosyncratic risk. I
then added a one-shot temporal diffusion pilot, asset-level VaR and Expected
Shortfall, CVaR and Wasserstein-DRO decisions, and a separate counterfactual module
validated in a known semi-synthetic structural model.

The strongest result was a negative one. On 74 validation origins, conventional
filtered historical and moving-block methods beat the switching-factor model on
energy and variogram scores. Historical CVaR also had lower realized Expected
Shortfall, drawdown, and turnover than the scenario-based portfolios. My diffusion
pipeline runs correctly, but with only four reporting origins I do not claim that
it wins. The counterfactual code exactly recovered a known oracle, while deliberate
structural misspecification created large errors.

The research conclusion is that model sophistication, distributional realism, and
decision quality are different objectives. The project is valuable because it
tests that distinction honestly, keeps the test set sealed, and makes every result
reproducible. Here “sealed” means governed exclusion from model use, not that Phase
0 never constructed or quality-checked those rows.

## Thirty-second version

CrisisForge is a reproducible Python study of generative financial stress
scenarios. It combines regime-switching factors, a stochastic asset mapping,
one-shot diffusion, VaR/ES, robust CVaR decisions, and a known-SCM counterfactual
extension. The important finding is that the complex model did not broadly beat
strong historical baselines and did not improve validation portfolio tail risk.
That negative result showed that plausible generation and useful decisions are not
the same objective—and all model stages kept the post-2019 rows held out.
