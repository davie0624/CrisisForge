# Literature Review and Evidence Map

## CrisisForge: Decision-Focused Market Simulation under Regime Shifts

**Version:** 0.2  
**Evidence reviewed through:** 2026-07-28  
**Scope:** switching state-space models, dynamic factors, temporal diffusion,
factor-to-asset reconstruction, tail calibration, multivariate forecast evaluation,
CVaR/Wasserstein distributional robustness, and structural counterfactual simulation

## Executive conclusion

The components of CrisisForge each have credible precedents, but the complete system
does not. Peer-reviewed econometric work supports Markov-switching state-space and
regime-dependent factor models; peer-reviewed machine-learning work supports
non-autoregressive, full-horizon time-series diffusion; established financial-risk
research supports volatility-filtered extreme-value methods, joint VaR–ES
evaluation, and tractable Wasserstein distributionally robust optimization (DRO).
Recent work also supports interventional and counterfactual forecasting in general
time series.

The strongest direct precedents for **factor-structured financial diffusion** and
**causal diffusion market simulation**, however, remain preprints or workshop work.
They establish that the proposed questions are current, not that the answers are
known. No located peer-reviewed study jointly propagates regime uncertainty through
a factor generator and a stochastic asset observation equation, evaluates
asset-level tail risk and portfolio decisions, and then separately validates
counterfactuals against known causal ground truth.

CrisisForge therefore uses a deliberately modular design:

\[
\text{switching factor model}
\rightarrow
\text{soft posterior regime context}
\rightarrow
\text{one-shot factor-residual diffusion}
\rightarrow
\text{stochastic asset mapping}
\rightarrow
\text{tail risk}
\rightarrow
\text{portfolio decision}.
\]

The structural counterfactual module is a separate research track. Predictive
conditioning is never described as a causal intervention, and real-market
counterfactual outputs are described as **model-based structural interventions**
unless an external identification strategy is supplied.

## Evidence-status convention

- **Journal:** peer-reviewed journal article.
- **Conference:** peer-reviewed archival conference paper.
- **Workshop:** workshop publication; useful evidence of an emerging direction but
  weaker than an archival paper.
- **Preprint / working paper:** not treated as established evidence.
- **Accepted forthcoming:** accepted by the named venue but not necessarily assigned
  final volume/page metadata at the review date.

Dates below refer to the publication or stated venue year, not the date the project
accessed the source.

## Compact evidence matrix

| Architecture component | Most relevant source | Status | What the source supports | What it does **not** establish | CrisisForge implication |
|---|---|---|---|---|---|
| Regime switching | [Hamilton (1989), *Econometrica*](https://www.jstor.org/stable/1912559) | Journal | Latent Markov regimes can represent discrete changes in time-series dynamics | That inferred regimes are literal economic ground truth | Treat regimes as latent statistical states and assign semantic names only after diagnostics |
| Switching state-space filtering | [Kim (1994), *Journal of Econometrics*](https://www.sciencedirect.com/science/article/pii/0304407694900361) | Journal | Filtering/smoothing is feasible in dynamic linear models with Markov switching | That smoothed states are valid forecast-time inputs | Use filtered probabilities in forecasts; reserve smoothed probabilities for retrospective analysis |
| High-dimensional switching factors | [Urga and Wang (2024), *Journal of Econometrics*](https://openaccess.city.ac.uk/id/eprint/32040/) | Journal | Estimation and inference for high-dimensional factor models with regime switching | That factor dimension or state count can be selected without uncertainty | Search \(K,q\), check state occupancy, stabilize rotations and labels |
| Large switching factor systems | [Barigozzi and Massacci (2025), *Journal of Econometrics*](https://cris.unibo.it/handle/11585/1000607) | Journal | Large panels can be modeled with Markov-switching factor structure | That state-dependent loadings eliminate idiosyncratic tail risk | Retain \(D_z\epsilon_t\) and residual dependence after factor reconstruction |
| Multi-regime state uncertainty | [Hashimzade et al. (2026), *Journal of Business & Economic Statistics*](https://eprints.gla.ac.uk/381752/) | Journal, early online | Modern filtering and smoothing for state-space models with multiple regimes | That a hard decoded state is preferable to the posterior distribution | Pass soft filtered/predictive regime probabilities to the generator |
| Non-autoregressive diffusion | [Shen and Kwok (2023), *ICML*](https://proceedings.mlr.press/v202/shen23d.html) | Conference | Conditional diffusion can generate multi-step forecasts non-autoregressively | That full-horizon generation cannot drift or miscalibrate tails | Generate the full \(H\times q\) path in one shot and still test horizon-wise stability |
| Financial time-series diffusion | [Huang, Chen, and Qiao (2024), *ICLR*](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f90fc76b199fe6b0ec2a51aaf72c3277-Abstract-Conference.html) | Conference | Diffusion can be adapted to irregular, scale-sensitive financial patterns | That it dominates econometric baselines on tail-risk decisions | Compare against bootstrap, Student-\(t\), copula, VAR, and DCC-GARCH |
| Controllable financial generation | [Tanaka et al. (2025), *IJCAI*](https://www.ijcai.org/proceedings/2025/1040) | Conference | Financial diffusion can condition generation on user-specified controls | That a conditioned scenario has a calibrated probability or causal meaning | Separate predictive mode from stress-control mode |
| Factor diffusion | [Chen et al. (2025), arXiv:2504.06566](https://arxiv.org/abs/2504.06566) | Preprint | A low-dimensional factor structure may reduce the statistical burden of generating high-dimensional returns | Peer-reviewed empirical superiority or validity of the complete CrisisForge architecture | Treat factor generation as a testable hypothesis; include direct asset-diffusion ablation |
| Factor-dimension trade-off | [Bagchi et al. (2026), arXiv:2603.10385](https://arxiv.org/abs/2603.10385) | Preprint | Factor dimension may induce a bias–variance trade-off in diffusion portfolio models | A universally optimal \(q\) | Select \(q\) by nested validation and report sensitivity |
| EVT for conditional financial tails | [McNeil and Frey (2000), *Journal of Empirical Finance*](https://www.sciencedirect.com/science/article/abs/pii/S0927539800000128) | Journal | Peaks-over-threshold EVT is most defensible after filtering conditional heteroskedasticity | That raw-return exceedances are iid or that univariate EVT captures co-crashes | Fit POT-GPD to standardized portfolio losses/residual magnitudes and diagnose dependence |
| Adaptive online calibration | [Gibbs and Candès (2024), *JMLR*](https://www.jmlr.org/papers/v25/22-1218.html) | Journal | Online conformal methods can adapt coverage under distribution shift | An automatic Expected Shortfall guarantee for dependent financial returns | Calibrate VaR horizon by horizon; evaluate ES separately |
| Joint VaR–ES learning | [Barrera et al. (2026), *Mathematical Finance*](https://onlinelibrary.wiley.com/doi/full/10.1111/mafi.70000) | Journal | VaR and ES can be learned and evaluated jointly | That correct VaR coverage alone validates ES | Use joint VaR–ES scores and ES regression diagnostics |
| Wasserstein DRO tractability | [Mohajerin Esfahani and Kuhn (2018), *Mathematical Programming*](https://link.springer.com/article/10.1007/s10107-017-1172-1) | Journal | Wasserstein ambiguity sets can yield finite, tractable data-driven robust programs | That robustness around a misspecified generator implies robustness to reality | Tune radius out of sample and report generator/model-risk caveats |
| Regime-aware robust CVaR | [Pun, Wang, and Yan (2023), *Manufacturing & Service Operations Management*](https://pubsonline.informs.org/doi/10.1287/msom.2023.1229) | Journal | Regime-switching ambiguity sets can improve CVaR portfolio formulation | That the proposed diffusion scenarios are the correct center distribution | Compare historical and generated scenario centers and propagate state uncertainty |
| Counterfactual forecasting | [Wu, Qiu, and Xie (2026), *ICLR*](https://iclr.cc/virtual/2026/poster/10011565) | Accepted conference | Abduction–action–prediction can be implemented in generative time-series forecasting | Identification of monetary-policy effects in observational financial data | Validate first in a known semi-synthetic SCM; use DoFlow as a comparator |
| Causal market simulation | [Thumm and Ontaneda (2025), arXiv:2511.04469](https://arxiv.org/abs/2511.04469) | Preprint; ICAIF workshop | Causal market simulators are an active emerging direction | General validity in nonstationary, regime-switching real markets | Treat regime-switching causal simulation as an open extension, not an established capability |
| Causal Wasserstein geometry | [Cheridito and Eckstein (2025), *Bernoulli*](https://www.research-collection.ethz.ch/items/aead0077-2b62-4a1c-8080-9ff6c6e8efc1) | Journal | Causal models admit transport notions that preserve temporal/causal structure | Equivalence to ordinary Wasserstein fit or DRO ambiguity sets | Keep causal transport, distribution-fit distance, and DRO radius mathematically separate |

## 1. Regime switching and dynamic factor representation

### 1.1 What is established

[Hamilton (1989)](https://www.jstor.org/stable/1912559)
established the modern empirical use of latent Markov regimes in economic time
series. [Kim (1994)](https://www.sciencedirect.com/science/article/pii/0304407694900361)
extended the framework to dynamic linear state-space systems and made explicit the
distinction between filtering and smoothing. This distinction is operationally
critical: a filtered probability \(P(z_t\mid\mathcal F_t)\) is available at forecast
origin, whereas a smoothed probability \(P(z_t\mid\mathcal F_T)\), \(T>t\), uses
future information.

Recent peer-reviewed econometric research addresses the large-panel version of the
problem. [Urga and Wang (2024)](https://openaccess.city.ac.uk/id/eprint/32040/)
study estimation and inference for high-dimensional factor models with regime
switching. [Barigozzi and Massacci
(2025)](https://cris.unibo.it/handle/11585/1000607) develop Markov-switching
factor modeling for large-dimensional datasets.
[Hashimzade et al. (2026)](https://eprints.gla.ac.uk/381752/) provide contemporary
filtering and smoothing methods for state-space models with multiple regimes.
Together, these sources support a coherent switching factor model rather than an
unrelated HMM followed by a separately estimated PCA.

A recent financial application by [Istiaque, Pun, and Yong
(2025)](https://www.tandfonline.com/doi/abs/10.1080/14697688.2025.2511115) develops a
hidden-Markov generative stock-market simulator and applies it to risk measurement.
It is an important comparator because it connects regime generation directly to
financial-risk outputs without diffusion.

### 1.2 Design implications

CrisisForge implements the regime and factor layers as a jointly interpretable
switching state-space core:

\[
z_t\sim P(z_t\mid z_{t-1}),\qquad
f_t=c_{z_t}+\Phi_{z_t}f_{t-1}+G_{z_t}x_{t-1}+Q_{z_t}^{1/2}\eta_t.
\]

The evidence implies six safeguards:

1. **Forecast with filtered probabilities.** Smoothed state probabilities are
   restricted to retrospective regime plots and post-hoc diagnostics.
2. **Preserve uncertainty.** The diffusion context receives posterior regime
   probabilities and predictive state probabilities, not a single decoded label.
3. **Do not name states ex ante.** “Crisis,” “inflation,” or “normal” labels are
   assigned only after examining training-fold volatility, rates, spreads, duration,
   and occupancy.
4. **Control state proliferation.** Models with unstable states, extremely short
   durations, or less than the pre-specified occupancy threshold are rejected.
5. **Resolve non-identifiability.** Regime label switching and factor rotation are
   handled explicitly across posterior draws, folds, and random seeds.
6. **Validate uncertainty propagation.** Soft conditioning, hard-state conditioning,
   and posterior-mean plug-in variants are separate ablations.

### 1.3 Evidence gap

No latent regime is directly observed, so classification “accuracy” is generally
not a valid primary metric. Economic narratives attached to a state are
interpretations, not labels. The literature also does not guarantee that adding
more regimes or factors improves forecasting. CrisisForge therefore treats
\(K\) and \(q\) as nested-validation choices and reports state occupancy,
transition persistence, factor reconstruction, and downstream predictive scores.

## 2. One-shot temporal diffusion in factor space

### 2.1 Accepted time-series evidence

[TimeDiff (Shen and Kwok,
2023)](https://proceedings.mlr.press/v202/shen23d.html) provides direct
peer-reviewed support for non-autoregressive conditional diffusion in multi-step
time-series prediction. It motivates generating a future block jointly instead of
recursively feeding one simulated day into the next. This removes one source of
autoregressive error accumulation but does not prove that long generated paths
will have correct variance growth, autocorrelation, or tail dependence.

Several accepted papers broaden the design space:

- [FTS-Diffusion (Huang, Chen, and Qiao,
  2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f90fc76b199fe6b0ec2a51aaf72c3277-Abstract-Conference.html)
  addresses irregular and scale-invariant patterns in financial time series.
- [Time Weaver (Narasimhan et al.,
  2024)](https://proceedings.mlr.press/v235/narasimhan24a.html) studies
  controllable generation of synthetic time series.
- [Diffusion-TS (Yuan and Qiao,
  2024)](https://openreview.net/forum?id=4h1apFjO99) develops an interpretable
  diffusion framework for general time-series generation.
- [Crabbé et al.
  (2024)](https://proceedings.mlr.press/v235/crabbe24a.html) formulate diffusion
  in the frequency domain, relevant when temporal-frequency structure is missed by
  a purely local architecture.
- [CoFinDiff (Tanaka et al.,
  2025)](https://www.ijcai.org/proceedings/2025/1040) supplies a direct
  peer-reviewed precedent for controllable financial time-series diffusion.

These papers support the feasibility of conditional, controllable, full-path
generation. They do not establish that diffusion beats strong financial
econometric baselines on held-out tail risk or realized portfolio decisions.

### 2.2 Factor diffusion is promising but provisional

[Chen et al. (2025)](https://arxiv.org/abs/2504.06566) explicitly connect
diffusion to factor-structured high-dimensional return generation. Its core
motivation—effective complexity governed by factor dimension rather than raw asset
count—closely matches CrisisForge. At the review date it remains an arXiv preprint,
so it cannot be the sole basis for a claim that factor diffusion is statistically
superior.

[Bagchi et al. (2026)](https://arxiv.org/abs/2603.10385), also a preprint, focus
on factor dimensionality and the bias–variance trade-off in diffusion portfolio
models. This reinforces the need to vary \(q\), but it does not supply a universal
factor count.

### 2.3 CrisisForge specification

The switching state-space model provides a conditional factor-path mean and scale.
Diffusion models the standardized nonlinear residual path:

\[
Y_t^0=S_t^{-1}
\left(f_{t+1:t+H}-\mu^{\mathrm{SSM}}_{t,1:H}\right)
\in\mathbb R^{H\times q}.
\]

A temporal U-Net or temporal transformer denoises the entire \(H\times q\) tensor
in one shot. Conditioning includes only information available at origin \(t\):
filtered factor history, filtered regime probabilities, vintage-safe market
features, and predictive regime probabilities. Realized future states, future
covariates, or future-tail labels are forbidden.

This residual interface prevents the diffusion network from redundantly relearning
the same linear transition already represented in the state-space model. The core
ablations are:

1. direct asset-level diffusion;
2. unconditional factor diffusion;
3. hard-regime factor diffusion;
4. soft-regime factor diffusion; and
5. state-space-only factor simulation.

The project will not import training shortcuts that use actual future observations
to create inference-time context. In particular, any augmentation or “future
mixup” method is admissible only if its inference protocol can be reproduced using
\(\mathcal F_t\) alone.

### 2.4 Required failure tests

One-shot generation still may fail. CrisisForge records:

- variance and tail growth by horizon;
- return and squared-return autocorrelation;
- spectral/frequency mismatch;
- explosive-path and implausible-level rates;
- sensitivity to \(q\), \(H\), noise schedule, and architecture;
- condition adherence under stress controls; and
- whether improved sample realism survives asset reconstruction.

If direct asset generation outperforms factor diffusion on nested held-out proper
scores and tail metrics, the factor advantage hypothesis is rejected.

## 3. Factor-to-asset reconstruction

### 3.1 Why an observation equation is indispensable

The diffusion model generates latent factor paths, but VaR, ES, co-crash
probability, and portfolio loss are asset-level objects. The bridge is:

\[
\boxed{
r_t=\alpha_{z_t}+B_{z_t}f_t+D_{z_t}\epsilon_t
}
\]

with, in the initial model,

\[
\epsilon_t\sim t_{\nu_{z_t}}(0,R_{z_t}),\qquad
\Sigma_{\epsilon,z}=D_zR_zD_z.
\]

The regime-switching factor literature supports state-dependent factor structure.
It does **not** support dropping idiosyncratic risk. Omitting
\(D_z\epsilon_t\) makes assets nearly deterministic functions of a few common
factors, understates name-specific dispersion, and distorts diversification.
Assuming independent residuals is also too restrictive for co-crash measurement;
\(R_z\) represents residual dependence not absorbed by the common factors.

### 3.2 Mixture simulation, not parameter averaging

If the current posterior over regimes is uncertain, the predictive distribution is
a mixture:

\[
p(r_{t+h}\mid\mathcal F_t)
=\sum_k P(z_{t+h}=k\mid\mathcal F_t)\,
p(r_{t+h}\mid z_{t+h}=k,\mathcal F_t).
\]

It is generally incorrect to replace this mixture by averaged loadings, scales, or
Cholesky factors. CrisisForge instead draws a full future regime path and, at each
horizon, applies the corresponding
\(\alpha_{z_h},B_{z_h},D_{z_h},R_{z_h}\). Posterior draws of these parameters are
propagated where computationally feasible.

### 3.3 Identification and validation

Factor rotations and scale must be normalized before interpreting \(B_z\).
Regime-specific loadings are heavily regularized because crisis-state samples are
small. The required comparisons are:

- fixed versus regime-dependent \(B\);
- fixed versus regime-dependent \(D\);
- diagonal versus shrinkage \(R_z\);
- Gaussian versus Student-\(t\) residuals; and
- plug-in versus posterior-predictive mapping.

Success is judged at asset level through covariance breaks, marginal tail
calibration, co-crash scoring, and portfolio loss—not by factor reconstruction
alone.

## 4. Tail modeling and sequential calibration

### 4.1 Extreme values after volatility filtering

[McNeil and Frey
(2000)](https://www.sciencedirect.com/science/article/abs/pii/S0927539800000128) combine
conditional volatility modeling with extreme-value theory for tail-related
financial risk measures. The durable design lesson is that peaks-over-threshold
(POT) estimation is more defensible on approximately standardized innovations than
on raw heteroskedastic returns.

[James et al.
(2023)](https://www.sciencedirect.com/science/article/pii/S0927539823000026)
extend financial tail-risk forecasting with covariates. This supports allowing tail
parameters to vary with a small, pre-specified set of state variables, subject to
strong shrinkage.

CrisisForge therefore fits generalized Pareto tails to standardized portfolio loss
or a multivariate residual magnitude, not blindly to raw daily asset returns:

\[
L-u_z\mid L>u_z,z\sim\operatorname{GPD}(\xi_z,\beta_z).
\]

Thresholds are selected on training data and stress-tested over a pre-specified
range. Because state-specific extremes are sparse, the model uses hierarchical
pooling rather than fitting a separate 99% tail from a handful of crisis
observations.

### 4.2 Tail-aware training is a sampling problem as well as a loss problem

Naively adding a tail loss to score matching can produce gradient competition:
normal-market observations dominate numerically, while rare tail windows may
create unstable updates. CrisisForge first trains the base score objective, then
evaluates tail-conditioned fine-tuning and controlled tail-window oversampling.
Any unequal sampling is paired with inverse-probability correction when estimating
the unconditional predictive distribution. Otherwise, the model would report the
oversampled crisis frequency as if it were a calibrated forecast probability.

Stress mode may deliberately force a tail condition. Predictive mode may not use a
future realized tail indicator.

### 4.3 What conformal calibration can and cannot claim

[Zaffran et al.
(2022)](https://proceedings.mlr.press/v162/zaffran22a.html) study adaptive
conformal prediction for time series.
[Gibbs and Candès
(2024)](https://www.jmlr.org/papers/v25/22-1218.html) develop online conformal
inference under arbitrary distribution shifts.
[Hallberg
(2024)](https://proceedings.mlr.press/v230/hallberg-szabadvary24a.html) addresses
adaptive multi-step time-series forecasting online, and
[Angelopoulos et al.
(2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)
provide a general conformal risk-control framework.

These papers motivate adaptive, horizon-specific quantile correction, but the exact
coverage guarantee depends on the method's assumptions and update protocol.
Dependent, nonstationary financial returns are not treated as exchangeable by
default. CrisisForge reports empirical sequential coverage and the assumptions
under which a theorem applies.

Most importantly, ordinary conformal VaR calibration is **not** presented as an ES
guarantee. ES is evaluated separately using joint VaR–ES methods.

### 4.4 VaR and ES validation

VaR coverage and independence are diagnosed with:

- [Kupiec (1995)](https://www.fedinprint.org/item/fedgfe/34596/original) for
  unconditional coverage; and
- [Christoffersen
  (1998)](https://ideas.repec.org/a/ier/iecrev/v39y1998i4p841-62.html) for interval
  forecast evaluation and violation dependence.

These are diagnostics, not a complete model ranking. A model can attain the right
number of violations while misplacing them in time or misestimating loss severity.

Joint VaR–ES evaluation uses the elicitability literature, including [Fissler,
Ziegel, and Gneiting
(2015)](https://arxiv.org/abs/1507.00244), together with contemporary statistical
learning of the pair by [Barrera et al.
(2026)](https://onlinelibrary.wiley.com/doi/full/10.1111/mafi.70000). ES regression
backtesting follows [Bayer and Dimitriadis
(2022)](https://academic.oup.com/jfec/article/20/3/437/5912157).

The resulting tail scorecard includes:

- VaR pinball loss;
- unconditional coverage and violation independence;
- a strictly consistent joint VaR–ES score;
- ESR diagnostics;
- co-crash Brier/log score and reliability curves; and
- Monte Carlo uncertainty for simulated risk measures.

## 5. Distributional and path evaluation

### 5.1 Proper scores before visual similarity

A synthetic path that “looks realistic” is not sufficient evidence. CrisisForge
uses proper multivariate scoring rules as primary distributional outcomes.
[Gneiting and Raftery
(2007)](https://doi.org/10.1198/016214506000001437) provide the general foundation
for strictly proper scoring rules. The energy score assesses joint distributional
accuracy, while the variogram score from [Scheuerer and Hamill
(2015)](https://repository.library.noaa.gov/view/noaa/22327/) is more sensitive to
dependence relationships that an energy score may underemphasize.

Diagnostic—not sole ranking—measures include:

- sliced Wasserstein distance;
- maximum mean discrepancy;
- marginal moments and quantiles;
- return and squared-return ACF;
- dynamic correlation and lower-tail dependence;
- drawdown depth and duration;
- co-crash frequency; and
- condition-adherence and explosive-path rates.

### 5.2 Why the metric set must remain plural

Central fit, tail calibration, and decision quality can conflict. A model may
improve the 99% tail by sacrificing central-density accuracy, or match the
distribution closely while creating unstable portfolio weights. CrisisForge does
not predeclare a single universally best model. It estimates a Pareto frontier
across:

1. proper distributional scores;
2. tail-risk calibration;
3. realized portfolio decision loss; and
4. computational cost and stability.

This directly operationalizes the hypothesis that statistical realism and decision
value need not induce the same model ranking.

## 6. CVaR and Wasserstein-DRO portfolio decisions

### 6.1 Tractable robust optimization

[Mohajerin Esfahani and Kuhn
(2018)](https://link.springer.com/article/10.1007/s10107-017-1172-1) establish
tractable reformulations for broad classes of data-driven DRO problems under
Wasserstein ambiguity. [Zhang, Yang, and Gao
(2024)](https://pubsonline.informs.org/doi/full/10.1287/opre.2023.0135) provide a
short general duality proof that clarifies why many inner worst-case distribution
problems can be converted to finite programs.

This evidence supports solving a dualized single-level program rather than a naive
nested adversarial optimization over thousands of scenarios at every rebalance.
The initial CrisisForge decision problem is deliberately single-period: weights are
chosen at origin \(t\) for an \(H\)-day buy-and-hold horizon. A dynamic strategy
would require a scenario tree or another explicit non-anticipative policy class and
is outside the first release.

### 6.2 Regimes and downside risk

[Pun, Wang, and Yan
(2023)](https://pubsonline.informs.org/doi/10.1287/msom.2023.1229) directly connect
regime-switching ambiguity sets with CVaR portfolio optimization.
[Blanchet, Chen, and Zhou
(2022)](https://pubsonline.informs.org/doi/10.1287/mnsc.2021.4155) study
Wasserstein-robust mean–variance portfolio selection. A newer downside-risk
formulation by [Liu et al.
(2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6446118) is a working
paper and is treated as an extension rather than settled evidence.

### 6.3 The center-distribution caveat

Let \(\widehat P_\theta\) be the empirical distribution of generated scenarios and
\(\mathcal U_\rho(\widehat P_\theta)\) a Wasserstein ball. A robust allocation:

\[
\min_w\sup_{Q\in\mathcal U_\rho(\widehat P_\theta)}
\operatorname{CVaR}_{\alpha,Q}(-w^\top R)
\]

is robust around \(\widehat P_\theta\), not automatically around the unknown true
market distribution. If the generator omits a crisis mechanism, a small ball
around it preserves the omission.

CrisisForge therefore:

- compares historical, block-bootstrap, Student-\(t\), copula, and generated
  scenario centers;
- tunes \(\rho\) by nested out-of-sample decision loss, not in-sample fit;
- reports sensitivity over a radius grid;
- separates generator misspecification from finite Monte Carlo error;
- constrains positions, turnover, and concentration; and
- evaluates realized ES, drawdown, transaction cost, and weight stability.

The DRO layer remains outside the initial diffusion training loop. This preserves
error attribution and makes the robustness claim auditable.

## 7. Structural counterfactual extension

### 7.1 Prediction, stress, intervention, and counterfactual are different tasks

The following objects are not interchangeable:

\[
P(Y\mid X=x),\qquad
P(Y\mid do(X=x)),\qquad
P(Y_x\mid X=x',Y=y').
\]

The first is observational conditioning. The second is an intervention. The third
is an individual/unit-level counterfactual requiring abduction of latent
disturbances, an action on the structural model, and prediction under the modified
system.

A user-controlled “policy shock” supplied to a conditional diffusion model is
therefore a stress condition unless the data-generating equations and identifying
assumptions justify a \(do(\cdot)\) interpretation.

### 7.2 Relevant causal evidence

[DoFlow (Wu, Qiu, and Xie,
2026)](https://iclr.cc/virtual/2026/poster/10011565), accepted at ICLR 2026,
provides the strongest located archival precedent for interventional and
counterfactual generative forecasting in time series. It supports an
abduction–action–prediction implementation but is not, by itself, an identification
strategy for financial policy effects.

[Lorch, Krause, and Schölkopf
(2024)](https://proceedings.mlr.press/v238/lorch24a.html) study causal modeling with
stationary diffusions. [Cheridito and Eckstein
(2025)](https://www.research-collection.ethz.ch/items/aead0077-2b62-4a1c-8080-9ff6c6e8efc1)
develop optimal transport and Wasserstein distances for causal models.
[Backhoff et al.
(2017)](https://doi.org/10.1137/16M1080197) provide earlier foundations for causal
transport in discrete time.

In applied settings, [Gao, Mishra, and Ramazzotti
(2018)](https://doi.org/10.1016/j.jocs.2018.04.003) connect causal data science to
financial stress testing, while [Brodersen et al.
(2015)](https://research.google/pubs/inferring-causal-impact-using-bayesian-structural-time-series-models/)
show how Bayesian structural time-series models can support causal-impact analysis
under an explicit counterfactual design.

The most direct “causal market simulator” source, [Thumm and Ontaneda
(2025)](https://arxiv.org/abs/2511.04469), is an arXiv preprint associated with an
ICAIF workshop. Its controlled experiments help define a research direction, but
do not validate real-market, regime-switching counterfactuals.

The paper [*Causal Time Series Generation via Diffusion Models*
(2025)](https://arxiv.org/abs/2509.20846) is an unreviewed preprint and was reported
as desk-rejected at ICLR. It is catalogued for completeness but is **not** used as
evidence that causal diffusion claims are valid.

### 7.3 Time-unrolled structural model

Financial feedback loops make a static contemporaneous DAG implausible: policy
affects markets, while subsequent market stress affects later policy. CrisisForge
uses a time-unrolled graph:

\[
X_t\rightarrow X_{t+1}\rightarrow X_{t+2},
\]

so feedback is represented across time without creating a directed cycle within a
time slice. Contemporaneous edges, if used, require a defensible causal ordering or
simultaneous-equation identification.

The counterfactual workflow is:

1. **Abduction:** infer exogenous innovations from a factual path.
2. **Action:** replace one structural equation with the intervention.
3. **Prediction:** propagate the same exogenous innovations through the modified
   time-unrolled system.

Primary validation occurs in a semi-synthetic regime-switching market where the
structural equations, innovations, factual paths, and counterfactual ground truth
are known. Outcomes include average treatment-effect error, response-curve error,
counterfactual distribution distance, individual-path error, tail-effect error,
and sensitivity to DAG misspecification.

### 7.4 Real-market claim boundary

Without a defensible natural experiment, instrumental variable, high-frequency
event-study design, or other identification argument, the real-data module cannot
claim that a policy-rate intervention “caused” the generated equity or credit path.
It can show the implications of an explicit structural model under stated
assumptions. The correct phrase is:

> model-based structural intervention under the specified time-unrolled SCM.

This boundary is a strength of the research design, not a cosmetic disclaimer.

## 8. Three Wasserstein objects that must not be conflated

CrisisForge uses “Wasserstein” in three mathematically different roles:

| Role | Object | Purpose | Interpretation |
|---|---|---|---|
| Distribution diagnostic | \(W(P_{\mathrm{generated}},P_{\mathrm{heldout}})\) or a sliced approximation | Compare generated and observed distributions | Statistical fit; lower is better under the chosen ground metric |
| DRO ambiguity set | \(\{Q:W(Q,\widehat P_\theta)\le\rho\}\) | Protect a decision against nearby distributions | Decision robustness around a chosen center |
| Causal/temporal transport | A causality-constrained transport distance | Compare stochastic processes while preserving information flow | Structural/temporal discrepancy |

A low distribution-fit distance does not choose the correct DRO radius. A
Wasserstein ambiguity ball does not make a model causal. A causal Wasserstein
distance is not a substitute for an identification design. Results and notation
will keep these three objects separate.

## 9. Resulting benchmark hierarchy

The literature implies that diffusion must earn its complexity. CrisisForge uses
the following minimum benchmark ladder:

1. unconditional historical simulation;
2. filtered historical simulation and block bootstrap;
3. multivariate Gaussian and Student-\(t\);
4. Student-\(t\) copula;
5. VAR and DCC-GARCH;
6. switching state-space/factor simulation without diffusion;
7. the HMM generative risk comparator of Istiaque, Pun, and Yong;
8. direct asset-level diffusion;
9. unconditional factor diffusion;
10. hard-regime factor diffusion; and
11. soft posterior regime-conditioned factor diffusion.

The tail and decision layers are evaluated on every feasible scenario generator,
not only on the preferred diffusion model. A complex model advances only if it
improves held-out distributional, tail-risk, or decision outcomes in a way that
survives uncertainty and computational-cost checks.

## 10. What the literature does not establish

The review identifies the following unresolved questions:

1. **No validated end-to-end finance system.** No located peer-reviewed study
   combines switching factors, one-shot diffusion, regime-dependent stochastic
   asset mapping, tail calibration, asset-level risk, DRO decisions, and identified
   counterfactuals.
2. **Factor diffusion is not yet settled.** The most direct theoretical and
   portfolio-specific evidence remains preprint work.
3. **Regime uncertainty is often collapsed.** The value of passing posterior state
   uncertainty into a generator and decision layer remains an empirical question.
4. **Observation-layer uncertainty is under-evaluated.** Good latent-factor samples
   do not guarantee realistic asset-level tails after reconstruction.
5. **Central and tail fit conflict.** No method is expected to dominate every
   distributional and risk metric.
6. **Generator quality and decision quality can rank models differently.**
   Decision-focused evaluation is necessary.
7. **Robustness is local to the ambiguity center.** Wasserstein DRO cannot repair a
   structurally incomplete generator unless the ambiguity set covers the omitted
   mechanism.
8. **Conditional scenarios are not counterfactuals.** Real financial
   counterfactuals remain identification-limited.
9. **Crisis samples are intrinsically scarce.** State-specific EVT and high-capacity
   diffusion require pooling, shrinkage, uncertainty intervals, and honest failure
   reporting.
10. **Long-horizon one-shot diffusion can still fail.** Removing autoregression
    does not guarantee stable path dynamics.

These gaps define CrisisForge's research agenda. They do not pre-announce positive
results.

## 11. Evidence-to-design decisions

The literature review fixes the following project decisions:

| Decision | Adopted specification | Evidence-based reason |
|---|---|---|
| Training strategy | Multi-stage estimation with selective diffusion/tail fine-tuning | Preserves statistical diagnostics and error attribution while allowing focused nonlinear refinement |
| Regime input | Soft filtered and predictive state probabilities | Avoids hard-label error amplification and future leakage |
| Forecast representation | Joint \(H\times q\) factor-residual tensor | Supported by non-autoregressive diffusion; avoids recursive day-by-day simulation |
| Asset reconstruction | State-specific \(\alpha_z,B_z,D_z,R_z,\nu_z\) | Retains state-dependent exposures, idiosyncratic scale, residual dependence, and heavy tails |
| State uncertainty | Draw regime paths and parameters; do not average matrices | Preserves the predictive mixture distribution |
| Tail estimation | Volatility-standardized POT-GPD with hierarchical pooling | Aligns with conditional EVT and acknowledges scarce regime-tail observations |
| Tail sampling | Controlled oversampling with inverse-probability correction | Improves learning signal without misreporting crisis frequency |
| Calibration | Sequential, horizon-specific VaR correction | More appropriate than exchangeable conformal assumptions for drifting time series |
| ES evaluation | Joint VaR–ES scores and ESR tests | VaR coverage alone cannot validate ES |
| Generator evaluation | Energy and variogram scores plus stylized-fact diagnostics | Proper scores evaluate distributions; diagnostics reveal specific failure modes |
| Decision layer | Empirical CVaR first, dualized Wasserstein DRO second | Establishes a transparent baseline before adding ambiguity robustness |
| Initial portfolio | Single-period \(H\)-day buy-and-hold | Avoids implicit violation of non-anticipativity |
| Counterfactual validation | Time-unrolled SCM and semi-synthetic ground truth | Separates causal validity from observational fit |
| Real-data causal wording | “Model-based structural intervention” | Avoids unsupported causal claims |

## 12. Reference register by component

### Regime and factor models

- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle*. **Econometrica, journal.**
  [Source](https://www.jstor.org/stable/1912559)
- Kim, C.-J. (1994). *Dynamic Linear Models with Markov-Switching*. **Journal of
  Econometrics, journal.**
  [Source](https://www.sciencedirect.com/science/article/pii/0304407694900361)
- Urga, G., and Wang, F. (2024). *Estimation and Inference for High Dimensional
  Factor Model with Regime Switching*. **Journal of Econometrics, journal.**
  [Source](https://openaccess.city.ac.uk/id/eprint/32040/)
- Barigozzi, M., and Massacci, D. (2025). *Modelling Large Dimensional Datasets
  with Markov Switching Factor Models*. **Journal of Econometrics, journal.**
  [Source](https://cris.unibo.it/handle/11585/1000607)
- Hashimzade, N., et al. (2026). *Filtering and Smoothing in State-Space Models
  with Multiple Regimes*. **Journal of Business & Economic Statistics, journal,
  early online.** [Source](https://eprints.gla.ac.uk/381752/)
- Istiaque, M., Pun, C. S., and Yong, H. (2025). *Stock Market Simulator Using
  Hidden Markov Generative Model and Its Application in Risk Measurement*.
  **Quantitative Finance, journal.**
  [Source](https://www.tandfonline.com/doi/abs/10.1080/14697688.2025.2511115)

### Diffusion and time-series generation

- Shen, L., and Kwok, J. (2023). *Non-autoregressive Conditional Diffusion Models
  for Time Series Prediction*. **ICML, conference.**
  [Source](https://proceedings.mlr.press/v202/shen23d.html)
- Huang, Q., Chen, L., and Qiao, Z. (2024). *Generative Learning for Financial
  Time Series with Irregular and Scale-Invariant Patterns*. **ICLR, conference.**
  [Source](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f90fc76b199fe6b0ec2a51aaf72c3277-Abstract-Conference.html)
- Narasimhan, S., et al. (2024). *Time Weaver: A Conditional Time Series
  Generation Model*. **ICML, conference.**
  [Source](https://proceedings.mlr.press/v235/narasimhan24a.html)
- Yuan, X., and Qiao, Y. (2024). *Diffusion-TS: Interpretable Diffusion for
  General Time Series Generation*. **ICLR, conference.**
  [Source](https://openreview.net/forum?id=4h1apFjO99)
- Crabbé, J., et al. (2024). *Time Series Diffusion in the Frequency Domain*.
  **ICML, conference.**
  [Source](https://proceedings.mlr.press/v235/crabbe24a.html)
- Tanaka, R., et al. (2025). *CoFinDiff: Controllable Financial Diffusion Model
  for Time Series Generation*. **IJCAI, conference.**
  [Source](https://www.ijcai.org/proceedings/2025/1040)
- Chen, et al. (2025). *Diffusion Factor Models: Generating High-Dimensional
  Returns with Factor Structure*. **arXiv:2504.06566, preprint.**
  [Source](https://arxiv.org/abs/2504.06566)
- Bagchi, et al. (2026). *Factor Dimensionality and the Bias-Variance Tradeoff in
  Diffusion Portfolio Models*. **arXiv:2603.10385, preprint.**
  [Source](https://arxiv.org/abs/2603.10385)

### Tail risk, calibration, and evaluation

- McNeil, A. J., and Frey, R. (2000). *Estimation of Tail-Related Risk Measures
  for Heteroscedastic Financial Time Series: An Extreme Value Approach*.
  **Journal of Empirical Finance, journal.**
  [Source](https://www.sciencedirect.com/science/article/abs/pii/S0927539800000128)
- James, N., et al. (2023). *Forecasting Tail Risk Measures for Financial Time
  Series: An Extreme Value Approach with Covariates*. **Journal of Empirical
  Finance, journal.**
  [Source](https://www.sciencedirect.com/science/article/pii/S0927539823000026)
- Zaffran, M., et al. (2022). *Adaptive Conformal Predictions for Time Series*.
  **ICML, conference.**
  [Source](https://proceedings.mlr.press/v162/zaffran22a.html)
- Gibbs, I., and Candès, E. (2024). *Conformal Inference for Online Prediction
  with Arbitrary Distribution Shifts*. **JMLR, journal.**
  [Source](https://www.jmlr.org/papers/v25/22-1218.html)
- Hallberg Szabadváry, E. (2024). *Adaptive Conformal Inference for Multi-Step
  Ahead Time-Series Forecasting Online*. **COPA/PMLR, workshop proceedings.**
  [Source](https://proceedings.mlr.press/v230/hallberg-szabadvary24a.html)
- Angelopoulos, A. N., et al. (2024). *Conformal Risk Control*. **ICLR,
  conference.**
  [Source](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)
- Barrera, D., et al. (2026). *Statistical Learning of Value-at-Risk and Expected
  Shortfall*. **Mathematical Finance, journal.**
  [Source](https://onlinelibrary.wiley.com/doi/full/10.1111/mafi.70000)
- Kupiec, P. H. (1995). *Techniques for Verifying the Accuracy of Risk
  Measurement Models*. **Journal of Derivatives, journal.**
  [Source](https://www.fedinprint.org/item/fedgfe/34596/original)
- Christoffersen, P. F. (1998). *Evaluating Interval Forecasts*. **International
  Economic Review, journal.**
  [Source](https://ideas.repec.org/a/ier/iecrev/v39y1998i4p841-62.html)
- Fissler, T., Ziegel, J. F., and Gneiting, T. (2015). *Expected Shortfall Is
  Jointly Elicitable with Value at Risk—Implications for Backtesting*.
  **arXiv version of methodological work.**
  [Source](https://arxiv.org/abs/1507.00244)
- Bayer, S., and Dimitriadis, T. (2022). *Regression-Based Expected Shortfall
  Backtesting*. **Journal of Financial Econometrics, journal.**
  [Source](https://academic.oup.com/jfec/article/20/3/437/5912157)
- Gneiting, T., and Raftery, A. E. (2007). *Strictly Proper Scoring Rules,
  Prediction, and Estimation*. **Journal of the American Statistical Association,
  journal.** [Source](https://doi.org/10.1198/016214506000001437)
- Scheuerer, M., and Hamill, T. M. (2015). *Variogram-Based Proper Scoring Rules
  for Probabilistic Forecasts of Multivariate Quantities*. **Monthly Weather
  Review, journal.**
  [Source](https://repository.library.noaa.gov/view/noaa/22327/)

### Robust portfolio optimization

- Mohajerin Esfahani, P., and Kuhn, D. (2018). *Data-Driven Distributionally
  Robust Optimization Using the Wasserstein Metric*. **Mathematical Programming,
  journal.**
  [Source](https://link.springer.com/article/10.1007/s10107-017-1172-1)
- Pun, C. S., Wang, L., and Yan, X. (2023). *Data-Driven Distributionally Robust
  CVaR Portfolio Optimization Under a Regime-Switching Ambiguity Set*.
  **Manufacturing & Service Operations Management, journal.**
  [Source](https://pubsonline.informs.org/doi/10.1287/msom.2023.1229)
- Zhang, L., Yang, J., and Gao, R. (2024). *A Short and General Duality Proof for
  Wasserstein Distributionally Robust Optimization*. **Operations Research,
  journal.**
  [Source](https://pubsonline.informs.org/doi/full/10.1287/opre.2023.0135)
- Blanchet, J., Chen, L., and Zhou, X. Y. (2022). *Distributionally Robust
  Mean-Variance Portfolio Selection with Wasserstein Distances*. **Management
  Science, journal.**
  [Source](https://pubsonline.informs.org/doi/10.1287/mnsc.2021.4155)
- Liu, et al. (2026). *Distributionally Robust Downside Risk Optimization*.
  **SSRN 6446118, working paper.**
  [Source](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6446118)

### Causal and counterfactual methods

- Wu, Qiu, and Xie (2026). *DoFlow: Flow-Based Generative Models for
  Interventional and Counterfactual Forecasting on Time Series*. **ICLR 2026,
  accepted conference paper.**
  [Source](https://iclr.cc/virtual/2026/poster/10011565)
- Thumm, L., and Ontaneda, D. (2025). *Towards Causal Market Simulators*.
  **arXiv:2511.04469; ICAIF workshop paper.**
  [Source](https://arxiv.org/abs/2511.04469)
- *Causal Time Series Generation via Diffusion Models* (2025).
  **arXiv:2509.20846, preprint; reported ICLR desk rejection. Not used as
  validation evidence.** [Source](https://arxiv.org/abs/2509.20846)
- Cheridito, P., and Eckstein, S. (2025). *Optimal Transport and Wasserstein
  Distances for Causal Models*. **Bernoulli, journal.**
  [Source](https://www.research-collection.ethz.ch/items/aead0077-2b62-4a1c-8080-9ff6c6e8efc1)
- Lorch, L., Krause, A., and Schölkopf, B. (2024). *Causal Modeling with
  Stationary Diffusions*. **AISTATS, conference.**
  [Source](https://proceedings.mlr.press/v238/lorch24a.html)
- Gao, J., Mishra, S., and Ramazzotti, D. (2018). *Causal Data Science for
  Financial Stress Testing*. **Journal of Computational Science, journal.**
  [Source](https://doi.org/10.1016/j.jocs.2018.04.003)
- Brodersen, K. H., et al. (2015). *Inferring Causal Impact Using Bayesian
  Structural Time-Series Models*. **Annals of Applied Statistics, journal.**
  [Source](https://research.google/pubs/inferring-causal-impact-using-bayesian-structural-time-series-models/)
- Backhoff-Veraguas, J., et al. (2017). *Causal Transport in Discrete Time and
  Applications*. **SIAM Journal on Optimization, journal.**
  [Source](https://doi.org/10.1137/16M1080197)

## Final claim discipline

The literature supports saying that CrisisForge:

- integrates established regime-switching and factor-model ideas with accepted
  non-autoregressive diffusion methods;
- tests whether factor structure and soft regime conditioning improve multivariate
  financial scenario generation;
- reconstructs asset returns through an explicit stochastic observation equation;
- evaluates tail forecasts and portfolio decisions using established risk and
  optimization methods; and
- develops a separately validated structural counterfactual extension.

Until the empirical work is complete, it does **not** support saying that
CrisisForge:

- predicts the next financial crisis;
- proves diffusion is superior to econometric simulation;
- identifies a latent state as the true economic regime;
- guarantees ES through ordinary conformal calibration;
- is robust to the true distribution merely because it solves a Wasserstein-DRO
  problem; or
- estimates real-world causal effects merely by conditioning on a policy shock.

These boundaries determine both the research protocol and the language used in the
final report, README, presentation, CV, and LinkedIn material.
