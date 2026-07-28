# Mathematical Formulation

**Version:** 0.3
**Date:** 2026-07-28

## 1. Common-endpoint return panel, information set, and estimands

Let:

- \(e_0<e_1<\cdots<e_T\): target-panel endpoints shared by the industry-return and
  Treasury-yield source calendars;
- \(r_t\in(-1,\infty)^N\): decimal asset **simple** returns over the common interval
  \((e_{t-1},e_t]\);
- \(x_t^{\mathrm{avail}}\in\mathbb R^p\): context values available by the
  after-close decision time at endpoint \(e_t\);
- \(\mathcal F_t=\sigma(r_{1:t},x_{1:t}^{\mathrm{avail}})\): the after-close
  forecast-origin information set;
- \(z_t\in\{1,\dots,K\}\): latent market regime;
- \(f_t\in\mathbb R^q\): latent common factors;
- \(H\): forecast horizon; and
- \(w\in\mathbb R^N\): portfolio weights.

The raw Phase 0 targets are simple returns. They are not log returns and no
risk-free rate is subtracted. EFFR is a context variable, not a transformation that
turns \(r_t\) into an excess return.

If \(\widetilde r_{i,u}\) denotes a source-native simple return dated inside the
common interval, the aligned target is:

\[
r_{i,t}
=
\prod_{u\in(e_{t-1},e_t]\cap\mathcal C_i}
\left(1+\widetilde r_{i,u}\right)-1,
\]

where \(\mathcal C_i\) is the source observation calendar. Thus each row represents
the same endpoint-to-endpoint holding interval for all targets even when the number
of underlying source observations differs. It does not assert that every row is
exactly one identical exchange session. The interval audit stores \(e_{t-1}\),
\(e_t\), elapsed calendar days, and source-observation counts.

The primary estimand is:

\[
\mathcal P_t^H=P(r_{t+1:t+H}\mid\mathcal F_t).
\]

The default forecast is issued only after close at \(e_t\), after \(r_t\) and all
eligible context values are available. Its first target is \(r_{t+1}\); there is no
same-interval \(r_t\) prediction in the registered design. Treasury-yield context
and New York Fed EFFR enter after their catalogued one-model-session lag. Five
return-derived context variables may use \(r_{1:t}\) because they support only the
after-close \(t\rightarrow t+1\) forecast. OFR stress components are
validation-only labels and are not members of \(\mathcal F_t\). Cboe/OptionMetrics
variables are absent from the public core and may enter only a separately licensed
extension.

For Phase 0, \(x_t^{\mathrm{avail}}\) has ten columns: 2-, 5-, and 10-year Treasury
par-yield levels, the 10y–2y slope, EFFR, equal-weighted industry-market return,
20-observation realized volatility, 20-observation downside semivolatility,
20-observation industry dispersion, and 60-observation market drawdown. The clean
matrix has 6,465 common-endpoint rows from 2000-07-05 through 2026-05-29.

An optional model may use:

\[
y_t=\log(1+r_t).
\]

This is an explicitly named, within-fold model-space transform. Any learned
centering or scaling is estimated on that fold's training data only. If the
observation equation is fit to \(y_t\), every equation and artifact is labeled
accordingly, and simulated paths are returned to simple-return space with
\(r_t=\exp(y_t)-1\) before aggregation, risk measurement, or optimization. The
symbols “simple return,” “log return,” and “excess return” are never used
interchangeably.

Derived estimands are:

\[
\operatorname{VaR}_{\alpha,t}(w),\qquad
\operatorname{ES}_{\alpha,t}(w),\qquad
p_{\mathrm{co-crash},t},
\]

and realized out-of-sample decision loss for a strategy fixed at origin \(t\).

Counterfactual estimands exist only in the separately specified SCM:

\[
\tau_j(h;a,a')
=E[X_{j,t+h}^{do(A_t=a)}-X_{j,t+h}^{do(A_t=a')}].
\]

The public financial panel does not, by itself, identify this causal estimand.

## 2. Bayesian switching dynamic factor model

The regime, factor, and observation layers are estimated as one coherent switching
state-space model. They are not three unrelated plug-in stages.

### 2.1 Regime transition

\[
P(z_{t+1}=k\mid z_t=j,\mathcal F_t)
=
\frac{\exp(a_{jk}+\gamma_{jk}^{\top}x_t^{\mathrm{avail}})}
{\sum_{\ell=1}^{K}\exp(a_{j\ell}+\gamma_{j\ell}^{\top}x_t^{\mathrm{avail}})}.
\]

The core model sets \(\gamma=0\) and uses a homogeneous sticky transition matrix.
Covariate-dependent transitions are added only if the sample supports them. This
indexing makes explicit that after-close information at \(t\) can affect the
distribution of \(z_{t+1}\), not the already realized state/return at \(t\).

### 2.2 Factor transition

\[
f_{t+1}
=c_{z_{t+1}}+\Phi_{z_{t+1}}f_t+G_{z_{t+1}}x_t^{\mathrm{avail}}
+Q_{z_{t+1}}^{1/2}\eta_{t+1},
\qquad \eta_{t+1}\sim N(0,I_q).
\]

The spectral radius of \(\Phi_z\) is restricted below one in the initial
specification.

### 2.3 Factor-to-asset observation equation

\[
\boxed{
r_t=\alpha_{z_t}+B_{z_t}f_t+D_{z_t}\epsilon_t
}
\]

with

\[
\epsilon_t\sim t_{\nu_{z_t}}^{\mathrm{std}}(0,R_{z_t}),\qquad
D_z=\operatorname{diag}(d_{1z},\ldots,d_{Nz}),
\]

and

\[
\Sigma_{\epsilon,z}=D_zR_zD_z.
\]

\(t_\nu^{\mathrm{std}}\) denotes a multivariate Student distribution standardized
to unit marginal variance, so \(R_z\) is its correlation matrix and
\(\operatorname{Cov}(D_z\epsilon_t)=D_zR_zD_z\). \(D_z\) controls asset-specific
residual scale; using \(D_z\) with iid residuals would falsely eliminate residual
co-movement and understate co-crash risk.

In the default specification this equation is for decimal simple returns. Generated
values must satisfy \(r_{i,t}>-1\); the implementation must use a support-preserving
link or explicitly log and report any rejected invalid draw. If the
log-simple-return sensitivity model is used, the left-hand side is explicitly
changed to \(y_t=\log(1+r_t)\); it is not silently called \(r_t\), and all scenarios
are inverse-transformed before financial risk calculations.

### 2.4 Priors and regularization

- transition rows: sticky Dirichlet;
- \(B_z=B_0+\Delta B_z\): hierarchical shrinkage;
- \(R_z\): LKJ prior or frequentist shrinkage analogue;
- \(d_{iz}\): half-\(t\) or log-normal;
- \(\nu_z>2\);
- stable \(\Phi_z\);
- optional horseshoe prior for sparse macro effects \(G_z\).

### 2.5 Identifiability

Factor rotations are normalized with loading/sign anchors and aligned across folds
by orthogonal Procrustes rotation. Regime label switching is resolved by ordering
states using training-fold return-derived market volatility and drawdown summaries.
OFR stress labels may be used only for external, retrospective agreement checks;
they do not order states or enter estimation.

The forecasting output is the filtered probability:

\[
\pi_{t|t,k}=P(z_t=k\mid\mathcal F_t).
\]

The smoothed probability \(P(z_t=k\mid\mathcal F_T)\), \(T>t\), is forbidden in
forecasting and backtesting because it uses future information.

### 2.6 Soft posterior path mixture, not averaged loadings

Let \(\pi_{t|t,k}=P(z_t=k\mid\mathcal F_t)\). The one-horizon predictive
distribution is a mixture:

\[
p(r_{t+h}\mid\mathcal F_t)
=
\sum_{k=1}^K
\pi_{t+h|t,k}\,
p(r_{t+h}\mid z_{t+h}=k,\mathcal F_t).
\]

The joint \(H\)-interval forecast is a path mixture:

\[
\begin{aligned}
p(r_{t+1:t+H}\mid\mathcal F_t)
&=
\sum_{z_{t+1:t+H}}
p(z_{t+1:t+H}\mid\mathcal F_t)\\
&\quad\times
\int
p(r_{t+1:t+H}\mid z_{t+1:t+H},\omega,\mathcal F_t)
p(\omega\mid D_{\mathrm{train}})\,d\omega .
\end{aligned}
\]

It is generally incorrect to replace either mixture with a single path built from
posterior-probability-averaged \(B_k,D_k,R_k\), or Cholesky factors. Simulation
therefore draws a complete regime path and, where supported, a parameter draw for
each scenario, then applies the corresponding
\(\alpha_{z_{t+h}},B_{z_{t+h}},D_{z_{t+h}},R_{z_{t+h}}\) at every horizon. A hard
decoded state/path is retained only as a diagnostic ablation, not as the primary
forecast.

## 3. Diffusion interface

The state-space model supplies a conditional factor-path mean and scale. Diffusion
models nonlinear residual paths rather than duplicating the same factor transition.

Define:

\[
Y_t^0=S_t^{-1}
\left(
f_{t+1:t+H}-\mu^{\mathrm{SSM}}_{t,1:H}
\right)\in\mathbb R^{H\times q}.
\]

The context is:

\[
C_t=\operatorname{Enc}\left(
f_{t-L+1:t}^{\mathrm{filtered}},
\{\pi_{u|u}\}_{u=t-L+1}^{t},
x_{t-L+1:t}^{\mathrm{avail}},
\Pi_{t,1:H},
M_t
\right),
\]

where \(M_t\) is an optional stress-control vector and

\[
\Pi_{t,h}=\pi_{t|t}P^h
\]

is predictive regime probability derived from origin-\(t\) information. Realized
future regimes are not inputs. In predictive mode \(M_t\) is omitted or fixed using
origin-\(t\) information and carries no separately calibrated stress probability.
For covariate-dependent transitions, \(\Pi_{t,1:H}\) is obtained by forward
simulation rather than by inserting future realized context.

### 3.1 One-shot forward diffusion

Use \(s\) for diffusion time:

\[
q(Y^s\mid Y^0)=
N\left(\sqrt{\bar\alpha_s}Y^0,(1-\bar\alpha_s)I\right).
\]

The temporal U-Net or temporal transformer predicts noise for the full
\(H\times q\) tensor:

\[
\mathcal L_{\mathrm{DDPM}}=
E_{t,s,\varepsilon}
\left[
\left\|
\varepsilon-\varepsilon_\theta(Y^s,s,C_t)
\right\|_2^2
\right].
\]

One-shot generation removes recursive interval-by-interval forecast accumulation,
but it does not guarantee stable long-horizon paths. Variance growth, ACF, and
explosive-path rates are explicitly tested.

### 3.2 Posterior-predictive scenario generation

For scenario \(m\):

1. draw parameters
   \[
   \omega^{(m)}\sim p(\omega\mid D_{\mathrm{train}});
   \]
2. draw
   \[
   z_{t+1:t+H}^{(m)}
   \]
   from the soft posterior-predictive regime process, rather than decoding one hard
   path;
3. draw an entire diffusion residual path \(Y_t^{0,(m)}\);
4. reconstruct factor path \(f_{t+1:t+H}^{(m)}\);
5. draw residuals \(\epsilon_{t+h}^{(m)}\); and
6. map:
   \[
   r_{t+h}^{(m)}
   =
   \alpha_{z_{t+h}^{(m)}}+
   B_{z_{t+h}^{(m)}}f_{t+h}^{(m)}
   +
   D_{z_{t+h}^{(m)}}\epsilon_{t+h}^{(m)}.
   \]

This propagates regime, factor, loading, scale, and residual uncertainty. If the
model was trained on \(y=\log(1+r)\), step 6 produces \(y\), and a final
\(\exp(y)-1\) step returns every path to decimal simple-return space.

## 4. Tail calibration

### 4.1 Tail-conditioned mixture

Using a training-only reference portfolio and threshold:

\[
A_{t,H}
=
1\{L^{\mathrm{ref}}(r_{t+1:t+H})>u_{\mathrm{train}}\}.
\]

Estimate:

\[
p_\theta(Y\mid C,A),\qquad p_\phi(A=1\mid C).
\]

The honest forecast is:

\[
p(Y\mid C)=
\sum_{a\in\{0,1\}}
p_\phi(a\mid C)p_\theta(Y\mid C,a).
\]

Tail windows may be oversampled, but inverse-probability correction preserves the
target distribution. Stress mode may force \(A=1\); predictive mode may not use
the realized future \(A_{t,H}\) as origin-\(t\) context.

### 4.2 Peaks over threshold

For portfolio loss \(L\) in regime \(z\):

\[
L-u_z\mid L>u_z,z
\sim \operatorname{GPD}(\xi_z,\beta_z).
\]

The initial threshold search is 90–95% on training data. Hierarchical shrinkage is
required because a regime-specific 99% tail would have too few exceedances.

The research extension calibrates multivariate radial magnitude while preserving
the angular tail-dependence component:

\[
R=\|\Sigma_z^{-1/2}\ell\|,\qquad
W=\frac{\Sigma_z^{-1/2}\ell}{R}.
\]

### 4.3 Sequential VaR calibration

\[
\widehat q^{\mathrm{cal}}_{\alpha,t}
=\widehat q^{\mathrm{raw}}_{\alpha,t}
+Q_\alpha\left(
L_u-\widehat q^{\mathrm{raw}}_{\alpha,u}
\right)_{u\in\mathcal C_t}.
\]

Calibration is separate by horizon. Because financial time series are dependent
and non-exchangeable, only adaptive/sequential assumptions or empirical coverage
are claimed. Ordinary conformal calibration is not presented as an ES guarantee.

## 5. Asset-level risk engine

For scenario \(m\), define the geometrically compounded \(H\)-interval asset
simple return:

\[
G_{i,t,H}^{(m)}
=
\prod_{h=1}^{H}
\left(1+r_{i,t+h}^{(m)}\right)-1.
\]

The arithmetic sum of simple returns is not used. For initial buy-and-hold weights
\(w\), with no interim rebalancing, portfolio loss is:

\[
L_w^{(m)}=-w^\top G_{t,H}^{(m)}.
\]

If a later experiment rebalances within the horizon, its self-financing wealth
recursion, transaction timing, and non-anticipativity constraints must be specified
instead of reusing this buy-and-hold expression.

### 5.1 Value at Risk and Expected Shortfall

\[
\operatorname{VaR}_\alpha(L_w)
=\inf\{\ell:F_{L_w}(\ell)\ge\alpha\},
\]

\[
\operatorname{ES}_\alpha(L_w)
=\frac{1}{1-\alpha}
\int_\alpha^1\operatorname{VaR}_u(L_w)\,du.
\]

Reported estimates include Monte Carlo standard errors or bootstrap intervals.

### 5.2 Co-crash probability

Fix asset thresholds from training data:

\[
c_i=Q_{\alpha_0}(G_{i,H};D_{\mathrm{train}}).
\]

For a minimum simultaneous crash fraction \(\kappa\):

\[
p_{\mathrm{co-crash},t}
=
P\left(
\frac{1}{N}\sum_{i=1}^{N}
1\{G_{i,t,H}\le c_i\}\ge\kappa
\mid\mathcal F_t
\right).
\]

This probability is evaluated with Brier score, log score, and calibration curves.

## 6. CVaR portfolio problem

The initial problem is single-period buy-and-hold over \(H\) common-endpoint
intervals. Scenario \(G_i\) is the vector of compounded asset simple returns from
section 5:

\[
\min_{w,\tau,s_i}
\quad
\tau+\frac{1}{(1-\alpha)S}\sum_{i=1}^S s_i
+\sum_j c_j u_j
\]

subject to:

\[
s_i\ge-w^\top G_i-\tau,\qquad s_i\ge0,
\]

\[
\mathbf1^\top w=1,\qquad \ell\le w\le u.
\]

Here \(u_j\ge w_j-w_{t-1,j}\), \(u_j\ge-(w_j-w_{t-1,j})\), and
\(\sum_j u_j\le\kappa\). The implementation uses explicit L1 turnover and an
additive linear transaction-cost approximation. For a realized block return
\(G_{t,H}\),

\[
R^{\mathrm{net}}_{t,H}
=w^\top G_{t,H}-\sum_j c_j|w_j-w_{t-1,j}|.
\]

This is not a multiplicative wealth withdrawal, intrablock execution model, or
market-impact model. End-of-block risky-asset weights drift according to realized
gross asset returns before the next rebalance. A multistage policy is out of scope
until scenario-tree and non-anticipativity constraints are added.

## 7. Wasserstein-DRO CVaR

Let:

\[
\mathcal U_\rho(\widehat P)
=\{Q:W_1(Q,\widehat P)\le\rho\}.
\]

\(\widehat P\) is the empirical distribution of compounded simple-return vectors
\(G_i\). For unrestricted support and linear portfolio loss, the worst-case CVaR
admits a single-level convex reformulation of the form:

\[
\min_{w,\tau,s_i}
\quad
\tau+
\frac{1}{(1-\alpha)S}\sum_i s_i
+
\frac{\rho}{1-\alpha}\|w\|_*
+\text{costs and regularization}.
\]

The exact dual norm follows the selected ground norm. Support constraints can alter
the reformulation and must be stated. In particular, a formulation operating on
one-period simple returns must respect \(r>-1\), while the decision formulation
above operates on compounded scenario vectors \(G\).

### 7.1 Generator uncertainty

A Wasserstein ball centered on diffusion samples is robust to perturbations around
the model output, not automatically around the true market distribution. The radius
is decomposed conceptually as:

\[
\rho=
\rho_{\mathrm{generator\ misspecification}}
+
\rho_{\mathrm{Monte\ Carlo}}.
\]

Both are calibrated on validation origins. Test data never select \(\rho\).

## 8. Error propagation

For a fixed regime path, define:

\[
T_z(f,e)=\alpha_z+B_zf+D_ze.
\]

Then:

\[
W_1(T_{z\#}P,T_{z\#}Q)
\le
\|B_z\|_{\mathrm{op}}W_1(P_f,Q_f)
+
\|D_z\|_{\mathrm{op}}W_1(P_e,Q_e).
\]

A practical diagnostic bound including regime and parameter error is:

\[
W_1(P_r,Q_r)
\lesssim
B_{\max}\delta_f+
D_{\max}\delta_\epsilon+
M\delta_z+
E\|(\widehat B-B)f\|+
E\|(\widehat D-D)\epsilon\|.
\]

For fixed \(w\), portfolio loss is \(\|w\|_*\)-Lipschitz, giving:

\[
\left|
\operatorname{CVaR}_{\alpha}^{P}(L_w)
-
\operatorname{CVaR}_{\alpha}^{Q}(L_w)
\right|
\le
\frac{\|w\|_*}{1-\alpha}W_1(P_r,Q_r).
\]

This bounds objective-value sensitivity, not optimizer-weight sensitivity; hence
the strong convex regularizer and explicit weight-stability tests. The displayed
bound is for the one-interval observation map. Horizon loss uses the nonlinear
simple-return compounding map from section 5; its sensitivity is evaluated on the
registered bounded-support scenario set rather than silently replacing compounding
with an arithmetic sum.

## 9. Structural counterfactual extension

This is a separate semi-synthetic validation track, not an automatic causal
interpretation of the public forecasting panel. Its confirmatory experiments use a
known data-generating SCM so factual and counterfactual paths are both available.

Use a time-unrolled SCM:

\[
X_{j,t}=
g_{j,z_t}\left(
X_{\operatorname{Pa}(j),t-1},
A_t,U_{j,t}
\right).
\]

Feedback can occur across time while the unrolled graph remains acyclic.

Individual counterfactual generation follows:

1. **Abduction:** infer exogenous innovations \(U\) from factual history.
2. **Action:** replace the treatment equation with \(do(A_t=a)\).
3. **Prediction:** hold the same \(U\) fixed and propagate the alternate path.

Evaluate:

\[
\tau_j(h;a,a')=
E[X_{j,t+h}^{a}-X_{j,t+h}^{a'}],
\]

\[
\Delta ES_\alpha(a,a')=
ES_\alpha(L^a)-ES_\alpha(L^{a'}),
\]

plus counterfactual distribution and path error.

Two estimands are reported separately:

- **total effect:** intervention may change future regime transitions;
- **controlled effect:** regime path is held fixed.

Conditioning on a realized post-treatment regime is avoided because it can condition
on a mediator or collider.

In real data, a policy-rate level is endogenous. Monetary-policy surprise is the
preferred treatment only if a separately registered identification study is
attempted. The current public core supplies neither a treatment design nor
counterfactual ground truth. Without an instrument, natural experiment,
high-frequency event-study design, or another defensible strategy, its outputs are
called model-based structural interventions, not estimated real-world causal
effects. A \(+200\) bp shock outside support is explicitly called an extrapolative
stress intervention and is never assigned an observational forecast probability.

## 10. Experiment protocol

- Public-core dimensions: \(N=15\) simple-return targets and \(p=10\) predictive
  context variables; initial latent dimensions are \(q\in\{3,4,5\}\),
  \(K\in\{2,3\}\), and \(H=20\) common-endpoint intervals.
- Expand to \(H=60\) only after stability at \(H=20\).
- Use after-close expanding or rolling origins; the first target is \(t+1\), with
  purge and embargo \(\ge H\).
- Refit all learned preprocessing inside each fold.
- Select hyperparameters only on validation.
- Compare models on identical origins and seeds.
- Use paired stationary block bootstrap or HAC inference for overlapping horizons.
- Preserve every registered experiment, including failures.

## 11. Principal diagnostics

- posterior draw versus plug-in mean;
- hard-decoded diagnostic ablation versus soft posterior regime conditioning;
- transition-matrix tempering;
- \(B_z,D_z,R_z\) perturbation;
- fixed versus regime-dependent mapping;
- Gaussian versus Student-\(t\) residuals;
- controlled factor-path error inflation;
- \(\rho\), threshold, \(K,q,H\) grids;
- true-factor/generated-residual and generated-factor/true-residual module swaps;
- central-distribution, tail, and decision Pareto frontier.
