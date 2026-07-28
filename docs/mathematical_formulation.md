# Mathematical Formulation

**Version:** 0.3

**Date:** 2026-07-28

**Scope:** equations implemented in the public core, followed by explicitly
deferred target extensions

## 1. Decision clock and return panel

Let \(e_0<e_1<\cdots<e_T\) denote common target-panel endpoints and let

- \(r_t\in(-1,\infty)^N\) be decimal asset **simple** returns over
  \((e_{t-1},e_t]\);
- \(x_t\in\mathbb R^p\) be context available after close at \(e_t\);
- \(\mathcal F_t=\sigma(r_{1:t},x_{1:t})\);
- \(z_t\in\{1,\ldots,K\}\) be a latent state;
- \(f_t\in\mathbb R^q\) be a low-dimensional factor score;
- \(H=20\) be the registered validation horizon; and
- \(w\in\mathbb R^N\) be portfolio weights.

For a source-native return \(\widetilde r_{i,u}\), the aligned target is

\[
r_{i,t}=
\prod_{u\in(e_{t-1},e_t]\cap\mathcal C_i}(1+\widetilde r_{i,u})-1.
\]

A row is a common-endpoint holding interval, not necessarily one identical
exchange session for every source. The predictive estimand is

\[
\mathcal P_t^H=P(r_{t+1:t+H}\mid\mathcal F_t).
\]

The decision occurs after \(r_t\) is observed; the first target is \(r_{t+1}\).
All learned transformations use training data only. The registered public run
first transforms simple returns to

\[
y_t=\log(1+r_t),
\]

fits the linear factor and observation system in \(y\)-space, and converts
generated observations back with \(r_t=\operatorname{expm1}(y_t)\).

## 2. Implemented switching-factor estimator

The public-core estimator is a **multi-stage MAP/empirical-Bayes baseline**, not a
joint Bayesian posterior.

### 2.1 Train-only factor representation

Let \(\mu\) and \(s>0\) be the asset-wise training mean and scale of \(y_t\), and let
\[
\widetilde y_t=(y_t-\mu)\oslash s.
\]

Train-only singular-value decomposition yields the first \(q\) right singular
vectors \(V_q\). Factor scores are

\[
f_t=\widetilde y_t V_q^\top.
\]

Each PCA axis is oriented deterministically so its largest absolute loading is
positive. No fold-to-fold Procrustes alignment or posterior factor rotation is
implemented.

### 2.2 Sticky Gaussian HMM

The latent state follows a homogeneous Markov chain

\[
P(z_t=k\mid z_{t-1}=j)=A_{jk}.
\]

Factor emissions are Gaussian:

\[
f_t\mid z_t=k\sim N(\mu_k,\Sigma_k).
\]

Parameters are estimated by multi-start EM with covariance regularization and a
sticky Dirichlet-style transition pseudocount. This is MAP-like estimation. It does
not draw \(A,\mu_k,\Sigma_k\) from a posterior.

Forward filtering produces

\[
\pi_{t\mid t,k}=P(z_t=k\mid f_{1:t}).
\]

Training-only forward–backward smoothing produces
\(\gamma_{t,k}=P(z_t=k\mid f_{1:T_{\mathrm{train}}})\) for fitting downstream
state-weighted components. At validation origins only \(\pi_{t\mid t}\) is used.
State indices are ordered reproducibly by the first-factor emission mean; economic
labels are assigned only after descriptive diagnostics.

### 2.3 Regime-weighted factor VAR

For each state \(k\), fit the weighted ridge VAR(1)

\[
f_t=c_k+\Phi_k f_{t-1}+\eta_{t,k},\qquad
\eta_{t,k}\sim N(0,Q_k),
\]

using \(\gamma_{t,k}\) as observation weights. If a state has insufficient
effective observations, pooled parameters are used. If
\(\rho(\Phi_k)\) exceeds the configured stability cap, \(\Phi_k\) is rescaled so
that \(\rho(\Phi_k)<1\).

Given origin belief \(\pi_{t\mid t}\), a scenario draws

\[
z_{t+1:t+H}^{(m)}\sim P_A(\cdot\mid\pi_{t\mid t})
\]

and then recursively draws \(f_{t+h}^{(m)}\) from the exact state-specific VAR
parameters.

### 2.4 Regime-weighted factor-to-asset mapping

For each state \(k\), weighted ridge regression estimates

\[
y_t=\alpha_k+B_k f_t+u_{t,k},
\qquad u_{t,k}\sim N(0,\Omega_k),
\]

where \(y_t=\log(1+r_t)\) in the registered public run. The residual covariance is
shrunk toward its diagonal and projected to a positive-semidefinite matrix.

The literal implemented observation equation is therefore

\[
\boxed{
y_t=\alpha_{z_t}+B_{z_t}f_t+D_{z_t}\epsilon_t,
\qquad
r_t=\operatorname{expm1}(y_t)
}
\]

with

\[
\Omega_k=D_kR_kD_k,\qquad
D_k=\operatorname{diag}\!\left(\sqrt{\operatorname{diag}\Omega_k}\right),
\]

and \(\epsilon_t\sim N(0,R_k)\). In code, scenarios use a Cholesky factor
\(L_kL_k^\top=\Omega_k\); \(D_k\) and \(R_k\) are an interpretive decomposition,
not separately estimated posterior objects.

For scenario \(m\) and horizon step \(h\),

\[
y_{t+h}^{(m)}
=\alpha_{z_{t+h}^{(m)}}+
B_{z_{t+h}^{(m)}}f_{t+h}^{(m)}
+L_{z_{t+h}^{(m)}}\xi_{t+h}^{(m)},
\quad \xi\sim N(0,I).
\]

Exact state-specific parameters are applied after drawing a state path. Loading or
Cholesky matrices are never probability-averaged. The generated asset return is
\(r_{t+h}^{(m)}=\operatorname{expm1}(y_{t+h}^{(m)})\).

## 3. Implemented one-shot temporal diffusion pilot

The diffusion pilot models the full future factor tensor directly rather than a
state-space residual.

For an origin \(t\), let

\[
Y_t^0=f_{t+1:t+H}\in\mathbb R^{H\times q}.
\]

Train-only standardizers transform factor history, macro/context history, and
future factor tensors. The context encoder receives a lookback window

\[
C_t=\operatorname{Enc}\left(
f_{t-L+1:t},x_{t-L+1:t},\pi_{t\mid t}
\right),
\]

with \(L=60\). Future realized states never enter \(C_t\).

For diffusion time \(s\),

\[
q(Y_t^s\mid Y_t^0)=
N\!\left(\sqrt{\bar\alpha_s}Y_t^0,(1-\bar\alpha_s)I\right).
\]

A temporal denoiser predicts noise for the complete \(H\times q\) path. For
sample \(i\) in minibatch \(\mathcal B\), the literal implementation computes

\[
\ell_i=
\frac{1}{Hq}
\left\|
\varepsilon_i-\varepsilon_\theta(Y_i^s,s,C_i)
\right\|_F^2,
\qquad
\mathcal L_{\mathcal B}
=\frac{1}{|\mathcal B|}\sum_{i\in\mathcal B}\ell_i.
\]

Reverse diffusion produces a whole factor path in one call, avoiding
interval-by-interval autoregressive rollout. This does not prove long-horizon
stability; it only removes that specific recursive mechanism.

### 3.1 Tail importance weighting used in the pilot

Let \(S_i\) be the maximum, over the horizon, of the cross-factor
root-mean-square absolute standardized shock in training window \(i\). The
registered threshold is \(u=q_{0.90}(S)\), and the excess scale is

\[
a=\max\left(
\operatorname{median}\{S_i-u:S_i>u\},
\operatorname{sd}(S),
10^{-8}
\right).
\]

The raw fine-tuning weight is the capped linear excess weight

\[
\widetilde w_i=
\min\left\{8,\,
1+3\frac{(S_i-u)_+}{a}
\right\}.
\]

Raw weights are normalized to mean one over the complete training set, capped
again at eight, and renormalized to training mean one. The selected weights are
then renormalized within each minibatch. The literal weighted loss is

\[
\mathcal L_{\mathcal B}^{\mathrm{tail}}
=
\frac{\sum_{i\in\mathcal B}w_i\ell_i}
{\sum_{i\in\mathcal B}w_i}.
\]

This is importance-weighted denoising, not a completed EVT or conformal
calibration layer.

### 3.2 Pilot asset reconstruction boundary

The pilot generates \(f_{t+1:t+H}\) with diffusion and separately samples a future
HMM state path from the same origin belief. It then applies the Stage 1 observation
map. Thus

\[
f_{t+1:t+H}^{(m)}
\not\!\perp z_{t+1:t+H}^{(m)}
\quad\text{is desired but not enforced.}
\]

The current implementation does not guarantee their joint dynamic consistency.

## 4. Asset-level risk engine

For scenario \(m\), geometrically compound each asset's simple returns:

\[
G_{i,t,H}^{(m)}
=\prod_{h=1}^{H}(1+r_{i,t+h}^{(m)})-1.
\]

For a buy-and-hold portfolio,

\[
R_{w,t,H}^{(m)}=w^\top G_{t,H}^{(m)},
\qquad
L_{w,t,H}^{(m)}=-R_{w,t,H}^{(m)}.
\]

The code does not replace geometric compounding with an arithmetic sum.
For the registered Stage 0, Stage 1, and Stage 2 generator evaluations, energy
and variogram scores use the full 15-asset vector \(G_{t,H}\). VaR, ES, joint
VaR–ES, and violation rates use the frozen equal-weight portfolio
\(w_i=1/15\).

### 4.1 VaR and ES

\[
\operatorname{VaR}_\alpha(L)
=\inf\{\ell:F_L(\ell)\ge\alpha\},
\]

\[
\operatorname{ES}_\alpha(L)
=
\frac{1}{1-\alpha}
\int_\alpha^1 F_L^{-1}(u)\,du.
\]

The finite-sample implementation integrates the right-continuous empirical
quantile and therefore includes fractional mass at the threshold when
\((1-\alpha)n\) is non-integer. It is not generally the unweighted mean of every
observation satisfying \(L\ge\operatorname{VaR}_\alpha(L)\). The validation study
uses \(\alpha=0.95\).

Coverage diagnostics include Kupiec unconditional coverage and Christoffersen
independence/conditional coverage. Generator rankings also use energy,
variogram, and joint VaR–ES scores.

### 4.2 Co-crash

Let \(c_i\) be a training-only asset threshold and \(\kappa\) the required fraction
of simultaneously breached assets:

\[
C_{t,H}=
1\left\{
\frac1N\sum_{i=1}^N1\{G_{i,t,H}\le c_i\}\ge\kappa
\right\}.
\]

The scenario estimate is

\[
\widehat p_{\mathrm{co-crash},t}
=\frac1M\sum_{m=1}^M C_{t,H}^{(m)}.
\]

At every evaluated origin—74 Stage 0/Stage 1 origins and four Stage 2
origins—the realized co-crash label is zero. Generated scenario sets may still
assign nonzero probabilities, but the Brier scores cannot demonstrate event
discrimination.

## 5. Empirical CVaR decision problem

Let \(G_m\in\mathbb R^N\) be scenario compounded returns and
\(\ell_m(w)=-w^\top G_m\). The Rockafellar–Uryasev linear program is

\[
\min_{w,\tau,\{s_m\}_{m=1}^M,u}
\quad
\tau+\frac{1}{(1-\alpha)M}\sum_{m=1}^M s_m
+c^\top u
\]

subject to

\[
s_m\ge -w^\top G_m-\tau,\qquad s_m\ge0,
\]

\[
\mathbf1^\top w=1,\qquad 0\le w_i\le0.20,
\]

and, when previous weights \(w^{-}\) exist,

\[
u_i\ge w_i-w_i^{-},\qquad
u_i\ge-(w_i-w_i^{-}),\qquad
\mathbf1^\top u\le0.40.
\]

The validation cost rate is \(c_i=0.001\). Realized net block return is the additive
linear approximation

\[
R_{t,H}^{\mathrm{net}}
=w_t^\top G_{t,H}-c^\top|w_t-w_t^{-}|.
\]

It is not a multiplicative wealth withdrawal, execution model, or market-impact
model.

## 6. Implemented 1-Wasserstein robust CVaR

Define the L1-ground-cost ambiguity set

\[
\mathcal U_\rho(\widehat P)
=\{Q:W_1(Q,\widehat P)\le\rho\}.
\]

For linear loss and unrestricted support, strong duality adds the dual-norm
penalty

\[
\frac{\rho}{1-\alpha}\|w\|_\infty
\]

to empirical CVaR. The implemented single-level problem is therefore

\[
\min_w\quad
\widehat{\operatorname{CVaR}}_\alpha(-w^\top G)
+\frac{\rho}{1-\alpha}\|w\|_\infty
+\text{linear transaction cost},
\]

under the same portfolio constraints. The four tested radii
\(0.0001,0.00025,0.0005,0.001\) are independent validation sensitivities; no
radius selection rule is claimed.

A Wasserstein ball around generated scenarios protects against local perturbations
of the generator output. It is not automatically a confidence set for the true
market distribution.

The registered dual assumes unrestricted return support even though cumulative
simple returns are economically bounded below by \(-1\). This is a tractability
approximation: the ambiguity set can place probability mass outside feasible
simple-return support.

## 7. Paired model comparison

For model \(a\), baseline \(b\), metric \(j\), and common validation origin \(o\),
define

\[
d_{o,j}^{a,b}=S_{o,j}^{a}-S_{o,j}^{b}.
\]

All registered scores are oriented so lower is better. The reported estimand is
\(\bar d_j\). A circular moving-block bootstrap with block length four resamples
the ordered origin differences and forms percentile 95% intervals from 10,000
draws. These intervals are exploratory, not multiplicity-adjusted, and not causal
model-class effects.

## 8. Structural counterfactual extension

This is a separate known-ground-truth semi-synthetic track. Within a time slice the
structural order is

\[
A_t\rightarrow Y_t\rightarrow L_t\rightarrow C_t
\rightarrow E_t\rightarrow V_t,
\]

where the variables represent policy, yield, liquidity, credit, equity, and
volatility. Lagged equity affects future policy:

\[
E_{t-1}\rightarrow A_t.
\]

Time unrolling preserves feedback while keeping the graph acyclic.

Each structural equation has the form

\[
X_{j,t}=g_{j,z_t}(X_{\operatorname{pa}(j),t},
X_{\operatorname{pa}^{-}(j),t-1},U_{j,t}).
\]

For a factual path:

1. **Abduction:** invert the known equations to recover \(U_{1:H}\).
2. **Action:** replace scheduled policy equations with \(do(A_t=a_t)\).
3. **Prediction:** reuse the same \(U_{1:H}\) and regime path to propagate the
   alternate world.

For outcome \(X_j\), the paired mean effect is

\[
\tau_j(h)=E[X_{j,h}^{a}-X_{j,h}^{a'}],
\]

where \(a\) is the treated schedule and \(a'\) is the factual/reference schedule.
The reported terminal and cumulative effects are

\[
\tau_j^{\mathrm{terminal}}=\tau_j(H),
\qquad
\tau_j^{\mathrm{cumulative}}=
E\left[\sum_{h=1}^H(X_{j,h}^{a}-X_{j,h}^{a'})\right].
\]

Let \(s_{\mathrm{equity}}=-1\) and \(s_{\mathrm{volatility}}=+1\). The
path-level tail loss and paired tail effect are

\[
L_j^a=s_j\sum_{h=1}^H X_{j,h}^a,
\qquad
\tau_j^{\mathrm{tail}}=
\operatorname{ES}_{0.95}(L_j^a)
-\operatorname{ES}_{0.95}(L_j^{a'}).
\]

Thus lower cumulative equity outcomes are larger losses, while higher cumulative
volatility outcomes are larger stress losses. If \(\widehat\tau_j(h)\) is an
effect path recovered under an estimated or misspecified SCM and
\(\tau_j^\star(h)\) is the known truth, the reported path error is

\[
\operatorname{RMSE}_j=
\left[
\frac1H\sum_{h=1}^H
\{\widehat\tau_j(h)-\tau_j^\star(h)\}^2
\right]^{1/2},
\]

and tail-effect error is
\(\left|\widehat\tau_j^{\mathrm{tail}}-\tau_j^{\mathrm{tail},\star}\right|\).
Exact oracle recovery is expected by construction and validates the numerical
AAP implementation. Misspecification error measures dependence on structural
assumptions.

No equation in this section identifies a causal effect in the observed public
financial panel.

## 9. Error propagation

For a fixed state \(k\), define the affine observation-space map

\[
T_k(f,e)=\alpha_k+B_kf+L_ke.
\]

Let \(\zeta=(f,e)\), \(C_k=[B_k,L_k]\), and let \(P_\zeta,Q_\zeta\) be joint
factor/residual distributions under the same specified ground norm. The general
affine pushforward bound is

\[
W_1(T_{k\#}P_\zeta,T_{k\#}Q_\zeta)
\le
\|C_k\|_{\mathrm{op}}W_1(P_\zeta,Q_\zeta).
\]

Only under an additional product-coupling assumption
\(P_\zeta=P_f\otimes P_e\), \(Q_\zeta=Q_f\otimes Q_e\), together with a
compatible additive product ground cost, can this be separated as

\[
W_1(T_{k\#}P_\zeta,T_{k\#}Q_\zeta)
\le
\|B_k\|_{\mathrm{op}}W_1(P_f,Q_f)
+\|L_k\|_{\mathrm{op}}W_1(P_e,Q_e).
\]

Subject to those coupling and norm conventions, a heuristic diagnostic budget
with state and parameter error is

\[
W_1(P_y,Q_y)
\lesssim
B_{\max}\delta_f+L_{\max}\delta_e+M\delta_z
+E\|(\widehat B-B)f\|
+E\|(\widehat L-L)e\|.
\]

The latter display is a diagnostic decomposition, not a proved global theorem.
These are observation/log-return-space diagnostics only. They are not a global
end-to-end bound after the nonlinear \(\operatorname{expm1}\) transform,
multi-period compounding, and portfolio optimization. A simple-return bound would
require restricting \(y\) to a bounded domain or otherwise controlling the
exponential map.

Separately, if \(P_G,Q_G\) are already distributions of asset-level cumulative
simple returns, fixed-\(w\) linear portfolio loss is
\(\|w\|_*\)-Lipschitz, implying

\[
\left|
\operatorname{CVaR}_\alpha^P(L_w)
-\operatorname{CVaR}_\alpha^Q(L_w)
\right|
\le
\frac{\|w\|_*}{1-\alpha}W_1(P_G,Q_G).
\]

This second statement does not connect \(P_G,Q_G\) to the factor-space bound
without additional assumptions. It concerns objective-value sensitivity, not the
stability of the optimizer \(w^\star\). Position and turnover constraints are
therefore substantive stability controls.

## 10. Target extensions not implemented

The complete research target would replace plug-in estimates with

\[
\omega^{(m)}\sim p(\omega\mid D_{\mathrm{train}})
\]

and integrate over posterior parameter draws, jointly consistent factor/state
paths, Student-\(t\) residuals, and a hierarchical observation map. It would also
add:

- fixed-versus-regime mapping and direct-asset diffusion ablations;
- a tail-conditioned mixture with inverse-probability correction;
- regime-hierarchical POT/GPD;
- sequential conformal VaR calibration and ESR diagnostics;
- validation-calibrated Wasserstein radii; and
- a separately identified real-data causal design.

These equations are design targets. They are not descriptions of the current
release.

## 11. Experimental boundaries

- \(N=15\), \(p=10\), \(q=4\), \(K=4\), and \(H=20\) in the reported public core.
- All reported model comparisons use validation data only.
- Stage 0 refits a rolling 1,500-row window; Stage 1 freezes train parameters and
  filters forward. Their score difference combines estimator and update-policy
  differences.
- Stage 2 has four reporting origins and is an engineering pilot.
- Phase 0 constructs and quality-checks the post-2019 rows, but they remain
  governed-excluded from estimator fitting, checkpoint selection, model scoring,
  and portfolio evaluation.
