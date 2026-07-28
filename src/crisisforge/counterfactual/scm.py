"""A time-unrolled, regime-switching semi-synthetic market SCM.

This module provides known counterfactual ground truth for method validation.
It is not an identification strategy for observed financial-market data and
must not be used to claim that an intervention has a real-world causal effect.

Within a time slice the structural order is

    policy -> yield -> liquidity -> credit -> equity -> volatility.

The economically important feedback edge runs from ``equity[t-1]`` to
``policy[t]``.  Time unrolling therefore preserves feedback without creating a
cycle in the directed acyclic graph.  Regimes follow an exogenous Markov chain.
Paired simulations reuse every exogenous innovation and regime draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

VARIABLES = (
    "policy",
    "yield",
    "credit",
    "liquidity",
    "equity",
    "volatility",
)
_INNOVATION_INDEX = {name: index for index, name in enumerate(VARIABLES)}
_CONTROLLED_VARIABLES = frozenset({"yield", "credit", "liquidity", "volatility"})


def _softplus(value: np.ndarray | float) -> np.ndarray | float:
    return np.logaddexp(0.0, value)


def _inverse_softplus(value: np.ndarray | float) -> np.ndarray | float:
    positive = np.asarray(value, dtype=float)
    if (positive <= 0.0).any():
        raise ValueError("volatility must be strictly positive")
    result = positive + np.log(-np.expm1(-positive))
    if np.ndim(value) == 0:
        return float(result)
    return result


@dataclass(frozen=True)
class SCMParameters:
    """Ground-truth structural coefficients in standardized research units."""

    transition_matrix: tuple[tuple[float, ...], ...] = (
        (0.965, 0.020, 0.015),
        (0.055, 0.925, 0.020),
        (0.045, 0.020, 0.935),
    )
    initial_regime_probabilities: tuple[float, ...] = (0.80, 0.10, 0.10)

    policy_regime_intercept: tuple[float, ...] = (0.00, -0.10, 0.25)
    yield_regime_intercept: tuple[float, ...] = (0.00, -0.04, 0.20)
    liquidity_regime_intercept: tuple[float, ...] = (0.00, 0.28, 0.10)
    credit_regime_intercept: tuple[float, ...] = (0.00, 0.38, 0.14)
    equity_regime_intercept: tuple[float, ...] = (0.00, -0.22, -0.12)
    volatility_regime_intercept: tuple[float, ...] = (-1.55, -1.00, -1.25)

    policy_ar: float = 0.72
    policy_equity_feedback: float = 0.18
    yield_ar: float = 0.68
    yield_policy: float = 0.62
    liquidity_ar: float = 0.58
    liquidity_yield: float = 0.28
    credit_ar: float = 0.64
    credit_yield: float = 0.22
    credit_liquidity: float = 0.44
    equity_ar: float = 0.22
    equity_credit: float = 0.42
    equity_liquidity: float = 0.32
    equity_volatility: float = 0.14
    volatility_ar: float = 0.60
    volatility_equity: float = 0.38
    volatility_credit: float = 0.22
    volatility_liquidity: float = 0.18

    innovation_scales: tuple[float, ...] = (0.04, 0.04, 0.05, 0.05, 0.08, 0.06)
    initial_state: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.20)

    def __post_init__(self) -> None:
        transition = np.asarray(self.transition_matrix, dtype=float)
        regime_count = transition.shape[0]
        if transition.ndim != 2 or transition.shape[1] != regime_count:
            raise ValueError("transition_matrix must be square")
        if (transition < 0.0).any() or not np.allclose(
            transition.sum(axis=1),
            1.0,
        ):
            raise ValueError("transition_matrix rows must be probability vectors")
        initial = np.asarray(self.initial_regime_probabilities, dtype=float)
        if initial.shape != (regime_count,) or (initial < 0.0).any():
            raise ValueError("initial_regime_probabilities have the wrong shape")
        if not np.isclose(initial.sum(), 1.0):
            raise ValueError("initial_regime_probabilities must sum to one")
        regime_intercepts = (
            self.policy_regime_intercept,
            self.yield_regime_intercept,
            self.liquidity_regime_intercept,
            self.credit_regime_intercept,
            self.equity_regime_intercept,
            self.volatility_regime_intercept,
        )
        if any(len(values) != regime_count for values in regime_intercepts):
            raise ValueError("every regime-intercept vector must match the regime count")
        scales = np.asarray(self.innovation_scales, dtype=float)
        if scales.shape != (len(VARIABLES),) or (scales <= 0.0).any():
            raise ValueError("innovation_scales must contain six positive values")
        initial_state = np.asarray(self.initial_state, dtype=float)
        if initial_state.shape != (len(VARIABLES),):
            raise ValueError("initial_state must contain six values")
        if initial_state[_INNOVATION_INDEX["volatility"]] <= 0.0:
            raise ValueError("initial volatility must be strictly positive")


@dataclass(frozen=True)
class Intervention:
    """A policy action and optional controlled-mediator assignments.

    ``policy`` encodes ``do(policy[t] = value)``.  Assignments in
    ``controlled`` hold named mediators fixed and therefore define a controlled,
    rather than total, effect.  Missing time points follow their structural
    equations.
    """

    policy: dict[int, float]
    controlled: dict[str, dict[int, float]] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        _validate_schedule(self.policy, name="policy")
        unknown = set(self.controlled).difference(_CONTROLLED_VARIABLES)
        if unknown:
            raise ValueError(f"unsupported controlled variables: {sorted(unknown)}")
        for variable, schedule in self.controlled.items():
            _validate_schedule(schedule, name=variable)
            if variable == "volatility" and any(value <= 0.0 for value in schedule.values()):
                raise ValueError("controlled volatility must be strictly positive")

    @property
    def effect_type(self) -> Literal["total", "controlled"]:
        return "controlled" if self.controlled else "total"


def _validate_schedule(schedule: dict[int, float], *, name: str) -> None:
    for time, value in schedule.items():
        if isinstance(time, bool) or not isinstance(time, int) or time < 0:
            raise ValueError(f"{name} intervention times must be non-negative integers")
        if not np.isfinite(value):
            raise ValueError(f"{name} intervention values must be finite")


@dataclass(frozen=True)
class SCMExogenous:
    """Exogenous innovations used to construct paired potential outcomes."""

    innovations: np.ndarray
    regime_uniforms: np.ndarray
    initial_regimes: np.ndarray


@dataclass(frozen=True)
class AbductedExogenous:
    """Exogenous state inferred from an observational semi-synthetic path."""

    innovations: np.ndarray
    regimes: np.ndarray
    initial_state: np.ndarray


@dataclass(frozen=True)
class SCMPaths:
    """Simulated paths; all continuous arrays have shape (paths, horizon)."""

    policy: np.ndarray
    yield_rate: np.ndarray
    credit: np.ndarray
    liquidity: np.ndarray
    equity: np.ndarray
    volatility: np.ndarray
    regime: np.ndarray
    initial_state: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.policy.shape

    def variable(self, name: str) -> np.ndarray:
        """Return a named path; ``yield`` maps to the ``yield_rate`` field."""
        if name not in VARIABLES:
            raise ValueError(f"unknown variable: {name}")
        return self.yield_rate if name == "yield" else getattr(self, name)

    def as_array(self) -> np.ndarray:
        """Stack variables in the public ``VARIABLES`` order."""
        return np.stack([self.variable(name) for name in VARIABLES], axis=-1)


@dataclass(frozen=True)
class PairedSCMPaths:
    """Factual and interventional paths sharing all exogenous noise."""

    factual: SCMPaths
    counterfactual: SCMPaths
    intervention: Intervention


@dataclass(frozen=True)
class CausalEffect:
    """Paired total or controlled effect on a selected outcome."""

    effect_type: Literal["total", "controlled"]
    outcome: str
    mean_path: np.ndarray
    ate_terminal: float
    cumulative_ate: float
    tail_effect: float


@dataclass(frozen=True)
class CounterfactualError:
    """Errors against known semi-synthetic counterfactual ground truth."""

    ate_error: float
    path_rmse: float
    tail_effect_error: float


class RegimeSwitchingSCM:
    """Known-ground-truth simulator with a time-unrolled structural graph."""

    def __init__(self, parameters: SCMParameters | None = None) -> None:
        self.parameters = parameters or SCMParameters()

    @property
    def regime_count(self) -> int:
        return len(self.parameters.transition_matrix)

    def generate_exogenous(
        self,
        *,
        num_paths: int,
        horizon: int,
        seed: int,
    ) -> SCMExogenous:
        """Draw reproducible structural innovations and Markov uniforms."""
        if num_paths < 1 or horizon < 1:
            raise ValueError("num_paths and horizon must be positive")
        rng = np.random.default_rng(seed)
        scales = np.asarray(self.parameters.innovation_scales, dtype=float)
        innovations = rng.normal(size=(num_paths, horizon, len(VARIABLES))) * scales
        regime_uniforms = rng.random((num_paths, horizon))
        initial_regimes = rng.choice(
            self.regime_count,
            size=num_paths,
            p=np.asarray(self.parameters.initial_regime_probabilities),
        )
        return SCMExogenous(
            innovations=innovations,
            regime_uniforms=regime_uniforms,
            initial_regimes=initial_regimes,
        )

    def simulate(
        self,
        exogenous: SCMExogenous,
        intervention: Intervention | None = None,
    ) -> SCMPaths:
        """Simulate a factual or interventional world from fixed exogenous noise."""
        innovations, uniforms, initial_regimes = _validate_exogenous(
            exogenous,
            regime_count=self.regime_count,
        )
        regimes = self._draw_regimes(uniforms, initial_regimes)
        initial = np.repeat(
            np.asarray(self.parameters.initial_state, dtype=float)[None, :],
            innovations.shape[0],
            axis=0,
        )
        return self._simulate_with_regimes(
            innovations,
            regimes,
            initial,
            intervention=intervention,
        )

    def paired_simulation(
        self,
        *,
        num_paths: int,
        horizon: int,
        intervention: Intervention,
        seed: int,
    ) -> PairedSCMPaths:
        """Generate paired factual/interventional paths with common random numbers."""
        noise = self.generate_exogenous(
            num_paths=num_paths,
            horizon=horizon,
            seed=seed,
        )
        return PairedSCMPaths(
            factual=self.simulate(noise),
            counterfactual=self.simulate(noise, intervention),
            intervention=intervention,
        )

    def abduct(self, factual: SCMPaths) -> AbductedExogenous:
        """Infer the innovations that reproduce an observational factual path."""
        _validate_paths(factual, regime_count=self.regime_count)
        paths, horizon = factual.shape
        innovations = np.empty((paths, horizon, len(VARIABLES)), dtype=float)
        p = self.parameters

        for time in range(horizon):
            previous = (
                factual.initial_state
                if time == 0
                else np.column_stack(
                    [
                        factual.policy[:, time - 1],
                        factual.yield_rate[:, time - 1],
                        factual.credit[:, time - 1],
                        factual.liquidity[:, time - 1],
                        factual.equity[:, time - 1],
                        factual.volatility[:, time - 1],
                    ]
                )
            )
            regime = factual.regime[:, time]
            policy_mean = (
                p.policy_ar * previous[:, _INNOVATION_INDEX["policy"]]
                + p.policy_equity_feedback
                * previous[:, _INNOVATION_INDEX["equity"]]
                + _regime_values(p.policy_regime_intercept, regime)
            )
            innovations[:, time, _INNOVATION_INDEX["policy"]] = (
                factual.policy[:, time] - policy_mean
            )

            yield_mean = (
                p.yield_ar * previous[:, _INNOVATION_INDEX["yield"]]
                + p.yield_policy * factual.policy[:, time]
                + _regime_values(p.yield_regime_intercept, regime)
            )
            innovations[:, time, _INNOVATION_INDEX["yield"]] = (
                factual.yield_rate[:, time] - yield_mean
            )

            liquidity_mean = (
                p.liquidity_ar * previous[:, _INNOVATION_INDEX["liquidity"]]
                + p.liquidity_yield * factual.yield_rate[:, time]
                + _regime_values(p.liquidity_regime_intercept, regime)
            )
            innovations[:, time, _INNOVATION_INDEX["liquidity"]] = (
                factual.liquidity[:, time] - liquidity_mean
            )

            credit_mean = (
                p.credit_ar * previous[:, _INNOVATION_INDEX["credit"]]
                + p.credit_yield * factual.yield_rate[:, time]
                + p.credit_liquidity * factual.liquidity[:, time]
                + _regime_values(p.credit_regime_intercept, regime)
            )
            innovations[:, time, _INNOVATION_INDEX["credit"]] = (
                factual.credit[:, time] - credit_mean
            )

            equity_mean = (
                p.equity_ar * previous[:, _INNOVATION_INDEX["equity"]]
                - p.equity_credit * factual.credit[:, time]
                - p.equity_liquidity * factual.liquidity[:, time]
                - p.equity_volatility * previous[:, _INNOVATION_INDEX["volatility"]]
                + _regime_values(p.equity_regime_intercept, regime)
            )
            innovations[:, time, _INNOVATION_INDEX["equity"]] = (
                factual.equity[:, time] - equity_mean
            )

            volatility_mean = (
                p.volatility_ar * previous[:, _INNOVATION_INDEX["volatility"]]
                - p.volatility_equity * factual.equity[:, time]
                + p.volatility_credit * factual.credit[:, time]
                + p.volatility_liquidity * factual.liquidity[:, time]
                + _regime_values(p.volatility_regime_intercept, regime)
            )
            innovations[:, time, _INNOVATION_INDEX["volatility"]] = (
                _inverse_softplus(factual.volatility[:, time]) - volatility_mean
            )

        return AbductedExogenous(
            innovations=innovations,
            regimes=factual.regime.copy(),
            initial_state=factual.initial_state.copy(),
        )

    def predict(
        self,
        abducted: AbductedExogenous,
        intervention: Intervention,
    ) -> SCMPaths:
        """Action-prediction step using the abducted common exogenous state."""
        innovations = np.asarray(abducted.innovations, dtype=float)
        raw_regimes = np.asarray(abducted.regimes)
        numeric_regimes = np.asarray(raw_regimes, dtype=float)
        if not np.isfinite(numeric_regimes).all() or not np.equal(
            numeric_regimes,
            np.floor(numeric_regimes),
        ).all():
            raise ValueError("abducted regimes must contain integer labels")
        regimes = numeric_regimes.astype(int)
        initial = np.asarray(abducted.initial_state, dtype=float)
        if innovations.ndim != 3 or innovations.shape[2] != len(VARIABLES):
            raise ValueError("abducted innovations have an invalid shape")
        if regimes.shape != innovations.shape[:2]:
            raise ValueError("abducted regimes have an invalid shape")
        if initial.shape != (innovations.shape[0], len(VARIABLES)):
            raise ValueError("abducted initial_state has an invalid shape")
        if not np.isfinite(innovations).all() or not np.isfinite(initial).all():
            raise ValueError("abducted state contains non-finite values")
        if (regimes < 0).any() or (regimes >= self.regime_count).any():
            raise ValueError("abducted regimes are out of range")
        return self._simulate_with_regimes(
            innovations,
            regimes,
            initial,
            intervention=intervention,
        )

    def abduction_action_prediction(
        self,
        factual: SCMPaths,
        intervention: Intervention,
    ) -> SCMPaths:
        """Run the complete abduction-action-prediction counterfactual workflow."""
        return self.predict(self.abduct(factual), intervention)

    def _draw_regimes(
        self,
        uniforms: np.ndarray,
        initial_regimes: np.ndarray,
    ) -> np.ndarray:
        paths, horizon = uniforms.shape
        transition = np.asarray(self.parameters.transition_matrix, dtype=float)
        regimes = np.empty((paths, horizon), dtype=int)
        previous = initial_regimes.copy()
        for time in range(horizon):
            cumulative = np.cumsum(transition[previous], axis=1)
            current = (uniforms[:, time, None] > cumulative).sum(axis=1)
            current = np.minimum(current, self.regime_count - 1)
            regimes[:, time] = current
            previous = current
        return regimes

    def _simulate_with_regimes(
        self,
        innovations: np.ndarray,
        regimes: np.ndarray,
        initial_state: np.ndarray,
        *,
        intervention: Intervention | None,
    ) -> SCMPaths:
        paths, horizon, _ = innovations.shape
        values = {
            variable: np.empty((paths, horizon), dtype=float) for variable in VARIABLES
        }
        p = self.parameters

        for time in range(horizon):
            previous = (
                initial_state
                if time == 0
                else np.column_stack(
                    [values[variable][:, time - 1] for variable in VARIABLES]
                )
            )
            regime = regimes[:, time]
            policy = (
                p.policy_ar * previous[:, _INNOVATION_INDEX["policy"]]
                + p.policy_equity_feedback
                * previous[:, _INNOVATION_INDEX["equity"]]
                + _regime_values(p.policy_regime_intercept, regime)
                + innovations[:, time, _INNOVATION_INDEX["policy"]]
            )
            policy = _apply_intervention(policy, intervention, "policy", time)
            values["policy"][:, time] = policy

            yield_rate = (
                p.yield_ar * previous[:, _INNOVATION_INDEX["yield"]]
                + p.yield_policy * policy
                + _regime_values(p.yield_regime_intercept, regime)
                + innovations[:, time, _INNOVATION_INDEX["yield"]]
            )
            yield_rate = _apply_intervention(yield_rate, intervention, "yield", time)
            values["yield"][:, time] = yield_rate

            liquidity = (
                p.liquidity_ar * previous[:, _INNOVATION_INDEX["liquidity"]]
                + p.liquidity_yield * yield_rate
                + _regime_values(p.liquidity_regime_intercept, regime)
                + innovations[:, time, _INNOVATION_INDEX["liquidity"]]
            )
            liquidity = _apply_intervention(
                liquidity,
                intervention,
                "liquidity",
                time,
            )
            values["liquidity"][:, time] = liquidity

            credit = (
                p.credit_ar * previous[:, _INNOVATION_INDEX["credit"]]
                + p.credit_yield * yield_rate
                + p.credit_liquidity * liquidity
                + _regime_values(p.credit_regime_intercept, regime)
                + innovations[:, time, _INNOVATION_INDEX["credit"]]
            )
            credit = _apply_intervention(credit, intervention, "credit", time)
            values["credit"][:, time] = credit

            equity = (
                p.equity_ar * previous[:, _INNOVATION_INDEX["equity"]]
                - p.equity_credit * credit
                - p.equity_liquidity * liquidity
                - p.equity_volatility
                * previous[:, _INNOVATION_INDEX["volatility"]]
                + _regime_values(p.equity_regime_intercept, regime)
                + innovations[:, time, _INNOVATION_INDEX["equity"]]
            )
            values["equity"][:, time] = equity

            volatility_latent = (
                p.volatility_ar * previous[:, _INNOVATION_INDEX["volatility"]]
                - p.volatility_equity * equity
                + p.volatility_credit * credit
                + p.volatility_liquidity * liquidity
                + _regime_values(p.volatility_regime_intercept, regime)
                + innovations[:, time, _INNOVATION_INDEX["volatility"]]
            )
            volatility = np.asarray(_softplus(volatility_latent), dtype=float)
            volatility = _apply_intervention(
                volatility,
                intervention,
                "volatility",
                time,
            )
            values["volatility"][:, time] = volatility

        return SCMPaths(
            policy=values["policy"],
            yield_rate=values["yield"],
            credit=values["credit"],
            liquidity=values["liquidity"],
            equity=values["equity"],
            volatility=values["volatility"],
            regime=regimes.copy(),
            initial_state=initial_state.copy(),
        )


def _apply_intervention(
    structural_value: np.ndarray,
    intervention: Intervention | None,
    variable: str,
    time: int,
) -> np.ndarray:
    if intervention is None:
        return structural_value
    if variable == "policy" and time in intervention.policy:
        return np.full_like(structural_value, intervention.policy[time])
    controlled_schedule = intervention.controlled.get(variable, {})
    if time in controlled_schedule:
        return np.full_like(structural_value, controlled_schedule[time])
    return structural_value


def _regime_values(values: tuple[float, ...], regimes: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)[regimes]


def _validate_exogenous(
    exogenous: SCMExogenous,
    *,
    regime_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    innovations = np.asarray(exogenous.innovations, dtype=float)
    uniforms = np.asarray(exogenous.regime_uniforms, dtype=float)
    raw_initial = np.asarray(exogenous.initial_regimes)
    numeric_initial = np.asarray(raw_initial, dtype=float)
    if not np.isfinite(numeric_initial).all() or not np.equal(
        numeric_initial,
        np.floor(numeric_initial),
    ).all():
        raise ValueError("initial_regimes must contain integer labels")
    initial = numeric_initial.astype(int)
    if innovations.ndim != 3 or innovations.shape[2] != len(VARIABLES):
        raise ValueError("innovations must have shape (paths, horizon, 6)")
    if uniforms.shape != innovations.shape[:2]:
        raise ValueError("regime_uniforms must match the path and horizon dimensions")
    if initial.shape != (innovations.shape[0],):
        raise ValueError("initial_regimes must have shape (paths,)")
    if not np.isfinite(innovations).all() or not np.isfinite(uniforms).all():
        raise ValueError("exogenous arrays contain non-finite values")
    if (uniforms < 0.0).any() or (uniforms >= 1.0).any():
        raise ValueError("regime_uniforms must lie in [0, 1)")
    if (initial < 0).any() or (initial >= regime_count).any():
        raise ValueError("initial_regimes are out of range")
    return innovations, uniforms, initial


def _validate_paths(paths: SCMPaths, *, regime_count: int) -> None:
    shape = paths.policy.shape
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("SCM paths must have shape (paths, horizon)")
    for variable in VARIABLES:
        values = paths.variable(variable)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{variable} path has an invalid shape or values")
    if (paths.volatility <= 0.0).any():
        raise ValueError("volatility paths must be strictly positive")
    if paths.regime.shape != shape:
        raise ValueError("regime path has an invalid shape")
    if not np.issubdtype(np.asarray(paths.regime).dtype, np.integer):
        raise ValueError("regime path must contain integer labels")
    if (paths.regime < 0).any() or (paths.regime >= regime_count).any():
        raise ValueError("regime path contains an out-of-range state")
    if paths.initial_state.shape != (shape[0], len(VARIABLES)):
        raise ValueError("initial_state has an invalid shape")
    if not np.isfinite(paths.initial_state).all():
        raise ValueError("initial_state contains non-finite values")
    if (paths.initial_state[:, _INNOVATION_INDEX["volatility"]] <= 0.0).any():
        raise ValueError("initial volatility must be strictly positive")


def _expected_shortfall(losses: np.ndarray, confidence_level: float) -> float:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if losses.ndim != 1 or losses.size < 1 or not np.isfinite(losses).all():
        raise ValueError("losses must be a non-empty finite vector")
    threshold = float(np.quantile(losses, confidence_level, method="higher"))
    return float(threshold + np.maximum(losses - threshold, 0.0).mean() / (1.0 - confidence_level))


def estimate_causal_effect(
    factual: SCMPaths,
    counterfactual: SCMPaths,
    *,
    outcome: str = "equity",
    effect_type: Literal["total", "controlled"] = "total",
    confidence_level: float = 0.95,
    loss_sign: float = -1.0,
) -> CausalEffect:
    """Estimate a paired path effect and the change in path-loss ES.

    With ``loss_sign=-1``, lower cumulative equity outcomes are larger losses.
    The pairing must be created from common exogenous noise.
    """
    factual_values = factual.variable(outcome)
    counterfactual_values = counterfactual.variable(outcome)
    if factual_values.shape != counterfactual_values.shape:
        raise ValueError("factual and counterfactual paths must have the same shape")
    if not np.isfinite(factual_values).all() or not np.isfinite(
        counterfactual_values
    ).all():
        raise ValueError("paired outcome paths must be finite")
    if not np.array_equal(factual.regime, counterfactual.regime):
        raise ValueError(
            "paired effects require identical exogenous regime paths"
        )
    if not np.array_equal(factual.initial_state, counterfactual.initial_state):
        raise ValueError(
            "paired effects require identical initial states"
        )
    if effect_type not in {"total", "controlled"}:
        raise ValueError("effect_type must be 'total' or 'controlled'")
    if not np.isfinite(loss_sign):
        raise ValueError("loss_sign must be finite")
    difference = counterfactual_values - factual_values
    factual_loss = loss_sign * factual_values.sum(axis=1)
    counterfactual_loss = loss_sign * counterfactual_values.sum(axis=1)
    return CausalEffect(
        effect_type=effect_type,
        outcome=outcome,
        mean_path=difference.mean(axis=0),
        ate_terminal=float(difference[:, -1].mean()),
        cumulative_ate=float(difference.sum(axis=1).mean()),
        tail_effect=(
            _expected_shortfall(counterfactual_loss, confidence_level)
            - _expected_shortfall(factual_loss, confidence_level)
        ),
    )


def evaluate_counterfactual_error(
    *,
    factual: SCMPaths,
    estimated_counterfactual: SCMPaths,
    ground_truth_counterfactual: SCMPaths,
    outcome: str = "equity",
    effect_type: Literal["total", "controlled"] = "total",
    confidence_level: float = 0.95,
    loss_sign: float = -1.0,
) -> CounterfactualError:
    """Compare an estimated counterfactual with known semi-synthetic truth."""
    estimated = estimate_causal_effect(
        factual,
        estimated_counterfactual,
        outcome=outcome,
        effect_type=effect_type,
        confidence_level=confidence_level,
        loss_sign=loss_sign,
    )
    truth = estimate_causal_effect(
        factual,
        ground_truth_counterfactual,
        outcome=outcome,
        effect_type=effect_type,
        confidence_level=confidence_level,
        loss_sign=loss_sign,
    )
    return CounterfactualError(
        ate_error=abs(estimated.ate_terminal - truth.ate_terminal),
        path_rmse=float(np.sqrt(np.mean((estimated.mean_path - truth.mean_path) ** 2))),
        tail_effect_error=abs(estimated.tail_effect - truth.tail_effect),
    )
