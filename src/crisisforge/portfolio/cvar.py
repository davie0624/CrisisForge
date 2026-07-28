"""CVaR portfolio decisions solved as auditable linear programs.

The Wasserstein formulation uses a 1-Wasserstein ball on asset-return
scenarios with the ground metric ``||x-y||_1``.  Its dual norm is
``||w||_infinity``.  For the portfolio loss ``-w.T @ r`` and an unbounded
return support, strong duality gives the robust CVaR objective

    empirical_CVaR_alpha(w) + rho * ||w||_infinity / (1 - alpha).

The infinity norm is linearized with an epigraph variable.  Transaction
costs and turnover use the explicit L1 definition ``sum(abs(w-w_previous))``;
the commonly used one-way turnover convention would divide this quantity by
two and is intentionally not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog


class PortfolioOptimizationError(RuntimeError):
    """Raised when the linear program is infeasible or fails to solve."""


@dataclass(frozen=True)
class SolverDiagnostics:
    """Minimal HiGHS diagnostics retained for the experiment registry."""

    success: bool
    status: int
    message: str
    iterations: int | None
    crossover_iterations: int | None
    equality_residual_max: float | None
    inequality_slack_min: float | None


@dataclass(frozen=True)
class CVaRPortfolioResult:
    """An optimized portfolio and a decomposition of its objective."""

    weights: np.ndarray
    confidence_level: float
    var_threshold: float
    empirical_cvar: float
    robust_penalty: float
    transaction_cost: float
    objective_value: float
    l1_turnover: float
    wasserstein_radius: float
    transport_norm: str
    dual_norm: str
    diagnostics: SolverDiagnostics


@dataclass(frozen=True)
class _ValidatedInputs:
    returns: np.ndarray
    confidence_level: float
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    previous_weights: np.ndarray | None
    turnover_limit: float | None
    transaction_cost_rates: np.ndarray


def _as_asset_vector(
    value: float | np.ndarray,
    *,
    assets: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim == 0:
        vector = np.repeat(float(vector), assets)
    if vector.shape != (assets,):
        raise ValueError(f"{name} must be a scalar or have shape ({assets},)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _validate_inputs(
    scenario_returns: np.ndarray,
    *,
    confidence_level: float,
    lower_bounds: float | np.ndarray,
    upper_bounds: float | np.ndarray,
    previous_weights: np.ndarray | None,
    turnover_limit: float | None,
    transaction_cost_rates: float | np.ndarray,
) -> _ValidatedInputs:
    returns = np.asarray(scenario_returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError("scenario_returns must have shape (scenarios, assets)")
    scenarios, assets = returns.shape
    if scenarios < 2 or assets < 1:
        raise ValueError("at least two scenarios and one asset are required")
    if not np.isfinite(returns).all():
        raise ValueError("scenario_returns contain non-finite values")
    if (returns < -1.0).any():
        raise ValueError("simple scenario returns cannot be below -1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    lower = _as_asset_vector(lower_bounds, assets=assets, name="lower_bounds")
    upper = _as_asset_vector(upper_bounds, assets=assets, name="upper_bounds")
    if (lower < 0.0).any():
        raise ValueError("lower_bounds must be non-negative for a long-only portfolio")
    if (upper < lower).any():
        raise ValueError("upper_bounds must be greater than or equal to lower_bounds")
    tolerance = 1e-12
    if lower.sum() > 1.0 + tolerance or upper.sum() < 1.0 - tolerance:
        raise ValueError("position bounds are incompatible with full investment")

    previous: np.ndarray | None = None
    if previous_weights is not None:
        previous = _as_asset_vector(
            previous_weights,
            assets=assets,
            name="previous_weights",
        )
        if (previous < -tolerance).any() or not np.isclose(
            previous.sum(),
            1.0,
            atol=1e-8,
        ):
            raise ValueError("previous_weights must be a long-only fully invested vector")
    if turnover_limit is not None:
        if previous is None:
            raise ValueError("turnover_limit requires previous_weights")
        if not np.isfinite(turnover_limit) or turnover_limit < 0.0:
            raise ValueError("turnover_limit must be finite and non-negative")

    costs = _as_asset_vector(
        transaction_cost_rates,
        assets=assets,
        name="transaction_cost_rates",
    )
    if (costs < 0.0).any():
        raise ValueError("transaction_cost_rates must be non-negative")
    if costs.any() and previous is None:
        raise ValueError("non-zero transaction costs require previous_weights")

    return _ValidatedInputs(
        returns=returns,
        confidence_level=float(confidence_level),
        lower_bounds=lower,
        upper_bounds=upper,
        previous_weights=previous,
        turnover_limit=turnover_limit,
        transaction_cost_rates=costs,
    )


def _solver_diagnostics(result: Any) -> SolverDiagnostics:
    equality_residual = None
    inequality_slack = None
    if getattr(result, "eqlin", None) is not None:
        residual = np.asarray(result.eqlin.residual, dtype=float)
        if residual.size:
            equality_residual = float(np.max(np.abs(residual)))
    if getattr(result, "ineqlin", None) is not None:
        slack = np.asarray(result.ineqlin.residual, dtype=float)
        if slack.size:
            inequality_slack = float(np.min(slack))
    return SolverDiagnostics(
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nit) if getattr(result, "nit", None) is not None else None,
        crossover_iterations=(
            int(result.crossover_nit)
            if getattr(result, "crossover_nit", None) is not None
            else None
        ),
        equality_residual_max=equality_residual,
        inequality_slack_min=inequality_slack,
    )


def _solve_cvar_lp(
    scenario_returns: np.ndarray,
    *,
    confidence_level: float,
    lower_bounds: float | np.ndarray,
    upper_bounds: float | np.ndarray,
    previous_weights: np.ndarray | None,
    turnover_limit: float | None,
    transaction_cost_rates: float | np.ndarray,
    wasserstein_radius: float,
) -> CVaRPortfolioResult:
    validated = _validate_inputs(
        scenario_returns,
        confidence_level=confidence_level,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        previous_weights=previous_weights,
        turnover_limit=turnover_limit,
        transaction_cost_rates=transaction_cost_rates,
    )
    if not np.isfinite(wasserstein_radius) or wasserstein_radius < 0.0:
        raise ValueError("wasserstein_radius must be finite and non-negative")

    returns = validated.returns
    scenarios, assets = returns.shape
    has_turnover_variables = validated.previous_weights is not None
    has_robust_epigraph = wasserstein_radius > 0.0

    weight_slice = slice(0, assets)
    eta_index = assets
    excess_slice = slice(assets + 1, assets + 1 + scenarios)
    next_index = excess_slice.stop
    turnover_slice: slice | None = None
    if has_turnover_variables:
        turnover_slice = slice(next_index, next_index + assets)
        next_index = turnover_slice.stop
    robust_epigraph_index: int | None = None
    if has_robust_epigraph:
        robust_epigraph_index = next_index
        next_index += 1
    variable_count = next_index

    objective = np.zeros(variable_count, dtype=float)
    objective[eta_index] = 1.0
    objective[excess_slice] = 1.0 / (scenarios * (1.0 - validated.confidence_level))
    if turnover_slice is not None:
        objective[turnover_slice] = validated.transaction_cost_rates
    if robust_epigraph_index is not None:
        objective[robust_epigraph_index] = wasserstein_radius / (1.0 - validated.confidence_level)

    inequality_rows: list[np.ndarray] = []
    inequality_rhs: list[float] = []

    # -r_i.T w - eta - u_i <= 0
    for scenario in range(scenarios):
        row = np.zeros(variable_count, dtype=float)
        row[weight_slice] = -returns[scenario]
        row[eta_index] = -1.0
        row[excess_slice.start + scenario] = -1.0
        inequality_rows.append(row)
        inequality_rhs.append(0.0)

    if turnover_slice is not None:
        assert validated.previous_weights is not None
        for asset in range(assets):
            positive = np.zeros(variable_count, dtype=float)
            positive[asset] = 1.0
            positive[turnover_slice.start + asset] = -1.0
            inequality_rows.append(positive)
            inequality_rhs.append(float(validated.previous_weights[asset]))

            negative = np.zeros(variable_count, dtype=float)
            negative[asset] = -1.0
            negative[turnover_slice.start + asset] = -1.0
            inequality_rows.append(negative)
            inequality_rhs.append(float(-validated.previous_weights[asset]))

        if validated.turnover_limit is not None:
            turnover_row = np.zeros(variable_count, dtype=float)
            turnover_row[turnover_slice] = 1.0
            inequality_rows.append(turnover_row)
            inequality_rhs.append(float(validated.turnover_limit))

    if robust_epigraph_index is not None:
        # s >= |w_j| linearizes ||w||_infinity.  The negative branch is
        # redundant under long-only bounds but retained to state the dual norm
        # faithfully and to keep the formulation auditable.
        for asset in range(assets):
            positive = np.zeros(variable_count, dtype=float)
            positive[asset] = 1.0
            positive[robust_epigraph_index] = -1.0
            inequality_rows.append(positive)
            inequality_rhs.append(0.0)

            negative = np.zeros(variable_count, dtype=float)
            negative[asset] = -1.0
            negative[robust_epigraph_index] = -1.0
            inequality_rows.append(negative)
            inequality_rhs.append(0.0)

    equality = np.zeros((1, variable_count), dtype=float)
    equality[0, weight_slice] = 1.0

    bounds: list[tuple[float | None, float | None]] = [
        (float(validated.lower_bounds[j]), float(validated.upper_bounds[j])) for j in range(assets)
    ]
    bounds.append((None, None))
    bounds.extend([(0.0, None)] * scenarios)
    if turnover_slice is not None:
        bounds.extend([(0.0, None)] * assets)
    if robust_epigraph_index is not None:
        bounds.append((0.0, None))

    result = linprog(
        objective,
        A_ub=np.asarray(inequality_rows, dtype=float),
        b_ub=np.asarray(inequality_rhs, dtype=float),
        A_eq=equality,
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )
    diagnostics = _solver_diagnostics(result)
    if not result.success:
        raise PortfolioOptimizationError(
            f"CVaR linear program failed (status={diagnostics.status}): {diagnostics.message}"
        )

    solution = np.asarray(result.x, dtype=float)
    weights = solution[weight_slice].copy()
    eta = float(solution[eta_index])
    losses = -returns @ weights
    empirical_cvar = float(
        eta + np.maximum(losses - eta, 0.0).sum() / (scenarios * (1.0 - validated.confidence_level))
    )
    robust_penalty = float(
        wasserstein_radius * np.max(np.abs(weights)) / (1.0 - validated.confidence_level)
    )
    if validated.previous_weights is None:
        l1_turnover = 0.0
        transaction_cost = 0.0
    else:
        absolute_trade = np.abs(weights - validated.previous_weights)
        l1_turnover = float(absolute_trade.sum())
        transaction_cost = float(np.dot(validated.transaction_cost_rates, absolute_trade))
    decomposed_objective = empirical_cvar + robust_penalty + transaction_cost

    return CVaRPortfolioResult(
        weights=weights,
        confidence_level=validated.confidence_level,
        var_threshold=eta,
        empirical_cvar=empirical_cvar,
        robust_penalty=robust_penalty,
        transaction_cost=transaction_cost,
        objective_value=decomposed_objective,
        l1_turnover=l1_turnover,
        wasserstein_radius=float(wasserstein_radius),
        transport_norm="l1",
        dual_norm="linfinity",
        diagnostics=diagnostics,
    )


def solve_empirical_cvar(
    scenario_returns: np.ndarray,
    *,
    confidence_level: float = 0.95,
    lower_bounds: float | np.ndarray = 0.0,
    upper_bounds: float | np.ndarray = 1.0,
    previous_weights: np.ndarray | None = None,
    turnover_limit: float | None = None,
    transaction_cost_rates: float | np.ndarray = 0.0,
) -> CVaRPortfolioResult:
    """Minimize empirical CVaR for a fully invested long-only portfolio."""
    return _solve_cvar_lp(
        scenario_returns,
        confidence_level=confidence_level,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        previous_weights=previous_weights,
        turnover_limit=turnover_limit,
        transaction_cost_rates=transaction_cost_rates,
        wasserstein_radius=0.0,
    )


def solve_wasserstein_robust_cvar(
    scenario_returns: np.ndarray,
    *,
    wasserstein_radius: float,
    confidence_level: float = 0.95,
    lower_bounds: float | np.ndarray = 0.0,
    upper_bounds: float | np.ndarray = 1.0,
    previous_weights: np.ndarray | None = None,
    turnover_limit: float | None = None,
    transaction_cost_rates: float | np.ndarray = 0.0,
) -> CVaRPortfolioResult:
    """Minimize 1-Wasserstein robust CVaR using its strong-dual LP.

    ``wasserstein_radius`` is measured in the same return units as the scenario
    rows.  The ground transport cost is L1 and the dual portfolio norm is
    L-infinity.  Setting the radius to zero intentionally dispatches to the
    empirical LP so that the limiting case has identical numerical behavior.
    """
    if wasserstein_radius == 0.0:
        return solve_empirical_cvar(
            scenario_returns,
            confidence_level=confidence_level,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            previous_weights=previous_weights,
            turnover_limit=turnover_limit,
            transaction_cost_rates=transaction_cost_rates,
        )
    return _solve_cvar_lp(
        scenario_returns,
        confidence_level=confidence_level,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        previous_weights=previous_weights,
        turnover_limit=turnover_limit,
        transaction_cost_rates=transaction_cost_rates,
        wasserstein_radius=wasserstein_radius,
    )
