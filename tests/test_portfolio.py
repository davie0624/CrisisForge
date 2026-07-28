from __future__ import annotations

import numpy as np
import pytest

from crisisforge.portfolio import (
    solve_empirical_cvar,
    solve_wasserstein_robust_cvar,
)


@pytest.fixture
def scenario_returns() -> np.ndarray:
    return np.array(
        [
            [0.020, 0.010, -0.010],
            [-0.010, 0.005, 0.000],
            [0.030, 0.000, -0.020],
            [-0.040, 0.004, 0.010],
            [0.010, 0.003, 0.000],
            [-0.020, 0.006, 0.005],
        ],
        dtype=float,
    )


def test_empirical_cvar_respects_simplex_and_position_limits(
    scenario_returns: np.ndarray,
) -> None:
    result = solve_empirical_cvar(
        scenario_returns,
        confidence_level=0.80,
        lower_bounds=np.array([0.10, 0.00, 0.00]),
        upper_bounds=np.array([0.60, 0.70, 0.60]),
    )
    assert result.diagnostics.success
    assert np.isclose(result.weights.sum(), 1.0)
    assert (result.weights >= np.array([0.10, 0.00, 0.00]) - 1e-10).all()
    assert (result.weights <= np.array([0.60, 0.70, 0.60]) + 1e-10).all()
    assert result.transport_norm == "l1"
    assert result.dual_norm == "linfinity"
    assert result.robust_penalty == 0.0


def test_turnover_and_transaction_cost_are_explicit_l1(
    scenario_returns: np.ndarray,
) -> None:
    previous = np.array([0.80, 0.10, 0.10])
    cost_rate = np.array([0.001, 0.002, 0.0015])
    result = solve_empirical_cvar(
        scenario_returns,
        confidence_level=0.80,
        previous_weights=previous,
        turnover_limit=0.20,
        transaction_cost_rates=cost_rate,
    )
    absolute_trade = np.abs(result.weights - previous)
    assert result.l1_turnover <= 0.20 + 1e-9
    assert np.isclose(result.l1_turnover, absolute_trade.sum())
    assert np.isclose(result.transaction_cost, np.dot(cost_rate, absolute_trade))
    assert np.isclose(
        result.objective_value,
        result.empirical_cvar + result.transaction_cost,
    )


def test_zero_wasserstein_radius_is_exact_empirical_limit(
    scenario_returns: np.ndarray,
) -> None:
    empirical = solve_empirical_cvar(
        scenario_returns,
        confidence_level=0.80,
        upper_bounds=0.70,
    )
    robust_zero = solve_wasserstein_robust_cvar(
        scenario_returns,
        wasserstein_radius=0.0,
        confidence_level=0.80,
        upper_bounds=0.70,
    )
    assert np.array_equal(robust_zero.weights, empirical.weights)
    assert robust_zero.objective_value == empirical.objective_value
    assert robust_zero.robust_penalty == 0.0


def test_robust_objective_is_nondecreasing_in_radius(
    scenario_returns: np.ndarray,
) -> None:
    radii = [0.0, 0.0005, 0.0010, 0.0030]
    results = [
        solve_wasserstein_robust_cvar(
            scenario_returns,
            wasserstein_radius=radius,
            confidence_level=0.80,
            upper_bounds=0.70,
        )
        for radius in radii
    ]
    objectives = np.array([result.objective_value for result in results])
    assert (np.diff(objectives) >= -1e-10).all()
    for radius, result in zip(radii[1:], results[1:], strict=True):
        expected_penalty = (
            radius * np.max(np.abs(result.weights)) / (1.0 - 0.80)
        )
        assert np.isclose(result.robust_penalty, expected_penalty)


def test_invalid_turnover_request_fails_before_solver(
    scenario_returns: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="requires previous_weights"):
        solve_empirical_cvar(
            scenario_returns,
            turnover_limit=0.20,
        )


def test_robust_lp_matches_direct_grid_objective(
    scenario_returns: np.ndarray,
) -> None:
    confidence = 0.80
    radius = 0.002
    result = solve_wasserstein_robust_cvar(
        scenario_returns[:, :2],
        wasserstein_radius=radius,
        confidence_level=confidence,
    )

    grid = np.linspace(0.0, 1.0, 10001)
    direct_objectives = np.empty_like(grid)
    for index, first_weight in enumerate(grid):
        weights = np.array([first_weight, 1.0 - first_weight])
        losses = -scenario_returns[:, :2] @ weights
        candidates = np.unique(losses)
        empirical = min(
            eta
            + np.maximum(losses - eta, 0.0).mean() / (1.0 - confidence)
            for eta in candidates
        )
        direct_objectives[index] = (
            empirical
            + radius * np.max(np.abs(weights)) / (1.0 - confidence)
        )

    assert result.objective_value <= direct_objectives.min() + 2e-6
    assert np.isclose(
        result.robust_penalty,
        radius * np.max(result.weights) / (1.0 - confidence),
    )


def test_simple_return_domain_is_enforced() -> None:
    invalid = np.array([[0.0, 0.1], [-1.01, 0.2]])
    with pytest.raises(ValueError, match="cannot be below -1"):
        solve_empirical_cvar(invalid)
