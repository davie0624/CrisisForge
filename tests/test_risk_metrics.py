from __future__ import annotations

import numpy as np

from crisisforge.risk import (
    aggregate_path_returns,
    brier_score,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    co_crash_probability,
    empirical_expected_shortfall,
    empirical_var,
    energy_score,
    fit_co_crash_thresholds,
    joint_var_es_score,
    kupiec_unconditional_coverage_test,
    portfolio_losses,
    realized_co_crash,
    variogram_score,
)


def test_path_aggregation_and_portfolio_loss() -> None:
    paths = np.array(
        [
            [[0.10, 0.00], [-0.05, 0.02]],
            [[-0.10, -0.20], [0.00, 0.10]],
        ]
    )
    cumulative = aggregate_path_returns(paths, return_type="simple")
    losses = portfolio_losses(cumulative, np.array([0.5, 0.5]))
    assert np.allclose(cumulative[0], [0.045, 0.02])
    assert np.isclose(losses[0], -0.0325)


def test_empirical_var_and_expected_shortfall() -> None:
    losses = np.arange(1.0, 11.0)
    assert empirical_var(losses, 0.8) == 9.0
    assert np.isclose(empirical_expected_shortfall(losses, 0.8), 9.5)


def test_co_crash_probability() -> None:
    returns = np.array(
        [
            [-0.20, -0.30, 0.01],
            [-0.20, -0.30, -0.40],
            [0.01, 0.02, 0.03],
        ]
    )
    probability = co_crash_probability(
        returns,
        np.array([-0.10, -0.10, -0.10]),
        minimum_fraction=2.0 / 3.0,
    )
    assert np.isclose(probability, 2.0 / 3.0)


def test_kupiec_and_brier_outputs_are_valid() -> None:
    result = kupiec_unconditional_coverage_test(
        np.array([False] * 95 + [True] * 5),
        confidence_level=0.95,
    )
    assert np.isclose(result["observed_violation_rate"], 0.05)
    assert result["p_value"] > 0.99
    assert np.isclose(
        brier_score(np.array([0.1, 0.8]), np.array([0.0, 1.0])),
        0.025,
    )


def test_christoffersen_tests_return_finite_statistics() -> None:
    violations = np.array(
        [False, False, True, False, False, False, True, False, False, False]
    )
    independence = christoffersen_independence_test(violations)
    conditional = christoffersen_conditional_coverage_test(
        violations,
        confidence_level=0.8,
    )
    assert independence["lr_ind"] >= 0.0
    assert 0.0 <= independence["p_value"] <= 1.0
    assert conditional["lr_cc"] >= conditional["lr_uc"]
    assert 0.0 <= conditional["p_value_cc"] <= 1.0


def test_training_fixed_co_crash_thresholds_and_outcome() -> None:
    training = np.array(
        [
            [-0.30, -0.10, 0.00],
            [-0.20, -0.20, 0.01],
            [-0.10, 0.00, -0.10],
            [0.00, 0.10, 0.20],
        ]
    )
    thresholds = fit_co_crash_thresholds(training, marginal_quantile=0.25)
    assert realized_co_crash(
        np.array([-0.40, -0.40, 0.10]),
        thresholds,
        minimum_fraction=2.0 / 3.0,
    )


def test_proper_multivariate_scores_reward_exact_scenarios() -> None:
    observation = np.array([0.1, -0.2])
    exact = np.repeat(observation[None, :], 20, axis=0)
    shifted = exact + 0.5
    assert energy_score(exact, observation) < energy_score(shifted, observation)
    assert variogram_score(exact, observation) <= variogram_score(
        shifted,
        observation,
    )


def test_joint_var_es_score_prefers_calibrated_forecast_on_tail_sample() -> None:
    losses = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
    calibrated = joint_var_es_score(
        losses,
        np.repeat(0.3, len(losses)),
        np.repeat(1.0, len(losses)),
        confidence_level=0.8,
    )
    poor = joint_var_es_score(
        losses,
        np.repeat(0.1, len(losses)),
        np.repeat(0.2, len(losses)),
        confidence_level=0.8,
    )
    assert calibrated < poor
