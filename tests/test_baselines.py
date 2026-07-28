from __future__ import annotations

import numpy as np
import pytest

from crisisforge.baselines import (
    EWMFilteredHistoricalGenerator,
    GaussianScenarioGenerator,
    HistoricalScenarioGenerator,
    MovingBlockBootstrapGenerator,
    StudentTCopulaScenarioGenerator,
    StudentTScenarioGenerator,
    VARResidualBootstrapGenerator,
)


@pytest.fixture
def returns() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(size=(250, 3)) * 0.01


@pytest.mark.parametrize(
    "generator",
    [
        HistoricalScenarioGenerator(),
        MovingBlockBootstrapGenerator(block_length=10),
        GaussianScenarioGenerator(),
        StudentTScenarioGenerator(degrees_of_freedom=6.0),
        EWMFilteredHistoricalGenerator(),
        StudentTCopulaScenarioGenerator(degrees_of_freedom=6.0),
        VARResidualBootstrapGenerator(),
    ],
)
def test_scenario_generators_return_expected_shape(
    generator: object,
    returns: np.ndarray,
) -> None:
    fitted = generator.fit(returns)
    scenarios = fitted.sample(
        num_scenarios=25,
        horizon=20,
        rng=np.random.default_rng(11),
    )
    assert scenarios.shape == (25, 20, 3)
    assert np.isfinite(scenarios).all()


def test_block_bootstrap_uses_contiguous_observations() -> None:
    returns = np.arange(30, dtype=float).reshape(15, 2)
    generator = MovingBlockBootstrapGenerator(block_length=4).fit(returns)
    scenarios = generator.sample(
        num_scenarios=3,
        horizon=4,
        rng=np.random.default_rng(4),
    )
    for scenario in scenarios:
        assert np.all(np.diff(scenario[:, 0]) == 2.0)


def test_parametric_generators_respect_simple_return_domain() -> None:
    observations = np.array(
        [
            [-0.90, 0.02],
            [0.05, -0.03],
            [0.02, 0.01],
            [-0.04, 0.03],
        ]
    )
    for generator in (
        GaussianScenarioGenerator(),
        StudentTScenarioGenerator(degrees_of_freedom=6.0),
        VARResidualBootstrapGenerator(),
    ):
        scenarios = generator.fit(observations).sample(
            num_scenarios=50,
            horizon=5,
            rng=np.random.default_rng(8),
        )
        assert (scenarios > -1.0).all()


def test_t_copula_preserves_empirical_marginal_bounds(
    returns: np.ndarray,
) -> None:
    scenarios = (
        StudentTCopulaScenarioGenerator()
        .fit(returns)
        .sample(
            num_scenarios=100,
            horizon=2,
            rng=np.random.default_rng(9),
        )
    )
    assert np.all(scenarios.min(axis=(0, 1)) >= returns.min(axis=0))
    assert np.all(scenarios.max(axis=(0, 1)) <= returns.max(axis=0))
