from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crisisforge.factors import (
    DynamicFactorModel,
    RegimeObservationMapping,
    fit_regime_factor_var,
    fit_regime_observation_mapping,
)
from crisisforge.regimes import StickyGaussianHMM, SwitchingDynamicFactorBaseline


def _persistent_two_state_sample(
    *,
    n_observations: int = 500,
    random_state: int = 19,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    transition = np.array([[0.97, 0.03], [0.04, 0.96]])
    states = np.empty(n_observations, dtype=int)
    states[0] = 0
    for time in range(1, n_observations):
        states[time] = rng.choice(2, p=transition[states[time - 1]])
    means = np.array([[-1.2, -0.6], [1.1, 0.8]])
    covariance = np.array([[0.12, 0.03], [0.03, 0.10]])
    observations = means[states] + rng.multivariate_normal(
        np.zeros(2),
        covariance,
        size=n_observations,
    )
    return observations, states


def _switching_asset_returns(
    *,
    n_observations: int = 500,
    random_state: int = 101,
) -> np.ndarray:
    factors, states = _persistent_two_state_sample(
        n_observations=n_observations,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state + 1)
    loadings = np.array(
        [
            [0.010, -0.004],
            [0.008, 0.006],
            [-0.004, 0.010],
            [0.005, 0.004],
        ]
    )
    state_alpha = np.array(
        [
            [-0.002, -0.001, 0.001, -0.001],
            [0.002, 0.001, -0.001, 0.001],
        ]
    )
    returns = factors @ loadings.T + state_alpha[states]
    returns += rng.normal(scale=0.002, size=returns.shape)
    return returns


def test_factor_model_uses_fixed_training_center_scale_and_signs() -> None:
    rng = np.random.default_rng(8)
    training = rng.normal(size=(120, 5)) * 0.01
    validation = rng.normal(loc=0.20, size=(30, 5)) * 0.01

    first = DynamicFactorModel(n_factors=3).fit(training)
    center_before = first.center_.copy()
    scale_before = first.scale_.copy()
    validation_factors = first.transform(validation)
    second = DynamicFactorModel(n_factors=3).fit(training)

    assert validation_factors.shape == (30, 3)
    assert np.array_equal(first.center_, center_before)
    assert np.array_equal(first.scale_, scale_before)
    assert np.allclose(first.components_, second.components_)
    for component in first.components_:
        pivot = np.argmax(np.abs(component))
        assert component[pivot] >= 0.0

    named = DynamicFactorModel(n_factors=2).fit(
        pd.DataFrame(training, columns=list("abcde"))
    )
    with pytest.raises(ValueError, match="columns or their order"):
        named.transform(
            pd.DataFrame(validation, columns=list("bacde"))
        )


def test_log1p_factor_round_trip_and_domain_check() -> None:
    returns = np.array(
        [
            [0.01, -0.02],
            [0.03, 0.04],
            [-0.05, 0.01],
            [0.02, -0.01],
        ]
    )
    model = DynamicFactorModel(
        n_factors=2,
        return_transform="log1p",
    ).fit(returns)
    reconstructed = model.inverse_transform(model.transform(returns))
    assert np.allclose(reconstructed, returns, atol=1e-12)

    with pytest.raises(ValueError, match="greater than -1"):
        model.transform(np.array([[-1.0, 0.0], [0.0, 0.0]]))


def test_sticky_hmm_probabilities_are_valid_and_filter_is_non_anticipative() -> None:
    observations, true_states = _persistent_two_state_sample()
    model = StickyGaussianHMM(
        n_states=2,
        n_init=4,
        max_iter=150,
        sticky_pseudocount=6.0,
        random_state=47,
    ).fit(observations)

    result_full = model.forward_backward(observations)
    result_prefix = model.forward_backward(observations[:300])
    assert np.allclose(result_full.filtered_probabilities.sum(axis=1), 1.0)
    assert np.allclose(result_full.smoothed_probabilities.sum(axis=1), 1.0)
    assert np.allclose(
        result_full.filtered_probabilities[:300],
        result_prefix.filtered_probabilities,
        atol=1e-10,
    )
    assert np.mean(np.diag(model.transition_matrix_)) > 0.80
    inferred_states = result_full.smoothed_probabilities.argmax(axis=1)
    assert np.mean(inferred_states == true_states) > 0.90
    diagnostics = model.diagnostics()
    assert diagnostics["minimum_state_occupancy"] > 0.05
    assert diagnostics["n_initializations"] == 4


def test_hmm_is_reproducible_with_fixed_seed() -> None:
    observations, _ = _persistent_two_state_sample(n_observations=300)
    kwargs = {
        "n_states": 2,
        "n_init": 3,
        "max_iter": 120,
        "random_state": 22,
    }
    first = StickyGaussianHMM(**kwargs).fit(observations)
    second = StickyGaussianHMM(**kwargs).fit(observations)
    assert np.allclose(first.transition_matrix_, second.transition_matrix_)
    assert np.allclose(first.emission_means_, second.emission_means_)
    assert np.allclose(
        first.filtered_probabilities_,
        second.filtered_probabilities_,
    )


def test_regime_factor_var_reports_and_enforces_spectral_stability() -> None:
    rng = np.random.default_rng(3)
    factor = np.empty((180, 1))
    factor[0] = 0.001
    for time in range(1, len(factor)):
        factor[time] = 1.05 * factor[time - 1] + rng.normal(scale=0.001)
    probabilities = np.ones((len(factor), 1))
    dynamics = fit_regime_factor_var(
        factor,
        probabilities,
        maximum_spectral_radius=0.80,
    )
    assert dynamics.spectral_radii_before_stabilization[0] > 0.80
    assert dynamics.spectral_radii[0] <= 0.80 + 1e-12
    assert dynamics.diagnostics()["all_states_stable"]


def test_observation_mapping_selects_state_parameters_without_averaging() -> None:
    mapping = RegimeObservationMapping(
        intercepts=np.array([[0.0, 1.0], [10.0, 20.0]]),
        loadings=np.zeros((2, 2, 1)),
        idiosyncratic_scales=np.ones((2, 2)),
        residual_correlations=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
        effective_counts=np.array([10.0, 10.0]),
        weighted_reconstruction_rmse=np.zeros(2),
    )
    factor_paths = np.zeros((1, 2, 1))
    regime_paths = np.array([[0, 1]])
    expected = mapping.expected_paths(factor_paths, regime_paths)
    assert np.array_equal(expected[0, 0], np.array([0.0, 1.0]))
    assert np.array_equal(expected[0, 1], np.array([10.0, 20.0]))


def test_integrated_baseline_generates_reproducible_asset_paths() -> None:
    returns = _switching_asset_returns()
    model = SwitchingDynamicFactorBaseline(
        n_states=2,
        n_factors=2,
        hmm_n_init=4,
        hmm_max_iter=150,
        sticky_pseudocount=6.0,
        random_state=71,
    ).fit(returns)

    first = model.sample_joint_paths(
        n_paths=12,
        horizon=10,
        random_state=91,
    )
    second = model.sample_joint_paths(
        n_paths=12,
        horizon=10,
        random_state=91,
    )
    assert first.regime_paths.shape == (12, 10)
    assert first.factor_paths.shape == (12, 10, 2)
    assert first.asset_return_paths.shape == (12, 10, 4)
    assert np.isfinite(first.asset_return_paths).all()
    assert np.array_equal(first.regime_paths, second.regime_paths)
    assert np.allclose(first.asset_return_paths, second.asset_return_paths)

    diagnostics = model.diagnostics()
    assert diagnostics["estimator_class"].startswith("MAP/empirical-Bayes")
    assert diagnostics["factor_dynamics"]["all_states_stable"]


def test_validation_filter_continues_from_training_endpoint() -> None:
    returns = _switching_asset_returns(n_observations=420)
    training = returns[:320]
    validation = returns[320:]
    model = SwitchingDynamicFactorBaseline(
        n_states=2,
        n_factors=2,
        hmm_n_init=3,
        hmm_max_iter=120,
        random_state=29,
    ).fit(training)

    validation_probabilities = model.filter(validation)
    all_factors = np.vstack(
        [
            model.training_factors_,
            model.factor_model.transform(validation),
        ]
    )
    reference = model.regime_model.forward_backward(
        all_factors
    ).filtered_probabilities[-len(validation) :]
    assert np.allclose(validation_probabilities, reference, atol=1e-11)

    changed_future = validation.copy()
    changed_future[50:] += 0.30
    changed_probabilities = model.filter(changed_future)
    assert np.allclose(
        validation_probabilities[:50],
        changed_probabilities[:50],
        atol=1e-11,
    )


def test_zero_occupancy_state_uses_pooled_dynamics_and_mapping() -> None:
    rng = np.random.default_rng(801)
    factors = rng.normal(size=(120, 2))
    observations = factors @ np.array([[0.03, -0.02, 0.01], [0.01, 0.02, -0.01]])
    observations += rng.normal(scale=0.004, size=(120, 3))
    probabilities = np.column_stack([np.ones(120), np.zeros(120)])

    dynamics = fit_regime_factor_var(factors, probabilities)
    mapping = fit_regime_observation_mapping(
        observations,
        factors,
        probabilities,
    )

    assert np.allclose(
        dynamics.transition_matrices[1],
        dynamics.transition_matrices[0],
    )
    assert np.allclose(
        dynamics.innovation_covariances[1],
        dynamics.innovation_covariances[0],
    )
    assert np.allclose(mapping.loadings[1], mapping.loadings[0])
    assert np.allclose(
        mapping.idiosyncratic_scales[1],
        mapping.idiosyncratic_scales[0],
    )
    assert (mapping.idiosyncratic_scales[1] > 1e-4).all()
    assert dynamics.diagnostics()["pooled_fallback_states"].tolist() == [False, True]
    assert mapping.diagnostics()["pooled_fallback_states"].tolist() == [False, True]


def test_state_paths_reject_fractional_labels_and_zero_probability_mass() -> None:
    mapping = RegimeObservationMapping(
        intercepts=np.zeros((2, 1)),
        loadings=np.zeros((2, 1, 1)),
        idiosyncratic_scales=np.ones((2, 1)),
        residual_correlations=np.ones((2, 1, 1)),
        effective_counts=np.ones(2),
        weighted_reconstruction_rmse=np.zeros(2),
    )
    with pytest.raises(ValueError, match="integer state labels"):
        mapping.expected_paths(
            np.zeros((1, 1, 1)),
            np.array([[0.5]]),
        )
    with pytest.raises(ValueError, match="integer state labels"):
        mapping.sample_paths(
            np.zeros((1, 1, 1)),
            np.array([[1.9]]),
            rng=np.random.default_rng(1),
        )

    observations, _ = _persistent_two_state_sample(n_observations=250)
    hmm = StickyGaussianHMM(
        n_states=2,
        n_init=2,
        max_iter=80,
    ).fit(observations)
    with pytest.raises(ValueError, match="positive total mass"):
        hmm.sample_posterior_predictive_paths(
            n_paths=2,
            horizon=2,
            rng=np.random.default_rng(1),
            initial_filtered_probabilities=np.zeros(2),
        )


def test_log1p_joint_paths_respect_simple_return_domain() -> None:
    returns = _switching_asset_returns(n_observations=350)
    model = SwitchingDynamicFactorBaseline(
        n_states=2,
        n_factors=2,
        return_transform="log1p",
        hmm_n_init=3,
        hmm_max_iter=100,
        random_state=42,
    ).fit(returns)
    paths = model.sample_joint_paths(
        n_paths=100,
        horizon=20,
        random_state=7,
    )
    assert np.isfinite(paths.asset_return_paths).all()
    assert (paths.asset_return_paths >= -1.0).all()
