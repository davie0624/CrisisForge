from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crisisforge.evaluation.stage1 import (
    build_switching_factor_model,
    sample_fixed_estimation_paths,
)
from crisisforge.regimes import SwitchingDynamicFactorBaseline


def _returns(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = np.repeat([0, 1], 100)
    factors = rng.normal(size=(200, 2)) + states[:, None] * 1.5
    loadings = np.array(
        [
            [0.009, 0.003],
            [0.005, -0.007],
            [-0.004, 0.008],
        ]
    )
    values = factors @ loadings.T + rng.normal(scale=0.002, size=(200, 3))
    return pd.DataFrame(
        values,
        index=pd.date_range("2010-01-04", periods=200, freq="B"),
        columns=["asset__a", "asset__b", "asset__c"],
    )


def test_stage1_builder_preserves_registered_choices() -> None:
    configuration = {
        "data": {"return_transform": "simple"},
        "factor_model": {"n_factors": 2, "scale_floor": 1e-7},
        "regime_model": {
            "n_states": 2,
            "n_initializations": 2,
            "maximum_iterations": 30,
            "tolerance": 1e-5,
            "transition_pseudocount": 0.5,
            "sticky_pseudocount": 4.0,
            "minimum_covariance_eigenvalue": 1e-5,
            "minimum_state_weight": 2e-3,
            "random_seed": 12,
        },
        "factor_dynamics": {
            "ridge": 1e-3,
            "covariance_floor": 3e-8,
            "maximum_spectral_radius": 0.90,
        },
        "observation_mapping": {
            "ridge": 2e-3,
            "residual_correlation_shrinkage": 0.25,
            "idiosyncratic_scale_floor": 4e-6,
        },
    }
    model = build_switching_factor_model(configuration)
    assert model.n_states == 2
    assert model.n_factors == 2
    assert model.regime_model.n_init == 2
    assert model.regime_model.minimum_state_weight == 2e-3
    assert model.factor_model.scale_floor == 1e-7
    assert model.factor_covariance_floor == 3e-8
    assert model.observation_scale_floor == 4e-6
    assert model.maximum_spectral_radius == 0.90
    assert model.residual_correlation_shrinkage == 0.25


def test_fixed_estimation_sampling_is_reproducible_and_does_not_refit() -> None:
    returns = _returns()
    model = SwitchingDynamicFactorBaseline(
        n_states=2,
        n_factors=2,
        hmm_n_init=2,
        hmm_max_iter=50,
        sticky_pseudocount=4.0,
        random_state=9,
    ).fit(returns.iloc[:150])
    center_before = model.factor_model.center_.copy()
    transition_before = model.regime_model.transition_matrix_.copy()
    first, belief_first = sample_fixed_estimation_paths(
        model,
        returns.iloc[:175],
        num_scenarios=16,
        horizon=5,
        random_seed=44,
    )
    second, belief_second = sample_fixed_estimation_paths(
        model,
        returns.iloc[:175],
        num_scenarios=16,
        horizon=5,
        random_seed=44,
    )
    assert first.shape == (16, 5, 3)
    assert np.allclose(first, second)
    assert np.allclose(belief_first, belief_second)
    assert np.isclose(belief_first.sum(), 1.0)
    assert np.array_equal(model.factor_model.center_, center_before)
    assert np.array_equal(model.regime_model.transition_matrix_, transition_before)


def test_stage1_sampling_rejects_nonpositive_dimensions() -> None:
    returns = _returns()
    model = SwitchingDynamicFactorBaseline(
        n_states=2,
        n_factors=2,
        hmm_n_init=1,
        hmm_max_iter=20,
        random_state=2,
    ).fit(returns.iloc[:150])
    with pytest.raises(ValueError, match="positive"):
        sample_fixed_estimation_paths(
            model,
            returns.iloc[:151],
            num_scenarios=0,
            horizon=5,
            random_seed=1,
        )
