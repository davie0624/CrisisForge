from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from crisisforge.tail import (
    RollingConformalQuantileCalibrator,
    SparseTailError,
    fit_pot_gpd,
    fit_training_importance_weights,
    pot_threshold_diagnostics,
)


def test_tail_importance_weights_are_finite_capped_and_normalized() -> None:
    rng = np.random.default_rng(14)
    windows = rng.normal(size=(100, 8, 3))
    windows[-5:] *= 12.0
    result = fit_training_importance_weights(
        windows,
        tail_quantile=0.90,
        strength=4.0,
        maximum_weight=6.0,
    )
    assert np.isfinite(result.weights).all()
    assert np.all(result.weights >= 0.0)
    assert np.all(result.weights <= 6.0)
    assert result.weights.mean() == pytest.approx(1.0)
    assert result.tail_count > 0
    assert result.weights[-5:].mean() > result.weights[:50].mean()


def test_constant_windows_produce_safe_unit_weights() -> None:
    windows = np.ones((12, 4, 2))
    result = fit_training_importance_weights(windows)
    np.testing.assert_allclose(result.weights, 1.0)
    assert result.tail_count == 0


def test_sparse_evt_refuses_or_uses_empirical_fallback() -> None:
    losses = np.linspace(-0.02, 0.08, 40)
    with pytest.raises(SparseTailError) as error:
        fit_pot_gpd(
            losses,
            threshold_quantile=0.95,
            minimum_exceedances=10,
            sparse_policy="raise",
        )
    assert error.value.diagnostics.n_exceedances < 10

    fallback = fit_pot_gpd(
        losses,
        threshold_quantile=0.95,
        minimum_exceedances=10,
        sparse_policy="empirical",
    )
    assert not fallback.uses_gpd
    assert fallback.diagnostics.status == "sparse"
    assert np.isfinite(fallback.quantile(0.99))
    assert np.isfinite(fallback.expected_shortfall(0.95))


def test_gpd_fit_and_threshold_diagnostics_are_finite() -> None:
    rng = np.random.default_rng(22)
    losses = stats.genpareto.rvs(
        c=0.15,
        scale=0.02,
        size=2_000,
        random_state=rng,
    )
    model = fit_pot_gpd(
        losses,
        threshold_quantile=0.90,
        minimum_exceedances=50,
    )
    assert model.uses_gpd
    assert model.diagnostics.status == "fitted"
    assert model.quantile(0.99) > model.threshold
    assert model.expected_shortfall(0.99) > model.quantile(0.99)

    diagnostics = pot_threshold_diagnostics(
        losses,
        threshold_quantiles=(0.90, 0.95, 0.99),
        minimum_exceedances=30,
    )
    assert len(diagnostics) == 3
    assert diagnostics[-1].status == "sparse"


def test_rolling_conformal_updates_only_after_outcome_arrives() -> None:
    calibrator = RollingConformalQuantileCalibrator(
        coverage_level=0.80,
        window_size=3,
        minimum_history=1,
    )
    first = calibrator.issue("t0", base_quantile=1.0)
    second = calibrator.issue("t1", base_quantile=1.0)
    assert first.correction == 0.0
    assert second.correction == 0.0
    assert calibrator.calibration_scores == ()

    score = calibrator.observe("t0", realized_loss=2.5)
    assert score == pytest.approx(1.5)
    third = calibrator.issue("t2", base_quantile=1.0)
    assert third.correction == pytest.approx(1.5)
    assert third.calibrated_quantile == pytest.approx(2.5)
    assert third.n_calibration_scores == 1

    with pytest.raises(KeyError):
        calibrator.observe("never-issued", realized_loss=3.0)
    with pytest.raises(KeyError):
        calibrator.observe("t0", realized_loss=2.5)
