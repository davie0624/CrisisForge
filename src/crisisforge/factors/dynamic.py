"""Fold-local dynamic-factor components for the Stage 1 baseline.

The classes in this module deliberately use transparent linear estimators.  They
are intended as an auditable baseline for later switching state-space work, not
as a claim that a full Bayesian dynamic factor model has been fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.DataFrame
ReturnTransform = Literal["simple", "log1p"]


def _as_2d_finite(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one column")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _make_positive_semidefinite(
    matrix: np.ndarray,
    *,
    eigenvalue_floor: float,
) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, eigenvalue_floor)
    return (eigenvectors * clipped) @ eigenvectors.T


def _covariance_cholesky(covariance: np.ndarray) -> np.ndarray:
    covariance = _make_positive_semidefinite(covariance, eigenvalue_floor=1e-10)
    return np.linalg.cholesky(covariance)


def _as_integer_state_paths(
    values: np.ndarray,
    *,
    name: str,
    n_states: int,
) -> np.ndarray:
    """Validate state paths without silently truncating floating-point labels."""

    numeric = np.asarray(values, dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain integer state labels")
    states = numeric.astype(int)
    if np.any((states < 0) | (states >= n_states)):
        raise ValueError(f"{name} contains an invalid state")
    return states


@dataclass(frozen=True)
class FactorDiagnostics:
    """In-sample diagnostics for a fitted fold-local PCA representation."""

    n_observations: int
    n_assets: int
    n_factors: int
    total_explained_variance_ratio: float
    reconstruction_rmse: float
    maximum_absolute_reconstruction_error: float


class DynamicFactorModel:
    """Estimate a deterministic-sign PCA representation on one training fold.

    Parameters are learned only by :meth:`fit`.  Calling :meth:`transform` on
    validation or test data never updates the fitted center, scale, or loadings.
    With ``return_transform="log1p"``, the input is interpreted as simple returns
    and converted fold-locally to log returns before centering and scaling.
    """

    def __init__(
        self,
        n_factors: int,
        *,
        return_transform: ReturnTransform = "simple",
        scale_floor: float = 1e-8,
    ) -> None:
        if n_factors < 1:
            raise ValueError("n_factors must be positive")
        if return_transform not in {"simple", "log1p"}:
            raise ValueError("return_transform must be 'simple' or 'log1p'")
        if scale_floor <= 0.0:
            raise ValueError("scale_floor must be positive")
        self.n_factors = int(n_factors)
        self.return_transform = return_transform
        self.scale_floor = float(scale_floor)

    def _observation_transform(self, values: ArrayLike) -> np.ndarray:
        array = _as_2d_finite(values, name="returns")
        if self.return_transform == "log1p":
            if np.any(array <= -1.0):
                raise ValueError("simple returns must be greater than -1 for log1p")
            return np.log1p(array)
        return array.copy()

    def to_observation_space(self, values: ArrayLike) -> np.ndarray:
        """Return values in the linear observation space used by the model."""

        return self._observation_transform(values)

    def observation_to_simple_returns(self, values: ArrayLike) -> np.ndarray:
        """Convert model-space observations back to simple returns."""

        array = _as_2d_finite(values, name="observations")
        if self.return_transform == "log1p":
            with np.errstate(over="ignore"):
                converted = np.expm1(array)
        else:
            converted = array.copy()
        if not np.isfinite(converted).all():
            raise ValueError("model-space observations map to non-finite simple returns")
        if np.any(converted < -1.0):
            raise ValueError("model-space observations imply simple returns below -1")
        return converted

    def fit(self, returns: ArrayLike) -> DynamicFactorModel:
        """Fit fold-local transformation parameters and PCA loadings."""

        observations = self._observation_transform(returns)
        n_observations, n_assets = observations.shape
        if n_observations < 2:
            raise ValueError("the training fold must contain at least two rows")
        if self.n_factors > min(n_observations, n_assets):
            raise ValueError("n_factors exceeds the rank supported by the training fold")

        self.feature_names_ = (
            tuple(str(column) for column in returns.columns)
            if isinstance(returns, pd.DataFrame)
            else tuple(f"asset_{index}" for index in range(n_assets))
        )
        self.center_ = observations.mean(axis=0)
        raw_scale = observations.std(axis=0, ddof=0)
        self.scale_ = np.maximum(raw_scale, self.scale_floor)
        standardized = (observations - self.center_) / self.scale_

        _, singular_values, right_vectors = np.linalg.svd(
            standardized,
            full_matrices=False,
        )
        components = right_vectors[: self.n_factors].copy()

        # PCA axes are sign-indeterminate.  Orient every axis so its largest
        # absolute loading is positive, making repeated fits byte-stable.
        for factor_index in range(components.shape[0]):
            pivot = int(np.argmax(np.abs(components[factor_index])))
            if components[factor_index, pivot] < 0.0:
                components[factor_index] *= -1.0

        total_variation = float(np.square(singular_values).sum())
        selected_variation = np.square(singular_values[: self.n_factors])
        if total_variation <= 0.0:
            explained = np.zeros(self.n_factors)
        else:
            explained = selected_variation / total_variation

        self.components_ = components
        self.singular_values_ = singular_values[: self.n_factors].copy()
        self.explained_variance_ratio_ = explained
        self.n_features_in_ = n_assets
        self.n_observations_fit_ = n_observations
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("DynamicFactorModel must be fitted first")

    def transform(self, returns: ArrayLike) -> np.ndarray:
        """Project observations using parameters fixed on the training fold."""

        self._check_fitted()
        if isinstance(returns, pd.DataFrame):
            columns = tuple(str(column) for column in returns.columns)
            if columns != self.feature_names_:
                raise ValueError(
                    "DataFrame columns or their order differ from the training fold"
                )
        observations = self._observation_transform(returns)
        if observations.shape[1] != self.n_features_in_:
            raise ValueError("returns has a different number of assets than the fitted model")
        standardized = (observations - self.center_) / self.scale_
        return standardized @ self.components_.T

    def inverse_transform(
        self,
        factors: ArrayLike,
        *,
        output_simple_returns: bool = True,
    ) -> np.ndarray:
        """Reconstruct observations from factor scores.

        PCA truncation means this is a projection unless all supported factors
        are retained.
        """

        self._check_fitted()
        factor_array = _as_2d_finite(factors, name="factors")
        if factor_array.shape[1] != self.n_factors:
            raise ValueError("factors has an unexpected number of columns")
        standardized = factor_array @ self.components_
        observations = standardized * self.scale_ + self.center_
        if output_simple_returns:
            return self.observation_to_simple_returns(observations)
        return observations

    def diagnostics(self, returns: ArrayLike) -> FactorDiagnostics:
        """Compute reconstruction diagnostics without refitting."""

        observations = self._observation_transform(returns)
        factors = self.transform(returns)
        reconstructed = self.inverse_transform(
            factors,
            output_simple_returns=False,
        )
        residual = observations - reconstructed
        return FactorDiagnostics(
            n_observations=observations.shape[0],
            n_assets=observations.shape[1],
            n_factors=self.n_factors,
            total_explained_variance_ratio=float(
                self.explained_variance_ratio_.sum()
            ),
            reconstruction_rmse=float(np.sqrt(np.mean(np.square(residual)))),
            maximum_absolute_reconstruction_error=float(
                np.max(np.abs(residual))
            ),
        )


@dataclass(frozen=True)
class RegimeFactorVAR:
    """Regime-specific Gaussian factor VAR(1) parameters."""

    intercepts: np.ndarray
    transition_matrices: np.ndarray
    innovation_covariances: np.ndarray
    effective_counts: np.ndarray
    spectral_radii_before_stabilization: np.ndarray
    spectral_radii: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.intercepts.shape[0])

    @property
    def n_factors(self) -> int:
        return int(self.intercepts.shape[1])

    def sample_paths(
        self,
        initial_factors: np.ndarray,
        regime_paths: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample factor paths using the exact parameters of each drawn state."""

        states = _as_integer_state_paths(
            regime_paths,
            name="regime_paths",
            n_states=self.n_states,
        )
        if states.ndim != 2:
            raise ValueError("regime_paths must have shape (paths, horizon)")
        n_paths, horizon = states.shape

        initial = np.asarray(initial_factors, dtype=float)
        if initial.ndim == 1:
            if initial.shape[0] != self.n_factors:
                raise ValueError("initial_factors has an invalid width")
            previous = np.repeat(initial[None, :], n_paths, axis=0)
        elif initial.shape == (n_paths, self.n_factors):
            previous = initial.copy()
        else:
            raise ValueError(
                "initial_factors must have shape (factors,) or (paths, factors)"
            )

        output = np.empty((n_paths, horizon, self.n_factors), dtype=float)
        cholesky = np.stack(
            [
                _covariance_cholesky(self.innovation_covariances[state])
                for state in range(self.n_states)
            ]
        )
        for step in range(horizon):
            current_states = states[:, step]
            current = np.empty_like(previous)
            for state in range(self.n_states):
                selected = np.flatnonzero(current_states == state)
                if selected.size == 0:
                    continue
                mean = (
                    self.intercepts[state]
                    + previous[selected] @ self.transition_matrices[state].T
                )
                noise = rng.normal(size=(selected.size, self.n_factors))
                current[selected] = mean + noise @ cholesky[state].T
            output[:, step] = current
            previous = current
        return output

    def diagnostics(self) -> dict[str, Any]:
        return {
            "effective_counts": self.effective_counts.copy(),
            "pooled_fallback_states": (
                self.effective_counts <= self.n_factors + 1
            ),
            "spectral_radii_before_stabilization": (
                self.spectral_radii_before_stabilization.copy()
            ),
            "spectral_radii": self.spectral_radii.copy(),
            "all_states_stable": bool(np.all(self.spectral_radii < 1.0)),
        }


def fit_regime_factor_var(
    factors: ArrayLike,
    regime_probabilities: ArrayLike,
    *,
    ridge: float = 1e-4,
    covariance_floor: float = 1e-8,
    maximum_spectral_radius: float = 0.995,
) -> RegimeFactorVAR:
    """Fit weighted, regime-specific VAR(1) dynamics.

    Probabilities are normally smoothed training probabilities.  A small ridge
    penalty and an explicit spectral-radius cap provide a stable baseline for
    forward simulation; both choices must be disclosed in empirical results.
    """

    factor_array = _as_2d_finite(factors, name="factors")
    probabilities = _as_2d_finite(
        regime_probabilities,
        name="regime_probabilities",
    )
    if probabilities.shape[0] != factor_array.shape[0]:
        raise ValueError("factors and regime_probabilities must have equal rows")
    if factor_array.shape[0] < 3:
        raise ValueError("at least three factor observations are required")
    if np.any(probabilities < 0.0):
        raise ValueError("regime probabilities cannot be negative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-7):
        raise ValueError("regime probabilities must sum to one")
    if ridge < 0.0 or covariance_floor <= 0.0:
        raise ValueError("ridge must be non-negative and covariance_floor positive")
    if not 0.0 < maximum_spectral_radius < 1.0:
        raise ValueError("maximum_spectral_radius must be between zero and one")

    lagged = factor_array[:-1]
    targets = factor_array[1:]
    design = np.column_stack([np.ones(len(lagged)), lagged])
    weights_by_state = probabilities[1:]
    n_states = probabilities.shape[1]
    n_factors = factor_array.shape[1]

    intercepts = np.empty((n_states, n_factors))
    matrices = np.empty((n_states, n_factors, n_factors))
    covariances = np.empty((n_states, n_factors, n_factors))
    effective_counts = weights_by_state.sum(axis=0)
    radii_before = np.empty(n_states)
    radii_after = np.empty(n_states)

    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    pooled_gram = design.T @ design + penalty
    pooled_rhs = design.T @ targets
    pooled_coefficients = np.linalg.solve(pooled_gram, pooled_rhs)
    pooled_intercept = pooled_coefficients[0]
    pooled_matrix = pooled_coefficients[1:].T
    pooled_radius_before = float(
        np.max(np.abs(np.linalg.eigvals(pooled_matrix)))
    )
    if pooled_radius_before > maximum_spectral_radius:
        pooled_matrix = pooled_matrix * (
            maximum_spectral_radius / pooled_radius_before
        )
    pooled_radius_after = float(
        np.max(np.abs(np.linalg.eigvals(pooled_matrix)))
    )
    pooled_residuals = targets - (
        pooled_intercept + lagged @ pooled_matrix.T
    )
    pooled_covariance = np.atleast_2d(
        np.cov(pooled_residuals, rowvar=False, ddof=0)
    )
    pooled_covariance = _make_positive_semidefinite(
        pooled_covariance,
        eigenvalue_floor=covariance_floor,
    )

    for state in range(n_states):
        weights = weights_by_state[:, state]
        effective = float(weights.sum())
        if effective <= design.shape[1]:
            # A weakly occupied state cannot support its own VAR.  Reusing the
            # pooled fit keeps both the conditional mean and innovation scale
            # numerically meaningful; tiny pseudo-weights would spuriously
            # collapse the covariance toward zero.
            intercepts[state] = pooled_intercept
            matrices[state] = pooled_matrix
            covariances[state] = pooled_covariance
            radii_before[state] = pooled_radius_before
            radii_after[state] = pooled_radius_after
            continue
        weighted_design = design * weights[:, None]
        gram = design.T @ weighted_design + penalty
        rhs = design.T @ (targets * weights[:, None])
        coefficients = np.linalg.solve(gram, rhs)
        intercept = coefficients[0]
        matrix = coefficients[1:].T

        eigenvalues = np.linalg.eigvals(matrix)
        radius = float(np.max(np.abs(eigenvalues)))
        radii_before[state] = radius
        if radius > maximum_spectral_radius:
            matrix = matrix * (maximum_spectral_radius / radius)
        radii_after[state] = float(np.max(np.abs(np.linalg.eigvals(matrix))))

        residuals = targets - (intercept + lagged @ matrix.T)
        if effective > design.shape[1]:
            covariance = (residuals * weights[:, None]).T @ residuals
            covariance /= max(effective, 1.0)
            covariance = _make_positive_semidefinite(
                covariance,
                eigenvalue_floor=covariance_floor,
            )
        intercepts[state] = intercept
        matrices[state] = matrix
        covariances[state] = covariance

    return RegimeFactorVAR(
        intercepts=intercepts,
        transition_matrices=matrices,
        innovation_covariances=covariances,
        effective_counts=effective_counts,
        spectral_radii_before_stabilization=radii_before,
        spectral_radii=radii_after,
    )


@dataclass(frozen=True)
class RegimeObservationMapping:
    """Regime-dependent observation equation ``alpha_z + B_z f + D_z eps``."""

    intercepts: np.ndarray
    loadings: np.ndarray
    idiosyncratic_scales: np.ndarray
    residual_correlations: np.ndarray
    effective_counts: np.ndarray
    weighted_reconstruction_rmse: np.ndarray

    @property
    def n_states(self) -> int:
        return int(self.intercepts.shape[0])

    @property
    def n_assets(self) -> int:
        return int(self.intercepts.shape[1])

    @property
    def n_factors(self) -> int:
        return int(self.loadings.shape[2])

    def expected_paths(
        self,
        factor_paths: np.ndarray,
        regime_paths: np.ndarray,
    ) -> np.ndarray:
        """Map paths through the exact observation equation for each state."""

        factors = np.asarray(factor_paths, dtype=float)
        states = _as_integer_state_paths(
            regime_paths,
            name="regime_paths",
            n_states=self.n_states,
        )
        if factors.ndim != 3:
            raise ValueError("factor_paths must have shape (paths, horizon, factors)")
        if states.shape != factors.shape[:2]:
            raise ValueError("regime_paths shape must match factor path and horizon")
        if factors.shape[2] != self.n_factors:
            raise ValueError("factor_paths has an invalid factor width")

        output = np.empty((*states.shape, self.n_assets), dtype=float)
        for state in range(self.n_states):
            selected = states == state
            if not np.any(selected):
                continue
            output[selected] = (
                self.intercepts[state]
                + factors[selected] @ self.loadings[state].T
            )
        return output

    def sample_paths(
        self,
        factor_paths: np.ndarray,
        regime_paths: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add state-specific correlated idiosyncratic shocks to asset paths."""

        states = np.asarray(regime_paths, dtype=int)
        output = self.expected_paths(factor_paths, states)
        standard_normal = rng.normal(size=output.shape)
        for state in range(self.n_states):
            selected = states == state
            if not np.any(selected):
                continue
            correlation_cholesky = _covariance_cholesky(
                self.residual_correlations[state]
            )
            correlated = standard_normal[selected] @ correlation_cholesky.T
            output[selected] += correlated * self.idiosyncratic_scales[state]
        return output

    def diagnostics(self) -> dict[str, Any]:
        minimum_eigenvalues = np.array(
            [
                np.linalg.eigvalsh(correlation).min()
                for correlation in self.residual_correlations
            ]
        )
        return {
            "effective_counts": self.effective_counts.copy(),
            "pooled_fallback_states": (
                self.effective_counts <= self.n_factors + 1
            ),
            "weighted_reconstruction_rmse": (
                self.weighted_reconstruction_rmse.copy()
            ),
            "minimum_residual_correlation_eigenvalue": minimum_eigenvalues,
        }


def fit_regime_observation_mapping(
    observations: ArrayLike,
    factors: ArrayLike,
    regime_probabilities: ArrayLike,
    *,
    ridge: float = 1e-4,
    correlation_shrinkage: float = 0.20,
    scale_floor: float = 1e-6,
) -> RegimeObservationMapping:
    """Fit weighted state-specific ``alpha``, ``B``, ``D``, and shrunk ``R``.

    The observations must already be in the linear space desired by the caller
    (simple returns by default, log returns when the optional log1p transform is
    selected).  Shrinkage is toward the identity correlation matrix.
    """

    observation_array = _as_2d_finite(observations, name="observations")
    factor_array = _as_2d_finite(factors, name="factors")
    probabilities = _as_2d_finite(
        regime_probabilities,
        name="regime_probabilities",
    )
    if not (
        observation_array.shape[0]
        == factor_array.shape[0]
        == probabilities.shape[0]
    ):
        raise ValueError("observations, factors, and probabilities need equal rows")
    if observation_array.shape[0] < 2:
        raise ValueError("at least two observations are required")
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-7,
    ):
        raise ValueError("regime probabilities must be non-negative and sum to one")
    if ridge < 0.0 or scale_floor <= 0.0:
        raise ValueError("ridge must be non-negative and scale_floor positive")
    if not 0.0 <= correlation_shrinkage <= 1.0:
        raise ValueError("correlation_shrinkage must be between zero and one")

    n_observations, n_assets = observation_array.shape
    n_factors = factor_array.shape[1]
    n_states = probabilities.shape[1]
    design = np.column_stack([np.ones(n_observations), factor_array])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0

    intercepts = np.empty((n_states, n_assets))
    loadings = np.empty((n_states, n_assets, n_factors))
    scales = np.empty((n_states, n_assets))
    correlations = np.empty((n_states, n_assets, n_assets))
    effective_counts = probabilities.sum(axis=0)
    reconstruction_rmse = np.empty(n_states)

    pooled_gram = design.T @ design + penalty
    pooled_rhs = design.T @ observation_array
    pooled_coefficients = np.linalg.solve(pooled_gram, pooled_rhs)
    pooled_predicted = design @ pooled_coefficients
    pooled_residuals = observation_array - pooled_predicted
    pooled_covariance = (pooled_residuals.T @ pooled_residuals) / n_observations
    pooled_covariance = _make_positive_semidefinite(
        pooled_covariance,
        eigenvalue_floor=scale_floor**2,
    )
    pooled_rmse = float(np.sqrt(np.mean(np.square(pooled_residuals))))

    for state in range(n_states):
        weights = probabilities[:, state]
        original_effective = float(weights.sum())
        use_pooled_fallback = original_effective <= design.shape[1]
        if use_pooled_fallback:
            coefficients = pooled_coefficients
            predicted = pooled_predicted
            residuals = pooled_residuals
            covariance = pooled_covariance
            effective = float(n_observations)
        else:
            weighted_design = design * weights[:, None]
            gram = design.T @ weighted_design + penalty
            rhs = design.T @ (observation_array * weights[:, None])
            coefficients = np.linalg.solve(gram, rhs)
            predicted = design @ coefficients
            residuals = observation_array - predicted
            effective = original_effective
            covariance = (residuals * weights[:, None]).T @ residuals
            covariance /= effective
            covariance = _make_positive_semidefinite(
                covariance,
                eigenvalue_floor=scale_floor**2,
            )
        state_scales = np.maximum(np.sqrt(np.diag(covariance)), scale_floor)
        raw_correlation = covariance / np.outer(state_scales, state_scales)
        raw_correlation = np.clip(raw_correlation, -1.0, 1.0)
        shrunk = (
            (1.0 - correlation_shrinkage) * raw_correlation
            + correlation_shrinkage * np.eye(n_assets)
        )
        shrunk = _make_positive_semidefinite(shrunk, eigenvalue_floor=1e-8)
        normalizer = np.sqrt(np.diag(shrunk))
        shrunk /= np.outer(normalizer, normalizer)

        intercepts[state] = coefficients[0]
        loadings[state] = coefficients[1:].T
        scales[state] = state_scales
        correlations[state] = shrunk
        if use_pooled_fallback:
            reconstruction_rmse[state] = pooled_rmse
        else:
            reconstruction_rmse[state] = float(
                np.sqrt(
                    np.sum(weights[:, None] * np.square(residuals))
                    / (effective * n_assets)
                )
            )

    return RegimeObservationMapping(
        intercepts=intercepts,
        loadings=loadings,
        idiosyncratic_scales=scales,
        residual_correlations=correlations,
        effective_counts=effective_counts,
        weighted_reconstruction_rmse=reconstruction_rmse,
    )
