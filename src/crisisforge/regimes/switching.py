"""Auditable switching-regime baseline for CrisisForge Stage 1.

This module implements a Gaussian hidden Markov model estimated by MAP-like EM
with a sticky Dirichlet-style transition penalty.  It is an empirical-Bayes
baseline: it does not perform MCMC, variational Bayes, or full posterior
inference over state-space parameters.  The distinction is intentional and
should be retained in reports and model cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from crisisforge.factors import (
    DynamicFactorModel,
    RegimeFactorVAR,
    RegimeObservationMapping,
    fit_regime_factor_var,
    fit_regime_observation_mapping,
)

ArrayLike = np.ndarray | pd.DataFrame


def _as_finite_matrix(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalize_probabilities(values: np.ndarray, *, axis: int) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=float), 1e-300)
    return clipped / clipped.sum(axis=axis, keepdims=True)


def _validate_probability_vector(
    values: np.ndarray,
    *,
    name: str,
    n_states: int,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (n_states,):
        raise ValueError(f"{name} has invalid shape")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
        raise ValueError(f"{name} must be non-negative with positive total mass")
    return probabilities / probabilities.sum()


def _regularize_covariance(
    covariance: np.ndarray,
    *,
    minimum_eigenvalue: float,
) -> np.ndarray:
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, minimum_eigenvalue)
    return (eigenvectors * clipped) @ eigenvectors.T


def _log_gaussian_density(
    observations: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    n_observations, n_features = observations.shape
    n_states = means.shape[0]
    output = np.empty((n_observations, n_states), dtype=float)
    constant = n_features * np.log(2.0 * np.pi)
    for state in range(n_states):
        covariance = covariances[state]
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise np.linalg.LinAlgError("emission covariance is not positive definite")
        centered = observations - means[state]
        solved = np.linalg.solve(covariance, centered.T).T
        mahalanobis = np.einsum("ij,ij->i", centered, solved)
        output[:, state] = -0.5 * (constant + log_determinant + mahalanobis)
    return output


@dataclass(frozen=True)
class ForwardBackwardResult:
    """Probabilities and sufficient statistics from one forward-backward pass."""

    log_likelihood: float
    filtered_probabilities: np.ndarray
    smoothed_probabilities: np.ndarray
    expected_transition_counts: np.ndarray


@dataclass(frozen=True)
class JointPathSample:
    """Posterior-predictive paths produced without averaging regime parameters."""

    regime_paths: np.ndarray
    factor_paths: np.ndarray
    asset_return_paths: np.ndarray


class StickyGaussianHMM:
    """Gaussian HMM fitted with multi-start MAP-like EM.

    ``transition_pseudocount`` and ``sticky_pseudocount`` are Dirichlet-style
    pseudo-counts added to expected transition counts in the M-step.  They are
    regularizers, not draws from a Bayesian posterior.
    """

    def __init__(
        self,
        n_states: int,
        *,
        n_init: int = 8,
        max_iter: int = 250,
        tolerance: float = 1e-6,
        transition_pseudocount: float = 0.5,
        sticky_pseudocount: float = 8.0,
        minimum_covar: float = 1e-6,
        minimum_state_weight: float = 1e-3,
        random_state: int = 1729,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be positive")
        if n_init < 1 or max_iter < 1:
            raise ValueError("n_init and max_iter must be positive")
        if tolerance <= 0.0 or minimum_covar <= 0.0:
            raise ValueError("tolerance and minimum_covar must be positive")
        if transition_pseudocount < 0.0 or sticky_pseudocount < 0.0:
            raise ValueError("transition pseudo-counts cannot be negative")
        if minimum_state_weight <= 0.0:
            raise ValueError("minimum_state_weight must be positive")

        self.n_states = int(n_states)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.transition_pseudocount = float(transition_pseudocount)
        self.sticky_pseudocount = float(sticky_pseudocount)
        self.minimum_covar = float(minimum_covar)
        self.minimum_state_weight = float(minimum_state_weight)
        self.random_state = int(random_state)

    def _initial_parameters(
        self,
        observations: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_observations, n_features = observations.shape
        if n_observations < self.n_states:
            raise ValueError("training observations must be at least n_states")

        # Stratified random seeds on the first principal direction reduce
        # immediate state collapse while preserving independent restarts.
        centered = observations - observations.mean(axis=0)
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        projection = centered @ right_vectors[0]
        order = np.argsort(projection)
        bins = np.array_split(order, self.n_states)
        mean_indices = np.array(
            [rng.choice(bin_indices) for bin_indices in bins],
            dtype=int,
        )
        means = observations[mean_indices].copy()

        global_covariance = np.atleast_2d(np.cov(observations, rowvar=False))
        global_covariance = _regularize_covariance(
            global_covariance,
            minimum_eigenvalue=self.minimum_covar,
        )
        covariances = np.repeat(
            global_covariance[None, :, :],
            self.n_states,
            axis=0,
        )

        transition_mass = (
            np.ones((self.n_states, self.n_states))
            + self.sticky_pseudocount * np.eye(self.n_states)
            + rng.uniform(0.0, 0.25, size=(self.n_states, self.n_states))
        )
        transition = _normalize_probabilities(transition_mass, axis=1)
        initial = np.repeat(1.0 / self.n_states, self.n_states)
        if n_features == 1:
            means = means.reshape(self.n_states, 1)
        return initial, transition, means, covariances

    @staticmethod
    def _forward_backward_with_parameters(
        observations: np.ndarray,
        initial: np.ndarray,
        transition: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> ForwardBackwardResult:
        log_emission = _log_gaussian_density(observations, means, covariances)
        log_initial = np.log(np.maximum(initial, 1e-300))
        log_transition = np.log(np.maximum(transition, 1e-300))
        n_observations, n_states = log_emission.shape

        log_alpha = np.empty((n_observations, n_states))
        log_alpha[0] = log_initial + log_emission[0]
        for time in range(1, n_observations):
            log_alpha[time] = log_emission[time] + logsumexp(
                log_alpha[time - 1, :, None] + log_transition,
                axis=0,
            )
        log_likelihood = float(logsumexp(log_alpha[-1]))
        filtered = np.exp(log_alpha - logsumexp(log_alpha, axis=1)[:, None])

        log_beta = np.zeros((n_observations, n_states))
        for time in range(n_observations - 2, -1, -1):
            log_beta[time] = logsumexp(
                log_transition + log_emission[time + 1][None, :] + log_beta[time + 1][None, :],
                axis=1,
            )
        log_gamma = log_alpha + log_beta
        smoothed = np.exp(log_gamma - logsumexp(log_gamma, axis=1)[:, None])

        expected_transition_counts = np.zeros((n_states, n_states))
        for time in range(n_observations - 1):
            log_xi = (
                log_alpha[time, :, None]
                + log_transition
                + log_emission[time + 1][None, :]
                + log_beta[time + 1][None, :]
            )
            log_xi -= logsumexp(log_xi)
            expected_transition_counts += np.exp(log_xi)

        return ForwardBackwardResult(
            log_likelihood=log_likelihood,
            filtered_probabilities=filtered,
            smoothed_probabilities=smoothed,
            expected_transition_counts=expected_transition_counts,
        )

    def _posterior_objective(
        self,
        log_likelihood: float,
        transition: np.ndarray,
    ) -> float:
        pseudo = np.full_like(transition, self.transition_pseudocount)
        pseudo += self.sticky_pseudocount * np.eye(self.n_states)
        return float(log_likelihood + np.sum(pseudo * np.log(transition)))

    def _fit_one(
        self,
        observations: np.ndarray,
        *,
        seed: np.random.SeedSequence,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        initial, transition, means, covariances = self._initial_parameters(
            observations,
            rng,
        )
        global_covariance = _regularize_covariance(
            np.atleast_2d(np.cov(observations, rowvar=False)),
            minimum_eigenvalue=self.minimum_covar,
        )

        history: list[float] = []
        converged = False
        previous_objective = -np.inf
        for _iteration in range(self.max_iter):
            result = self._forward_backward_with_parameters(
                observations,
                initial,
                transition,
                means,
                covariances,
            )
            weights = result.smoothed_probabilities
            state_weights = weights.sum(axis=0)

            initial = _normalize_probabilities(
                weights[0] + 1.0 / self.n_states,
                axis=0,
            )
            transition_mass = (
                result.expected_transition_counts
                + self.transition_pseudocount
                + self.sticky_pseudocount * np.eye(self.n_states)
            )
            transition = _normalize_probabilities(transition_mass, axis=1)

            updated_means = np.empty_like(means)
            updated_covariances = np.empty_like(covariances)
            minimum_weight = max(
                self.minimum_state_weight * len(observations),
                observations.shape[1] + 1.0,
            )
            for state in range(self.n_states):
                state_weight = float(state_weights[state])
                if state_weight < minimum_weight:
                    updated_means[state] = observations[rng.integers(len(observations))]
                    updated_covariances[state] = global_covariance
                    continue
                state_probabilities = weights[:, state]
                mean = (state_probabilities[:, None] * observations).sum(axis=0) / state_weight
                centered = observations - mean
                covariance = ((centered * state_probabilities[:, None]).T @ centered) / state_weight
                updated_means[state] = mean
                updated_covariances[state] = _regularize_covariance(
                    covariance,
                    minimum_eigenvalue=self.minimum_covar,
                )
            means = updated_means
            covariances = updated_covariances

            updated_result = self._forward_backward_with_parameters(
                observations,
                initial,
                transition,
                means,
                covariances,
            )
            objective = self._posterior_objective(
                updated_result.log_likelihood,
                transition,
            )
            history.append(objective)
            if np.isfinite(previous_objective):
                improvement = abs(objective - previous_objective)
                if improvement <= self.tolerance * (1.0 + abs(previous_objective)):
                    converged = True
                    break
            previous_objective = objective

        final_result = self._forward_backward_with_parameters(
            observations,
            initial,
            transition,
            means,
            covariances,
        )
        return {
            "initial": initial,
            "transition": transition,
            "means": means,
            "covariances": covariances,
            "result": final_result,
            "objective": self._posterior_objective(
                final_result.log_likelihood,
                transition,
            ),
            "history": np.asarray(history),
            "converged": converged,
            "n_iter": _iteration + 1,
        }

    def fit(self, observations: ArrayLike) -> StickyGaussianHMM:
        """Fit all restarts and retain the highest regularized objective."""

        observation_array = _as_finite_matrix(observations, name="observations")
        if len(observation_array) < self.n_states * 3:
            raise ValueError("too few observations for the requested number of states")

        seeds = np.random.SeedSequence(self.random_state).spawn(self.n_init)
        candidates = [self._fit_one(observation_array, seed=seed) for seed in seeds]
        objectives = np.asarray([candidate["objective"] for candidate in candidates])
        best_index = int(np.argmax(objectives))
        best = candidates[best_index]

        # Deterministic state labels: increasing mean on the first fitted
        # feature.  Economic names are deliberately not inferred here.
        order = np.argsort(best["means"][:, 0], kind="stable")
        self.initial_probabilities_ = best["initial"][order]
        self.initial_probabilities_ /= self.initial_probabilities_.sum()
        self.transition_matrix_ = best["transition"][np.ix_(order, order)]
        self.emission_means_ = best["means"][order]
        self.emission_covariances_ = best["covariances"][order]
        self.n_features_in_ = observation_array.shape[1]

        result = self._forward_backward_with_parameters(
            observation_array,
            self.initial_probabilities_,
            self.transition_matrix_,
            self.emission_means_,
            self.emission_covariances_,
        )
        self.filtered_probabilities_ = result.filtered_probabilities
        self.smoothed_probabilities_ = result.smoothed_probabilities
        self.expected_transition_counts_ = result.expected_transition_counts
        self.log_likelihood_ = result.log_likelihood
        self.regularized_objective_ = self._posterior_objective(
            self.log_likelihood_,
            self.transition_matrix_,
        )
        self.objective_by_initialization_ = objectives
        self.best_initialization_ = best_index
        self.objective_history_ = best["history"]
        self.converged_ = bool(best["converged"])
        self.n_iter_ = int(best["n_iter"])
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("StickyGaussianHMM must be fitted first")

    def forward_backward(
        self,
        observations: ArrayLike,
        *,
        initial_probabilities: np.ndarray | None = None,
    ) -> ForwardBackwardResult:
        """Evaluate filtered and smoothed state probabilities.

        ``initial_probabilities`` is the prior for the first supplied
        observation.  Passing it is required when a new validation sequence
        should continue from a fitted training endpoint rather than restart at
        the training sample's initial distribution.
        """

        self._check_fitted()
        observation_array = _as_finite_matrix(observations, name="observations")
        if observation_array.shape[1] != self.n_features_in_:
            raise ValueError("observations has an invalid feature width")
        initial = (
            self.initial_probabilities_
            if initial_probabilities is None
            else _validate_probability_vector(
                initial_probabilities,
                name="initial_probabilities",
                n_states=self.n_states,
            )
        )
        return self._forward_backward_with_parameters(
            observation_array,
            initial,
            self.transition_matrix_,
            self.emission_means_,
            self.emission_covariances_,
        )

    def sample_posterior_predictive_paths(
        self,
        *,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator,
        initial_filtered_probabilities: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw future state paths conditional on the latest filtered belief.

        The first returned state is at forecast time ``t+1``.  A current state is
        first drawn from ``p(z_t | y_1:t)`` and then propagated through the
        transition matrix.
        """

        self._check_fitted()
        if n_paths < 1 or horizon < 1:
            raise ValueError("n_paths and horizon must be positive")
        if initial_filtered_probabilities is None:
            current_probabilities = self.filtered_probabilities_[-1]
        else:
            current_probabilities = _validate_probability_vector(
                initial_filtered_probabilities,
                name="initial_filtered_probabilities",
                n_states=self.n_states,
            )

        current_states = rng.choice(
            self.n_states,
            size=n_paths,
            p=current_probabilities,
        )
        paths = np.empty((n_paths, horizon), dtype=int)
        for step in range(horizon):
            next_states = np.empty(n_paths, dtype=int)
            for state in range(self.n_states):
                selected = np.flatnonzero(current_states == state)
                if selected.size:
                    next_states[selected] = rng.choice(
                        self.n_states,
                        size=selected.size,
                        p=self.transition_matrix_[state],
                    )
            paths[:, step] = next_states
            current_states = next_states
        return paths

    def diagnostics(self) -> dict[str, Any]:
        """Return state occupancy, covariance, and convergence diagnostics."""

        self._check_fitted()
        state_occupancy = self.smoothed_probabilities_.mean(axis=0)
        minimum_covariance_eigenvalue = np.array(
            [np.linalg.eigvalsh(covariance).min() for covariance in self.emission_covariances_]
        )
        return {
            "state_occupancy": state_occupancy,
            "minimum_state_occupancy": float(state_occupancy.min()),
            "transition_diagonal": np.diag(self.transition_matrix_).copy(),
            "minimum_covariance_eigenvalue": minimum_covariance_eigenvalue,
            "log_likelihood": float(self.log_likelihood_),
            "regularized_objective": float(self.regularized_objective_),
            "converged": self.converged_,
            "n_iter": self.n_iter_,
            "best_initialization": self.best_initialization_,
            "n_initializations": self.n_init,
        }


class SwitchingDynamicFactorBaseline:
    """Multi-stage MAP/empirical-Bayes switching factor baseline.

    The estimator deliberately separates factor extraction, MAP-like HMM
    estimation, regime-specific VAR dynamics, and the observation equation.
    Training uses smoothed state probabilities; forecasting is conditioned only
    on the latest filtered probabilities.  Simulation draws full state paths and
    never averages regime-specific ``B``, ``D``, or residual correlations.
    """

    def __init__(
        self,
        *,
        n_states: int,
        n_factors: int,
        return_transform: str = "simple",
        factor_scale_floor: float = 1e-8,
        hmm_n_init: int = 8,
        hmm_max_iter: int = 250,
        hmm_tolerance: float = 1e-6,
        transition_pseudocount: float = 0.5,
        sticky_pseudocount: float = 8.0,
        minimum_covar: float = 1e-6,
        hmm_minimum_state_weight: float = 1e-3,
        var_ridge: float = 1e-4,
        factor_covariance_floor: float = 1e-8,
        maximum_spectral_radius: float = 0.995,
        observation_ridge: float = 1e-4,
        residual_correlation_shrinkage: float = 0.20,
        observation_scale_floor: float = 1e-6,
        random_state: int = 1729,
    ) -> None:
        self.n_states = int(n_states)
        self.n_factors = int(n_factors)
        self.return_transform = return_transform
        self.var_ridge = float(var_ridge)
        self.factor_covariance_floor = float(factor_covariance_floor)
        self.maximum_spectral_radius = float(maximum_spectral_radius)
        self.observation_ridge = float(observation_ridge)
        self.observation_scale_floor = float(observation_scale_floor)
        self.residual_correlation_shrinkage = float(residual_correlation_shrinkage)
        self.random_state = int(random_state)
        self.factor_model = DynamicFactorModel(
            n_factors=n_factors,
            return_transform=return_transform,  # type: ignore[arg-type]
            scale_floor=factor_scale_floor,
        )
        self.regime_model = StickyGaussianHMM(
            n_states=n_states,
            n_init=hmm_n_init,
            max_iter=hmm_max_iter,
            tolerance=hmm_tolerance,
            transition_pseudocount=transition_pseudocount,
            sticky_pseudocount=sticky_pseudocount,
            minimum_covar=minimum_covar,
            minimum_state_weight=hmm_minimum_state_weight,
            random_state=random_state,
        )

    def fit(self, returns: ArrayLike) -> SwitchingDynamicFactorBaseline:
        """Fit every stage on the supplied training fold."""

        return_array = _as_finite_matrix(returns, name="returns")
        self.factor_model.fit(returns)
        factors = self.factor_model.transform(returns)
        observations = self.factor_model.to_observation_space(returns)
        self.regime_model.fit(factors)
        training_probabilities = self.regime_model.smoothed_probabilities_

        self.factor_dynamics_: RegimeFactorVAR = fit_regime_factor_var(
            factors,
            training_probabilities,
            ridge=self.var_ridge,
            covariance_floor=self.factor_covariance_floor,
            maximum_spectral_radius=self.maximum_spectral_radius,
        )
        self.observation_mapping_: RegimeObservationMapping = fit_regime_observation_mapping(
            observations,
            factors,
            training_probabilities,
            ridge=self.observation_ridge,
            correlation_shrinkage=self.residual_correlation_shrinkage,
            scale_floor=self.observation_scale_floor,
        )
        self.training_factors_ = factors
        self.training_returns_ = return_array.copy()
        self.filtered_probabilities_ = self.regime_model.filtered_probabilities_.copy()
        self.smoothed_probabilities_ = self.regime_model.smoothed_probabilities_.copy()
        self.last_factor_ = factors[-1].copy()
        self.n_assets_ = return_array.shape[1]
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("SwitchingDynamicFactorBaseline must be fitted first")

    def filter(
        self,
        returns: ArrayLike,
        *,
        continue_from_training: bool = True,
    ) -> np.ndarray:
        """Filter a sequence without using its future observations.

        By default the first supplied row is treated as the observation
        immediately after the training fold.  Set ``continue_from_training`` to
        false only when evaluating an independent sequence.
        """

        self._check_fitted()
        factors = self.factor_model.transform(returns)
        initial = None
        if continue_from_training:
            initial = (
                self.regime_model.filtered_probabilities_[-1] @ self.regime_model.transition_matrix_
            )
        return self.regime_model.forward_backward(
            factors,
            initial_probabilities=initial,
        ).filtered_probabilities

    def smooth(
        self,
        returns: ArrayLike,
        *,
        continue_from_training: bool = True,
    ) -> np.ndarray:
        """Smooth a sequence, optionally continued from the training endpoint."""

        self._check_fitted()
        factors = self.factor_model.transform(returns)
        initial = None
        if continue_from_training:
            initial = (
                self.regime_model.filtered_probabilities_[-1] @ self.regime_model.transition_matrix_
            )
        return self.regime_model.forward_backward(
            factors,
            initial_probabilities=initial,
        ).smoothed_probabilities

    def sample_joint_paths(
        self,
        *,
        n_paths: int,
        horizon: int,
        random_state: int | None = None,
        initial_filtered_probabilities: np.ndarray | None = None,
    ) -> JointPathSample:
        """Generate joint regime, factor, and asset-level simple-return paths."""

        self._check_fitted()
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        regimes = self.regime_model.sample_posterior_predictive_paths(
            n_paths=n_paths,
            horizon=horizon,
            rng=rng,
            initial_filtered_probabilities=initial_filtered_probabilities,
        )
        factors = self.factor_dynamics_.sample_paths(
            self.last_factor_,
            regimes,
            rng=rng,
        )
        observations = self.observation_mapping_.sample_paths(
            factors,
            regimes,
            rng=rng,
        )
        flat_observations = observations.reshape(-1, self.n_assets_)
        simple_returns = self.factor_model.observation_to_simple_returns(flat_observations).reshape(
            observations.shape
        )
        return JointPathSample(
            regime_paths=regimes,
            factor_paths=factors,
            asset_return_paths=simple_returns,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Collect auditable diagnostics from every fitted stage."""

        self._check_fitted()
        factor_diagnostics = self.factor_model.diagnostics(self.training_returns_)
        return {
            "estimator_class": "MAP/empirical-Bayes multi-stage baseline",
            "factor": {
                "n_observations": factor_diagnostics.n_observations,
                "n_assets": factor_diagnostics.n_assets,
                "n_factors": factor_diagnostics.n_factors,
                "total_explained_variance_ratio": (
                    factor_diagnostics.total_explained_variance_ratio
                ),
                "reconstruction_rmse": factor_diagnostics.reconstruction_rmse,
                "maximum_absolute_reconstruction_error": (
                    factor_diagnostics.maximum_absolute_reconstruction_error
                ),
            },
            "regime": self.regime_model.diagnostics(),
            "factor_dynamics": self.factor_dynamics_.diagnostics(),
            "observation_mapping": self.observation_mapping_.diagnostics(),
        }
