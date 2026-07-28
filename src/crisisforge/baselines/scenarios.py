from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _validate_returns(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("returns must have shape (time, assets)")
    if values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("returns require at least three rows and one asset")
    if not np.isfinite(values).all():
        raise ValueError("returns contain missing or non-finite values")
    return values


def _validate_sample_request(num_scenarios: int, horizon: int) -> None:
    if num_scenarios < 1:
        raise ValueError("num_scenarios must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")


@dataclass
class HistoricalScenarioGenerator:
    """I.i.d. historical simulation benchmark."""

    returns_: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> HistoricalScenarioGenerator:
        self.returns_ = _validate_returns(returns)
        return self

    def sample(
        self,
        *,
        num_scenarios: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _validate_sample_request(num_scenarios, horizon)
        if self.returns_ is None:
            raise RuntimeError("fit must be called before sample")
        indices = rng.integers(
            0,
            len(self.returns_),
            size=(num_scenarios, horizon),
        )
        return self.returns_[indices]


@dataclass
class MovingBlockBootstrapGenerator:
    """Moving-block bootstrap preserving short-range serial dependence."""

    block_length: int = 20
    returns_: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> MovingBlockBootstrapGenerator:
        values = _validate_returns(returns)
        if self.block_length < 1 or self.block_length > len(values):
            raise ValueError("block_length must be between 1 and the number of rows")
        self.returns_ = values
        return self

    def sample(
        self,
        *,
        num_scenarios: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _validate_sample_request(num_scenarios, horizon)
        if self.returns_ is None:
            raise RuntimeError("fit must be called before sample")
        time_rows, assets = self.returns_.shape
        blocks_needed = int(np.ceil(horizon / self.block_length))
        max_start = time_rows - self.block_length
        output = np.empty((num_scenarios, horizon, assets), dtype=float)
        for scenario in range(num_scenarios):
            starts = rng.integers(0, max_start + 1, size=blocks_needed)
            blocks = [self.returns_[start : start + self.block_length] for start in starts]
            output[scenario] = np.concatenate(blocks, axis=0)[:horizon]
        return output


@dataclass
class GaussianScenarioGenerator:
    """Shrinkage multivariate-Gaussian benchmark."""

    shrinkage: float = 0.05
    model_log_returns: bool = True
    mean_: np.ndarray | None = None
    covariance_: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> GaussianScenarioGenerator:
        values = _validate_returns(returns)
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        model_values = _to_model_space(values, self.model_log_returns)
        covariance = np.cov(model_values, rowvar=False)
        covariance = np.atleast_2d(covariance)
        diagonal = np.diag(np.diag(covariance))
        covariance = (1.0 - self.shrinkage) * covariance + self.shrinkage * diagonal
        covariance += np.eye(covariance.shape[0]) * 1e-10
        self.mean_ = model_values.mean(axis=0)
        self.covariance_ = covariance
        return self

    def sample(
        self,
        *,
        num_scenarios: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _validate_sample_request(num_scenarios, horizon)
        if self.mean_ is None or self.covariance_ is None:
            raise RuntimeError("fit must be called before sample")
        draws = rng.multivariate_normal(
            self.mean_,
            self.covariance_,
            size=num_scenarios * horizon,
        )
        paths = draws.reshape(num_scenarios, horizon, -1)
        return _from_model_space(paths, self.model_log_returns)


@dataclass
class StudentTScenarioGenerator:
    """Elliptical multivariate-Student-t benchmark with covariance shrinkage."""

    shrinkage: float = 0.05
    degrees_of_freedom: float | None = None
    model_log_returns: bool = True
    location_: np.ndarray | None = None
    shape_: np.ndarray | None = None
    fitted_degrees_of_freedom_: float | None = None

    def fit(self, returns: np.ndarray) -> StudentTScenarioGenerator:
        values = _validate_returns(returns)
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        model_values = _to_model_space(values, self.model_log_returns)
        df = (
            float(self.degrees_of_freedom)
            if self.degrees_of_freedom is not None
            else _kurtosis_df_estimate(model_values)
        )
        if df <= 2.0:
            raise ValueError("degrees_of_freedom must exceed 2")

        covariance = np.atleast_2d(np.cov(model_values, rowvar=False))
        diagonal = np.diag(np.diag(covariance))
        covariance = (1.0 - self.shrinkage) * covariance + self.shrinkage * diagonal
        covariance += np.eye(covariance.shape[0]) * 1e-10

        # scipy's multivariate_t uses a shape matrix whose covariance is
        # shape * df / (df - 2).
        self.location_ = model_values.mean(axis=0)
        self.shape_ = covariance * (df - 2.0) / df
        self.fitted_degrees_of_freedom_ = df
        return self

    def sample(
        self,
        *,
        num_scenarios: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _validate_sample_request(num_scenarios, horizon)
        if self.location_ is None or self.shape_ is None or self.fitted_degrees_of_freedom_ is None:
            raise RuntimeError("fit must be called before sample")
        draws = stats.multivariate_t.rvs(
            loc=self.location_,
            shape=self.shape_,
            df=self.fitted_degrees_of_freedom_,
            size=num_scenarios * horizon,
            random_state=rng,
        )
        paths = np.asarray(draws).reshape(num_scenarios, horizon, -1)
        return _from_model_space(paths, self.model_log_returns)


@dataclass
class EWMFilteredHistoricalGenerator:
    """EWMA filtered historical simulation with joint residual resampling."""

    decay: float = 0.94
    model_log_returns: bool = True
    residuals_: np.ndarray | None = None
    last_variance_: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> EWMFilteredHistoricalGenerator:
        values = _validate_returns(returns)
        if not 0.0 < self.decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        model_values = _to_model_space(values, self.model_log_returns)
        variance = np.var(model_values, axis=0, ddof=1)
        variance = np.maximum(variance, 1e-12)
        residuals = np.empty_like(model_values)
        for row, observation in enumerate(model_values):
            sigma = np.sqrt(np.maximum(variance, 1e-12))
            residuals[row] = observation / sigma
            variance = self.decay * variance + (1.0 - self.decay) * observation**2
        residuals -= residuals.mean(axis=0, keepdims=True)
        self.residuals_ = residuals
        self.last_variance_ = variance
        return self

    def sample(
        self,
        *,
        num_scenarios: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _validate_sample_request(num_scenarios, horizon)
        if self.residuals_ is None or self.last_variance_ is None:
            raise RuntimeError("fit must be called before sample")
        assets = self.residuals_.shape[1]
        output = np.empty((num_scenarios, horizon, assets), dtype=float)
        variances = np.repeat(self.last_variance_[None, :], num_scenarios, axis=0)
        for step in range(horizon):
            indices = rng.integers(0, len(self.residuals_), size=num_scenarios)
            innovations = self.residuals_[indices]
            observations = np.sqrt(np.maximum(variances, 1e-12)) * innovations
            output[:, step, :] = observations
            variances = self.decay * variances + (1.0 - self.decay) * observations**2
        return _from_model_space(output, self.model_log_returns)


@dataclass
class StudentTCopulaScenarioGenerator:
    """Empirical marginals joined by a shrinkage Student-t copula."""

    degrees_of_freedom: float = 6.0
    shrinkage: float = 0.05
    sorted_marginals_: np.ndarray | None = None
    correlation_: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> StudentTCopulaScenarioGenerator:
        values = _validate_returns(returns)
        if self.degrees_of_freedom <= 2.0:
            raise ValueError("degrees_of_freedom must exceed 2")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        sample_size = len(values)
        ranks = np.column_stack(
            [
                stats.rankdata(values[:, column], method="average")
                for column in range(values.shape[1])
            ]
        )
        uniforms = (ranks - 0.5) / sample_size
        latent = stats.t.ppf(uniforms, df=self.degrees_of_freedom)
        correlation = np.atleast_2d(np.corrcoef(latent, rowvar=False))
        correlation = (1.0 - self.shrinkage) * correlation + self.shrinkage * np.eye(
            correlation.shape[0]
        )
        correlation += np.eye(correlation.shape[0]) * 1e-10
        self.sorted_marginals_ = np.sort(values, axis=0)
        self.correlation_ = correlation
        return self

    def sample(
        self,
        *,
        num_scenarios: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _validate_sample_request(num_scenarios, horizon)
        if self.sorted_marginals_ is None or self.correlation_ is None:
            raise RuntimeError("fit must be called before sample")
        total = num_scenarios * horizon
        latent = stats.multivariate_t.rvs(
            loc=np.zeros(self.correlation_.shape[0]),
            shape=self.correlation_,
            df=self.degrees_of_freedom,
            size=total,
            random_state=rng,
        )
        latent = np.asarray(latent).reshape(total, -1)
        uniforms = stats.t.cdf(latent, df=self.degrees_of_freedom)
        output = np.empty_like(uniforms)
        probability_grid = (np.arange(len(self.sorted_marginals_), dtype=float) + 0.5) / len(
            self.sorted_marginals_
        )
        for column in range(output.shape[1]):
            output[:, column] = np.interp(
                uniforms[:, column],
                probability_grid,
                self.sorted_marginals_[:, column],
            )
        return output.reshape(num_scenarios, horizon, -1)


@dataclass
class VARResidualBootstrapGenerator:
    """Ridge VAR(1) with joint residual bootstrap."""

    ridge: float = 1e-6
    model_log_returns: bool = True
    intercept_: np.ndarray | None = None
    coefficients_: np.ndarray | None = None
    residuals_: np.ndarray | None = None
    last_state_: np.ndarray | None = None

    def fit(self, returns: np.ndarray) -> VARResidualBootstrapGenerator:
        values = _validate_returns(returns)
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        model_values = _to_model_space(values, self.model_log_returns)
        design = np.column_stack([np.ones(len(model_values) - 1), model_values[:-1]])
        response = model_values[1:]
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ response)
        fitted = design @ beta
        residuals = response - fitted
        residuals -= residuals.mean(axis=0, keepdims=True)
        self.intercept_ = beta[0]
        self.coefficients_ = beta[1:]
        self.residuals_ = residuals
        self.last_state_ = model_values[-1]
        return self

    def sample(
        self,
        *,
        num_scenarios: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        _validate_sample_request(num_scenarios, horizon)
        if (
            self.intercept_ is None
            or self.coefficients_ is None
            or self.residuals_ is None
            or self.last_state_ is None
        ):
            raise RuntimeError("fit must be called before sample")
        assets = len(self.last_state_)
        states = np.repeat(self.last_state_[None, :], num_scenarios, axis=0)
        output = np.empty((num_scenarios, horizon, assets), dtype=float)
        for step in range(horizon):
            residual_indices = rng.integers(
                0,
                len(self.residuals_),
                size=num_scenarios,
            )
            states = (
                self.intercept_ + states @ self.coefficients_ + self.residuals_[residual_indices]
            )
            output[:, step, :] = states
        return _from_model_space(output, self.model_log_returns)


def _to_model_space(values: np.ndarray, use_log_returns: bool) -> np.ndarray:
    if not use_log_returns:
        return values
    if (values <= -1.0).any():
        raise ValueError("simple returns must exceed -100% for log1p modeling")
    return np.log1p(values)


def _from_model_space(values: np.ndarray, use_log_returns: bool) -> np.ndarray:
    return np.expm1(values) if use_log_returns else values


def _kurtosis_df_estimate(returns: np.ndarray) -> float:
    excess = stats.kurtosis(returns, axis=0, fisher=True, bias=False, nan_policy="omit")
    positive = excess[np.isfinite(excess) & (excess > 0.0)]
    if len(positive) == 0:
        return 30.0
    # Student-t excess kurtosis is 6/(nu-4) for nu>4.
    estimate = 6.0 / float(np.median(positive)) + 4.0
    return float(np.clip(estimate, 4.1, 30.0))
