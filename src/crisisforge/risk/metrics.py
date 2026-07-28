from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from scipy.special import xlogy


@dataclass(frozen=True)
class PortfolioRisk:
    confidence_level: float
    value_at_risk: float
    expected_shortfall: float
    mean_loss: float
    loss_standard_deviation: float
    scenario_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def aggregate_path_returns(
    scenario_paths: np.ndarray,
    *,
    return_type: str = "simple",
) -> np.ndarray:
    """Aggregate scenario paths of shape (scenarios, horizon, assets)."""
    paths = np.asarray(scenario_paths, dtype=float)
    if paths.ndim != 3:
        raise ValueError("scenario_paths must have shape (scenarios, horizon, assets)")
    if not np.isfinite(paths).all():
        raise ValueError("scenario_paths contain missing or non-finite values")
    if return_type == "simple":
        if (paths <= -1.0).any():
            raise ValueError("simple returns cannot be less than or equal to -100%")
        return np.prod(1.0 + paths, axis=1) - 1.0
    if return_type == "log":
        return np.sum(paths, axis=1)
    raise ValueError("return_type must be 'simple' or 'log'")


def portfolio_losses(
    cumulative_returns: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    returns = np.asarray(cumulative_returns, dtype=float)
    portfolio_weights = np.asarray(weights, dtype=float)
    if returns.ndim != 2:
        raise ValueError("cumulative_returns must have shape (scenarios, assets)")
    if portfolio_weights.shape != (returns.shape[1],):
        raise ValueError("weights must have one entry per asset")
    if not np.isfinite(returns).all() or not np.isfinite(portfolio_weights).all():
        raise ValueError("returns and weights must be finite")
    return -(returns @ portfolio_weights)


def empirical_var(losses: np.ndarray, confidence_level: float) -> float:
    values = _validate_losses(losses, confidence_level)
    return float(np.quantile(values, confidence_level, method="higher"))


def empirical_expected_shortfall(losses: np.ndarray, confidence_level: float) -> float:
    """Exact integral of the right-continuous empirical loss quantile."""
    values = np.sort(_validate_losses(losses, confidence_level))
    sample_size = len(values)
    left = np.arange(sample_size, dtype=float) / sample_size
    right = (np.arange(sample_size, dtype=float) + 1.0) / sample_size
    weights = np.maximum(right - np.maximum(left, confidence_level), 0.0)
    return float(np.sum(weights * values) / (1.0 - confidence_level))


def estimate_portfolio_risk(
    losses: np.ndarray,
    confidence_level: float,
) -> PortfolioRisk:
    values = _validate_losses(losses, confidence_level)
    return PortfolioRisk(
        confidence_level=confidence_level,
        value_at_risk=empirical_var(values, confidence_level),
        expected_shortfall=empirical_expected_shortfall(values, confidence_level),
        mean_loss=float(values.mean()),
        loss_standard_deviation=float(values.std(ddof=1)),
        scenario_count=int(len(values)),
    )


def co_crash_probability(
    cumulative_returns: np.ndarray,
    thresholds: np.ndarray,
    *,
    minimum_fraction: float,
) -> float:
    returns = np.asarray(cumulative_returns, dtype=float)
    cutoffs = np.asarray(thresholds, dtype=float)
    if returns.ndim != 2:
        raise ValueError("cumulative_returns must have shape (scenarios, assets)")
    if cutoffs.shape != (returns.shape[1],):
        raise ValueError("thresholds must have one entry per asset")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must lie in (0, 1]")
    crash_fraction = np.mean(returns <= cutoffs, axis=1)
    return float(np.mean(crash_fraction >= minimum_fraction))


def fit_co_crash_thresholds(
    training_cumulative_returns: np.ndarray,
    *,
    marginal_quantile: float = 0.05,
) -> np.ndarray:
    """Freeze marginal crash thresholds using training outcomes only."""
    returns = np.asarray(training_cumulative_returns, dtype=float)
    if returns.ndim != 2 or len(returns) == 0:
        raise ValueError(
            "training_cumulative_returns must have shape (observations, assets)"
        )
    if not np.isfinite(returns).all():
        raise ValueError("training cumulative returns must be finite")
    if not 0.0 < marginal_quantile < 0.5:
        raise ValueError("marginal_quantile must lie in (0, 0.5)")
    return np.quantile(returns, marginal_quantile, axis=0, method="linear")


def realized_co_crash(
    cumulative_return: np.ndarray,
    thresholds: np.ndarray,
    *,
    minimum_fraction: float,
) -> bool:
    realized = np.asarray(cumulative_return, dtype=float)
    cutoffs = np.asarray(thresholds, dtype=float)
    if realized.ndim != 1 or realized.shape != cutoffs.shape:
        raise ValueError("cumulative_return and thresholds must be equal-length vectors")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must lie in (0, 1]")
    return bool(np.mean(realized <= cutoffs) >= minimum_fraction)


def kupiec_unconditional_coverage_test(
    violations: np.ndarray,
    confidence_level: float,
) -> dict[str, float | int]:
    hits = np.asarray(violations, dtype=bool)
    if hits.ndim != 1 or len(hits) == 0:
        raise ValueError("violations must be a non-empty one-dimensional array")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    expected_probability = 1.0 - confidence_level
    count = int(hits.sum())
    total = int(len(hits))
    observed_probability = count / total
    eps = np.finfo(float).eps
    observed_clipped = np.clip(observed_probability, eps, 1.0 - eps)
    null_log_likelihood = (total - count) * np.log(1.0 - expected_probability)
    null_log_likelihood += count * np.log(expected_probability)
    alternative_log_likelihood = (total - count) * np.log(1.0 - observed_clipped)
    alternative_log_likelihood += count * np.log(observed_clipped)
    statistic = float(-2.0 * (null_log_likelihood - alternative_log_likelihood))
    return {
        "observations": total,
        "violations": count,
        "expected_violation_rate": expected_probability,
        "observed_violation_rate": observed_probability,
        "lr_uc": statistic,
        "p_value": float(stats.chi2.sf(statistic, df=1)),
    }


def christoffersen_independence_test(
    violations: np.ndarray,
) -> dict[str, float | int]:
    """Likelihood-ratio test for serial independence of VaR violations."""
    hits = np.asarray(violations, dtype=bool)
    if hits.ndim != 1 or len(hits) < 2:
        raise ValueError("violations must contain at least two observations")
    previous = hits[:-1].astype(int)
    current = hits[1:].astype(int)
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    total = n00 + n01 + n10 + n11
    unconditional = (n01 + n11) / total
    probability_01 = _safe_probability(n01, n00 + n01)
    probability_11 = _safe_probability(n11, n10 + n11)
    log_null = xlogy(n00 + n10, 1.0 - unconditional) + xlogy(
        n01 + n11,
        unconditional,
    )
    log_alternative = (
        xlogy(n00, 1.0 - probability_01)
        + xlogy(n01, probability_01)
        + xlogy(n10, 1.0 - probability_11)
        + xlogy(n11, probability_11)
    )
    statistic = float(max(-2.0 * (log_null - log_alternative), 0.0))
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_ind": statistic,
        "p_value": float(stats.chi2.sf(statistic, df=1)),
    }


def christoffersen_conditional_coverage_test(
    violations: np.ndarray,
    confidence_level: float,
) -> dict[str, float | int]:
    unconditional = kupiec_unconditional_coverage_test(
        violations,
        confidence_level,
    )
    independence = christoffersen_independence_test(violations)
    statistic = float(unconditional["lr_uc"] + independence["lr_ind"])
    return {
        **unconditional,
        **{key: value for key, value in independence.items() if key != "p_value"},
        "lr_cc": statistic,
        "p_value_cc": float(stats.chi2.sf(statistic, df=2)),
        "p_value_independence": float(independence["p_value"]),
    }


def joint_var_es_score(
    realized_losses: np.ndarray,
    value_at_risk_forecasts: np.ndarray,
    expected_shortfall_forecasts: np.ndarray,
    confidence_level: float,
) -> float:
    """A strictly consistent exponential Fissler–Ziegel VaR–ES score.

    Upper-tail losses are mapped to the equivalent lower-tail return problem.
    Lower scores are better and comparisons are meaningful only on the same
    realized series and confidence level.
    """
    losses = np.asarray(realized_losses, dtype=float)
    var = np.asarray(value_at_risk_forecasts, dtype=float)
    es = np.asarray(expected_shortfall_forecasts, dtype=float)
    if losses.ndim != 1 or losses.shape != var.shape or losses.shape != es.shape:
        raise ValueError("losses, VaR, and ES forecasts must be equal-length vectors")
    if not np.isfinite(losses).all() or not np.isfinite(var).all() or not np.isfinite(es).all():
        raise ValueError("losses, VaR, and ES forecasts must be finite")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if (es < var).any():
        raise ValueError("Expected Shortfall forecasts must not be below VaR forecasts")
    tail_probability = 1.0 - confidence_level
    transformed_y = -losses
    transformed_q = -var
    transformed_e = -es
    indicator = transformed_y <= transformed_q
    score = np.exp(transformed_e) * (
        transformed_e
        - transformed_q
        + indicator * (transformed_q - transformed_y) / tail_probability
        - 1.0
    )
    return float(np.mean(score))


def energy_score(
    samples: np.ndarray,
    observation: np.ndarray,
    *,
    beta: float = 1.0,
    max_pair_draws: int = 50_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Multivariate energy score with bounded pairwise Monte Carlo cost."""
    scenarios = np.asarray(samples, dtype=float)
    realized = np.asarray(observation, dtype=float)
    if scenarios.ndim != 2 or realized.shape != (scenarios.shape[1],):
        raise ValueError("samples must be (scenarios, assets) and observation an asset vector")
    if not np.isfinite(scenarios).all() or not np.isfinite(realized).all():
        raise ValueError("samples and observation must be finite")
    if not 0.0 < beta < 2.0:
        raise ValueError("beta must lie in (0, 2)")
    first_term = float(np.mean(np.linalg.norm(scenarios - realized, axis=1) ** beta))
    pair_count = len(scenarios) ** 2
    if pair_count <= max_pair_draws:
        pair_distances = np.linalg.norm(
            scenarios[:, None, :] - scenarios[None, :, :],
            axis=2,
        )
        second_term = float(np.mean(pair_distances**beta))
    else:
        generator = rng or np.random.default_rng(0)
        left = generator.integers(0, len(scenarios), size=max_pair_draws)
        right = generator.integers(0, len(scenarios), size=max_pair_draws)
        second_term = float(
            np.mean(np.linalg.norm(scenarios[left] - scenarios[right], axis=1) ** beta)
        )
    return first_term - 0.5 * second_term


def variogram_score(
    samples: np.ndarray,
    observation: np.ndarray,
    *,
    order: float = 0.5,
) -> float:
    """Unweighted multivariate variogram score."""
    scenarios = np.asarray(samples, dtype=float)
    realized = np.asarray(observation, dtype=float)
    if scenarios.ndim != 2 or realized.shape != (scenarios.shape[1],):
        raise ValueError("samples must be (scenarios, assets) and observation an asset vector")
    if not np.isfinite(scenarios).all() or not np.isfinite(realized).all():
        raise ValueError("samples and observation must be finite")
    if order <= 0.0:
        raise ValueError("order must be positive")
    realized_pairwise = np.abs(realized[:, None] - realized[None, :]) ** order
    simulated_pairwise = np.mean(
        np.abs(scenarios[:, :, None] - scenarios[:, None, :]) ** order,
        axis=0,
    )
    upper = np.triu_indices(len(realized), k=1)
    return float(np.sum((realized_pairwise[upper] - simulated_pairwise[upper]) ** 2))


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    forecasts = np.asarray(probabilities, dtype=float)
    labels = np.asarray(outcomes, dtype=float)
    if forecasts.shape != labels.shape:
        raise ValueError("probabilities and outcomes must have the same shape")
    if ((forecasts < 0.0) | (forecasts > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("outcomes must be binary")
    return float(np.mean((forecasts - labels) ** 2))


def _validate_losses(losses: np.ndarray, confidence_level: float) -> np.ndarray:
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("losses must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("losses contain missing or non-finite values")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    return values


def _safe_probability(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
