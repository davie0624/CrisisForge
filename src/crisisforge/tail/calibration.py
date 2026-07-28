"""Training-fold tail weighting, POT-GPD fitting, and sequential calibration.

The tools in this module are deliberately separate:

* importance weights alter only a second-stage diffusion training objective;
* POT-GPD estimates a univariate upper loss tail and refuses unsupported fits;
* rolling conformal correction updates only after a registered forecast outcome
  has arrived.

None of these components supplies a causal interpretation or an Expected
Shortfall coverage guarantee.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy import stats

TailDirection = Literal["two_sided", "downside", "upside"]
SparseTailPolicy = Literal["raise", "empirical"]


def _finite_array(
    values: np.ndarray | Iterable[float],
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if array.size < 1:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class TailImportanceResult:
    """Fold-local severity scores and normalized training weights."""

    severity: np.ndarray
    threshold: float
    weights: np.ndarray
    raw_weights: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    tail_count: int
    direction: TailDirection


def training_window_severity(
    training_windows: np.ndarray,
    *,
    direction: TailDirection = "two_sided",
    scale_floor: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute robust severity using statistics fitted on training windows only.

    Input shape is ``(windows, horizon, features)``. For each path, severity is
    the largest cross-feature root-mean-square standardized shock across the
    horizon. ``downside`` retains only negative standardized shocks and
    ``upside`` only positive shocks. The default is suitable for latent factors
    whose economic sign may not be identified.
    """

    windows = _finite_array(training_windows, name="training_windows", ndim=3)
    if windows.shape[0] < 2:
        raise ValueError("at least two training windows are required")
    if direction not in {"two_sided", "downside", "upside"}:
        raise ValueError("direction must be 'two_sided', 'downside', or 'upside'")
    if scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive")

    flattened = windows.reshape(-1, windows.shape[-1])
    center = np.median(flattened, axis=0)
    median_absolute_deviation = np.median(
        np.abs(flattened - center),
        axis=0,
    )
    robust_scale = 1.4826 * median_absolute_deviation
    standard_scale = flattened.std(axis=0, ddof=0)
    scale = np.where(
        robust_scale > scale_floor,
        robust_scale,
        np.maximum(standard_scale, scale_floor),
    )
    standardized = (windows - center) / scale
    if direction == "two_sided":
        shocks = np.abs(standardized)
    elif direction == "downside":
        shocks = np.maximum(-standardized, 0.0)
    else:
        shocks = np.maximum(standardized, 0.0)
    severity = np.sqrt(np.mean(np.square(shocks), axis=2)).max(axis=1)
    return severity, center, scale


def fit_training_importance_weights(
    training_windows: np.ndarray,
    *,
    tail_quantile: float = 0.90,
    strength: float = 3.0,
    maximum_weight: float = 8.0,
    direction: TailDirection = "two_sided",
    scale_floor: float = 1.0e-8,
) -> TailImportanceResult:
    """Fit capped, mean-one weights from training-window severity.

    The returned weights are finite, non-negative, capped before and after
    normalization, and have mean one up to floating-point error. If all
    severities are equal, the function safely returns unit weights.
    """

    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must lie in (0, 1)")
    if strength < 0.0:
        raise ValueError("strength cannot be negative")
    if maximum_weight < 1.0:
        raise ValueError("maximum_weight must be at least one")
    severity, center, scale = training_window_severity(
        training_windows,
        direction=direction,
        scale_floor=scale_floor,
    )
    threshold = float(np.quantile(severity, tail_quantile))
    excess = np.maximum(severity - threshold, 0.0)
    positive_excess = excess[excess > 0.0]
    if positive_excess.size:
        excess_scale = max(
            float(np.median(positive_excess)),
            float(np.std(severity)),
            scale_floor,
        )
        raw_weights = 1.0 + strength * excess / excess_scale
    else:
        raw_weights = np.ones_like(severity)
    raw_weights = np.nan_to_num(
        raw_weights,
        nan=1.0,
        posinf=maximum_weight,
        neginf=1.0,
    )
    raw_weights = np.clip(raw_weights, 1.0, maximum_weight)

    weights = raw_weights / max(float(raw_weights.mean()), scale_floor)
    weights = np.clip(weights, 0.0, maximum_weight)
    weights /= max(float(weights.mean()), scale_floor)
    if not np.isfinite(weights).all():
        raise FloatingPointError("importance-weight normalization was not finite")
    return TailImportanceResult(
        severity=severity,
        threshold=threshold,
        weights=weights,
        raw_weights=raw_weights,
        center=center,
        scale=scale,
        tail_count=int(np.count_nonzero(severity > threshold)),
        direction=direction,
    )


@dataclass(frozen=True)
class POTThresholdDiagnostics:
    """Diagnostics for a single peaks-over-threshold candidate."""

    threshold: float
    threshold_quantile: float
    n_observations: int
    n_exceedances: int
    exceedance_rate: float
    mean_excess: float | None
    shape: float | None
    scale: float | None
    status: str
    message: str


class SparseTailError(ValueError):
    """Raised when a requested GPD fit lacks enough exceedances."""

    def __init__(self, diagnostics: POTThresholdDiagnostics) -> None:
        super().__init__(diagnostics.message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class POTGPDModel:
    """Univariate upper-loss POT model, possibly using empirical fallback."""

    threshold: float
    threshold_probability: float
    shape: float | None
    scale: float | None
    empirical_losses: np.ndarray
    diagnostics: POTThresholdDiagnostics

    @property
    def uses_gpd(self) -> bool:
        return self.shape is not None and self.scale is not None

    def quantile(self, probability: float) -> float:
        """Estimate an upper-loss quantile."""

        if not 0.0 < probability < 1.0:
            raise ValueError("probability must lie in (0, 1)")
        if not self.uses_gpd or probability <= self.threshold_probability:
            return float(
                np.quantile(self.empirical_losses, probability, method="higher")
            )
        conditional_probability = (
            probability - self.threshold_probability
        ) / (1.0 - self.threshold_probability)
        excess_quantile = stats.genpareto.ppf(
            conditional_probability,
            c=float(self.shape),
            loc=0.0,
            scale=float(self.scale),
        )
        estimate = self.threshold + float(excess_quantile)
        if not math.isfinite(estimate):
            raise FloatingPointError("GPD quantile is non-finite")
        return estimate

    def expected_shortfall(self, probability: float) -> float:
        """Estimate ES when the fitted GPD has finite mean.

        Empirical fallback averages observations at or above the empirical
        quantile. This estimator does not inherit conformal coverage.
        """

        quantile = self.quantile(probability)
        if not self.uses_gpd or probability <= self.threshold_probability:
            tail = self.empirical_losses[self.empirical_losses >= quantile]
            return float(tail.mean())
        shape = float(self.shape)
        scale = float(self.scale)
        if shape >= 1.0:
            return math.inf
        return float((quantile + scale - shape * self.threshold) / (1.0 - shape))


def _sparse_diagnostics(
    *,
    threshold: float,
    threshold_quantile: float,
    losses: np.ndarray,
    exceedances: np.ndarray,
    minimum_exceedances: int,
    status: str = "sparse",
) -> POTThresholdDiagnostics:
    message = (
        f"POT fit refused: {len(exceedances)} exceedances are below the "
        f"minimum of {minimum_exceedances}"
    )
    return POTThresholdDiagnostics(
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        n_observations=len(losses),
        n_exceedances=len(exceedances),
        exceedance_rate=len(exceedances) / len(losses),
        mean_excess=float(exceedances.mean()) if len(exceedances) else None,
        shape=None,
        scale=None,
        status=status,
        message=message,
    )


def fit_pot_gpd(
    losses: np.ndarray | Iterable[float],
    *,
    threshold_quantile: float = 0.95,
    minimum_exceedances: int = 25,
    sparse_policy: SparseTailPolicy = "raise",
) -> POTGPDModel:
    """Fit a zero-location GPD above an empirical loss threshold.

    ``sparse_policy="raise"`` is the default because silent parametric
    extrapolation from a tiny tail is unsafe. ``"empirical"`` returns a model
    whose quantile and ES methods use the observed sample instead.
    """

    values = _finite_array(losses, name="losses", ndim=1)
    if len(values) < 3:
        raise ValueError("at least three loss observations are required")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must lie in (0, 1)")
    if minimum_exceedances < 3:
        raise ValueError("minimum_exceedances must be at least three")
    if sparse_policy not in {"raise", "empirical"}:
        raise ValueError("sparse_policy must be 'raise' or 'empirical'")

    ordered = np.sort(values)
    threshold = float(np.quantile(ordered, threshold_quantile))
    exceedances = ordered[ordered > threshold] - threshold
    threshold_probability = 1.0 - len(exceedances) / len(ordered)
    if len(exceedances) < minimum_exceedances:
        diagnostics = _sparse_diagnostics(
            threshold=threshold,
            threshold_quantile=threshold_quantile,
            losses=ordered,
            exceedances=exceedances,
            minimum_exceedances=minimum_exceedances,
        )
        if sparse_policy == "raise":
            raise SparseTailError(diagnostics)
        return POTGPDModel(
            threshold=threshold,
            threshold_probability=threshold_probability,
            shape=None,
            scale=None,
            empirical_losses=ordered,
            diagnostics=diagnostics,
        )

    try:
        shape, _, scale = stats.genpareto.fit(exceedances, floc=0.0)
    except (FloatingPointError, RuntimeError, ValueError) as error:
        diagnostics = _sparse_diagnostics(
            threshold=threshold,
            threshold_quantile=threshold_quantile,
            losses=ordered,
            exceedances=exceedances,
            minimum_exceedances=minimum_exceedances,
            status="fit_failed",
        )
        diagnostics = replace(diagnostics, message=f"GPD fit failed: {error}")
        if sparse_policy == "raise":
            raise SparseTailError(diagnostics) from error
        return POTGPDModel(
            threshold=threshold,
            threshold_probability=threshold_probability,
            shape=None,
            scale=None,
            empirical_losses=ordered,
            diagnostics=diagnostics,
        )

    if not np.isfinite([shape, scale]).all() or scale <= 0.0:
        diagnostics = _sparse_diagnostics(
            threshold=threshold,
            threshold_quantile=threshold_quantile,
            losses=ordered,
            exceedances=exceedances,
            minimum_exceedances=minimum_exceedances,
            status="fit_failed",
        )
        diagnostics = replace(
            diagnostics,
            message="GPD fit produced invalid parameters",
        )
        if sparse_policy == "raise":
            raise SparseTailError(diagnostics)
        return POTGPDModel(
            threshold=threshold,
            threshold_probability=threshold_probability,
            shape=None,
            scale=None,
            empirical_losses=ordered,
            diagnostics=diagnostics,
        )

    diagnostics = POTThresholdDiagnostics(
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        n_observations=len(ordered),
        n_exceedances=len(exceedances),
        exceedance_rate=len(exceedances) / len(ordered),
        mean_excess=float(exceedances.mean()),
        shape=float(shape),
        scale=float(scale),
        status="fitted",
        message="GPD fit completed; inspect threshold stability before use",
    )
    return POTGPDModel(
        threshold=threshold,
        threshold_probability=threshold_probability,
        shape=float(shape),
        scale=float(scale),
        empirical_losses=ordered,
        diagnostics=diagnostics,
    )


def pot_threshold_diagnostics(
    losses: np.ndarray | Iterable[float],
    *,
    threshold_quantiles: Iterable[float] = (0.90, 0.925, 0.95, 0.975),
    minimum_exceedances: int = 25,
) -> tuple[POTThresholdDiagnostics, ...]:
    """Return comparable diagnostics across candidate thresholds."""

    values = _finite_array(losses, name="losses", ndim=1)
    diagnostics: list[POTThresholdDiagnostics] = []
    for threshold_quantile in threshold_quantiles:
        model = fit_pot_gpd(
            values,
            threshold_quantile=float(threshold_quantile),
            minimum_exceedances=minimum_exceedances,
            sparse_policy="empirical",
        )
        diagnostics.append(model.diagnostics)
    return tuple(diagnostics)


@dataclass(frozen=True)
class ConformalQuantileForecast:
    """Audit record created before a realized outcome is available."""

    forecast_id: Hashable
    base_quantile: float
    correction: float
    calibrated_quantile: float
    n_calibration_scores: int


class RollingConformalQuantileCalibrator:
    """Sequential upper-quantile correction with delayed outcome updates.

    A forecast must first be registered with :meth:`issue`. Its score can enter
    the rolling calibration window only through :meth:`observe`, which uses the
    base quantile saved at issue time. This prevents accidental same-period
    outcome leakage. It does not establish exchangeability and does not
    calibrate Expected Shortfall.
    """

    def __init__(
        self,
        *,
        coverage_level: float,
        window_size: int = 250,
        minimum_history: int = 20,
        cold_start_correction: float = 0.0,
    ) -> None:
        if not 0.0 < coverage_level < 1.0:
            raise ValueError("coverage_level must lie in (0, 1)")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0 <= minimum_history <= window_size:
            raise ValueError("minimum_history must lie between zero and window_size")
        if not math.isfinite(cold_start_correction):
            raise ValueError("cold_start_correction must be finite")
        self.coverage_level = float(coverage_level)
        self.window_size = int(window_size)
        self.minimum_history = int(minimum_history)
        self.cold_start_correction = float(cold_start_correction)
        self._scores: deque[float] = deque(maxlen=window_size)
        self._pending: dict[Hashable, float] = {}
        self._seen_ids: set[Hashable] = set()

    @property
    def calibration_scores(self) -> tuple[float, ...]:
        return tuple(self._scores)

    @property
    def pending_forecast_ids(self) -> tuple[Hashable, ...]:
        return tuple(self._pending)

    def _correction(self) -> float:
        if len(self._scores) < self.minimum_history or not self._scores:
            return self.cold_start_correction
        ordered = np.sort(np.asarray(self._scores, dtype=float))
        rank = math.ceil((len(ordered) + 1) * self.coverage_level)
        rank = min(max(rank, 1), len(ordered))
        return float(ordered[rank - 1])

    def issue(
        self,
        forecast_id: Hashable,
        *,
        base_quantile: float,
    ) -> ConformalQuantileForecast:
        """Register and return a calibrated forecast before its outcome."""

        if forecast_id in self._seen_ids:
            raise ValueError("forecast_id has already been used")
        if not math.isfinite(base_quantile):
            raise ValueError("base_quantile must be finite")
        correction = self._correction()
        self._pending[forecast_id] = float(base_quantile)
        self._seen_ids.add(forecast_id)
        return ConformalQuantileForecast(
            forecast_id=forecast_id,
            base_quantile=float(base_quantile),
            correction=correction,
            calibrated_quantile=float(base_quantile + correction),
            n_calibration_scores=len(self._scores),
        )

    def observe(self, forecast_id: Hashable, *, realized_loss: float) -> float:
        """Resolve one pending forecast and only then append its conformity score."""

        if forecast_id not in self._pending:
            raise KeyError("forecast_id is not pending; issue it before observing")
        if not math.isfinite(realized_loss):
            raise ValueError("realized_loss must be finite")
        base_quantile = self._pending.pop(forecast_id)
        score = float(realized_loss - base_quantile)
        self._scores.append(score)
        return score
