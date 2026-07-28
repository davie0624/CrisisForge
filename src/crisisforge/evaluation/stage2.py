"""Leakage-audited Stage 2 pilot for one-shot factor-path diffusion.

The runner fits the Stage 1 factor/HMM/observation system on the chronological
training split only, constructs one-shot ``H x q`` factor targets, trains a
conditional temporal DDPM, and reports only on a late validation segment.  The
test split is never used for fitting, checkpoint selection, or evaluation.

This is deliberately an engineering pilot.  In particular, future HMM regimes
are sampled independently of diffusion factor paths conditional on the same
origin belief, and Stage 1 is MAP/empirical-Bayes rather than a full Bayesian
parameter posterior.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from crisisforge.config import load_yaml, project_root_from_module
from crisisforge.data.validation import hash_file
from crisisforge.diffusion import ConditionalTemporalDDPM
from crisisforge.evaluation.rolling import rolling_cumulative_returns
from crisisforge.evaluation.stage1 import build_switching_factor_model
from crisisforge.regimes import SwitchingDynamicFactorBaseline
from crisisforge.risk import (
    aggregate_path_returns,
    brier_score,
    christoffersen_conditional_coverage_test,
    co_crash_probability,
    empirical_expected_shortfall,
    empirical_var,
    energy_score,
    fit_co_crash_thresholds,
    joint_var_es_score,
    portfolio_losses,
    realized_co_crash,
    variogram_score,
)
from crisisforge.tail import fit_training_importance_weights


@dataclass(frozen=True)
class TrainOnlyStandardizer:
    """Feature-wise center and scale fitted on an explicitly supplied train set."""

    mean: np.ndarray
    scale: np.ndarray
    n_fit_rows: int

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        *,
        scale_floor: float = 1.0e-8,
    ) -> TrainOnlyStandardizer:
        array = _finite_matrix(values, name="standardizer_values")
        if scale_floor <= 0.0:
            raise ValueError("scale_floor must be positive")
        if len(array) < 2:
            raise ValueError("a standardizer requires at least two training rows")
        scale = np.maximum(array.std(axis=0, ddof=0), scale_floor)
        return cls(
            mean=array.mean(axis=0),
            scale=scale,
            n_fit_rows=len(array),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != len(self.mean):
            raise ValueError("standardizer feature width does not match")
        transformed = (array - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("standardized values contain non-finite entries")
        return transformed

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != len(self.mean):
            raise ValueError("standardizer feature width does not match")
        restored = array * self.scale + self.mean
        if not np.isfinite(restored).all():
            raise ValueError("inverse-standardized values contain non-finite entries")
        return restored


@dataclass(frozen=True)
class OneShotWindowBundle:
    """Arrays plus explicit positional boundaries for leakage audits."""

    past_context: np.ndarray
    future_factors: np.ndarray
    regime_probabilities: np.ndarray
    origin_positions: np.ndarray
    context_start_positions: np.ndarray
    context_end_positions: np.ndarray
    target_start_positions: np.ndarray
    target_end_positions: np.ndarray
    origin_dates: np.ndarray
    context_end_dates: np.ndarray
    target_start_dates: np.ndarray
    target_end_dates: np.ndarray

    def __len__(self) -> int:
        return int(len(self.origin_positions))

    def standardized(
        self,
        *,
        context_standardizer: TrainOnlyStandardizer,
        factor_standardizer: TrainOnlyStandardizer,
    ) -> OneShotWindowBundle:
        return OneShotWindowBundle(
            past_context=context_standardizer.transform(self.past_context),
            future_factors=factor_standardizer.transform(self.future_factors),
            regime_probabilities=self.regime_probabilities.copy(),
            origin_positions=self.origin_positions.copy(),
            context_start_positions=self.context_start_positions.copy(),
            context_end_positions=self.context_end_positions.copy(),
            target_start_positions=self.target_start_positions.copy(),
            target_end_positions=self.target_end_positions.copy(),
            origin_dates=self.origin_dates.copy(),
            context_end_dates=self.context_end_dates.copy(),
            target_start_dates=self.target_start_dates.copy(),
            target_end_dates=self.target_end_dates.copy(),
        )

    def boundary_frame(self, *, partition: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "partition": partition,
                "origin_position": self.origin_positions,
                "context_start_position": self.context_start_positions,
                "context_end_position": self.context_end_positions,
                "target_start_position": self.target_start_positions,
                "target_end_position": self.target_end_positions,
                "origin_date": self.origin_dates,
                "context_end_date": self.context_end_dates,
                "target_start_date": self.target_start_dates,
                "target_end_date": self.target_end_dates,
            }
        )


class OneShotFactorDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic tensor view over pre-built one-shot windows."""

    def __init__(
        self,
        windows: OneShotWindowBundle,
        *,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        if len(windows) < 1:
            raise ValueError("the dataset requires at least one window")
        if sample_weights is None:
            weights = np.ones(len(windows), dtype=np.float32)
        else:
            weights = np.asarray(sample_weights, dtype=float)
            if weights.shape != (len(windows),):
                raise ValueError("sample_weights must have one value per window")
            if not np.isfinite(weights).all() or np.any(weights < 0.0):
                raise ValueError("sample_weights must be finite and non-negative")
        self.clean_paths = torch.as_tensor(
            np.ascontiguousarray(windows.future_factors),
            dtype=torch.float32,
        )
        self.past_context = torch.as_tensor(
            np.ascontiguousarray(windows.past_context),
            dtype=torch.float32,
        )
        self.regime_probabilities = torch.as_tensor(
            np.ascontiguousarray(windows.regime_probabilities),
            dtype=torch.float32,
        )
        self.sample_weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.clean_paths.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "clean_paths": self.clean_paths[index],
            "past_context": self.past_context[index],
            "regime_probabilities": self.regime_probabilities[index],
            "sample_weight": self.sample_weights[index],
        }


def _finite_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_index(index: pd.Index, *, expected_rows: int) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(index)
    if len(dates) != expected_rows:
        raise ValueError("index length does not match the arrays")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("index must be unique and increasing")
    return dates


def eligible_origin_positions(
    *,
    first_origin: int,
    last_target_position: int,
    history_length: int,
    horizon: int,
    stride: int,
) -> np.ndarray:
    """Return origins whose history and complete future target both exist."""

    if history_length < 1 or horizon < 1 or stride < 1:
        raise ValueError("history_length, horizon, and stride must be positive")
    start = max(int(first_origin), history_length - 1)
    final_origin = int(last_target_position) - horizon
    if final_origin < start:
        return np.empty(0, dtype=int)
    return np.arange(start, final_origin + 1, stride, dtype=int)


def build_one_shot_windows(
    *,
    index: pd.Index,
    factors: np.ndarray,
    macro_features: np.ndarray,
    filtered_probabilities: np.ndarray,
    origin_positions: np.ndarray,
    history_length: int,
    horizon: int,
) -> OneShotWindowBundle:
    """Build strict ``history -> future`` windows with auditable boundaries.

    For origin position ``t`` the context is exactly
    ``[t-history_length+1, ..., t]`` and the target is exactly
    ``[t+1, ..., t+horizon]``.  The regime vector is the one-sided filtered
    probability at ``t``.  No future macro or future regime input is accepted.
    """

    factor_array = _finite_matrix(factors, name="factors")
    macro_array = _finite_matrix(macro_features, name="macro_features")
    regime_array = _finite_matrix(
        filtered_probabilities,
        name="filtered_probabilities",
    )
    if not (len(factor_array) == len(macro_array) == len(regime_array)):
        raise ValueError("factors, macro_features, and probabilities need equal rows")
    if np.any(regime_array < 0.0) or not np.allclose(
        regime_array.sum(axis=1),
        1.0,
        atol=1.0e-6,
    ):
        raise ValueError("filtered probabilities must be non-negative and sum to one")
    dates = _validate_index(index, expected_rows=len(factor_array))
    origins = np.asarray(origin_positions)
    if origins.ndim != 1 or len(origins) < 1:
        raise ValueError("origin_positions must be a non-empty one-dimensional array")
    if not np.issubdtype(origins.dtype, np.integer):
        raise ValueError("origin_positions must contain integers")
    origins = origins.astype(int, copy=False)
    if np.any(np.diff(origins) <= 0):
        raise ValueError("origin_positions must be strictly increasing")
    if history_length < 1 or horizon < 1:
        raise ValueError("history_length and horizon must be positive")

    context_starts = origins - history_length + 1
    context_ends = origins.copy()
    target_starts = origins + 1
    target_ends = origins + horizon
    if np.any(context_starts < 0) or np.any(target_ends >= len(factor_array)):
        raise ValueError("an origin does not have the requested history and future target")
    if not np.array_equal(context_ends, origins):
        raise AssertionError("context end must equal forecast origin")
    if not np.array_equal(target_starts, origins + 1):
        raise AssertionError("target must begin one row after forecast origin")

    context_rows = np.concatenate((factor_array, macro_array), axis=1)
    contexts = np.stack(
        [
            context_rows[start : origin + 1]
            for start, origin in zip(context_starts, origins, strict=True)
        ]
    )
    targets = np.stack(
        [
            factor_array[start : end + 1]
            for start, end in zip(target_starts, target_ends, strict=True)
        ]
    )
    probabilities = regime_array[origins].copy()
    return OneShotWindowBundle(
        past_context=contexts,
        future_factors=targets,
        regime_probabilities=probabilities,
        origin_positions=origins.copy(),
        context_start_positions=context_starts,
        context_end_positions=context_ends,
        target_start_positions=target_starts,
        target_end_positions=target_ends,
        origin_dates=dates[origins].to_numpy(),
        context_end_dates=dates[context_ends].to_numpy(),
        target_start_dates=dates[target_starts].to_numpy(),
        target_end_dates=dates[target_ends].to_numpy(),
    )


def fit_train_only_standardizers(
    *,
    factors: np.ndarray,
    macro_features: np.ndarray,
    train_last_position: int,
) -> tuple[TrainOnlyStandardizer, TrainOnlyStandardizer]:
    """Fit context and target transforms using rows no later than train end."""

    factor_array = _finite_matrix(factors, name="factors")
    macro_array = _finite_matrix(macro_features, name="macro_features")
    if len(factor_array) != len(macro_array):
        raise ValueError("factors and macro_features need equal rows")
    if not 1 <= train_last_position < len(factor_array):
        raise ValueError("train_last_position is outside the available sample")
    train_slice = slice(0, train_last_position + 1)
    context_values = np.concatenate(
        (factor_array[train_slice], macro_array[train_slice]),
        axis=1,
    )
    return (
        TrainOnlyStandardizer.fit(context_values),
        TrainOnlyStandardizer.fit(factor_array[train_slice]),
    )


def _seed_everything(seed: int, *, torch_num_threads: int | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_num_threads is not None:
        if torch_num_threads < 1:
            raise ValueError("torch_num_threads must be positive")
        torch.set_num_threads(torch_num_threads)
    torch.use_deterministic_algorithms(True)


def _deterministic_validation_loss(
    model: ConditionalTemporalDDPM,
    dataset: OneShotFactorDataset,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    generator = torch.Generator(device=device).manual_seed(seed)
    losses: list[float] = []
    counts: list[int] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                clean = batch["clean_paths"].to(device)
                context = batch["past_context"].to(device)
                regimes = batch["regime_probabilities"].to(device)
                loss = model.training_loss(
                    clean,
                    context,
                    regimes,
                    generator=generator,
                )
                losses.append(float(loss))
                counts.append(len(clean))
    finally:
        model.train(was_training)
    return float(np.average(losses, weights=counts))


def fit_ddpm_checkpoint(
    model: ConditionalTemporalDDPM,
    train_dataset: OneShotFactorDataset,
    validation_dataset: OneShotFactorDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    random_seed: int,
    device: torch.device,
    checkpoint_path: Path,
    use_sample_weights: bool,
    stage_name: str,
) -> list[dict[str, Any]]:
    """Train one stage and select a checkpoint on deterministic tuning loss."""

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    shuffle_generator = torch.Generator().manual_seed(random_seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=shuffle_generator,
    )
    noise_generator = torch.Generator(device=device).manual_seed(random_seed + 1)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_counts: list[int] = []
        for batch in loader:
            clean = batch["clean_paths"].to(device)
            context = batch["past_context"].to(device)
            regimes = batch["regime_probabilities"].to(device)
            weights = batch["sample_weight"].to(device) if use_sample_weights else None
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(
                clean,
                context,
                regimes,
                sample_weights=weights,
                generator=noise_generator,
            )
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite diffusion gradient norm")
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_counts.append(len(clean))
        training_loss = float(np.average(epoch_losses, weights=epoch_counts))
        validation_loss = _deterministic_validation_loss(
            model,
            validation_dataset,
            batch_size=batch_size,
            seed=random_seed + 100_000,
            device=device,
        )
        is_best = validation_loss < best_loss
        row = {
            "stage": stage_name,
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_denoising_loss": validation_loss,
            "selected_checkpoint": is_best,
        }
        history.append(row)
        if is_best:
            best_loss = validation_loss
            torch.save(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "validation_denoising_loss": validation_loss,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return history


def _build_ddpm(
    *,
    horizon: int,
    factor_dim: int,
    context_dim: int,
    regime_dim: int,
    settings: dict[str, Any],
) -> ConditionalTemporalDDPM:
    return ConditionalTemporalDDPM(
        horizon=horizon,
        factor_dim=factor_dim,
        context_dim=context_dim,
        regime_dim=regime_dim,
        num_diffusion_steps=int(settings["diffusion_steps"]),
        hidden_channels=int(settings["hidden_channels"]),
        time_embedding_dim=int(settings["time_embedding_dim"]),
        num_residual_blocks=int(settings["residual_blocks"]),
        beta_start=float(settings["beta_start"]),
        beta_end=float(settings["beta_end"]),
    )


def _filtered_probabilities_fixed_parameters(
    model: SwitchingDynamicFactorBaseline,
    factors: np.ndarray,
) -> np.ndarray:
    """One-sided state beliefs from fixed training parameters."""

    result = model.regime_model.forward_backward(factors)
    probabilities = result.filtered_probabilities
    if probabilities.shape != (len(factors), model.n_states):
        raise AssertionError("unexpected filtered-probability shape")
    return probabilities


def generate_asset_scenarios(
    *,
    model: ConditionalTemporalDDPM,
    stage1_model: SwitchingDynamicFactorBaseline,
    standardized_context: np.ndarray,
    filtered_probability: np.ndarray,
    factor_standardizer: TrainOnlyStandardizer,
    num_scenarios: int,
    random_seed: int,
    standardized_factor_clip: float | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate factors one-shot, then independently draw and apply HMM states."""

    if standardized_context.ndim != 2:
        raise ValueError("standardized_context must be (history, context_dim)")
    if filtered_probability.shape != (stage1_model.n_states,):
        raise ValueError("filtered_probability has unexpected shape")
    context = torch.as_tensor(
        np.repeat(standardized_context[None, :, :], num_scenarios, axis=0),
        dtype=torch.float32,
        device=device,
    )
    regimes = torch.as_tensor(
        np.repeat(filtered_probability[None, :], num_scenarios, axis=0),
        dtype=torch.float32,
        device=device,
    )
    standardized_factors = (
        model.sample(context, regimes, seed=random_seed).detach().cpu().numpy()
    )
    clipped_fraction = 0.0
    if standardized_factor_clip is not None:
        if standardized_factor_clip <= 0.0:
            raise ValueError("standardized_factor_clip must be positive")
        clipped_fraction = float(
            np.mean(np.abs(standardized_factors) > standardized_factor_clip)
        )
        standardized_factors = np.clip(
            standardized_factors,
            -standardized_factor_clip,
            standardized_factor_clip,
        )
    factor_paths = factor_standardizer.inverse_transform(standardized_factors)
    rng = np.random.default_rng(random_seed + 10_000)
    regime_paths = stage1_model.regime_model.sample_posterior_predictive_paths(
        n_paths=num_scenarios,
        horizon=model.horizon,
        rng=rng,
        initial_filtered_probabilities=filtered_probability,
    )
    observations = stage1_model.observation_mapping_.sample_paths(
        factor_paths,
        regime_paths,
        rng=rng,
    )
    flat = observations.reshape(-1, observations.shape[-1])
    simple_returns = stage1_model.factor_model.observation_to_simple_returns(flat)
    asset_paths = simple_returns.reshape(observations.shape)
    return asset_paths, regime_paths, clipped_fraction


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _split_validation_origins(
    origins: np.ndarray,
    *,
    tuning_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(origins) < 2:
        raise ValueError("at least two validation origins are needed")
    if not 0.0 < tuning_fraction < 1.0:
        raise ValueError("validation_tuning_fraction must lie in (0, 1)")
    split = int(np.floor(len(origins) * tuning_fraction))
    split = min(max(split, 1), len(origins) - 1)
    return origins[:split], origins[split:]


def _save_standardizers(
    path: Path,
    *,
    context: TrainOnlyStandardizer,
    factor: TrainOnlyStandardizer,
    context_columns: list[str],
    factor_columns: list[str],
) -> None:
    np.savez_compressed(
        path,
        context_mean=context.mean,
        context_scale=context.scale,
        context_n_fit_rows=np.asarray(context.n_fit_rows),
        context_columns=np.asarray(context_columns),
        factor_mean=factor.mean,
        factor_scale=factor.scale,
        factor_n_fit_rows=np.asarray(factor.n_fit_rows),
        factor_columns=np.asarray(factor_columns),
    )


def run_stage2_evaluation(
    project_root: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run the explicitly small, sealed-test Stage 2 public-core pilot."""

    config_path = config_path or project_root / "configs/stage2_evaluation.yaml"
    configuration = load_yaml(config_path)
    experiment = configuration["experiment"]
    if experiment["evaluation_split"] != "validation":
        raise ValueError("Stage 2 keeps the test split sealed; use validation only")
    if experiment["test_set_policy"] != "sealed":
        raise ValueError("test_set_policy must remain sealed")
    if not bool(experiment["pilot"]):
        raise ValueError("this runner is registered only as a public-core pilot")

    paths = configuration["paths"]
    pipeline_path = project_root / "configs/pipeline.yaml"
    stage1_config_path = project_root / paths["stage1_model_config"]
    stage2_model_config_path = project_root / paths["stage2_model_config"]
    matrix_path = project_root / paths["model_matrix"]
    manifest_path = project_root / paths["phase0_manifest"]
    output_root = project_root / paths["output"]
    checkpoint_root = output_root / "checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "evaluation_config": config_path,
        "pipeline_config": pipeline_path,
        "stage1_model_config": stage1_config_path,
        "stage2_model_config": stage2_model_config_path,
        "model_matrix": matrix_path,
        "phase0_manifest": manifest_path,
    }
    input_hashes = {
        key: {"path": str(path.relative_to(project_root)), "sha256": hash_file(path)}
        for key, path in input_paths.items()
    }
    pipeline = load_yaml(pipeline_path)
    stage1_configuration = load_yaml(stage1_config_path)
    registered_stage2 = load_yaml(stage2_model_config_path)
    matrix = pd.read_parquet(matrix_path).sort_index()
    if matrix.index.has_duplicates or not matrix.index.is_monotonic_increasing:
        raise ValueError("model_matrix index must be unique and increasing")
    asset_columns = [column for column in matrix if column.startswith("asset__")]
    macro_columns = [column for column in matrix if column.startswith("macro__")]
    expected_macro = int(configuration["windows"]["expected_macro_feature_count"])
    if len(macro_columns) != expected_macro:
        raise ValueError(
            f"expected {expected_macro} macro__ columns, found {len(macro_columns)}"
        )
    if not asset_columns:
        raise ValueError("model_matrix contains no asset__ columns")

    train_end = pd.Timestamp(pipeline["splits"]["train_end"])
    validation_end = pd.Timestamp(pipeline["splits"]["validation_end"])
    usable = matrix.loc[matrix.index <= validation_end, asset_columns + macro_columns]
    if usable.empty or usable.isna().any().any():
        raise ValueError("Stage 2 requires a complete train-plus-validation panel")
    if (usable.index > validation_end).any():
        raise AssertionError("test observations entered the Stage 2 sample")
    train_mask = usable.index <= train_end
    train_last = int(np.flatnonzero(train_mask)[-1])
    if train_last < 2:
        raise ValueError("chronological training split is too short")

    seed = int(experiment["random_seed"])
    training_configuration = configuration["training"]
    _seed_everything(
        seed,
        torch_num_threads=int(training_configuration["torch_num_threads"]),
    )
    device = torch.device(str(training_configuration["device"]))
    if device.type != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {device} is unavailable")

    stage1_model = build_switching_factor_model(stage1_configuration)
    fit_start = time.perf_counter()
    train_returns = usable.iloc[: train_last + 1][asset_columns]
    stage1_model.fit(train_returns)
    stage1_fit_seconds = time.perf_counter() - fit_start
    returns = usable[asset_columns]
    macro = usable[macro_columns].to_numpy(dtype=float)
    factors = stage1_model.factor_model.transform(returns)
    filtered = _filtered_probabilities_fixed_parameters(stage1_model, factors)

    context_standardizer, factor_standardizer = fit_train_only_standardizers(
        factors=factors,
        macro_features=macro,
        train_last_position=train_last,
    )
    context_columns = [
        *(f"factor_{index}" for index in range(stage1_model.n_factors)),
        *macro_columns,
    ]
    factor_columns = [f"factor_{index}" for index in range(stage1_model.n_factors)]
    _save_standardizers(
        output_root / "train_only_standardizers.npz",
        context=context_standardizer,
        factor=factor_standardizer,
        context_columns=context_columns,
        factor_columns=factor_columns,
    )

    windows = configuration["windows"]
    history_length = int(windows["history_length"])
    horizon = int(windows["horizon"])
    train_origins = eligible_origin_positions(
        first_origin=history_length - 1,
        last_target_position=train_last,
        history_length=history_length,
        horizon=horizon,
        stride=int(windows["training_origin_stride"]),
    )
    validation_origins = eligible_origin_positions(
        first_origin=train_last,
        last_target_position=len(usable) - 1,
        history_length=history_length,
        horizon=horizon,
        stride=int(windows["validation_origin_stride"]),
    )
    tuning_origins, reporting_pool = _split_validation_origins(
        validation_origins,
        tuning_fraction=float(windows["validation_tuning_fraction"]),
    )
    train_bundle = build_one_shot_windows(
        index=usable.index,
        factors=factors,
        macro_features=macro,
        filtered_probabilities=filtered,
        origin_positions=train_origins,
        history_length=history_length,
        horizon=horizon,
    ).standardized(
        context_standardizer=context_standardizer,
        factor_standardizer=factor_standardizer,
    )
    tuning_bundle = build_one_shot_windows(
        index=usable.index,
        factors=factors,
        macro_features=macro,
        filtered_probabilities=filtered,
        origin_positions=tuning_origins,
        history_length=history_length,
        horizon=horizon,
    ).standardized(
        context_standardizer=context_standardizer,
        factor_standardizer=factor_standardizer,
    )
    boundaries = pd.concat(
        (
            train_bundle.boundary_frame(partition="train"),
            tuning_bundle.boundary_frame(partition="validation_tuning"),
        ),
        ignore_index=True,
    )

    tail_configuration = configuration["tail_fine_tuning"]
    importance = fit_training_importance_weights(
        train_bundle.future_factors,
        tail_quantile=float(tail_configuration["severity_quantile"]),
        strength=float(tail_configuration["importance_strength"]),
        maximum_weight=float(tail_configuration["maximum_weight"]),
        direction=str(tail_configuration["direction"]),  # type: ignore[arg-type]
    )
    train_base_dataset = OneShotFactorDataset(train_bundle)
    train_tail_dataset = OneShotFactorDataset(
        train_bundle,
        sample_weights=importance.weights,
    )
    tuning_dataset = OneShotFactorDataset(tuning_bundle)

    neural_settings = configuration["model"]
    registered_dimensions = registered_stage2["diffusion"]
    if int(registered_dimensions["factor_dim"]) != stage1_model.n_factors:
        raise ValueError("registered Stage 2 factor_dim differs from fitted Stage 1")
    if int(registered_dimensions["context_dim"]) != len(context_columns):
        raise ValueError("registered Stage 2 context_dim differs from built context")
    if int(registered_dimensions["regime_dim"]) != stage1_model.n_states:
        raise ValueError("registered Stage 2 regime_dim differs from fitted Stage 1")
    model = _build_ddpm(
        horizon=horizon,
        factor_dim=stage1_model.n_factors,
        context_dim=len(context_columns),
        regime_dim=stage1_model.n_states,
        settings=neural_settings,
    )
    histories: list[dict[str, Any]] = []
    training_start = time.perf_counter()
    base_checkpoint = checkpoint_root / "base.pt"
    histories.extend(
        fit_ddpm_checkpoint(
            model,
            train_base_dataset,
            tuning_dataset,
            epochs=int(training_configuration["base_epochs"]),
            batch_size=int(training_configuration["batch_size"]),
            learning_rate=float(training_configuration["learning_rate"]),
            weight_decay=float(training_configuration["weight_decay"]),
            gradient_clip_norm=float(training_configuration["gradient_clip_norm"]),
            random_seed=seed,
            device=device,
            checkpoint_path=base_checkpoint,
            use_sample_weights=False,
            stage_name="base",
        )
    )
    base_state = copy.deepcopy(model.state_dict())
    tail_checkpoint = checkpoint_root / "tail_weighted.pt"
    if bool(tail_configuration["enabled"]):
        histories.extend(
            fit_ddpm_checkpoint(
                model,
                train_tail_dataset,
                tuning_dataset,
                epochs=int(training_configuration["tail_fine_tune_epochs"]),
                batch_size=int(training_configuration["batch_size"]),
                learning_rate=float(training_configuration["learning_rate"]),
                weight_decay=float(training_configuration["weight_decay"]),
                gradient_clip_norm=float(training_configuration["gradient_clip_norm"]),
                random_seed=seed + 1_000,
                device=device,
                checkpoint_path=tail_checkpoint,
                use_sample_weights=True,
                stage_name="tail_weighted",
            )
        )
    else:
        torch.save(
            {
                "stage": "tail_weighted_disabled",
                "epoch": 0,
                "validation_denoising_loss": None,
                "model_state_dict": copy.deepcopy(base_state),
            },
            tail_checkpoint,
        )
    training_seconds = time.perf_counter() - training_start
    pd.DataFrame(histories).to_csv(output_root / "training_history.csv", index=False)

    evaluation_configuration = configuration["evaluation"]
    evaluation_stride = int(evaluation_configuration["origin_stride"])
    validation_stride = int(windows["validation_origin_stride"])
    reporting_step = max(
        int(np.ceil(evaluation_stride / validation_stride)),
        1,
    )
    reporting_origins = reporting_pool[::reporting_step]
    maximum_origins = int(evaluation_configuration["maximum_origins"])
    reporting_origins = reporting_origins[:maximum_origins]
    if len(reporting_origins) < 2:
        raise ValueError("the late-validation reporting segment needs at least two origins")
    reporting_bundle = build_one_shot_windows(
        index=usable.index,
        factors=factors,
        macro_features=macro,
        filtered_probabilities=filtered,
        origin_positions=reporting_origins,
        history_length=history_length,
        horizon=horizon,
    ).standardized(
        context_standardizer=context_standardizer,
        factor_standardizer=factor_standardizer,
    )
    boundaries = pd.concat(
        (
            boundaries,
            reporting_bundle.boundary_frame(partition="validation_reporting"),
        ),
        ignore_index=True,
    )
    boundaries.to_parquet(output_root / "window_boundaries.parquet", index=False)

    threshold_outcomes = rolling_cumulative_returns(
        train_returns.to_numpy(dtype=float),
        horizon=horizon,
    )
    risk_configuration = configuration["risk"]
    thresholds = fit_co_crash_thresholds(
        threshold_outcomes,
        marginal_quantile=float(risk_configuration["co_crash_marginal_quantile"]),
    )
    confidence = float(risk_configuration["confidence_level"])
    minimum_fraction = float(risk_configuration["co_crash_minimum_fraction"])
    equal_weights = np.repeat(1.0 / len(asset_columns), len(asset_columns))
    variants = tuple(str(value) for value in evaluation_configuration["variants"])
    variant_checkpoints = {
        "base": base_checkpoint,
        "tail_weighted": tail_checkpoint,
    }
    if not set(variants).issubset(variant_checkpoints):
        raise ValueError("unknown Stage 2 evaluation variant")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    scenario_blocks: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
    origin_date_blocks: dict[str, list[str]] = {variant: [] for variant in variants}
    evaluation_start = time.perf_counter()
    for variant in variants:
        checkpoint = torch.load(
            variant_checkpoints[variant],
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        for origin_number, origin in enumerate(reporting_origins):
            origin_date = usable.index[origin]
            # Pair reverse-diffusion and future-regime random numbers across
            # variants so the small pilot is not dominated by Monte Carlo noise.
            seed_at_origin = seed + origin_number
            actual_path = returns.iloc[
                origin + 1 : origin + 1 + horizon
            ].to_numpy(dtype=float)
            try:
                standardized_context = reporting_bundle.past_context[origin_number]
                filtered_probability = reporting_bundle.regime_probabilities[origin_number]
                scenario_paths, regime_paths, clipped_fraction = (
                    generate_asset_scenarios(
                        model=model,
                        stage1_model=stage1_model,
                        standardized_context=standardized_context,
                        filtered_probability=filtered_probability,
                        factor_standardizer=factor_standardizer,
                        num_scenarios=int(evaluation_configuration["num_scenarios"]),
                        random_seed=seed_at_origin,
                        standardized_factor_clip=float(
                            neural_settings["standardized_factor_clip"]
                        ),
                        device=device,
                    )
                )
                cumulative = aggregate_path_returns(scenario_paths)
                actual_cumulative = aggregate_path_returns(actual_path[None, :, :])[0]
                losses = portfolio_losses(cumulative, equal_weights)
                actual_loss = float(-(actual_cumulative @ equal_weights))
                value_at_risk = empirical_var(losses, confidence)
                expected_shortfall = empirical_expected_shortfall(losses, confidence)
                records.append(
                    {
                        "model_variant": variant,
                        "origin_date": origin_date.date().isoformat(),
                        "horizon_end_date": usable.index[
                            origin + horizon
                        ].date().isoformat(),
                        "seed": seed_at_origin,
                        "scenario_count": len(cumulative),
                        "energy_score": energy_score(
                            cumulative,
                            actual_cumulative,
                            rng=np.random.default_rng(seed_at_origin + 17),
                        ),
                        "variogram_score": variogram_score(
                            cumulative,
                            actual_cumulative,
                        ),
                        "value_at_risk": value_at_risk,
                        "expected_shortfall": expected_shortfall,
                        "actual_loss": actual_loss,
                        "var_violation": actual_loss > value_at_risk,
                        "predicted_co_crash_probability": co_crash_probability(
                            cumulative,
                            thresholds,
                            minimum_fraction=minimum_fraction,
                        ),
                        "actual_co_crash": realized_co_crash(
                            actual_cumulative,
                            thresholds,
                            minimum_fraction=minimum_fraction,
                        ),
                        "filtered_state_entropy": float(
                            -np.sum(
                                filtered_probability
                                * np.log(np.maximum(filtered_probability, 1.0e-300))
                            )
                        ),
                        "generated_factor_clip_fraction": clipped_fraction,
                        "mean_future_state_switches": float(
                            np.mean(np.sum(np.diff(regime_paths, axis=1) != 0, axis=1))
                        ),
                    }
                )
                scenario_blocks[variant].append(cumulative)
                origin_date_blocks[variant].append(origin_date.date().isoformat())
            except Exception as exc:
                failures.append(
                    {
                        "model_variant": variant,
                        "origin_date": origin_date.date().isoformat(),
                        "seed": seed_at_origin,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    evaluation_seconds = time.perf_counter() - evaluation_start
    detail = pd.DataFrame(records)
    if detail.empty:
        raise RuntimeError("every Stage 2 validation evaluation failed")
    detail.to_csv(output_root / "rolling_results.csv", index=False)
    for variant in variants:
        if scenario_blocks[variant]:
            np.savez_compressed(
                output_root / f"cumulative_asset_scenarios_{variant}.npz",
                scenarios=np.stack(scenario_blocks[variant]),
                origin_dates=np.asarray(origin_date_blocks[variant]),
                asset_columns=np.asarray(asset_columns),
                units=np.asarray("decimal_simple_cumulative_return"),
            )
    summary_rows: list[dict[str, Any]] = []
    for variant, group in detail.groupby("model_variant", sort=True):
        if len(group) < 2:
            failures.append(
                {
                    "model_variant": variant,
                    "error_type": "InsufficientCoverageOrigins",
                    "error": "coverage tests require at least two successful origins",
                }
            )
            coverage: dict[str, Any] = {"status": "insufficient_origins"}
        else:
            coverage = christoffersen_conditional_coverage_test(
                group["var_violation"].to_numpy(dtype=bool),
                confidence,
            )
        actual_crashes = group["actual_co_crash"].to_numpy(dtype=float)
        predicted_crashes = group[
            "predicted_co_crash_probability"
        ].to_numpy(dtype=float)
        summary_rows.append(
            {
                "model_variant": variant,
                "pilot": True,
                "superiority_claim_permitted": False,
                "origins": int(len(group)),
                "mean_energy_score": float(group["energy_score"].mean()),
                "mean_variogram_score": float(group["variogram_score"].mean()),
                "joint_var_es_score": joint_var_es_score(
                    group["actual_loss"].to_numpy(dtype=float),
                    group["value_at_risk"].to_numpy(dtype=float),
                    group["expected_shortfall"].to_numpy(dtype=float),
                    confidence,
                ),
                "mean_brier_score": brier_score(
                    predicted_crashes,
                    actual_crashes,
                ),
                "actual_co_crash_count": int(actual_crashes.sum()),
                "mean_generated_factor_clip_fraction": float(
                    group["generated_factor_clip_fraction"].mean()
                ),
                "coverage": coverage,
            }
        )
    (output_root / "failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "experiment_id": experiment["id"],
        "label": experiment["label"],
        "pilot": True,
        "evaluation_split": "validation_reporting_segment",
        "test_set_opened": False,
        "superiority_claim_permitted": False,
        "stage1_estimator": "MAP/empirical-Bayes multi-stage baseline",
        "full_bayesian_parameter_posterior": False,
        "future_regime_factor_joint_consistency": False,
        "stage1_fit_seconds": stage1_fit_seconds,
        "diffusion_training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "training_windows": len(train_bundle),
        "validation_tuning_windows": len(tuning_bundle),
        "validation_reporting_origins": len(reporting_origins),
        "tail_importance": {
            "threshold": importance.threshold,
            "tail_count": importance.tail_count,
            "minimum_weight": float(importance.weights.min()),
            "maximum_weight": float(importance.weights.max()),
            "mean_weight": float(importance.weights.mean()),
        },
        "variants": summary_rows,
        "known_limitations": configuration["known_limitations"],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    diagnostics = {
        "stage1": stage1_model.diagnostics(),
        "stage2_registered_scope": registered_stage2["estimator_scope"],
        "effective_pilot_neural_settings": neural_settings,
        "standardizers": {
            "context": asdict(context_standardizer),
            "factor": asdict(factor_standardizer),
        },
        "window_contract": {
            "last_context_time": "origin_t",
            "first_target_time": "origin_t_plus_1",
            "future_context": False,
            "regime_condition": "filtered_probability_at_origin",
        },
        "tail_weights_fitted_on": "training_windows_only",
        "independent_future_regime_draws": True,
        "full_bayesian_parameter_posterior": False,
    }
    (output_root / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    receipt = {
        "status": "completed" if not failures else "completed_with_failures",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment["id"],
        "pilot": True,
        "input_hashes": input_hashes,
        "checkpoint_hashes": {
            variant: hash_file(path)
            for variant, path in variant_checkpoints.items()
        },
        "standardizer_sha256": hash_file(
            output_root / "train_only_standardizers.npz"
        ),
        "test_set_opened": False,
        "summary": summary,
    }
    (output_root / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sealed-test Stage 2 public-core DDPM pilot"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else project_root_from_module()
    )
    receipt = run_stage2_evaluation(project_root, config_path=args.config)
    print(json.dumps(receipt, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
