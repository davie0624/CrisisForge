from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from crisisforge.data.validation import hash_file  # noqa: E402
from crisisforge.diffusion import ConditionalTemporalDDPM  # noqa: E402
from crisisforge.evaluation.stage2 import (  # noqa: E402
    OneShotFactorDataset,
    build_one_shot_windows,
    eligible_origin_positions,
    fit_ddpm_checkpoint,
    fit_train_only_standardizers,
    run_stage2_evaluation,
)


def _synthetic_arrays(
    rows: int = 36,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    index = pd.date_range("2018-01-02", periods=rows, freq="B")
    row = np.arange(rows, dtype=float)
    factors = np.column_stack((row, -row))
    macro = np.column_stack((100.0 + row, 200.0 + row, 300.0 + row))
    probability = np.linspace(0.1, 0.9, rows)
    regimes = np.column_stack((probability, 1.0 - probability))
    return index, factors, macro, regimes


def test_window_boundaries_end_context_at_origin_and_start_target_next_row() -> None:
    index, factors, macro, regimes = _synthetic_arrays()
    origins = np.array([7, 13, 20])
    bundle = build_one_shot_windows(
        index=index,
        factors=factors,
        macro_features=macro,
        filtered_probabilities=regimes,
        origin_positions=origins,
        history_length=5,
        horizon=4,
    )
    np.testing.assert_array_equal(bundle.context_end_positions, origins)
    np.testing.assert_array_equal(bundle.target_start_positions, origins + 1)
    np.testing.assert_array_equal(bundle.target_end_positions, origins + 4)
    np.testing.assert_array_equal(bundle.context_end_dates, index[origins])
    np.testing.assert_array_equal(bundle.target_start_dates, index[origins + 1])
    np.testing.assert_allclose(bundle.past_context[:, -1, :2], factors[origins])
    np.testing.assert_allclose(bundle.future_factors[:, 0], factors[origins + 1])
    np.testing.assert_allclose(bundle.regime_probabilities, regimes[origins])


def test_window_builder_has_no_interface_for_future_context() -> None:
    index, factors, macro, regimes = _synthetic_arrays()
    with pytest.raises(TypeError):
        build_one_shot_windows(
            index=index,
            factors=factors,
            macro_features=macro,
            filtered_probabilities=regimes,
            origin_positions=np.array([10]),
            history_length=5,
            horizon=3,
            future_context=np.zeros((1, 3, 1)),
        )


def test_standardizers_are_train_only_and_late_outlier_cannot_change_them() -> None:
    _, factors, macro, _ = _synthetic_arrays()
    first = fit_train_only_standardizers(
        factors=factors,
        macro_features=macro,
        train_last_position=19,
    )
    factors_with_future_outlier = factors.copy()
    macro_with_future_outlier = macro.copy()
    factors_with_future_outlier[20:] += 1_000_000.0
    macro_with_future_outlier[20:] -= 1_000_000.0
    second = fit_train_only_standardizers(
        factors=factors_with_future_outlier,
        macro_features=macro_with_future_outlier,
        train_last_position=19,
    )
    for first_scaler, second_scaler in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_scaler.mean, second_scaler.mean)
        np.testing.assert_array_equal(first_scaler.scale, second_scaler.scale)
        assert first_scaler.n_fit_rows == 20


def test_eligible_origins_require_complete_history_and_target() -> None:
    origins = eligible_origin_positions(
        first_origin=0,
        last_target_position=19,
        history_length=5,
        horizon=3,
        stride=4,
    )
    np.testing.assert_array_equal(origins, np.array([4, 8, 12, 16]))
    assert np.all(origins - 5 + 1 >= 0)
    assert np.all(origins + 3 <= 19)


def test_tiny_checkpoint_training_is_deterministic_and_saves_file(
    tmp_path: Path,
) -> None:
    index, factors, macro, regimes = _synthetic_arrays(rows=32)
    context_scaler, factor_scaler = fit_train_only_standardizers(
        factors=factors,
        macro_features=macro,
        train_last_position=23,
    )
    train = build_one_shot_windows(
        index=index,
        factors=factors,
        macro_features=macro,
        filtered_probabilities=regimes,
        origin_positions=np.array([5, 8, 11, 14, 17]),
        history_length=5,
        horizon=3,
    ).standardized(
        context_standardizer=context_scaler,
        factor_standardizer=factor_scaler,
    )
    validation = build_one_shot_windows(
        index=index,
        factors=factors,
        macro_features=macro,
        filtered_probabilities=regimes,
        origin_positions=np.array([23, 26]),
        history_length=5,
        horizon=3,
    ).standardized(
        context_standardizer=context_scaler,
        factor_standardizer=factor_scaler,
    )

    def train_once(path: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        torch.manual_seed(77)
        model = ConditionalTemporalDDPM(
            horizon=3,
            factor_dim=2,
            context_dim=5,
            regime_dim=2,
            num_diffusion_steps=3,
            hidden_channels=8,
            time_embedding_dim=8,
            num_residual_blocks=1,
        )
        history = fit_ddpm_checkpoint(
            model,
            OneShotFactorDataset(train),
            OneShotFactorDataset(validation),
            epochs=1,
            batch_size=2,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            random_seed=42,
            device=torch.device("cpu"),
            checkpoint_path=path,
            use_sample_weights=False,
            stage_name="base",
        )
        return history, torch.load(path, map_location="cpu", weights_only=False)

    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    first_history, first_checkpoint = train_once(first_path)
    second_history, second_checkpoint = train_once(second_path)
    assert first_path.exists() and second_path.exists()
    assert first_history == second_history
    assert first_checkpoint["epoch"] == 1
    assert first_checkpoint["validation_denoising_loss"] == pytest.approx(
        second_checkpoint["validation_denoising_loss"],
        abs=0.0,
    )
    for name, value in first_checkpoint["model_state_dict"].items():
        torch.testing.assert_close(
            value,
            second_checkpoint["model_state_dict"][name],
            rtol=0.0,
            atol=0.0,
        )


def _write_yaml(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def test_tiny_full_runner_saves_audited_validation_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(2026)
    index = pd.date_range("2018-01-02", periods=120, freq="B")
    latent = rng.normal(scale=0.01, size=(120, 2))
    loadings = np.array([[0.7, 0.2], [-0.4, 0.8], [0.3, -0.5]])
    returns = latent @ loadings.T + rng.normal(scale=0.002, size=(120, 3))
    macro = rng.normal(size=(120, 3))
    matrix = pd.DataFrame(
        np.column_stack((returns, macro)),
        index=index,
        columns=[
            "asset__a",
            "asset__b",
            "asset__c",
            "macro__m1",
            "macro__m2",
            "macro__m3",
        ],
    )
    matrix.loc[index[110] :, :] = 999_999.0  # Sealed test rows must be ignored.
    matrix_path = tmp_path / "data/processed/model_matrix.parquet"
    matrix_path.parent.mkdir(parents=True)
    matrix.to_parquet(matrix_path)
    manifest_path = tmp_path / "artifacts/phase0/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_schema_version": "1.0",
                "files": [
                    {
                        "path": "data/processed/model_matrix.parquet",
                        "sha256": hash_file(matrix_path),
                        "bytes": matrix_path.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _write_yaml(
        tmp_path / "configs/pipeline.yaml",
        {
            "splits": {
                "train_end": index[59].date().isoformat(),
                "validation_end": index[109].date().isoformat(),
            }
        },
    )
    _write_yaml(
        tmp_path / "configs/stage1_model.yaml",
        {
            "data": {"return_transform": "log1p"},
            "factor_model": {"n_factors": 2, "scale_floor": 1.0e-8},
            "regime_model": {
                "n_states": 2,
                "n_initializations": 1,
                "maximum_iterations": 20,
                "tolerance": 1.0e-5,
                "transition_pseudocount": 0.5,
                "sticky_pseudocount": 2.0,
                "minimum_covariance_eigenvalue": 1.0e-5,
                "minimum_state_weight": 1.0e-3,
                "random_seed": 7,
            },
            "factor_dynamics": {
                "ridge": 1.0e-4,
                "covariance_floor": 1.0e-8,
                "maximum_spectral_radius": 0.95,
            },
            "observation_mapping": {
                "ridge": 1.0e-4,
                "residual_correlation_shrinkage": 0.2,
                "idiosyncratic_scale_floor": 1.0e-6,
            },
        },
    )
    _write_yaml(
        tmp_path / "configs/stage2_diffusion.yaml",
        {
            "estimator_scope": "synthetic-test-pilot",
            "diffusion": {"factor_dim": 2, "context_dim": 5, "regime_dim": 2},
        },
    )
    evaluation_config = {
        "experiment": {
            "id": "tiny_stage2_test",
            "label": "public_core_pilot",
            "pilot": True,
            "evaluation_split": "validation",
            "test_set_policy": "sealed",
            "random_seed": 31,
        },
        "windows": {
            "history_length": 5,
            "horizon": 3,
            "training_origin_stride": 3,
            "validation_origin_stride": 3,
            "validation_tuning_fraction": 0.4,
            "expected_macro_feature_count": 3,
        },
        "model": {
            "diffusion_steps": 3,
            "hidden_channels": 8,
            "time_embedding_dim": 8,
            "residual_blocks": 1,
            "beta_start": 1.0e-4,
            "beta_end": 2.0e-2,
            "standardized_factor_clip": 5.0,
        },
        "training": {
            "device": "cpu",
            "torch_num_threads": 1,
            "batch_size": 8,
            "base_epochs": 1,
            "tail_fine_tune_epochs": 1,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        },
        "tail_fine_tuning": {
            "enabled": True,
            "direction": "two_sided",
            "severity_quantile": 0.8,
            "importance_strength": 2.0,
            "maximum_weight": 4.0,
        },
        "evaluation": {
            "origin_stride": 6,
            "maximum_origins": 2,
            "num_scenarios": 8,
            "variants": ["base", "tail_weighted"],
        },
        "risk": {
            "confidence_level": 0.8,
            "co_crash_marginal_quantile": 0.1,
            "co_crash_minimum_fraction": 0.5,
        },
        "paths": {
            "model_matrix": "data/processed/model_matrix.parquet",
            "phase0_manifest": "artifacts/phase0/manifest.json",
            "stage1_model_config": "configs/stage1_model.yaml",
            "stage2_model_config": "configs/stage2_diffusion.yaml",
            "output": "artifacts/stage2_test",
        },
        "known_limitations": [
            "independent future regime and factor draws",
            "no full Bayesian parameter posterior",
        ],
    }
    evaluation_path = tmp_path / "configs/stage2_evaluation.yaml"
    _write_yaml(evaluation_path, evaluation_config)

    receipt = run_stage2_evaluation(tmp_path, config_path=evaluation_path)
    output = tmp_path / "artifacts/stage2_test"
    assert receipt["status"] == "completed"
    assert receipt["test_set_opened"] is False
    assert receipt["summary"]["superiority_claim_permitted"] is False
    assert (output / "checkpoints/base.pt").exists()
    assert (output / "checkpoints/tail_weighted.pt").exists()
    assert (output / "train_only_standardizers.npz").exists()
    assert (output / "cumulative_asset_scenarios_base.npz").exists()
    assert (output / "cumulative_asset_scenarios_tail_weighted.npz").exists()
    boundaries = pd.read_parquet(output / "window_boundaries.parquet")
    reporting = boundaries.loc[boundaries["partition"] == "validation_reporting"]
    assert pd.to_datetime(reporting["target_end_date"]).max() <= index[109]
