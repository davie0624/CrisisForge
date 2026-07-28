from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crisisforge.config import load_yaml, project_root_from_module
from crisisforge.data.providers import (
    download_macro_source,
    download_return_source,
    polite_pause,
    read_cached_snapshot,
    write_snapshot,
)
from crisisforge.data.validation import (
    DataQualityError,
    assert_chronological_splits,
    finite_fraction,
    frame_profile,
    hash_file,
    validate_return_bounds,
    write_manifest,
)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for optional licensed/applied price panels."""
    if (prices <= 0).any().any():
        raise DataQualityError("Log returns require strictly positive prices")
    return np.log(prices).diff()


def transform_macro(values: pd.Series, transform: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if transform == "level":
        return numeric
    if transform == "difference":
        return numeric.diff()
    if transform == "bps_difference":
        return numeric.diff() * 100.0
    if transform == "log_difference":
        if (numeric.dropna() <= 0).any():
            raise DataQualityError("log_difference requires positive observations")
        return np.log(numeric).diff()
    if transform == "pct_change":
        return numeric.pct_change(fill_method=None)
    raise ValueError(f"Unsupported macro transform: {transform}")


def duration_convexity_return_proxy(
    yields_percent: pd.Series,
    maturity_years: int,
    *,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """Approximate a constant-maturity Treasury return from official par yields.

    This is an explicitly labeled research proxy, not an observed bond total return.
    The approximation uses carry, modified duration, and convexity under an annual
    zero-coupon simplification:

        r_t ≈ y_{t-1} Δτ_t - D*_{t-1} Δy_t + 0.5 C_{t-1}(Δy_t)^2.
    """
    yields = pd.to_numeric(yields_percent, errors="coerce") / 100.0
    previous = yields.shift(1)
    delta = yields.diff()
    modified_duration = maturity_years / (1.0 + previous)
    convexity = maturity_years * (maturity_years + 1.0) / (1.0 + previous) ** 2
    if isinstance(yields.index, pd.DatetimeIndex):
        elapsed_years = yields.index.to_series().diff().dt.total_seconds()
        elapsed_years = elapsed_years / (365.25 * 24.0 * 60.0 * 60.0)
        elapsed_years = elapsed_years.reindex(yields.index)
    else:
        elapsed_years = pd.Series(1.0 / trading_days_per_year, index=yields.index)
        elapsed_years.iloc[0] = np.nan
    carry = previous * elapsed_years
    return carry - modified_duration * delta + 0.5 * convexity * delta.pow(2)


def aggregate_simple_returns_to_endpoints(
    returns: pd.DataFrame,
    endpoints: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compound returns over intervals defined by common market endpoints.

    If two markets observe different holidays, retaining only same-date rows would
    silently compare different holding periods. This function aggregates every
    source return in ``(endpoint[t-1], endpoint[t]]``.
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns require a DatetimeIndex")
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("returns index must be sorted and unique")
    common = pd.DatetimeIndex(endpoints).sort_values().unique()
    if len(common) < 2:
        raise ValueError("at least two common endpoints are required")
    if not common.isin(returns.index).all():
        raise ValueError("every endpoint must exist in the source return index")
    if (returns <= -1.0).any().any():
        raise ValueError("simple returns cannot be less than or equal to -100%")
    cumulative_log_growth = np.log1p(returns).cumsum()
    endpoint_growth = cumulative_log_growth.reindex(common)
    return np.expm1(endpoint_growth.diff())


def build_common_interval_audit(
    endpoints: pd.DatetimeIndex,
    source_calendars: dict[str, pd.DatetimeIndex],
) -> pd.DataFrame:
    """Describe the exact holding interval represented by every target row."""
    common = pd.DatetimeIndex(endpoints).sort_values().unique()
    if len(common) < 2:
        raise ValueError("at least two common endpoints are required")
    audit = pd.DataFrame(index=common)
    audit.index.name = "interval_end"
    audit["interval_start"] = pd.Series(common, index=common).shift(1)
    audit["calendar_days"] = (
        pd.Series(common, index=common).diff().dt.total_seconds() / 86_400.0
    )
    for name, calendar in source_calendars.items():
        source_calendar = pd.DatetimeIndex(calendar).sort_values().unique()
        endpoint_present = common.isin(source_calendar)
        audit[f"endpoint_present__{name}"] = endpoint_present
        positions = source_calendar.searchsorted(common, side="right")
        counts = np.full(len(common), np.nan)
        counts[1:] = np.diff(positions)
        audit[f"observations__{name}"] = counts
    endpoint_columns = [
        column for column in audit if column.startswith("endpoint_present__")
    ]
    audit["all_source_endpoints_present"] = audit[endpoint_columns].all(axis=1)
    return audit


def build_derived_market_features(
    returns: pd.DataFrame,
    specifications: list[dict[str, Any]],
    *,
    interval_years: pd.Series | None = None,
) -> pd.DataFrame:
    """Build after-close, backward-looking stress features from target returns."""
    outputs: dict[str, pd.Series] = {}
    for specification in specifications:
        feature = specification["feature"]
        operation = specification["operation"]
        lookback = int(specification.get("lookback", 1))
        output_name = f"macro__{feature}"

        if operation == "cross_sectional_mean":
            columns = [
                column
                for column in returns
                if column.startswith(specification["source_prefix"])
            ]
            if not columns:
                raise DataQualityError(
                    f"No return columns match {specification['source_prefix']}"
                )
            outputs[output_name] = returns[columns].mean(axis=1)
            continue

        if operation == "rolling_cross_sectional_dispersion":
            columns = [
                column
                for column in returns
                if column.startswith(specification["source_prefix"])
            ]
            if not columns:
                raise DataQualityError(
                    f"No return columns match {specification['source_prefix']}"
                )
            daily_dispersion = returns[columns].std(axis=1, ddof=1)
            outputs[output_name] = daily_dispersion.rolling(
                lookback,
                min_periods=lookback,
            ).mean()
            continue

        source_name = f"macro__{specification['source']}"
        if source_name not in outputs:
            raise DataQualityError(
                f"Derived market feature {feature} requires {source_name}"
            )
        source = outputs[source_name]
        if operation == "rolling_volatility":
            if interval_years is not None:
                elapsed = interval_years.reindex(source.index)
                log_returns = np.log1p(source)
                outputs[output_name] = np.sqrt(
                    log_returns.pow(2)
                    .rolling(lookback, min_periods=lookback)
                    .sum()
                    / elapsed.rolling(lookback, min_periods=lookback).sum()
                )
            else:
                annualization = float(specification["annualization"])
                outputs[output_name] = (
                    source.rolling(lookback, min_periods=lookback).std(ddof=1)
                    * np.sqrt(annualization)
                )
        elif operation == "rolling_downside_semivolatility":
            if interval_years is not None:
                elapsed = interval_years.reindex(source.index)
                downside_squared = np.log1p(source).clip(upper=0.0).pow(2)
                outputs[output_name] = np.sqrt(
                    downside_squared.rolling(
                        lookback,
                        min_periods=lookback,
                    ).sum()
                    / elapsed.rolling(lookback, min_periods=lookback).sum()
                )
            else:
                annualization = float(specification["annualization"])
                downside_squared = source.clip(upper=0.0).pow(2)
                outputs[output_name] = np.sqrt(
                    downside_squared.rolling(
                        lookback,
                        min_periods=lookback,
                    ).mean()
                    * annualization
                )
        elif operation == "rolling_drawdown":
            wealth = (1.0 + source).cumprod()
            rolling_peak = wealth.rolling(lookback, min_periods=lookback).max()
            outputs[output_name] = wealth / rolling_peak - 1.0
        else:
            raise ValueError(f"Unsupported derived market operation: {operation}")
    return pd.DataFrame(outputs, index=returns.index)


def asof_align_feature(
    target_dates: pd.DatetimeIndex,
    source: pd.DataFrame,
    *,
    feature_name: str,
    availability_lag_sessions: int,
    max_staleness_calendar_days: int,
    value_prefix: str = "macro",
) -> pd.DataFrame:
    """Backward as-of join on the model calendar without weekend-only BDay logic.

    Availability lags are measured in actual model observation sessions. This
    respects holidays present in ``target_dates`` and is intentionally
    conservative when source and model calendars differ.
    """
    prepared = source[["date", "value"]].dropna().copy()
    prepared["source_date"] = pd.to_datetime(prepared["date"])
    model_calendar = pd.DatetimeIndex(target_dates).sort_values().unique()
    if availability_lag_sessions < 0:
        raise ValueError("availability_lag_sessions must be non-negative")
    base_positions = model_calendar.searchsorted(
        pd.DatetimeIndex(prepared["source_date"]),
        side="left",
    )
    availability_positions = base_positions + availability_lag_sessions
    valid = availability_positions < len(model_calendar)
    prepared = prepared.loc[valid].copy()
    prepared["available_date"] = model_calendar[availability_positions[valid]]
    prepared = prepared.sort_values(["available_date", "source_date"]).drop_duplicates(
        "available_date",
        keep="last",
    )

    target = pd.DataFrame({"date": model_calendar})
    merged = pd.merge_asof(
        target,
        prepared[["available_date", "source_date", "value"]],
        left_on="date",
        right_on="available_date",
        direction="backward",
        allow_exact_matches=True,
    )
    age = (merged["date"] - merged["source_date"]).dt.days
    stale = merged["source_date"].isna() | (age > max_staleness_calendar_days)
    merged.loc[stale, "value"] = np.nan
    merged = merged.set_index("date")
    return pd.DataFrame(
        {
            f"{value_prefix}__{feature_name}": merged["value"],
            f"source_date__{feature_name}": merged["source_date"],
            f"available_date__{feature_name}": merged["available_date"],
            f"stale__{feature_name}": stale.to_numpy(),
        },
        index=merged.index,
    )


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_end: str,
    validation_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_cut = pd.Timestamp(train_end)
    validation_cut = pd.Timestamp(validation_end)
    train = frame.loc[frame.index <= train_cut].copy()
    validation = frame.loc[
        (frame.index > train_cut) & (frame.index <= validation_cut)
    ].copy()
    test = frame.loc[frame.index > validation_cut].copy()
    assert_chronological_splits(train, validation, test)
    return train, validation, test


def _fetch_downloaded_returns(
    project_root: Path,
    catalog: dict[str, Any],
    pipeline: dict[str, Any],
    *,
    refresh: bool,
    allow_network: bool,
    raw_base: Path | None = None,
) -> dict[str, pd.DataFrame]:
    raw_root = (
        raw_base
        if raw_base is not None
        else project_root / pipeline["paths"]["raw"]
    ) / "returns"
    start_date = pipeline["sample"]["start_date"]
    end_date = pipeline["sample"]["end_date"] or datetime.now(UTC).date().isoformat()
    frames: dict[str, pd.DataFrame] = {}

    for source in catalog["target_return_sources"]:
        if source["provider"].startswith("derived_"):
            continue
        source_id = source["id"]
        path = raw_root / f"{source_id}.csv"
        if refresh or not path.exists():
            if not allow_network:
                raise FileNotFoundError(f"No cached return snapshot for {source_id}: {path}")
            download = download_return_source(source, start_date, end_date)
            write_snapshot(
                download,
                path,
                {
                    "source_id": source_id,
                    "name": source["name"],
                    "page_url": source["page_url"],
                    "native_frequency": source["native_frequency"],
                    "units": source["units"],
                    "requested_start": start_date,
                    "requested_end": end_date,
                },
            )
            polite_pause()
        frames[source_id] = read_cached_snapshot(
            path,
            expected_source_id=source_id,
            expected_provider=source["provider"],
            expected_columns=[
                "date",
                *[item["source_column"] for item in source["columns"]],
            ],
            expected_page_url=source["page_url"],
            start_date=start_date,
            end_date=end_date,
        )
    return frames


def _fetch_macro(
    project_root: Path,
    catalog: dict[str, Any],
    pipeline: dict[str, Any],
    *,
    refresh: bool,
    allow_network: bool,
    raw_base: Path | None = None,
) -> dict[str, pd.DataFrame]:
    raw_root = (
        raw_base
        if raw_base is not None
        else project_root / pipeline["paths"]["raw"]
    ) / "macro"
    start_date = pipeline["sample"]["start_date"]
    end_date = pipeline["sample"]["end_date"] or datetime.now(UTC).date().isoformat()
    frames: dict[str, pd.DataFrame] = {}

    for source in catalog["macro_sources"]:
        source_id = source["id"]
        path = raw_root / f"{source_id}.csv"
        if refresh or not path.exists():
            if not allow_network:
                raise FileNotFoundError(
                    f"No cached direct-origin snapshot for {source_id}: {path}"
                )
            download = download_macro_source(source, start_date, end_date)
            write_snapshot(
                download,
                path,
                {
                    "source_id": source_id,
                    "name": source["name"],
                    "page_url": source["page_url"],
                    "native_frequency": source["native_frequency"],
                    "requested_start": start_date,
                    "requested_end": end_date,
                },
            )
            polite_pause()
        frames[source_id] = read_cached_snapshot(
            path,
            expected_source_id=source_id,
            expected_provider=source["provider"],
            expected_columns=[
                "date",
                *[item["source_column"] for item in source["columns"]],
            ],
            expected_page_url=source["page_url"],
            start_date=start_date,
            end_date=end_date,
        )
    return frames


def _target_return_quality(returns: pd.DataFrame) -> dict[str, Any]:
    return {
        column: {
            "observed": int(returns[column].notna().sum()),
            "missing_rate": float(returns[column].isna().mean()),
            "first_date": (
                returns[column].first_valid_index().date().isoformat()
                if returns[column].first_valid_index() is not None
                else None
            ),
            "last_date": (
                returns[column].last_valid_index().date().isoformat()
                if returns[column].last_valid_index() is not None
                else None
            ),
        }
        for column in returns
    }


def _build_target_return_panel(
    catalog: dict[str, Any],
    downloaded_returns: dict[str, pd.DataFrame],
    macro_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    downloaded_parts: list[pd.DataFrame] = []
    derived_sources: list[dict[str, Any]] = []
    for source in catalog["target_return_sources"]:
        if source["provider"] == "french_research_zip":
            raw = downloaded_returns[source["id"]].set_index("date")
            renamed = {
                item["source_column"]: f"asset__{item['asset']}"
                for item in source["columns"]
            }
            downloaded_parts.append(raw[list(renamed)].rename(columns=renamed))
            continue
        if source["provider"] == "derived_treasury_return":
            derived_sources.append(source)
            continue
        raise ValueError(f"Unsupported target-return provider: {source['provider']}")
    if not downloaded_parts:
        raise ValueError("At least one downloaded return panel is required")
    base_returns = (
        pd.concat(downloaded_parts, axis=1, join="inner")
        .sort_index()
        .dropna(how="any")
    )
    parts: list[pd.DataFrame] = []
    interval_audit: pd.DataFrame | None = None
    for source in derived_sources:
        raw = macro_frames[source["source_id"]].set_index("date").sort_index()
        required_yields = [item["source_column"] for item in source["columns"]]
        valid_raw = raw.dropna(subset=required_yields)
        base_calendar = base_returns.index
        treasury_calendar = valid_raw.index
        common_endpoints = base_calendar.intersection(treasury_calendar).sort_values()
        interval_audit = build_common_interval_audit(
            common_endpoints,
            {
                "industry_returns": base_calendar,
                "treasury_yields": treasury_calendar,
            },
        )
        synchronized_base = aggregate_simple_returns_to_endpoints(
            base_returns,
            common_endpoints,
        )
        raw = valid_raw.reindex(common_endpoints)
        derived: dict[str, pd.Series] = {}
        for item in source["columns"]:
            derived[f"asset__{item['asset']}"] = duration_convexity_return_proxy(
                raw[item["source_column"]],
                maturity_years=int(item["maturity_years"]),
            )
        base_returns = synchronized_base
        parts = [base_returns, pd.DataFrame(derived, index=common_endpoints)]
    if not derived_sources:
        parts = [base_returns]
        interval_audit = build_common_interval_audit(
            base_returns.index,
            {"industry_returns": base_returns.index},
        )
    if interval_audit is None:
        raise DataQualityError("Target interval audit was not constructed")
    panel = pd.concat(parts, axis=1, join="inner").sort_index()
    return panel, interval_audit.reindex(panel.index)


def _build_context_panels(
    target_dates: pd.DatetimeIndex,
    macro_frames: dict[str, pd.DataFrame],
    macro_sources: list[dict[str, Any]],
    derived_features: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aligned_parts: list[pd.DataFrame] = []
    model_features: list[str] = []
    validation_features: list[str] = []
    diagnostic_features: list[str] = []
    for source in macro_sources:
        raw = macro_frames[source["id"]].copy()
        for item in source["columns"]:
            series = raw[["date", item["source_column"]]].rename(
                columns={item["source_column"]: "value"}
            )
            series["value"] = transform_macro(series["value"], item["transform"])
            use = item.get("use", "model")
            if use == "validation_only":
                prefix = "validation"
            elif use == "optional_diagnostic":
                prefix = "diagnostic"
            else:
                prefix = "macro"
            aligned = asof_align_feature(
                target_dates,
                series,
                feature_name=item["feature"],
                availability_lag_sessions=source["availability_lag_model_sessions"],
                max_staleness_calendar_days=source[
                    "max_staleness_calendar_days"
                ],
                value_prefix=prefix,
            )
            aligned_parts.append(aligned)
            value_column = f"{prefix}__{item['feature']}"
            if use == "validation_only":
                validation_features.append(value_column)
            elif use == "optional_diagnostic":
                diagnostic_features.append(value_column)
            else:
                model_features.append(value_column)
    audit = pd.concat(aligned_parts, axis=1)
    for item in derived_features:
        if item["operation"] != "subtract":
            raise ValueError(f"Unsupported derived macro operation: {item['operation']}")
        left = f"macro__{item['left']}"
        right = f"macro__{item['right']}"
        output = f"macro__{item['feature']}"
        audit[output] = audit[left] - audit[right]
        model_features.append(output)
    return (
        audit[model_features].copy(),
        audit[validation_features].copy(),
        audit[diagnostic_features].copy(),
        audit,
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)}")


def _activate_raw_snapshot(staging_root: Path, raw_root: Path) -> Path | None:
    """Activate a validated candidate while retaining a rollback snapshot."""
    if raw_root.name != "raw":
        raise ValueError(f"Refusing to replace unexpected raw path: {raw_root}")
    if staging_root.parent != raw_root.parent:
        raise ValueError("Raw staging and destination must share a parent filesystem")
    backup_root = raw_root.parent / f".raw-backup-{os.getpid()}"
    if backup_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing backup: {backup_root}")
    moved_old = False
    try:
        if raw_root.exists():
            os.replace(raw_root, backup_root)
            moved_old = True
        os.replace(staging_root, raw_root)
        (raw_root / ".gitkeep").touch()
    except Exception:
        if moved_old and not raw_root.exists() and backup_root.exists():
            os.replace(backup_root, raw_root)
        raise
    return backup_root if moved_old else None


def _finalize_raw_snapshot(backup_root: Path | None) -> None:
    if backup_root is not None and backup_root.exists():
        shutil.rmtree(backup_root)


def _rollback_raw_snapshot(raw_root: Path, backup_root: Path | None) -> None:
    """Restore the previous raw cache after any post-activation failure."""
    if raw_root.name != "raw":
        raise ValueError(f"Refusing to roll back unexpected raw path: {raw_root}")
    failed_root = raw_root.parent / f".raw-failed-{os.getpid()}"
    if failed_root.exists():
        shutil.rmtree(failed_root)
    if raw_root.exists():
        os.replace(raw_root, failed_root)
    if backup_root is not None and backup_root.exists():
        os.replace(backup_root, raw_root)
    if failed_root.exists():
        shutil.rmtree(failed_root)


def _begin_publication_transaction(
    publication_roots: list[Path],
) -> dict[Path, Path | None]:
    """Retain exact prior outputs so a failed publication can be rolled back.

    Phase 0 publishes three directory trees (interim, processed, and its
    artifacts).  They must be treated as a single transaction: a manifest
    failure after Parquet files have been written must not leave new outputs
    paired with the old raw cache and old manifest.
    """
    allowed_names = {"interim", "processed", "phase0"}
    backups: dict[Path, Path | None] = {}
    try:
        for root in publication_roots:
            if root.name not in allowed_names:
                raise ValueError(f"Refusing to transact unexpected path: {root}")
            gitkeep_bytes = (
                (root / ".gitkeep").read_bytes()
                if (root / ".gitkeep").exists()
                else b""
            )
            backup = root.parent / f".{root.name}-backup-{os.getpid()}"
            if backup.exists():
                raise RuntimeError(f"Refusing to overwrite existing backup: {backup}")
            moved_old = False
            if root.exists():
                os.replace(root, backup)
                moved_old = True
            root.mkdir(parents=True, exist_ok=False)
            if root.name in {"interim", "processed"}:
                (root / ".gitkeep").write_bytes(gitkeep_bytes)
            backups[root] = backup if moved_old else None
    except Exception:
        _rollback_publication_transaction(backups)
        raise
    return backups


def _finalize_publication_transaction(
    backups: dict[Path, Path | None],
) -> None:
    for backup in backups.values():
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def _rollback_publication_transaction(
    backups: dict[Path, Path | None],
) -> None:
    """Restore every previously published Phase 0 directory exactly."""
    for root, backup in reversed(list(backups.items())):
        if root.exists():
            shutil.rmtree(root)
        if backup is not None and backup.exists():
            os.replace(backup, root)


def _load_source_metadata(raw_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(raw_root.glob("*/*.meta.json")):
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        record["metadata_path"] = str(metadata_path.relative_to(raw_root.parent.parent))
        records.append(record)
    return records


def _raw_allowlist_report(
    raw_root: Path,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    expected: set[str] = set()
    for source in catalog["target_return_sources"]:
        if source["provider"].startswith("derived_"):
            continue
        stem = f"returns/{source['id']}"
        expected.update({f"{stem}.csv", f"{stem}.meta.json"})
    for source in catalog["macro_sources"]:
        stem = f"macro/{source['id']}"
        expected.update({f"{stem}.csv", f"{stem}.meta.json"})
    actual = {
        str(path.relative_to(raw_root))
        for path in raw_root.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    }
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    return {
        "passed": not unexpected and not missing,
        "expected_files": sorted(expected),
        "unexpected_files": unexpected,
        "missing_files": missing,
    }


def _target_calendar_quality(
    catalog: dict[str, Any],
    downloaded_returns: dict[str, pd.DataFrame],
    macro_frames: dict[str, pd.DataFrame],
    *,
    requested_start: pd.Timestamp,
) -> dict[str, Any]:
    calendars: dict[str, pd.DatetimeIndex] = {}
    for source in catalog["target_return_sources"]:
        if source["provider"] == "french_research_zip":
            columns = [item["source_column"] for item in source["columns"]]
            frame = downloaded_returns[source["id"]].dropna(subset=columns)
            calendars[source["id"]] = pd.DatetimeIndex(frame["date"])
        elif source["provider"] == "derived_treasury_return":
            columns = [item["source_column"] for item in source["columns"]]
            frame = macro_frames[source["source_id"]].dropna(subset=columns)
            calendars[source["id"]] = pd.DatetimeIndex(frame["date"])
        else:
            raise ValueError(
                f"Unsupported target source for calendar audit: {source['provider']}"
            )
    if not calendars:
        raise DataQualityError("No target calendars are configured")
    source_metrics: dict[str, dict[str, Any]] = {}
    for source_id, dates in calendars.items():
        unique_dates = dates.sort_values().unique()
        if len(unique_dates) < 2:
            max_gap = float("inf")
        else:
            max_gap = float(
                pd.Series(unique_dates).diff().dt.total_seconds().max() / 86_400.0
            )
        source_metrics[source_id] = {
            "observations": int(len(unique_dates)),
            "first_date": unique_dates.min().date().isoformat(),
            "last_date": unique_dates.max().date().isoformat(),
            "start_delay_calendar_days": int(
                (unique_dates.min().normalize() - requested_start.normalize()).days
            ),
            "maximum_internal_gap_calendar_days": max_gap,
        }
    overlap_start = max(dates.min() for dates in calendars.values())
    overlap_end = min(dates.max() for dates in calendars.values())
    overlap_counts = {
        source_id: int(((dates >= overlap_start) & (dates <= overlap_end)).sum())
        for source_id, dates in calendars.items()
    }
    maximum_count = max(overlap_counts.values())
    relative_density = (
        min(overlap_counts.values()) / maximum_count if maximum_count else 0.0
    )
    return {
        "sources": source_metrics,
        "overlap_start": overlap_start.date().isoformat(),
        "overlap_end": overlap_end.date().isoformat(),
        "overlap_observation_counts": overlap_counts,
        "minimum_relative_density": float(relative_density),
    }


def _aligned_context_calendar_quality(
    alignment_audit: pd.DataFrame,
    macro_sources: list[dict[str, Any]],
    target_dates: pd.DatetimeIndex,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Measure usable aligned coverage and gaps for every context source.

    Freshness checks only inspect the last observation.  A source can still have
    a year-long hole in the middle, so model and validation sources receive
    independent coverage and internal-gap diagnostics after availability lags
    and staleness rules have been applied.
    """
    aligned = alignment_audit.reindex(pd.DatetimeIndex(target_dates))
    roles: dict[str, dict[str, dict[str, Any]]] = {
        "model": {},
        "validation": {},
        "diagnostic": {},
    }
    prefix_for_use = {
        "model": "macro",
        "validation_only": "validation",
        "optional_diagnostic": "diagnostic",
    }
    role_for_use = {
        "model": "model",
        "validation_only": "validation",
        "optional_diagnostic": "diagnostic",
    }
    for source in macro_sources:
        columns_by_use: dict[str, list[str]] = {}
        for item in source["columns"]:
            use = item.get("use", "model")
            prefix = prefix_for_use[use]
            columns_by_use.setdefault(use, []).append(
                f"{prefix}__{item['feature']}"
            )
        for use, value_columns in columns_by_use.items():
            missing = sorted(set(value_columns) - set(aligned.columns))
            if missing:
                raise DataQualityError(
                    f"Aligned audit is missing {source['id']} columns: {missing}"
                )
            usable = aligned[value_columns].notna().all(axis=1)
            usable_dates = aligned.index[usable]
            maximum_gap: float | None
            if len(usable_dates) < 2:
                maximum_gap = None
            else:
                maximum_gap = float(
                    pd.Series(usable_dates).diff().dt.total_seconds().max()
                    / 86_400.0
                )
            roles[role_for_use[use]][source["id"]] = {
                "features": value_columns,
                "usable_observations": int(usable.sum()),
                "coverage_on_target_calendar": float(usable.mean()),
                "first_usable_date": (
                    usable_dates.min().date().isoformat()
                    if len(usable_dates)
                    else None
                ),
                "last_usable_date": (
                    usable_dates.max().date().isoformat()
                    if len(usable_dates)
                    else None
                ),
                "maximum_internal_gap_calendar_days": maximum_gap,
            }
    return roles


def _source_freshness(
    metadata_records: list[dict[str, Any]],
    *,
    used_last_dates: dict[str, pd.Timestamp],
    reference_date: pd.Timestamp,
    warning_after_days: int,
    fail_after_days: int,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    failures: list[str] = []
    for record in metadata_records:
        source_id = record["source_id"]
        if source_id not in used_last_dates:
            failures.append(f"{source_id} has no filtered observations")
            continue
        last_date = pd.Timestamp(used_last_dates[source_id])
        age_days = int((reference_date.normalize() - last_date.normalize()).days)
        status = "current"
        message = (
            f"{source_id} ends {last_date.date().isoformat()} "
            f"({age_days} calendar days before the requested end)"
        )
        if age_days < 0:
            status = "failed"
            failures.append(
                f"{source_id} filtered end exceeds requested end by {-age_days} days"
            )
        elif age_days > fail_after_days:
            status = "failed"
            failures.append(message)
        elif age_days > warning_after_days:
            status = "warning"
            warnings.append(message)
        rows.append(
            {
                "source_id": source_id,
                "last_date": last_date.date().isoformat(),
                "age_calendar_days": age_days,
                "status": status,
            }
        )
    return rows, warnings, failures


def _runtime_receipt(project_root: Path) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in ("numpy", "pandas", "pyarrow", "PyYAML", "requests", "scipy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    git_commit: str | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        git_commit = result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git_commit,
    }


def _persist_phase0_run(
    *,
    project_root: Path,
    raw_root: Path,
    artifacts_root: Path,
    paths: dict[str, Path],
    frame_outputs: dict[str, pd.DataFrame],
    quality_report: dict[str, Any],
    receipt_fields: dict[str, Any],
) -> dict[str, Any]:
    for name, frame in frame_outputs.items():
        _write_parquet(frame, paths[name])

    quality_json = artifacts_root / "quality_report.json"
    quality_json.write_text(
        json.dumps(quality_report, indent=2, sort_keys=True, default=_json_safe),
        encoding="utf-8",
    )
    quality_md = artifacts_root / "quality_report.md"
    quality_md.write_text(_quality_markdown(quality_report), encoding="utf-8")

    raw_paths = sorted(path for path in raw_root.rglob("*") if path.is_file())
    governed_paths: list[Path] = []
    for directory_name in (
        "configs",
        "docs",
        "experiments",
        "linkedin",
        "notebooks",
        "reports",
        "scripts",
        "slides",
        "src",
        "tests",
    ):
        directory = project_root / directory_name
        if not directory.exists():
            continue
        governed_paths.extend(
            path
            for path in directory.rglob("*")
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                # The registry is a mutable index of downstream runs. Including
                # it in the Phase 0 manifest creates a circular dependency:
                # each run records the manifest hash, which would then change
                # the manifest itself. Exact run provenance remains in each
                # stage's immutable local receipt.
                and path != project_root / "experiments" / "registry.csv"
            )
        )
    for relative_path in (
        ".gitignore",
        ".python-version",
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
    ):
        candidate = project_root / relative_path
        if candidate.exists():
            governed_paths.append(candidate)
    manifest = write_manifest(
        [
            *raw_paths,
            *paths.values(),
            quality_json,
            quality_md,
            *governed_paths,
        ],
        artifacts_root / "manifest.json",
        {
            "pipeline_version": "phase0-v0.3.0",
            "configuration": {
                "pipeline": "configs/pipeline.yaml",
                "catalog": "configs/data_catalog.yaml",
            },
        },
        base_path=project_root,
    )
    receipt = {
        **receipt_fields,
        "manifest": str((artifacts_root / "manifest.json").relative_to(project_root)),
        "manifest_sha256": hash_file(artifacts_root / "manifest.json"),
        "manifest_file_count": len(manifest["files"]),
        "runtime": _runtime_receipt(project_root),
    }
    (artifacts_root / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return receipt


def run_phase0(
    project_root: Path,
    *,
    refresh: bool = False,
    allow_network: bool = True,
) -> dict[str, Any]:
    if refresh and not allow_network:
        raise ValueError("--refresh and --offline are mutually exclusive")
    pipeline = load_yaml(project_root / "configs" / "pipeline.yaml")
    catalog = load_yaml(project_root / "configs" / "data_catalog.yaml")
    interim_root = project_root / pipeline["paths"]["interim"]
    processed_root = project_root / pipeline["paths"]["processed"]
    artifacts_root = project_root / pipeline["paths"]["artifacts"]
    artifacts_root.mkdir(parents=True, exist_ok=True)
    raw_root = project_root / pipeline["paths"]["raw"]

    staging_context: tempfile.TemporaryDirectory[str] | None = None
    candidate_raw_root = raw_root
    if refresh:
        raw_root.parent.mkdir(parents=True, exist_ok=True)
        staging_context = tempfile.TemporaryDirectory(
            prefix=".raw-staging-",
            dir=raw_root.parent,
        )
        candidate_raw_root = Path(staging_context.name)
        downloaded_returns = _fetch_downloaded_returns(
            project_root,
            catalog,
            pipeline,
            refresh=True,
            allow_network=True,
            raw_base=candidate_raw_root,
        )
        macro_frames = _fetch_macro(
            project_root,
            catalog,
            pipeline,
            refresh=True,
            allow_network=True,
            raw_base=candidate_raw_root,
        )
    else:
        downloaded_returns = _fetch_downloaded_returns(
            project_root,
            catalog,
            pipeline,
            refresh=False,
            allow_network=allow_network,
        )
        macro_frames = _fetch_macro(
            project_root,
            catalog,
            pipeline,
            refresh=False,
            allow_network=allow_network,
        )
    raw_allowlist = _raw_allowlist_report(candidate_raw_root, catalog)
    if not raw_allowlist["passed"]:
        raise DataQualityError(
            "Raw cache does not match the public-core allowlist: "
            f"unexpected={raw_allowlist['unexpected_files']}, "
            f"missing={raw_allowlist['missing_files']}"
        )
    target_calendar_quality = _target_calendar_quality(
        catalog,
        downloaded_returns,
        macro_frames,
        requested_start=pd.Timestamp(pipeline["sample"]["start_date"]),
    )

    returns, interval_audit = _build_target_return_panel(
        catalog,
        downloaded_returns,
        macro_frames,
    )
    return_flags = validate_return_bounds(
        returns,
        maximum_abs=pipeline["quality"]["maximum_daily_return_abs"],
    )

    (
        macro_features,
        validation_labels,
        optional_diagnostics,
        alignment_audit,
    ) = _build_context_panels(
        returns.index,
        macro_frames,
        catalog["macro_sources"],
        catalog.get("derived_macro_features", []),
    )
    derived_market_features = build_derived_market_features(
        returns,
        catalog.get("derived_market_features", []),
        interval_years=interval_audit["calendar_days"] / 365.25,
    )
    macro_features = macro_features.join(derived_market_features, how="left")
    derived_alignment = pd.DataFrame(index=returns.index)
    for item in catalog.get("derived_market_features", []):
        feature = item["feature"]
        derived_alignment[f"source_date__{feature}"] = returns.index
        derived_alignment[f"available_date__{feature}"] = returns.index
        derived_alignment[f"stale__{feature}"] = False
    alignment_audit = alignment_audit.join(derived_alignment, how="left")
    model_matrix = returns.join(macro_features, how="left")

    # Target returns are never imputed. All targets must be jointly observed.
    asset_columns = [column for column in model_matrix if column.startswith("asset__")]
    model_matrix = model_matrix.dropna(subset=asset_columns)
    # Model context is backward-as-of aligned and is never backward-filled.
    macro_columns = [column for column in model_matrix if column.startswith("macro__")]
    model_matrix = model_matrix.dropna(subset=macro_columns)
    context_calendar_quality = _aligned_context_calendar_quality(
        alignment_audit,
        catalog["macro_sources"],
        returns.index,
    )

    source_date_columns = [
        column for column in alignment_audit if column.startswith("source_date__")
    ]
    available_date_columns = [
        column for column in alignment_audit if column.startswith("available_date__")
    ]
    audit_on_model_dates = alignment_audit.reindex(model_matrix.index)
    source_dates_ok = all(
        not (audit_on_model_dates[column] > audit_on_model_dates.index)
        .fillna(False)
        .any()
        for column in source_date_columns
    )
    availability_dates_ok = all(
        not (audit_on_model_dates[column] > audit_on_model_dates.index)
        .fillna(False)
        .any()
        for column in available_date_columns
    )

    train, validation, test = chronological_split(
        model_matrix,
        train_end=pipeline["splits"]["train_end"],
        validation_end=pipeline["splits"]["validation_end"],
    )

    paths = {
        "target_returns": interim_root / "target_returns.parquet",
        "target_interval_audit": interim_root / "target_interval_audit.parquet",
        "macro_features": interim_root / "macro_features.parquet",
        "validation_labels": interim_root / "validation_labels.parquet",
        "optional_diagnostics": interim_root / "optional_diagnostics.parquet",
        "alignment_audit": interim_root / "availability_alignment_audit.parquet",
        "model_matrix": processed_root / "model_matrix.parquet",
        "train": processed_root / "splits" / "train.parquet",
        "validation": processed_root / "splits" / "validation.parquet",
        "test": processed_root / "splits" / "test.parquet",
    }

    asset_coverage = {
        column: float(returns[column].notna().mean()) for column in asset_columns
    }
    interval_rows = interval_audit.loc[interval_audit["interval_start"].notna()]
    metadata_records = _load_source_metadata(candidate_raw_root)
    used_last_dates = {
        source_id: pd.Timestamp(frame["date"].max())
        for source_id, frame in {
            **downloaded_returns,
            **macro_frames,
        }.items()
    }
    requested_end = pd.Timestamp(
        pipeline["sample"]["end_date"] or datetime.now(UTC).date().isoformat()
    )
    freshness_rows, freshness_warnings, freshness_failures = _source_freshness(
        metadata_records,
        used_last_dates=used_last_dates,
        reference_date=requested_end,
        warning_after_days=int(
            pipeline["quality"]["warning_after_source_staleness_calendar_days"]
        ),
        fail_after_days=int(
            pipeline["quality"]["fail_after_source_staleness_calendar_days"]
        ),
    )
    finite_model_fraction = finite_fraction(model_matrix)
    source_internal_gaps = [
        float(item["maximum_internal_gap_calendar_days"])
        for item in target_calendar_quality["sources"].values()
    ]
    source_start_delays = [
        int(item["start_delay_calendar_days"])
        for item in target_calendar_quality["sources"].values()
    ]
    observation_count_columns = [
        column
        for column in interval_rows
        if column.startswith("observations__")
    ]
    interval_observation_count_gap = (
        interval_rows[observation_count_columns].max(axis=1)
        - interval_rows[observation_count_columns].min(axis=1)
    )
    model_context_metrics = list(context_calendar_quality["model"].values())
    validation_context_metrics = list(
        context_calendar_quality["validation"].values()
    )
    model_index_gap = (
        float(
            pd.Series(model_matrix.index).diff().dt.total_seconds().max()
            / 86_400.0
        )
        if len(model_matrix) >= 2
        else float("inf")
    )
    model_index_relative_density = len(model_matrix) / max(len(returns), 1)
    gate_results = {
        "raw_cache_matches_catalog_allowlist": bool(raw_allowlist["passed"]),
        "minimum_asset_coverage": (
            bool(asset_coverage)
            and min(asset_coverage.values())
            >= float(pipeline["quality"]["minimum_asset_coverage"])
        ),
        "target_index_unique": not returns.index.has_duplicates,
        "target_index_increasing": returns.index.is_monotonic_increasing,
        "common_interval_endpoints_aligned": (
            not interval_rows.empty
            and bool(interval_rows["all_source_endpoints_present"].all())
            and bool((interval_rows["calendar_days"] > 0).all())
        ),
        "target_source_internal_gaps_within_limit": (
            bool(source_internal_gaps)
            and max(source_internal_gaps)
            <= float(
                pipeline["quality"][
                    "maximum_target_source_internal_gap_calendar_days"
                ]
            )
        ),
        "target_source_start_delays_within_limit": (
            bool(source_start_delays)
            and max(source_start_delays)
            <= int(
                pipeline["quality"][
                    "maximum_target_source_start_delay_calendar_days"
                ]
            )
        ),
        "relative_target_calendar_density": (
            target_calendar_quality["minimum_relative_density"]
            >= float(
                pipeline["quality"]["minimum_relative_target_calendar_density"]
            )
        ),
        "model_context_source_coverage": (
            bool(model_context_metrics)
            and min(
                item["coverage_on_target_calendar"]
                for item in model_context_metrics
            )
            >= float(
                pipeline["quality"]["minimum_model_context_source_coverage"]
            )
        ),
        "model_context_source_internal_gaps_within_limit": (
            bool(model_context_metrics)
            and all(
                item["maximum_internal_gap_calendar_days"] is not None
                and item["maximum_internal_gap_calendar_days"]
                <= float(
                    pipeline["quality"][
                        "maximum_model_context_source_internal_gap_calendar_days"
                    ]
                )
                for item in model_context_metrics
            )
        ),
        "validation_source_coverage": (
            not validation_context_metrics
            or min(
                item["coverage_on_target_calendar"]
                for item in validation_context_metrics
            )
            >= float(pipeline["quality"]["minimum_validation_source_coverage"])
        ),
        "validation_source_internal_gaps_within_limit": (
            not validation_context_metrics
            or all(
                item["maximum_internal_gap_calendar_days"] is not None
                and item["maximum_internal_gap_calendar_days"]
                <= float(
                    pipeline["quality"][
                        "maximum_validation_source_internal_gap_calendar_days"
                    ]
                )
                for item in validation_context_metrics
            )
        ),
        "model_index_relative_density": (
            model_index_relative_density
            >= float(pipeline["quality"]["minimum_model_index_relative_density"])
        ),
        "model_index_internal_gap_within_limit": (
            model_index_gap
            <= float(
                pipeline["quality"][
                    "maximum_model_index_internal_gap_calendar_days"
                ]
            )
        ),
        "common_interval_calendar_gap_within_limit": (
            not interval_rows.empty
            and float(interval_rows["calendar_days"].max())
            <= float(
                pipeline["quality"]["maximum_common_interval_calendar_days"]
            )
        ),
        "interval_source_observation_count_gap_within_limit": (
            not interval_observation_count_gap.empty
            and float(interval_observation_count_gap.max())
            <= float(
                pipeline["quality"][
                    "maximum_interval_source_observation_count_gap"
                ]
            )
        ),
        "model_matrix_finite": finite_model_fraction == 1.0,
        "model_index_unique": not model_matrix.index.has_duplicates,
        "source_dates_not_after_model_date": source_dates_ok,
        "availability_dates_not_after_model_date": availability_dates_ok,
        "strict_chronological_splits": (
            train.index.max() < validation.index.min() < test.index.min()
        ),
        "validation_only_features_excluded": not any(
            column.startswith("validation__") for column in model_matrix
        ),
        "source_freshness_within_failure_limit": not freshness_failures,
    }
    failed_gates = [name for name, passed in gate_results.items() if not passed]
    warnings = list(freshness_warnings)
    if return_flags:
        warnings.append(
            f"{len(return_flags)} returns exceed the review threshold; "
            "they were retained and flagged"
        )
    status = "failed" if failed_gates else ("passed_with_warnings" if warnings else "passed")

    quality_report = {
        "status": status,
        "panel": catalog["panel"],
        "decision_timestamp": pipeline["project"]["decision_timestamp"],
        "sample_requested": pipeline["sample"],
        "target_quality": _target_return_quality(returns),
        "target_calendar_quality": {
            **target_calendar_quality,
            "maximum_interval_calendar_days": float(
                interval_rows["calendar_days"].max()
            ),
            "maximum_interval_source_observation_count_gap": float(
                interval_observation_count_gap.max()
            ),
        },
        "context_calendar_quality": context_calendar_quality,
        "model_calendar_quality": {
            "observations": int(len(model_matrix)),
            "target_calendar_observations": int(len(returns)),
            "relative_density": float(model_index_relative_density),
            "maximum_internal_gap_calendar_days": float(model_index_gap),
        },
        "raw_allowlist": raw_allowlist,
        "asset_coverage": asset_coverage,
        "source_freshness": freshness_rows,
        "warnings": warnings,
        "failed_gates": failed_gates,
        "quality_gates": gate_results,
        "potential_bad_ticks": return_flags,
        "profiles": {
            "target_returns": frame_profile(returns),
            "target_interval_audit": frame_profile(interval_audit),
            "macro_features": frame_profile(macro_features),
            "validation_labels": frame_profile(validation_labels),
            "optional_diagnostics": frame_profile(optional_diagnostics),
            "model_matrix": frame_profile(model_matrix),
        },
        "finite_fraction_model_matrix": finite_model_fraction,
        "splits": {
            "train": frame_profile(train),
            "validation": frame_profile(validation),
            "test": frame_profile(test),
        },
        "leakage_checks": {
            "source_dates_not_after_model_date": source_dates_ok,
            "availability_dates_not_after_model_date": availability_dates_ok,
            "strict_chronological_splits": gate_results[
                "strict_chronological_splits"
            ],
            "validation_only_ofr_excluded_from_model_matrix": gate_results[
                "validation_only_features_excluded"
            ],
            "derived_market_features_use_after_close_t_to_predict_t_plus_1": (
                pipeline["project"]["decision_timestamp"] == "after_close"
            ),
            "scaling_or_imputation_fit": "not performed in Phase 0",
        },
        "known_limitations": [
            (
                "French industry portfolios are retrospective research portfolios, "
                "not investable ETFs, and the source file is periodically reconstructed."
            ),
            (
                "Treasury target returns are transparent duration-convexity proxies "
                "derived from par yields, not observed bond total returns."
            ),
            (
                "Direct official histories can be revised; raw snapshots, retrieval "
                "timestamps, source URLs, and hashes are therefore retained locally."
            ),
            (
                "The public core omits Cboe option-index histories because the website "
                "terms are not treated as permission for public ML-data redistribution. "
                "Licensed Cboe DataShop or OptionMetrics data remain optional extensions."
            ),
            (
                "OFR FSI is retained only as an external validation label because its "
                "components overlap market targets and would create circular predictors."
            ),
            (
                "FRED is deliberately excluded because its current Terms of Use prohibit "
                "using FRED Services/Content to develop or train ML or generative-AI software."
            ),
            (
                "A publication-grade applied panel requires licensed, point-in-time "
                "asset total returns such as CRSP/WRDS."
            ),
            (
                "Rows are common-endpoint holding intervals, not an assertion that every "
                "row represents exactly one identical exchange session. The interval "
                "audit records calendar duration and source observation counts."
            ),
        ],
    }
    if failed_gates:
        failure_root = artifacts_root.parent / "phase0_failures"
        failure_root.mkdir(parents=True, exist_ok=True)
        failure_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        (failure_root / f"{failure_timestamp}.json").write_text(
            json.dumps(
                quality_report,
                indent=2,
                sort_keys=True,
                default=_json_safe,
            ),
            encoding="utf-8",
        )
        if staging_context is not None:
            staging_context.cleanup()
        raise DataQualityError(
            "Phase 0 quality gates failed: " + ", ".join(failed_gates)
        )

    backup_root: Path | None = None
    publication_backups: dict[Path, Path | None] = {}
    frame_outputs = {
        "target_returns": returns,
        "target_interval_audit": interval_audit,
        "macro_features": macro_features,
        "validation_labels": validation_labels,
        "optional_diagnostics": optional_diagnostics,
        "alignment_audit": alignment_audit,
        "model_matrix": model_matrix,
        "train": train,
        "validation": validation,
        "test": test,
    }
    receipt_fields = {
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_mode": "refresh" if refresh else ("online-cache" if allow_network else "offline"),
        "panel": catalog["panel"]["id"],
        "model_rows": int(len(model_matrix)),
        "model_columns": int(model_matrix.shape[1]),
        "asset_columns": len(asset_columns),
        "macro_columns": len(macro_columns),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "first_model_date": model_matrix.index.min().date().isoformat(),
        "last_model_date": model_matrix.index.max().date().isoformat(),
        "warnings": warnings,
        "failed_gates": failed_gates,
    }
    try:
        if refresh:
            backup_root = _activate_raw_snapshot(candidate_raw_root, raw_root)
        publication_backups = _begin_publication_transaction(
            [interim_root, processed_root, artifacts_root]
        )
        receipt = _persist_phase0_run(
            project_root=project_root,
            raw_root=raw_root,
            artifacts_root=artifacts_root,
            paths=paths,
            frame_outputs=frame_outputs,
            quality_report=quality_report,
            receipt_fields=receipt_fields,
        )
    except Exception:
        _rollback_publication_transaction(publication_backups)
        if refresh:
            _rollback_raw_snapshot(raw_root, backup_root)
        raise
    else:
        _finalize_publication_transaction(publication_backups)
        if refresh:
            _finalize_raw_snapshot(backup_root)
    finally:
        if staging_context is not None:
            staging_context.cleanup()
    return receipt


def _quality_markdown(report: dict[str, Any]) -> str:
    splits = report["splits"]
    bad_ticks = report["potential_bad_ticks"]
    gates = report["quality_gates"]
    warnings = report["warnings"]
    return "\n".join(
        [
            "# Phase 0 Data Quality Report",
            "",
            f"Status: **{report['status']}**",
            "",
            f"Panel: `{report['panel']['id']}`",
            "",
            f"Decision timestamp: `{report['decision_timestamp']}`",
            "",
            "## Quality gates",
            "",
            "| Gate | Result |",
            "|---|---|",
            *[
                f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
                for name, passed in gates.items()
            ],
            "",
            "## Chronological splits",
            "",
            "| Split | Rows | First date | Last date |",
            "|---|---:|---|---|",
            *[
                (
                    f"| {name} | {profile['rows']} | "
                    f"{profile['first_date']} | {profile['last_date']} |"
                )
                for name, profile in splits.items()
            ],
            "",
            "## Leakage controls",
            "",
            "- Source and assumed-availability dates never exceed the model observation date.",
            "- Train, validation, and test periods are strictly chronological and non-overlapping.",
            "- OFR stress components are validation-only and excluded from the model matrix.",
            (
                "- Return-derived market features at close t are used only as context "
                "for targets beginning at t+1."
            ),
            "- No scaler, imputer, factor model, or regime model is fit in Phase 0.",
            "",
            "## Data checks",
            "",
            (
                "- Finite fraction in the final model matrix: "
                f"{report['finite_fraction_model_matrix']:.6f}"
            ),
            (
                "- Potential absolute daily returns above the configured threshold: "
                f"{len(bad_ticks)}"
            ),
            "- Potential bad ticks are reported for review and are not silently winsorized.",
            "",
            "## Warnings",
            "",
            *([f"- {item}" for item in warnings] if warnings else ["- None."]),
            "",
            "## Known limitations",
            "",
            *[f"- {item}" for item in report["known_limitations"]],
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CrisisForge Phase 0 data pipeline")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root_from_module(),
        help="Repository root",
    )
    source_mode = parser.add_mutually_exclusive_group()
    source_mode.add_argument(
        "--refresh",
        action="store_true",
        help="Atomically refresh all source snapshots",
    )
    source_mode.add_argument(
        "--offline",
        action="store_true",
        help="Require existing raw snapshots and make no network calls",
    )
    args = parser.parse_args()
    receipt = run_phase0(
        args.project_root.resolve(),
        refresh=args.refresh,
        allow_network=not args.offline,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
