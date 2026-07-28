from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DataQualityError(RuntimeError):
    """Raised when an invariant required for research validity fails."""


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prices(
    frame: pd.DataFrame,
    *,
    symbol: str,
    minimum_coverage: float,
    fail_on_duplicate_dates: bool,
    fail_on_nonpositive_prices: bool,
) -> dict[str, Any]:
    required = {"date", "adj_close"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise DataQualityError(f"{symbol} is missing columns: {sorted(missing_columns)}")

    duplicate_count = int(frame["date"].duplicated().sum())
    nonpositive_count = int((frame["adj_close"].dropna() <= 0).sum())
    observed_count = int(frame["adj_close"].notna().sum())
    coverage = observed_count / max(len(frame), 1)

    if fail_on_duplicate_dates and duplicate_count:
        raise DataQualityError(f"{symbol} has {duplicate_count} duplicate dates")
    if fail_on_nonpositive_prices and nonpositive_count:
        raise DataQualityError(f"{symbol} has {nonpositive_count} non-positive prices")
    if coverage < minimum_coverage:
        raise DataQualityError(
            f"{symbol} adjusted-close coverage {coverage:.3f} is below {minimum_coverage:.3f}"
        )

    return {
        "rows": int(len(frame)),
        "observed_adjusted_close": observed_count,
        "coverage": coverage,
        "duplicate_dates": duplicate_count,
        "nonpositive_prices": nonpositive_count,
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
    }


def validate_return_bounds(returns: pd.DataFrame, maximum_abs: float) -> list[dict[str, Any]]:
    """Return potential bad ticks; they are surfaced, not silently deleted."""
    flags: list[dict[str, Any]] = []
    for column in returns.columns:
        mask = returns[column].abs() > maximum_abs
        for timestamp, value in returns.loc[mask, column].items():
            flags.append(
                {
                    "date": pd.Timestamp(timestamp).date().isoformat(),
                    "series": column,
                    "log_return": float(value),
                    "threshold": maximum_abs,
                }
            )
    return flags


def frame_profile(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "first_date": frame.index.min().date().isoformat() if len(frame) else None,
        "last_date": frame.index.max().date().isoformat() if len(frame) else None,
        "duplicate_index": int(frame.index.duplicated().sum()),
        "missing_rate_by_column": {
            column: float(rate) for column, rate in frame.isna().mean().items()
        },
    }


def write_manifest(
    paths: list[Path],
    output_path: Path,
    extra: dict[str, Any],
    *,
    base_path: Path | None = None,
) -> dict[str, Any]:
    records = []
    for path in sorted(paths):
        records.append(
            {
                "path": str(path.relative_to(base_path) if base_path is not None else path),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )
    manifest = {
        "manifest_schema_version": "1.0",
        "files": records,
        **extra,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def assert_chronological_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    if train.empty or validation.empty or test.empty:
        raise DataQualityError("At least one chronological split is empty")
    if not train.index.max() < validation.index.min():
        raise DataQualityError("Train and validation periods overlap")
    if not validation.index.max() < test.index.min():
        raise DataQualityError("Validation and test periods overlap")


def finite_fraction(frame: pd.DataFrame) -> float:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.size == 0:
        return 1.0
    return float(np.isfinite(numeric.to_numpy()).mean())
