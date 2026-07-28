from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and fail early on malformed configuration."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def project_root_from_module() -> Path:
    """Locate a CrisisForge checkout without assuming editable installation."""
    candidates: list[Path] = []
    for starting_point in (Path.cwd().resolve(), Path(__file__).resolve()):
        candidates.extend([starting_point, *starting_point.parents])
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "configs/pipeline.yaml"
        ).is_file():
            return candidate
    raise RuntimeError(
        "Could not locate a CrisisForge checkout. Run from the repository or "
        "pass --project-root explicitly."
    )


def resolve_config_path(
    project_root: Path,
    config_path: Path | None,
    *,
    default_relative: str,
) -> Path:
    """Resolve CLI/API config paths consistently against the project root."""
    candidate = Path(default_relative) if config_path is None else Path(config_path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def display_path(path: Path, *, project_root: Path) -> str:
    """Prefer a repository-relative path while supporting external configs."""
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())
