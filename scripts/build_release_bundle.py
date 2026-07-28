"""Build a deterministic CrisisForge source, evidence, and data bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA_VERSION = "crisisforge.release-bundle.v1"
ARCHIVE_ROOT = "CrisisForge"
FIXED_ZIP_TIMESTAMP = (2026, 7, 28, 0, 0, 0)
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "dist",
        "mlruns",
        "work",
        "wandb",
    }
)
EXCLUDED_NAMES = frozenset({".DS_Store"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _included_files(project_root: Path) -> list[Path]:
    root = project_root.resolve()
    included: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.name in EXCLUDED_NAMES
            or any(part in EXCLUDED_PARTS for part in relative.parts)
        ):
            continue
        included.append(path)
    return sorted(included, key=lambda item: item.relative_to(root).as_posix())


def _zip_info(archive_path: str, *, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(archive_path, FIXED_ZIP_TIMESTAMP)
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    return info


def build_release_bundle(
    project_root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Package the repository without environments, caches, Git internals, or recursion."""

    root = project_root.resolve()
    destination = (
        output_path.resolve() if output_path is not None else root / "dist/CrisisForge-v0.3.0.zip"
    )
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("output_path must stay inside the project root") from exc

    files = _included_files(root)
    records = [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "archive_root": ARCHIVE_ROOT,
        "file_count": len(records),
        "files": records,
        "schema_version": BUNDLE_SCHEMA_VERSION,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            archive.writestr(
                _zip_info(f"{ARCHIVE_ROOT}/{relative}", executable=executable),
                path.read_bytes(),
            )
        archive.writestr(
            _zip_info(f"{ARCHIVE_ROOT}/BUNDLE_MANIFEST.json"),
            manifest_bytes,
        )
    os.replace(temporary, destination)
    return {
        "archive": destination.relative_to(root).as_posix(),
        "bytes": destination.stat().st_size,
        "file_count": len(records),
        "sha256": sha256_file(destination),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = build_release_bundle(
        args.project_root,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
