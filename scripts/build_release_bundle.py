"""Build a deterministic, public-safe CrisisForge source bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA_VERSION = "crisisforge.release-bundle.v1"
ARCHIVE_ROOT = "CrisisForge"
FIXED_ZIP_TIMESTAMP = (2026, 7, 28, 0, 0, 0)
ALLOWLISTED_ROOT_FILES = frozenset(
    {
        ".gitattributes",
        ".gitignore",
        ".python-version",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "LICENSE.md",
        "Makefile",
        "NOTICE",
        "README.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "uv.lock",
    }
)
ALLOWLISTED_TOP_LEVEL_DIRECTORIES = frozenset(
    {
        ".github",
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
    }
)
DENIED_PARTS = frozenset(
    {
        ".aws",
        ".git",
        ".gnupg",
        ".pytest_cache",
        ".ruff_cache",
        ".ssh",
        ".venv",
        "__pycache__",
        "artifacts",
        "checkpoints",
        "dist",
        "mlruns",
        "work",
        "wandb",
    }
)
DENIED_PATH_PREFIXES = (
    ("data", "raw"),
    ("data", "interim"),
    ("data", "processed"),
)
DENIED_NAMES = frozenset(
    {
        ".DS_Store",
        ".RData",
        ".Rhistory",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secret.json",
        "secrets.json",
        "service-account.json",
        "service_account.json",
    }
)
DENIED_SUFFIXES = (
    ".ckpt",
    ".jks",
    ".key",
    ".keystore",
    ".npz",
    ".p12",
    ".parquet",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".tar",
    ".tar.bz2",
    ".tar.gz",
    ".tgz",
    ".whl",
    ".zip",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_tracked_paths(project_root: Path) -> list[Path]:
    root = project_root.resolve()
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("a Git repository is required to build a release bundle") from exc
    return [Path(os.fsdecode(raw_path)) for raw_path in result.stdout.split(b"\0") if raw_path]


def _is_allowlisted(relative: Path) -> bool:
    if len(relative.parts) == 1:
        return relative.name in ALLOWLISTED_ROOT_FILES
    return relative.parts[0] in ALLOWLISTED_TOP_LEVEL_DIRECTORIES


def _is_denied(relative: Path) -> bool:
    lowered_parts = tuple(part.casefold() for part in relative.parts)
    lowered_name = relative.name.casefold()
    if any(part in {value.casefold() for value in DENIED_PARTS} for part in lowered_parts):
        return True
    if any(
        lowered_parts[: len(prefix)] == tuple(part.casefold() for part in prefix)
        for prefix in DENIED_PATH_PREFIXES
    ):
        return True
    if lowered_name.startswith(".env"):
        return True
    if lowered_name in {value.casefold() for value in DENIED_NAMES}:
        return True
    return any(lowered_name.endswith(suffix) for suffix in DENIED_SUFFIXES)


def _included_files(project_root: Path) -> list[Path]:
    root = project_root.resolve()
    included: list[Path] = []
    for relative in _git_tracked_paths(root):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Git reported an unsafe path: {relative}")
        if not _is_allowlisted(relative) or _is_denied(relative):
            continue
        path = root / relative
        if path.is_symlink():
            raise ValueError(f"release bundle refuses tracked symlink: {relative.as_posix()}")
        if not path.is_file():
            raise FileNotFoundError(
                f"allowlisted Git-tracked file is missing: {relative.as_posix()}"
            )
        included.append(path)
    return sorted(included, key=lambda item: item.relative_to(root).as_posix())


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


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
    """Package only Git-tracked, allowlisted, public-safe source files."""

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
    archive_sha256 = sha256_file(destination)
    checksum_path = destination.parent / "SHA256SUMS"
    _atomic_write(
        checksum_path,
        f"{archive_sha256}  {destination.name}\n".encode(),
    )
    return {
        "archive": destination.relative_to(root).as_posix(),
        "bytes": destination.stat().st_size,
        "checksum_file": checksum_path.relative_to(root).as_posix(),
        "checksum_sha256": sha256_file(checksum_path),
        "file_count": len(records),
        "sha256": archive_sha256,
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
