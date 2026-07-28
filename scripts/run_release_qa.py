"""Run release QA and save a deterministic, timing-free JSON receipt."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

QA_SCHEMA_VERSION = "crisisforge.release-qa-receipt.v1"


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("release QA requires a Git repository") from exc
    return {"commit": commit, "dirty": bool(status.strip())}


def _repo_relative(project_root: Path, path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"path must be repository-relative: {path}")
    resolved = (project_root / candidate).resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path escapes the repository: {path}") from exc


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_check(
    *,
    identifier: str,
    execute: Sequence[str] | None,
    display: Sequence[str],
    project_root: Path,
    required: bool,
    unavailable_reason: str | None = None,
) -> tuple[dict[str, Any], bool]:
    if execute is None:
        status = "failed" if required else "skipped"
        return (
            {
                "command": list(display),
                "id": identifier,
                "reason": unavailable_reason or "tool_unavailable",
                "required": required,
                "status": status,
            },
            not required,
        )
    completed = subprocess.run(
        list(execute),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    passed = completed.returncode == 0
    if not passed:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return (
        {
            "command": list(display),
            "id": identifier,
            "required": required,
            "returncode": completed.returncode,
            "status": "passed" if passed else "failed",
        },
        passed,
    )


def run_release_qa(
    project_root: Path,
    *,
    output_path: str = "artifacts/release/qa_receipt.json",
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run mandatory QA checks and write a receipt without timestamps or durations."""
    root = project_root.resolve()
    output_relative = _repo_relative(root, output_path)
    git = _git_state(root)
    if git["dirty"] and not allow_dirty:
        raise RuntimeError("refusing to run release QA from a dirty Git worktree")

    pytest_command = (
        [sys.executable, "-m", "pytest"] if importlib.util.find_spec("pytest") is not None else None
    )
    ruff_command = (
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "src",
            "tests",
            "scripts",
        ]
        if importlib.util.find_spec("ruff") is not None
        else None
    )
    checks_to_run: list[tuple[str, Sequence[str] | None, Sequence[str], bool, str | None]] = [
        (
            "pytest",
            pytest_command,
            ["python", "-m", "pytest"],
            True,
            "pytest_module_unavailable",
        ),
        (
            "ruff",
            ruff_command,
            ["python", "-m", "ruff", "check", "src", "tests", "scripts"],
            True,
            "ruff_module_unavailable",
        ),
        (
            "compileall",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "src",
                "tests",
                "scripts",
            ],
            ["python", "-m", "compileall", "-q", "src", "tests", "scripts"],
            True,
            None,
        ),
    ]

    cffconvert = shutil.which("cffconvert")
    checks_to_run.append(
        (
            "citation_cff",
            [cffconvert, "--validate", "-i", "CITATION.cff"] if cffconvert else None,
            ["cffconvert", "--validate", "-i", "CITATION.cff"],
            False,
            "cffconvert_unavailable",
        )
    )

    checks: list[dict[str, Any]] = []
    passed = True
    for identifier, execute, display, required, unavailable_reason in checks_to_run:
        receipt, check_passed = _run_check(
            identifier=identifier,
            execute=execute,
            display=display,
            project_root=root,
            required=required,
            unavailable_reason=unavailable_reason,
        )
        checks.append(receipt)
        passed = passed and check_passed

    payload = {
        "checks": checks,
        "git": git,
        "schema_version": QA_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
    }
    _atomic_json(root / output_relative, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        default="artifacts/release/qa_receipt.json",
        help="Repository-relative receipt path.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development only: permit a dirty Git worktree.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    payload = run_release_qa(
        args.project_root,
        output_path=args.output,
        allow_dirty=args.allow_dirty,
    )
    print(
        json.dumps(
            {
                "receipt": _repo_relative(args.project_root.resolve(), args.output),
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
