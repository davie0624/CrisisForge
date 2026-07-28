from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/build_release_bundle.py"
_SPEC = importlib.util.spec_from_file_location("build_release_bundle", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

ARCHIVE_ROOT = _MODULE.ARCHIVE_ROOT
build_release_bundle = _MODULE.build_release_bundle


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_bundle_is_deterministic_and_excludes_runtime_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/model.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/result.json").write_text('{"ok": true}\n', encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv/secret.txt").write_text("exclude\n", encoding="utf-8")
    (tmp_path / "work").mkdir()
    (tmp_path / "work/scratch.txt").write_text("exclude\n", encoding="utf-8")
    (tmp_path / "dist").mkdir()

    first = build_release_bundle(tmp_path)
    archive_path = tmp_path / first["archive"]
    first_digest = _digest(archive_path)
    second = build_release_bundle(tmp_path)

    assert first_digest == _digest(archive_path)
    assert first["sha256"] == second["sha256"]
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert f"{ARCHIVE_ROOT}/src/model.py" in names
        assert f"{ARCHIVE_ROOT}/artifacts/result.json" in names
        assert f"{ARCHIVE_ROOT}/BUNDLE_MANIFEST.json" in names
        assert all("/.venv/" not in name for name in names)
        assert all("/work/" not in name for name in names)
        assert all("/dist/" not in name for name in names)
        manifest = json.loads(archive.read(f"{ARCHIVE_ROOT}/BUNDLE_MANIFEST.json"))
        assert manifest["file_count"] == 2
