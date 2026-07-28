from __future__ import annotations

from pathlib import Path

from crisisforge.config import (
    display_path,
    project_root_from_module,
    resolve_config_path,
)


def test_project_root_searches_upward_from_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/pipeline.yaml").write_text("project: {}\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    nested = tmp_path / "nested/deeper"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert project_root_from_module() == tmp_path


def test_config_path_resolution_and_display_support_external_files(
    tmp_path: Path,
) -> None:
    internal = resolve_config_path(
        tmp_path,
        Path("configs/example.yaml"),
        default_relative="unused.yaml",
    )
    assert internal == (tmp_path / "configs/example.yaml").resolve()
    assert display_path(internal, project_root=tmp_path) == "configs/example.yaml"

    external = tmp_path.parent / "external-crisisforge-test.yaml"
    resolved = resolve_config_path(
        tmp_path,
        external,
        default_relative="unused.yaml",
    )
    assert resolved == external.resolve()
    assert display_path(resolved, project_root=tmp_path) == str(external.resolve())
