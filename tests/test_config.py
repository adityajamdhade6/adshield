"""Tests for YAML config loading."""

from pathlib import Path

import pytest

from adshield.config import CONFIG_DIR_ENV, default_config_dir, load_yaml


def test_loads_a_mapping(tmp_path: Path) -> None:
    (tmp_path / "x.yaml").write_text("a: 1\nb: [2, 3]\n")
    assert load_yaml("x.yaml", tmp_path) == {"a": 1, "b": [2, 3]}


def test_rejects_non_mapping_and_empty_files(tmp_path: Path) -> None:
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    (tmp_path / "empty.yaml").write_text("")
    for name in ("list.yaml", "empty.yaml"):
        with pytest.raises(ValueError, match="mapping"):
            load_yaml(name, tmp_path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml("nope.yaml", tmp_path)


def test_env_var_overrides_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "x.yaml").write_text("k: v\n")
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    assert default_config_dir() == tmp_path
    assert load_yaml("x.yaml") == {"k": "v"}


def test_default_config_dir_is_the_repo_configs_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    assert (default_config_dir() / "rules.yaml").exists()
