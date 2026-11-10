"""Configuration loading, path resolution and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from doculens.config import BASELINE_METHOD, ConfigError, load_config


def _write_config(tmp_path: Path, source: Path, mutate) -> Path:
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    mutate(raw)
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    path = configs / "test.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_default_config_loads(config):
    assert config.seed == 42
    assert config.data.n_docs >= 1
    assert BASELINE_METHOD in config.eval.methods


def test_paths_resolve_against_the_project_root(config):
    assert config.paths.results_long.is_absolute()
    assert config.paths.results_long.parent == config.paths.results_dir


def test_pipelines_build_from_config(config):
    pipeline = config.build_pipeline("pipeline_full")
    assert pipeline.stage_names[0] == "grayscale"
    assert len(pipeline) == len(config.pipelines["pipeline_full"])


def test_unknown_stage_name_is_rejected(tmp_path, default_config_path):
    path = _write_config(
        tmp_path,
        default_config_path,
        lambda raw: raw["pipelines"].__setitem__("pipeline_basic", [{"stage": "nope"}]),
    )
    with pytest.raises(ConfigError, match="unknown stage"):
        load_config(path)


def test_unknown_stage_parameter_is_rejected(tmp_path, default_config_path):
    path = _write_config(
        tmp_path,
        default_config_path,
        lambda raw: raw["pipelines"].__setitem__(
            "pipeline_basic", [{"stage": "median_blur", "params": {"bogus": 3}}]
        ),
    )
    with pytest.raises(ConfigError, match="invalid pipeline"):
        load_config(path)


def test_invalid_stage_parameter_value_is_rejected(tmp_path, default_config_path):
    path = _write_config(
        tmp_path,
        default_config_path,
        lambda raw: raw["pipelines"].__setitem__(
            "pipeline_basic", [{"stage": "median_blur", "params": {"ksize": 4}}]
        ),
    )
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_degradation_is_rejected(tmp_path, default_config_path):
    path = _write_config(
        tmp_path,
        default_config_path,
        lambda raw: raw["degradations"].__setitem__("warp", {"levels": {1: {}, 2: {}, 3: {}}}),
    )
    with pytest.raises(ConfigError, match="unknown degradation"):
        load_config(path)


def test_missing_severity_level_is_rejected(tmp_path, default_config_path):
    def mutate(raw):
        del raw["degradations"]["blur"]["levels"][3]

    with pytest.raises(ConfigError, match="exactly levels"):
        load_config(_write_config(tmp_path, default_config_path, mutate))


def test_unknown_method_is_rejected(tmp_path, default_config_path):
    path = _write_config(
        tmp_path,
        default_config_path,
        lambda raw: raw["eval"].__setitem__("methods", ["pipeline_x"]),
    )
    with pytest.raises(ConfigError, match="neither"):
        load_config(path)


def test_split_fractions_must_sum_to_one(tmp_path, default_config_path):
    path = _write_config(
        tmp_path, default_config_path, lambda raw: raw["data"]["split"].__setitem__("train", 0.9)
    )
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(path)


def test_missing_section_is_rejected(tmp_path, default_config_path):
    path = _write_config(tmp_path, default_config_path, lambda raw: raw.pop("ocr"))
    with pytest.raises(ConfigError, match="missing the required key 'ocr'"):
        load_config(path)


def test_missing_file_is_rejected(tmp_path, default_config_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_unimplemented_ocr_engine_is_rejected(tmp_path, default_config_path):
    path = _write_config(
        tmp_path, default_config_path, lambda raw: raw["ocr"].__setitem__("engine", "trocr")
    )
    with pytest.raises(ConfigError, match="not implemented"):
        load_config(path)
