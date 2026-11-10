"""Pipeline composition, ablation variants and stage capture."""

from __future__ import annotations

import numpy as np
import pytest

from doculens.dip.pipeline import Pipeline
from doculens.dip.registry import UnknownStageError, available_stages, build_pipeline, create_stage

SPECS = [
    {"stage": "grayscale"},
    {"stage": "median_blur", "params": {"ksize": 3}},
    {"stage": "otsu_threshold"},
]


def test_pipeline_from_config_runs_end_to_end_on_a_tiny_image(tiny_colour_page):
    pipeline = build_pipeline("test", SPECS)
    out = pipeline(tiny_colour_page)
    assert out.ndim == 2
    assert out.shape == tiny_colour_page.shape[:2]
    assert out.dtype == np.uint8


def test_default_pipelines_run_end_to_end(config, tiny_page):
    for name in config.pipelines:
        out = config.build_pipeline(name)(tiny_page)
        assert out.shape == tiny_page.shape, name


def test_pipeline_preserves_stage_order():
    assert build_pipeline("test", SPECS).stage_names == [
        "grayscale",
        "median_blur",
        "otsu_threshold",
    ]


def test_empty_pipeline_is_the_identity(tiny_page):
    assert np.array_equal(Pipeline("empty", [])(tiny_page), tiny_page)


def test_run_capturing_returns_one_image_per_stage(tiny_page):
    pipeline = build_pipeline("test", SPECS)
    final, captured = pipeline.run_capturing(tiny_page)
    assert len(captured) == len(pipeline)
    assert [c.name for c in captured] == pipeline.stage_names
    assert np.array_equal(captured[-1].image, final)


def test_run_saving_stages_writes_numbered_files(tiny_page, tmp_path):
    pipeline = build_pipeline("test", SPECS)
    pipeline.run_saving_stages(tiny_page, tmp_path / "doc")
    names = sorted(p.name for p in (tmp_path / "doc").glob("*.png"))
    assert names == [
        "00_input.png",
        "01_grayscale.png",
        "02_median_blur.png",
        "03_otsu_threshold.png",
    ]


def test_pipeline_does_not_mutate_its_input(tiny_page):
    original = tiny_page.copy()
    build_pipeline("test", SPECS)(tiny_page)
    assert np.array_equal(tiny_page, original)


def test_without_stage_drops_exactly_one_stage():
    pipeline = build_pipeline("test", SPECS)
    ablated = pipeline.without_stage(1)
    assert ablated.stage_names == ["grayscale", "otsu_threshold"]
    assert pipeline.stage_names == ["grayscale", "median_blur", "otsu_threshold"]
    assert ablated.name == "test__minus_01_median_blur"


def test_without_stage_rejects_an_out_of_range_index():
    with pytest.raises(IndexError):
        build_pipeline("test", SPECS).without_stage(9)


def test_describe_reports_stages_and_parameters():
    described = build_pipeline("test", SPECS).describe()
    assert described[1] == {"index": 1, "stage": "median_blur", "params": {"ksize": 3}}


def test_unknown_stage_is_rejected():
    with pytest.raises(UnknownStageError):
        create_stage("does_not_exist")


def test_pipeline_entry_with_unexpected_key_is_rejected():
    with pytest.raises(ValueError, match="unexpected keys"):
        build_pipeline("test", [{"stage": "grayscale", "typo": 1}])


def test_pipeline_entry_without_stage_key_is_rejected():
    with pytest.raises(ValueError, match="missing the 'stage' key"):
        build_pipeline("test", [{"params": {}}])


def test_registry_lists_the_expected_stages():
    assert {"grayscale", "clahe", "deskew", "otsu_threshold"} <= set(available_stages())
