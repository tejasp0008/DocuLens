"""Evaluation runner and summaries, exercised with a stub OCR engine.

These tests deliberately avoid Tesseract so that the suite runs anywhere and
stays fast; what is under test is the harness, not the recogniser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from doculens.config import BASELINE_METHOD
from doculens.dip.registry import build_pipeline
from doculens.eval.ablation import FULL_VARIANT, build_ablation_variants, summarise_ablation
from doculens.eval.failures import select_failures
from doculens.eval.runner import (
    EvalItem,
    build_methods,
    evaluate_image,
    run_evaluation,
    summarise_by_degradation,
    summarise_by_level,
    summarise_overall,
)
from doculens.ocr.base import OCREngine

GROUND_TRUTH = "Policy Number: POL-1\nCity: Pune"


@dataclass(frozen=True)
class StubEngine:
    """An OCR engine that returns a fixed string, ignoring the image."""

    text: str
    name: str = "stub"

    def read(self, image: np.ndarray) -> str:
        assert isinstance(image, np.ndarray)
        return self.text


@pytest.fixture
def items(tiny_page):
    return [
        EvalItem(
            doc_id=f"doc_{i:03d}",
            doc_type="claim_form",
            split="test",
            degradation=degradation,
            level=level,
            image_path=None,
            ground_truth=GROUND_TRUTH,
        )
        for i, (degradation, level) in enumerate([("none", 0), ("blur", 1), ("blur", 2)], start=1)
    ]


def test_stub_engine_satisfies_the_protocol():
    assert isinstance(StubEngine("x"), OCREngine)


def test_evaluate_image_scores_a_perfect_read(tiny_page, config):
    cer, wer, pre_ms, ocr_ms, text = evaluate_image(
        tiny_page, GROUND_TRUTH, None, StubEngine(GROUND_TRUTH), config.eval.normalisation
    )
    assert cer == 0.0
    assert wer == 0.0
    assert text == GROUND_TRUTH
    assert pre_ms >= 0.0
    assert ocr_ms >= 0.0


def test_evaluate_image_runs_the_pipeline(tiny_page, config):
    pipeline = config.build_pipeline("pipeline_basic")
    _, _, pre_ms, _, _ = evaluate_image(
        tiny_page, GROUND_TRUTH, pipeline, StubEngine(GROUND_TRUTH), config.eval.normalisation
    )
    assert pre_ms > 0.0


def test_build_methods_maps_the_baseline_to_no_pipeline(config):
    methods = build_methods(config)
    assert methods[BASELINE_METHOD] is None
    assert methods["pipeline_full"].stage_names[0] == "grayscale"


def _run(items, config, monkeypatch, jobs=1, text=GROUND_TRUTH):
    import doculens.eval.runner as runner_module

    monkeypatch.setattr(runner_module, "read_image", lambda path: np.full((60, 80), 255, np.uint8))
    return runner_module.run_evaluation(
        items=items,
        methods={BASELINE_METHOD: None, "basic": build_pipeline("basic", [{"stage": "grayscale"}])},
        engine=StubEngine(text),
        normalisation=config.eval.normalisation,
        jobs=jobs,
        show_progress=False,
    )


def test_run_evaluation_produces_one_row_per_item_and_method(items, config, monkeypatch):
    results, predictions = _run(items, config, monkeypatch)
    assert len(results) == len(items) * 2
    assert len(predictions) == len(items) * 2
    assert set(results["method"]) == {BASELINE_METHOD, "basic"}
    assert (results["cer"] == 0.0).all()


def test_parallel_and_serial_runs_agree(items, config, monkeypatch):
    serial, _ = _run(items, config, monkeypatch, jobs=1)
    parallel, _ = _run(items, config, monkeypatch, jobs=4)
    columns = ["doc_id", "degradation", "level", "method", "cer", "wer"]
    assert serial[columns].equals(parallel[columns])


def test_run_evaluation_rejects_an_empty_item_list(config):
    with pytest.raises(ValueError, match="no items"):
        run_evaluation([], {}, StubEngine(""), config.eval.normalisation)


def test_summaries_aggregate_the_expected_groups(items, config, monkeypatch):
    results, _ = _run(items, config, monkeypatch, text="completely wrong text")
    assert len(summarise_overall(results)) == 2
    assert len(summarise_by_degradation(results)) == 2 * results["degradation"].nunique()
    assert len(summarise_by_level(results)) == 2 * 3
    assert summarise_overall(results)["cer_mean"].between(0, 5).all()


def test_summarise_overall_is_sorted_best_first(items, config, monkeypatch):
    results, _ = _run(items, config, monkeypatch)
    overall = summarise_overall(results)
    assert overall["cer_mean"].is_monotonic_increasing


def test_ablation_variants_drop_exactly_one_stage_each(config):
    pipeline = config.build_pipeline("pipeline_full")
    variants = build_ablation_variants(pipeline)
    assert len(variants) == len(pipeline) + 1
    assert len(variants[FULL_VARIANT]) == len(pipeline)
    for name, variant in variants.items():
        if name != FULL_VARIANT:
            assert len(variant) == len(pipeline) - 1


def test_ablation_summary_computes_deltas_against_the_full_pipeline(items, config, monkeypatch):
    import doculens.eval.runner as runner_module

    monkeypatch.setattr(runner_module, "read_image", lambda path: np.full((60, 80), 255, np.uint8))
    pipeline = build_pipeline("p", [{"stage": "grayscale"}, {"stage": "otsu_threshold"}])
    results, _ = runner_module.run_evaluation(
        items=items,
        methods=build_ablation_variants(pipeline),
        engine=StubEngine(GROUND_TRUTH),
        normalisation=config.eval.normalisation,
        jobs=1,
        show_progress=False,
    )
    summary = summarise_ablation(results, pipeline)
    assert list(summary["removed_stage"]) == ["", "grayscale", "otsu_threshold"]
    assert summary.loc[summary["variant"] == FULL_VARIANT, "delta_cer_vs_full"].iloc[0] == 0.0


def test_select_failures_returns_the_worst_cases_first(items, config, monkeypatch):
    results, _ = _run(items, config, monkeypatch, text="wrong")
    worst = select_failures(results, BASELINE_METHOD, top_k=2)
    assert len(worst) == 2
    assert list(worst["rank"]) == [1, 2]
    assert worst["cer"].is_monotonic_decreasing


def test_select_failures_rejects_an_unknown_method(items, config, monkeypatch):
    results, _ = _run(items, config, monkeypatch)
    with pytest.raises(ValueError, match="no results for method"):
        select_failures(results, "not_a_method", top_k=1)
