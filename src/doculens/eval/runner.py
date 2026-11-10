"""Runs OCR methods over a corpus and produces tidy result rows.

The runner knows nothing about Tesseract: it is handed an
:class:`~doculens.ocr.base.OCREngine` and a mapping of method name to
:class:`~doculens.dip.pipeline.Pipeline`. Swapping in PaddleOCR or TrOCR later
therefore changes nothing in this module.

Results are always written long-format -- one row per
``(doc_id, degradation, level, method)`` -- so that every table and chart in
the report is a ``groupby`` of the same file.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from doculens.config import BASELINE_METHOD, Config, NormalisationConfig
from doculens.data.corpus import read_image
from doculens.dip.pipeline import Pipeline
from doculens.eval.metrics import score
from doculens.ocr.base import OCREngine

logger = logging.getLogger(__name__)

RESULT_COLUMNS = (
    "doc_id",
    "doc_type",
    "split",
    "degradation",
    "level",
    "method",
    "cer",
    "wer",
    "runtime_ms",
    "preprocess_ms",
    "ocr_ms",
    "n_chars_gt",
    "n_chars_pred",
)
"""Column order of the tidy long-format results table."""

PREDICTION_COLUMNS = ("doc_id", "degradation", "level", "method", "text")
"""Column order of the saved OCR transcripts."""

_MS_PER_S = 1000.0


@dataclass(frozen=True)
class EvalItem:
    """One corpus image together with its ground truth."""

    doc_id: str
    doc_type: str
    split: str
    degradation: str
    level: int
    image_path: Path
    ground_truth: str


def default_jobs() -> int:
    """Return a sensible default worker count for OCR.

    Returns:
        The CPU count capped at 8, or 1 if it cannot be determined.
    """
    return max(1, min(8, os.cpu_count() or 1))


def build_methods(
    config: Config, methods: Sequence[str] | None = None
) -> dict[str, Pipeline | None]:
    """Resolve method names to pipelines.

    Args:
        config: Validated configuration.
        methods: Method names. Defaults to ``eval.methods``.

    Returns:
        A mapping of method name to pipeline, with ``None`` for the reserved
        ``baseline_raw`` method, which performs no preprocessing.
    """
    names = tuple(methods) if methods is not None else config.eval.methods
    resolved: dict[str, Pipeline | None] = {}
    for name in names:
        resolved[name] = None if name == BASELINE_METHOD else config.build_pipeline(name)
    return resolved


def items_from_manifest(manifest: pd.DataFrame, root: Path) -> list[EvalItem]:
    """Convert manifest rows into evaluation items, reading the ground truth.

    Args:
        manifest: Degraded-corpus manifest rows.
        root: Project root that manifest paths are relative to.

    Returns:
        The evaluation items in manifest order.
    """
    items: list[EvalItem] = []
    cache: dict[str, str] = {}
    for row in manifest.itertuples(index=False):
        if row.txt not in cache:
            cache[row.txt] = (root / row.txt).read_text(encoding="utf-8")
        items.append(
            EvalItem(
                doc_id=str(row.doc_id),
                doc_type=str(row.doc_type),
                split=str(row.split),
                degradation=str(row.degradation),
                level=int(row.level),
                image_path=root / row.image,
                ground_truth=cache[row.txt],
            )
        )
    return items


def evaluate_image(
    image: np.ndarray,
    ground_truth: str,
    pipeline: Pipeline | None,
    engine: OCREngine,
    normalisation: NormalisationConfig,
) -> tuple[float, float, float, float, str]:
    """Preprocess, recognise and score a single image.

    This is the smallest reusable unit of evaluation and takes only plain
    inputs, so a future agent or MCP tool can call it directly without any
    filesystem layout.

    Args:
        image: The input image. Not mutated.
        ground_truth: Raw reference text.
        pipeline: Preprocessing pipeline, or ``None`` for the raw baseline.
        engine: OCR engine to read with.
        normalisation: Text normalisation settings.

    Returns:
        A tuple ``(cer, wer, preprocess_ms, ocr_ms, predicted_text)``.
    """
    start = time.perf_counter()
    processed = image if pipeline is None else pipeline(image)
    after_preprocess = time.perf_counter()
    text = engine.read(processed)
    after_ocr = time.perf_counter()
    cer, wer = score(ground_truth, text, normalisation)
    return (
        cer,
        wer,
        (after_preprocess - start) * _MS_PER_S,
        (after_ocr - after_preprocess) * _MS_PER_S,
        text,
    )


def _evaluate_item(
    item: EvalItem,
    methods: Mapping[str, Pipeline | None],
    engine: OCREngine,
    normalisation: NormalisationConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate every method on one corpus image.

    Args:
        item: The corpus image and its ground truth.
        methods: Method name to pipeline (``None`` for the raw baseline).
        engine: OCR engine.
        normalisation: Text normalisation settings.

    Returns:
        A tuple of (result rows, prediction rows) for this image.
    """
    image = read_image(item.image_path)
    results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for method, pipeline in methods.items():
        cer, wer, pre_ms, ocr_ms, text = evaluate_image(
            image, item.ground_truth, pipeline, engine, normalisation
        )
        results.append(
            {
                "doc_id": item.doc_id,
                "doc_type": item.doc_type,
                "split": item.split,
                "degradation": item.degradation,
                "level": item.level,
                "method": method,
                "cer": cer,
                "wer": wer,
                "runtime_ms": pre_ms + ocr_ms,
                "preprocess_ms": pre_ms,
                "ocr_ms": ocr_ms,
                "n_chars_gt": len(item.ground_truth),
                "n_chars_pred": len(text),
            }
        )
        predictions.append(
            {
                "doc_id": item.doc_id,
                "degradation": item.degradation,
                "level": item.level,
                "method": method,
                "text": text,
            }
        )
    return results, predictions


def run_evaluation(
    items: Sequence[EvalItem],
    methods: Mapping[str, Pipeline | None],
    engine: OCREngine,
    normalisation: NormalisationConfig,
    jobs: int = 1,
    show_progress: bool = True,
    description: str = "evaluating",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every method on every item.

    Work is distributed over a thread pool because each OCR call spends its
    time in a Tesseract subprocess. Rows are collected per item and sorted
    afterwards, so the output is byte-identical regardless of ``jobs``; only
    the timing columns vary with parallelism.

    Args:
        items: Corpus images to evaluate.
        methods: Method name to pipeline (``None`` for the raw baseline).
        engine: OCR engine.
        normalisation: Text normalisation settings.
        jobs: Number of worker threads.
        show_progress: Whether to display a progress bar.
        description: Progress bar label.

    Returns:
        A tuple ``(results, predictions)`` of tidy long-format DataFrames.
    """
    if not items:
        raise ValueError("no items to evaluate")
    worker_count = max(1, int(jobs))
    logger.info(
        "evaluating %d images x %d methods (%d OCR calls) with %d worker(s)",
        len(items),
        len(methods),
        len(items) * len(methods),
        worker_count,
    )

    results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []

    def work(item: EvalItem) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        return _evaluate_item(item, methods, engine, normalisation)

    if worker_count == 1:
        iterator = (work(item) for item in items)
    else:
        pool = ThreadPoolExecutor(max_workers=worker_count)
        iterator = pool.map(work, items)
    progress = tqdm(
        iterator, total=len(items), desc=description, unit="img", disable=not show_progress
    )
    try:
        for item_results, item_predictions in progress:
            results.extend(item_results)
            predictions.extend(item_predictions)
    finally:
        progress.close()
        if worker_count != 1:
            pool.shutdown()

    results_df = pd.DataFrame(results, columns=list(RESULT_COLUMNS))
    results_df = results_df.sort_values(["doc_id", "degradation", "level", "method"]).reset_index(
        drop=True
    )
    predictions_df = pd.DataFrame(predictions, columns=list(PREDICTION_COLUMNS))
    predictions_df = predictions_df.sort_values(
        ["doc_id", "degradation", "level", "method"]
    ).reset_index(drop=True)
    return results_df, predictions_df


def summarise_by_degradation(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean CER/WER per degradation and method.

    Args:
        results: Tidy long-format results.

    Returns:
        One row per ``(degradation, method)`` with means and sample counts.
    """
    return (
        results.groupby(["degradation", "method"], as_index=False)
        .agg(
            cer_mean=("cer", "mean"),
            cer_median=("cer", "median"),
            cer_std=("cer", "std"),
            wer_mean=("wer", "mean"),
            runtime_ms_mean=("runtime_ms", "mean"),
            n=("cer", "size"),
        )
        .sort_values(["degradation", "method"])
        .reset_index(drop=True)
    )


def summarise_by_level(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean CER/WER per degradation, severity level and method.

    Args:
        results: Tidy long-format results.

    Returns:
        One row per ``(degradation, level, method)``.
    """
    return (
        results.groupby(["degradation", "level", "method"], as_index=False)
        .agg(
            cer_mean=("cer", "mean"),
            wer_mean=("wer", "mean"),
            runtime_ms_mean=("runtime_ms", "mean"),
            n=("cer", "size"),
        )
        .sort_values(["degradation", "level", "method"])
        .reset_index(drop=True)
    )


def summarise_overall(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean CER/WER per method across the whole evaluated corpus.

    Args:
        results: Tidy long-format results.

    Returns:
        One row per method, sorted by mean CER ascending (best first).
    """
    return (
        results.groupby("method", as_index=False)
        .agg(
            cer_mean=("cer", "mean"),
            cer_median=("cer", "median"),
            wer_mean=("wer", "mean"),
            runtime_ms_mean=("runtime_ms", "mean"),
            preprocess_ms_mean=("preprocess_ms", "mean"),
            n=("cer", "size"),
        )
        .sort_values("cer_mean")
        .reset_index(drop=True)
    )


def run_and_save(
    config: Config,
    manifest: pd.DataFrame,
    engine: OCREngine,
    methods: Sequence[str] | None = None,
    jobs: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run the configured methods over a manifest and write every results file.

    Writes ``results_long.csv``, ``predictions.csv``,
    ``summary_by_degradation.csv``, ``summary_by_level.csv`` and
    ``summary_overall.csv`` into the configured results directory.

    Args:
        config: Validated configuration.
        manifest: Corpus rows to evaluate.
        engine: OCR engine.
        methods: Method names. Defaults to ``eval.methods``.
        jobs: Worker threads. Defaults to :func:`default_jobs`.
        show_progress: Whether to display a progress bar.

    Returns:
        The tidy long-format results.
    """
    config.paths.ensure_dirs()
    items = items_from_manifest(manifest, config.paths.root)
    resolved = build_methods(config, methods)
    results, predictions = run_evaluation(
        items=items,
        methods=resolved,
        engine=engine,
        normalisation=config.eval.normalisation,
        jobs=default_jobs() if jobs is None else jobs,
        show_progress=show_progress,
    )
    results.to_csv(config.paths.results_long, index=False)
    predictions.to_csv(config.paths.results_dir / "predictions.csv", index=False)
    summarise_by_degradation(results).to_csv(
        config.paths.results_dir / "summary_by_degradation.csv", index=False
    )
    summarise_by_level(results).to_csv(
        config.paths.results_dir / "summary_by_level.csv", index=False
    )
    summarise_overall(results).to_csv(config.paths.results_dir / "summary_overall.csv", index=False)
    logger.info("wrote %d result rows -> %s", len(results), config.paths.results_long)
    return results
