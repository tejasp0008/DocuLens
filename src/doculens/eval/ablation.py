"""Leave-one-stage-out ablation of a configured pipeline.

The ablation reads the same YAML pipeline definition as the main run and drops
one stage at a time, so the two can never drift apart: adding a stage to
``pipeline_full`` in the config automatically adds a row to the ablation table.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from doculens.config import Config
from doculens.dip.pipeline import Pipeline
from doculens.eval.runner import EvalItem, default_jobs, items_from_manifest, run_evaluation
from doculens.ocr.base import OCREngine

logger = logging.getLogger(__name__)

FULL_VARIANT = "full"
"""Label used for the unablated reference pipeline."""

ABLATION_COLUMNS = (
    "variant",
    "removed_stage",
    "stage_index",
    "cer_mean",
    "cer_median",
    "wer_mean",
    "delta_cer_vs_full",
    "delta_wer_vs_full",
    "runtime_ms_mean",
    "n",
)
"""Column order of the ablation summary table."""


def build_ablation_variants(pipeline: Pipeline) -> dict[str, Pipeline | None]:
    """Build the full pipeline plus one variant per removed stage.

    Args:
        pipeline: The pipeline to ablate.

    Returns:
        A mapping of variant name to pipeline. The key :data:`FULL_VARIANT`
        holds the unablated pipeline; the others are named
        ``minus_<NN>_<stage>``.
    """
    variants: dict[str, Pipeline | None] = {FULL_VARIANT: pipeline}
    for index, stage in enumerate(pipeline.stages):
        variants[f"minus_{index:02d}_{stage.name}"] = pipeline.without_stage(index)
    return variants


def summarise_ablation(results: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Aggregate ablation results and compare each variant with the full pipeline.

    A positive ``delta_cer_vs_full`` means removing the stage made recognition
    worse, so the stage was earning its place. A negative delta means the
    pipeline is better without that stage, which is reported as found.

    Args:
        results: Tidy long-format results whose ``method`` column holds variant
            names.
        pipeline: The pipeline that was ablated, used to recover stage order.

    Returns:
        The ablation summary, ordered with the full pipeline first and then by
        pipeline stage order.
    """
    grouped = (
        results.groupby("method", as_index=False)
        .agg(
            cer_mean=("cer", "mean"),
            cer_median=("cer", "median"),
            wer_mean=("wer", "mean"),
            runtime_ms_mean=("runtime_ms", "mean"),
            n=("cer", "size"),
        )
        .rename(columns={"method": "variant"})
    )
    full_row = grouped.loc[grouped["variant"] == FULL_VARIANT]
    if full_row.empty:
        raise ValueError("ablation results do not contain the full pipeline variant")
    full_cer = float(full_row["cer_mean"].iloc[0])
    full_wer = float(full_row["wer_mean"].iloc[0])

    stage_names = pipeline.stage_names

    def _removed(variant: str) -> str:
        if variant == FULL_VARIANT:
            return ""
        return stage_names[int(variant.split("_")[1])]

    def _index(variant: str) -> int:
        return -1 if variant == FULL_VARIANT else int(variant.split("_")[1])

    grouped["removed_stage"] = grouped["variant"].map(_removed)
    grouped["stage_index"] = grouped["variant"].map(_index)
    grouped["delta_cer_vs_full"] = grouped["cer_mean"] - full_cer
    grouped["delta_wer_vs_full"] = grouped["wer_mean"] - full_wer
    return grouped.sort_values("stage_index")[list(ABLATION_COLUMNS)].reset_index(drop=True)


def run_ablation(
    config: Config,
    manifest: pd.DataFrame,
    engine: OCREngine,
    pipeline_name: str | None = None,
    jobs: int | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the leave-one-stage-out ablation and write its tables.

    Writes ``ablation_long.csv`` (one row per image and variant) and
    ``ablation.csv`` (the summary) into the configured results directory.

    Args:
        config: Validated configuration.
        manifest: Corpus rows to evaluate.
        engine: OCR engine.
        pipeline_name: Pipeline to ablate. Defaults to ``eval.ablation.pipeline``.
        jobs: Worker threads. Defaults to :func:`~doculens.eval.runner.default_jobs`.
        show_progress: Whether to display a progress bar.

    Returns:
        A tuple ``(summary, long_results)``.
    """
    config.paths.ensure_dirs()
    name = pipeline_name or config.eval.ablation.pipeline
    pipeline = config.build_pipeline(name)
    variants = build_ablation_variants(pipeline)
    logger.info("ablating %r: %d variants over %d stages", name, len(variants), len(pipeline))

    items: Sequence[EvalItem] = items_from_manifest(manifest, config.paths.root)
    results, _ = run_evaluation(
        items=items,
        methods=variants,
        engine=engine,
        normalisation=config.eval.normalisation,
        jobs=default_jobs() if jobs is None else jobs,
        show_progress=show_progress,
        description="ablation",
    )
    results["ablated_pipeline"] = name
    results.to_csv(config.paths.results_dir / "ablation_long.csv", index=False)
    summary = summarise_ablation(results, pipeline)
    summary.insert(0, "ablated_pipeline", name)
    summary.to_csv(config.paths.ablation_csv, index=False)
    logger.info("wrote ablation summary -> %s", config.paths.ablation_csv)
    return summary, results
