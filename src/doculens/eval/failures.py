"""Selection and visualisation of the worst-performing cases.

Averages hide the interesting part. This module picks the highest-CER cases for
one method and renders each as a single figure holding the degraded input, what
the pipeline turned it into, and the ground truth beside the OCR output -- which
is usually enough to see *why* a case failed rather than only that it did.
"""

from __future__ import annotations

import logging
import textwrap

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from doculens.config import BASELINE_METHOD, Config  # noqa: E402
from doculens.data.corpus import read_image  # noqa: E402
from doculens.eval.charts import (  # noqa: E402
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    apply_theme,
    save_figure,
)
from doculens.eval.metrics import normalise_text  # noqa: E402

logger = logging.getLogger(__name__)

TEXT_PANEL_LINES = 16
"""Number of transcript lines shown in a failure figure."""

TEXT_PANEL_WIDTH = 46
"""Wrap width, in characters, for the transcript panels."""

FAILURE_COLUMNS = (
    "rank",
    "doc_id",
    "doc_type",
    "degradation",
    "level",
    "method",
    "cer",
    "wer",
    "baseline_cer",
    "figure",
)
"""Column order of the failure summary table."""


def select_failures(results: pd.DataFrame, method: str, top_k: int) -> pd.DataFrame:
    """Return the ``top_k`` highest-CER rows for one method.

    Ties are broken by WER and then by identifier, so selection is
    deterministic across runs.

    Args:
        results: Tidy long-format results.
        method: Method whose failures to select.
        top_k: How many cases to return.

    Returns:
        The selected rows, worst first, with a 1-based ``rank`` column.

    Raises:
        ValueError: If the method is absent from the results.
    """
    subset = results[results["method"] == method]
    if subset.empty:
        raise ValueError(
            f"no results for method {method!r}; available methods are "
            f"{sorted(results['method'].unique())}"
        )
    ordered = subset.sort_values(
        ["cer", "wer", "doc_id", "degradation", "level"], ascending=[False, False, True, True, True]
    ).head(int(top_k))
    ordered = ordered.reset_index(drop=True)
    ordered.insert(0, "rank", range(1, len(ordered) + 1))
    return ordered


def _wrap_transcript(text: str, max_lines: int, width: int) -> str:
    """Wrap and clip a transcript for display in a figure panel.

    Args:
        text: Raw transcript.
        max_lines: Maximum number of displayed lines.
        width: Wrap width in characters.

    Returns:
        The clipped text, with an ellipsis line if truncated.
    """
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.rstrip()
        if not stripped:
            continue
        lines.extend(textwrap.wrap(stripped, width=width) or [""])
    if not lines:
        return "(no text recognised)"
    if len(lines) > max_lines:
        return "\n".join([*lines[:max_lines], f"... (+{len(lines) - max_lines} more lines)"])
    return "\n".join(lines)


def build_failure_figure(
    config: Config,
    row: pd.Series,
    image_path_relative: str,
    ground_truth: str,
    prediction: str,
) -> Figure:
    """Render one failure case as a before / after / transcript figure.

    Args:
        config: Validated configuration.
        row: The selected results row.
        image_path_relative: Corpus image path relative to the project root.
        ground_truth: Raw ground-truth text.
        prediction: Raw OCR output for this case.

    Returns:
        The finished figure.
    """
    apply_theme()
    degraded = read_image(config.paths.root / image_path_relative)
    method = str(row["method"])
    if method == BASELINE_METHOD:
        processed = degraded
        processed_title = "no preprocessing (baseline)"
    else:
        processed = config.build_pipeline(method)(degraded)
        processed_title = f"after {method}"

    figure = plt.figure(figsize=(13.5, 6.4))
    grid = figure.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.15, 1.15], wspace=0.12)

    ax_in = figure.add_subplot(grid[0, 0])
    ax_in.imshow(degraded, cmap="gray", vmin=0, vmax=255)
    ax_in.set_title(f"input: {row['degradation']} level {row['level']}", fontsize=10)

    ax_out = figure.add_subplot(grid[0, 1])
    ax_out.imshow(processed, cmap="gray", vmin=0, vmax=255)
    ax_out.set_title(processed_title, fontsize=10)

    for ax in (ax_in, ax_out):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        for spine in ax.spines.values():
            spine.set_color(INK_MUTED)

    normalisation = config.eval.normalisation
    panels = (
        (grid[0, 2], "ground truth", normalise_text(ground_truth, normalisation)),
        (grid[0, 3], "OCR output", normalise_text(prediction, normalisation)),
    )
    for cell, title, text in panels:
        ax = figure.add_subplot(cell)
        ax.axis("off")
        ax.set_title(title, fontsize=10)
        ax.text(
            0.0,
            1.0,
            _wrap_transcript(text, TEXT_PANEL_LINES, TEXT_PANEL_WIDTH),
            va="top",
            ha="left",
            family="monospace",
            fontsize=7.4,
            color=INK_SECONDARY,
            transform=ax.transAxes,
        )

    figure.suptitle(
        f"#{int(row['rank'])}  {row['doc_id']} ({row['doc_type']})  -  {method}  -  "
        f"CER {float(row['cer']):.3f}  WER {float(row['wer']):.3f}",
        fontsize=12,
        fontweight="bold",
        color=INK_PRIMARY,
    )
    figure.patch.set_facecolor(SURFACE)
    # subplots_adjust rather than tight_layout: the transcript panels have their
    # axes turned off, which tight_layout cannot measure.
    figure.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.14)
    return figure


def generate_failure_assets(
    config: Config,
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Select the worst cases and write one figure each, plus a summary table.

    Args:
        config: Validated configuration.
        results: Tidy long-format results.
        predictions: Saved OCR transcripts.
        manifest: Degraded-corpus manifest, used to locate images and truth.

    Returns:
        The failure summary table, which is also written to
        ``results/failures/failures.csv``.
    """
    method = config.eval.failures.method
    selected = select_failures(results, method, config.eval.failures.top_k)
    config.paths.failures_dir.mkdir(parents=True, exist_ok=True)

    image_lookup = {
        (r.doc_id, r.degradation, int(r.level)): (r.image, r.txt)
        for r in manifest.itertuples(index=False)
    }
    prediction_lookup = {
        (r.doc_id, r.degradation, int(r.level), r.method): ("" if pd.isna(r.text) else str(r.text))
        for r in predictions.itertuples(index=False)
    }
    baseline_lookup = {
        (r.doc_id, r.degradation, int(r.level)): float(r.cer)
        for r in results[results["method"] == BASELINE_METHOD].itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        key = (row.doc_id, row.degradation, int(row.level))
        image_relative, txt_relative = image_lookup[key]
        ground_truth = (config.paths.root / txt_relative).read_text(encoding="utf-8")
        prediction = prediction_lookup.get((*key, row.method), "")
        figure = build_failure_figure(
            config, pd.Series(row._asdict()), image_relative, ground_truth, prediction
        )
        name = f"{int(row.rank):02d}_{row.doc_id}__{row.degradation}__{int(row.level)}.png"
        path = save_figure(figure, config.paths.failures_dir / name, config.eval.report.dpi)
        rows.append(
            {
                "rank": int(row.rank),
                "doc_id": row.doc_id,
                "doc_type": row.doc_type,
                "degradation": row.degradation,
                "level": int(row.level),
                "method": row.method,
                "cer": float(row.cer),
                "wer": float(row.wer),
                "baseline_cer": baseline_lookup.get(key, float("nan")),
                "figure": str(path.relative_to(config.paths.root)),
            }
        )

    summary = pd.DataFrame(rows, columns=list(FAILURE_COLUMNS))
    summary.to_csv(config.paths.failures_dir / "failures.csv", index=False)
    logger.info("wrote %d failure figures -> %s", len(summary), config.paths.failures_dir)
    return summary
