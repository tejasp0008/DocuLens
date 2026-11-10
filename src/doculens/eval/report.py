"""Generation of every report asset: tables, charts, figures and a filled report.

Everything in ``results/`` is produced here from the tidy results written by
:mod:`doculens.eval.runner`, so the tables, the charts and the prose quote the
same numbers by construction. Nothing in this module recomputes a metric.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from doculens.config import BASELINE_METHOD, Config  # noqa: E402
from doculens.data.corpus import CLEAN_CONDITION, read_image  # noqa: E402
from doculens.eval.charts import (  # noqa: E402
    BASELINE,
    CATEGORICAL_SLOTS,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    ablation_chart,
    apply_theme,
    grouped_bar_chart,
    save_figure,
    severity_line_chart,
)

logger = logging.getLogger(__name__)

_FLOAT_PRECISION = 4
_MONTAGE_COLUMNS = 4
_DIAGRAM_FUTURE_COLOUR = "#6250d6"


def dataframe_to_markdown(frame: pd.DataFrame, precision: int = _FLOAT_PRECISION) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Written by hand rather than via ``DataFrame.to_markdown`` so that the
    project does not take a dependency on ``tabulate`` just to format tables.

    Args:
        frame: Table to render.
        precision: Decimal places for floating-point columns.

    Returns:
        The markdown table, or a placeholder if the frame is empty.
    """
    if frame.empty:
        return "_(no rows)_"

    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:.{precision}f}"
        return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)

    header = list(frame.columns)
    lines = ["| " + " | ".join(str(c) for c in header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def _headline_table(results: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    """Build the baseline-versus-pipeline CER table, one row per degradation.

    Args:
        results: Tidy long-format results.
        methods: Method names in configured order.

    Returns:
        A wide table of mean CER per degradation and method.
    """
    pivot = results.pivot_table(index="degradation", columns="method", values="cer", aggfunc="mean")
    ordered = [m for m in methods if m in pivot.columns]
    pivot = pivot[ordered].reset_index()
    pivot.columns.name = None
    return pivot


def write_tables(config: Config, results: pd.DataFrame, ablation: pd.DataFrame | None) -> Path:
    """Write every results table as markdown, ready to paste into the report.

    Args:
        config: Validated configuration.
        results: Tidy long-format results.
        ablation: Ablation summary, or ``None`` if the ablation has not been run.

    Returns:
        The path of the written markdown file.
    """
    from doculens.eval.runner import (
        summarise_by_degradation,
        summarise_by_level,
        summarise_overall,
    )

    methods = list(config.eval.methods)
    sections = [
        "# DocuLens result tables",
        "",
        f"Split(s) evaluated: `{', '.join(config.eval.report_splits)}`. "
        f"Rows: {len(results)}. Documents: {results['doc_id'].nunique()}.",
        "",
        "## Mean CER by degradation and method",
        "",
        dataframe_to_markdown(_headline_table(results, methods)),
        "",
        "## Overall, per method",
        "",
        dataframe_to_markdown(summarise_overall(results)),
        "",
        "## By degradation",
        "",
        dataframe_to_markdown(summarise_by_degradation(results)),
        "",
        "## By degradation and severity level",
        "",
        dataframe_to_markdown(summarise_by_level(results)),
    ]
    if ablation is not None:
        sections += [
            "",
            f"## Leave-one-stage-out ablation of `{config.eval.ablation.pipeline}`",
            "",
            dataframe_to_markdown(ablation),
        ]
    path = config.paths.results_dir / "tables.md"
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    logger.info("wrote markdown tables -> %s", path)
    return path


def generate_charts(
    config: Config, results: pd.DataFrame, ablation: pd.DataFrame | None
) -> list[Path]:
    """Write the CER charts for the report.

    Args:
        config: Validated configuration.
        results: Tidy long-format results.
        ablation: Ablation summary, or ``None`` to skip the ablation chart.

    Returns:
        The paths of the figures written.
    """
    from doculens.eval.runner import summarise_by_degradation, summarise_by_level

    methods = [m for m in config.eval.methods if m in set(results["method"])]
    dpi = config.eval.report.dpi
    written: list[Path] = []

    written.append(
        save_figure(
            grouped_bar_chart(
                summarise_by_degradation(results),
                methods=methods,
                value_column="cer_mean",
                title="Mean character error rate by degradation (test split)",
                y_label="mean CER (lower is better)",
            ),
            config.paths.figures_dir / "cer_by_degradation.png",
            dpi,
        )
    )
    written.append(
        save_figure(
            grouped_bar_chart(
                summarise_by_degradation(results),
                methods=methods,
                value_column="wer_mean",
                title="Mean word error rate by degradation (test split)",
                y_label="mean WER (lower is better)",
            ),
            config.paths.figures_dir / "wer_by_degradation.png",
            dpi,
        )
    )
    written.append(
        save_figure(
            severity_line_chart(
                summarise_by_level(results),
                methods=methods,
                value_column="cer_mean",
                title="Character error rate against degradation severity",
                y_label="mean CER",
            ),
            config.paths.figures_dir / "cer_vs_severity.png",
            dpi,
        )
    )
    if ablation is not None:
        written.append(
            save_figure(
                ablation_chart(
                    ablation,
                    f"Leave-one-stage-out ablation of {config.eval.ablation.pipeline}",
                ),
                config.paths.figures_dir / "ablation.png",
                dpi,
            )
        )
    return written


def generate_stage_figures(config: Config, manifest: pd.DataFrame) -> list[Path]:
    """Save per-stage images and a montage figure for a few sample documents.

    Intermediate images go to ``outputs/stages/<doc_id>/NN_<stage>.png`` (this
    is the ``save_stages`` path required by the pipeline) and the montage goes
    to ``results/figures/``.

    Args:
        config: Validated configuration.
        manifest: Degraded-corpus manifest.

    Returns:
        The paths of the montage figures written.
    """
    settings = config.eval.report
    pipeline = config.build_pipeline(settings.headline_method)
    candidates = manifest[
        (manifest["degradation"] == settings.stage_sample_degradation)
        & (manifest["level"] == settings.stage_sample_level)
        & (manifest["split"].isin(config.eval.report_splits))
    ].sort_values("doc_id")
    if candidates.empty:
        logger.warning(
            "no corpus rows for the configured stage sample condition %s level %d",
            settings.stage_sample_degradation,
            settings.stage_sample_level,
        )
        return []

    # Remove montages from a previous run: a change to n_docs, the split or the
    # sample condition would otherwise leave stale figures beside the new ones.
    for stale in config.paths.figures_dir.glob("stages_*.png"):
        stale.unlink()

    written: list[Path] = []
    for row in candidates.head(settings.stage_sample_docs).itertuples(index=False):
        image = read_image(config.paths.root / row.image)
        out_dir = config.paths.stages_dir / f"{row.doc_id}__{row.degradation}__{int(row.level)}"
        pipeline.run_saving_stages(image, out_dir)
        figure = _montage_figure(
            out_dir,
            title=(
                f"{settings.headline_method} stage by stage - {row.doc_id}, "
                f"{row.degradation} level {int(row.level)}"
            ),
        )
        written.append(
            save_figure(
                figure,
                config.paths.figures_dir
                / f"stages_{row.doc_id}__{row.degradation}__{int(row.level)}.png",
                settings.dpi,
            )
        )
    return written


def _montage_figure(stage_dir: Path, title: str) -> Figure:
    """Lay out every saved stage image of one document in a grid.

    Args:
        stage_dir: Directory holding ``NN_<stage>.png`` images.
        title: Figure title.

    Returns:
        The finished figure.
    """
    apply_theme()
    paths = sorted(stage_dir.glob("*.png"))
    columns = min(_MONTAGE_COLUMNS, max(len(paths), 1))
    rows = (len(paths) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 4.4 * rows), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax, path in zip(flat, paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(path.stem.replace("_", " "), fontsize=9)
    for ax in flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        for spine in ax.spines.values():
            spine.set_color(INK_MUTED)
    for ax in flat[len(paths) :]:
        ax.set_visible(False)
    figure.suptitle(title, fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def generate_segmentation_figure(config: Config, manifest: pd.DataFrame) -> Path | None:
    """Draw the connected-component field regions on one clean sample page.

    Args:
        config: Validated configuration.
        manifest: Degraded-corpus manifest.

    Returns:
        The path written, or ``None`` if segmentation is disabled or no clean
        sample is available.
    """
    if not config.segmentation.enabled:
        return None
    from doculens.dip.segmentation import draw_regions, segment_field_regions

    clean = manifest[
        (manifest["degradation"] == CLEAN_CONDITION)
        & (manifest["split"].isin(config.eval.report_splits))
    ].sort_values("doc_id")
    if clean.empty:
        return None
    row = clean.iloc[0]
    image = read_image(config.paths.root / row["image"])
    regions = segment_field_regions(image, config.segmentation)
    annotated = cv2.cvtColor(draw_regions(image, regions), cv2.COLOR_BGR2RGB)

    apply_theme()
    figure, ax = plt.subplots(figsize=(6.4, 8.6))
    ax.imshow(annotated)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)
    ax.set_title(
        f"Field-region segmentation: {len(regions)} regions on {row['doc_id']}",
        fontsize=11,
        fontweight="bold",
    )
    figure.tight_layout()
    return save_figure(
        figure, config.paths.figures_dir / "segmentation_regions.png", config.eval.report.dpi
    )


def generate_block_diagram(config: Config) -> Path:
    """Draw the system block diagram, including the dashed future branch.

    Args:
        config: Validated configuration.

    Returns:
        The path of the figure written.
    """
    apply_theme()
    figure, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    ax.grid(visible=False)

    built = [
        ("Input\ndocument image", CATEGORICAL_SLOTS[0]),
        ("DIP\npreprocessing", CATEGORICAL_SLOTS[2]),
        ("Segmentation\n(field regions)", CATEGORICAL_SLOTS[2]),
        ("OCR engine\n(Tesseract)", CATEGORICAL_SLOTS[1]),
        ("Evaluation\n(CER / WER)", CATEGORICAL_SLOTS[5]),
    ]
    future = [
        "Field\nextraction",
        "RAG over\npolicy rules",
        "Agent loop\n(validate claim)",
        "MCP server\n(tools)",
    ]

    box_width, box_height, gap = 15.4, 9.0, 4.0
    y_built = 29.0
    for index, (label, colour) in enumerate(built):
        x = 2.0 + index * (box_width + gap)
        _draw_box(ax, x, y_built, box_width, box_height, label, colour, dashed=False)
        if index:
            _draw_arrow(ax, x - gap, y_built + box_height / 2, x, y_built + box_height / 2, False)

    y_future = 9.0
    for index, label in enumerate(future):
        x = 2.0 + (index + 1) * (box_width + gap)
        _draw_box(
            ax, x, y_future, box_width, box_height, label, _DIAGRAM_FUTURE_COLOUR, dashed=True
        )
        if index:
            _draw_arrow(ax, x - gap, y_future + box_height / 2, x, y_future + box_height / 2, True)

    ocr_x = 2.0 + 3 * (box_width + gap)
    _draw_arrow(
        ax,
        ocr_x + box_width / 2,
        y_built,
        2.0 + box_width + gap + box_width / 2,
        y_future + box_height,
        True,
    )
    ax.text(
        2.0,
        y_future + box_height / 2,
        "future work\n(stubs only)",
        fontsize=10,
        style="italic",
        color=_DIAGRAM_FUTURE_COLOUR,
        va="center",
        ha="left",
    )
    ax.text(
        2.0,
        y_built + box_height + 3.0,
        "Implemented in this project",
        fontsize=11,
        fontweight="bold",
        color=INK_PRIMARY,
        va="center",
    )
    figure.tight_layout()
    return save_figure(
        figure, config.paths.figures_dir / "block_diagram.png", config.eval.report.dpi
    )


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    colour: str,
    dashed: bool,
) -> None:
    """Draw one labelled block of the diagram.

    Args:
        ax: Target axes.
        x: Left edge.
        y: Bottom edge.
        width: Box width.
        height: Box height.
        label: Text inside the box.
        colour: Outline colour.
        dashed: Whether to draw the outline dashed (used for future work).
    """
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.4,rounding_size=1.4",
            linewidth=1.8,
            edgecolor=colour,
            facecolor=SURFACE,
            linestyle="--" if dashed else "-",
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK_SECONDARY if dashed else INK_PRIMARY,
        fontweight="normal" if dashed else "bold",
    )


def _draw_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, dashed: bool) -> None:
    """Draw a connector arrow between two blocks.

    Args:
        ax: Target axes.
        x0: Start x.
        y0: Start y.
        x1: End x.
        y1: End y.
        dashed: Whether the arrow belongs to the future branch.
    """
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=_DIAGRAM_FUTURE_COLOUR if dashed else BASELINE,
            linestyle="--" if dashed else "-",
            shrinkA=0,
            shrinkB=2,
        )
    )


def _format_pct(value: float) -> str:
    """Format a fraction as a signed percentage string.

    Args:
        value: Fractional change.

    Returns:
        A string such as ``-91.3%``.
    """
    return f"{value * 100:+.1f}%"


def write_filled_report(
    config: Config,
    results: pd.DataFrame,
    ablation: pd.DataFrame | None,
    failures: pd.DataFrame | None,
    manifest: pd.DataFrame,
) -> Path:
    """Write ``results/REPORT.md``: the report skeleton filled with real numbers.

    ``docs/REPORT_SKELETON.md`` stays a hand-written outline under version
    control; this function produces the generated counterpart so that the
    numbers in the prose always come from the last run rather than from memory.

    Args:
        config: Validated configuration.
        results: Tidy long-format results.
        ablation: Ablation summary, or ``None``.
        failures: Failure summary, or ``None``.
        manifest: Degraded-corpus manifest.

    Returns:
        The path of the written report.
    """
    from doculens.eval.runner import summarise_overall

    headline = config.eval.report.headline_method
    overall = summarise_overall(results)
    baseline_cer = float(overall.loc[overall["method"] == BASELINE_METHOD, "cer_mean"].iloc[0])
    headline_cer = float(overall.loc[overall["method"] == headline, "cer_mean"].iloc[0])
    best_method = str(overall.iloc[0]["method"])
    best_cer = float(overall.iloc[0]["cer_mean"])

    per_degradation = _headline_table(results, list(config.eval.methods))
    wins = per_degradation[per_degradation[headline] < per_degradation[BASELINE_METHOD]]
    losses = per_degradation[per_degradation[headline] > per_degradation[BASELINE_METHOD]]

    stage_rows = pd.DataFrame(
        [
            {"#": entry["index"], "stage": entry["stage"], "parameters": entry["params"] or "-"}
            for entry in config.build_pipeline(headline).describe()
        ]
    )
    evaluated_docs = int(results["doc_id"].nunique())
    corpus_docs = int(manifest["doc_id"].nunique())
    n_methods = int(results["method"].nunique())
    conditions = manifest.groupby(["degradation", "level"]).ngroups

    def _case_lines(frame: pd.DataFrame, direction: str) -> str:
        if frame.empty:
            return f"- None: `{headline}` never {direction} the baseline on this corpus.\n"
        lines = []
        for row in frame.itertuples(index=False):
            base = getattr(row, BASELINE_METHOD)
            head = getattr(row, headline)
            change = (head - base) / base if base else float("nan")
            lines.append(
                f"- **{row.degradation}**: baseline CER {base:.4f} -> `{headline}` "
                f"{head:.4f} ({_format_pct(change) if base else 'n/a'})"
            )
        return "\n".join(lines) + "\n"

    parts = [
        "# DocuLens - generated results report",
        "",
        "_Generated by `doculens report`. Every number below is read from "
        "`results/results_long.csv`; nothing here is hand-entered._",
        "",
        "## 1. Problem",
        "",
        "Insurance claim intake in India receives claim forms and garage invoices as "
        "phone photographs and rescans: out of focus, noisy, skewed, unevenly lit and "
        "heavily recompressed. OCR on those images degrades badly, and every downstream "
        "step (field extraction, policy validation, settlement) inherits the error. This "
        "project asks a measurable question: **how much of that loss can classical digital "
        "image processing recover before the OCR engine ever sees the page?**",
        "",
        "## 2. Data",
        "",
        f"- {evaluated_docs} synthetic documents evaluated on the "
        f"`{', '.join(config.eval.report_splits)}` split "
        f"(corpus of {corpus_docs}, split "
        f"{config.data.split.train:.0%}/{config.data.split.val:.0%}/{config.data.split.test:.0%} "
        "train/val/test, fixed before any tuning).",
        f"- Document types: {', '.join(config.data.doc_types)}, rendered at "
        f"{config.data.canvas.width}x{config.data.canvas.height} px with Faker "
        f"`{config.data.locale}` content and exact ground truth.",
        f"- {conditions} capture conditions per document: the undegraded page plus "
        f"{len(config.degradations)} degradations "
        f"({', '.join(sorted(config.degradations))}) at 3 severity levels each.",
        f"- Total evaluated rows: {len(results)} "
        f"({n_methods} methods x {len(results) // max(n_methods, 1)} images).",
        "",
        "## 3. Method",
        "",
        f"Pipelines are ordered lists of stages defined in `{config.source_path.name}`. "
        f"The headline pipeline `{headline}` is:",
        "",
        dataframe_to_markdown(stage_rows),
        "",
        "Scoring normalises both the ground truth and the OCR output "
        f"(lowercase={config.eval.normalisation.lowercase}, "
        f"collapse_whitespace={config.eval.normalisation.collapse_whitespace}, "
        f"strip={config.eval.normalisation.strip}, "
        f"remove_punctuation={config.eval.normalisation.remove_punctuation}), then "
        "computes CER and WER as edit-distance ratios.",
        "",
        "## 4. Results",
        "",
        f"- Baseline (`{BASELINE_METHOD}`, no preprocessing): **mean CER {baseline_cer:.4f}**",
        f"- Headline pipeline (`{headline}`): **mean CER {headline_cer:.4f}** "
        f"({_format_pct((headline_cer - baseline_cer) / baseline_cer) if baseline_cer else 'n/a'} "
        "against the baseline)",
        f"- Best method overall on this corpus: **`{best_method}`** at mean CER {best_cer:.4f}",
        "",
        dataframe_to_markdown(overall),
        "",
        "Mean CER by degradation and method:",
        "",
        dataframe_to_markdown(per_degradation),
        "",
        "Figures: `results/figures/cer_by_degradation.png`, "
        "`results/figures/cer_vs_severity.png`, `results/figures/block_diagram.png`.",
        "",
        "## 5. Success cases",
        "",
        f"Degradations where `{headline}` beats the raw baseline:",
        "",
        _case_lines(wins, "beat"),
        "## 6. Failure cases",
        "",
        f"Degradations where `{headline}` is **worse** than the raw baseline:",
        "",
        _case_lines(losses, "lost to"),
    ]

    if ablation is not None:
        removed_stage = ablation["removed_stage"]
        hurting = ablation[
            removed_stage.notna()
            & (removed_stage.astype(str).str.strip() != "")
            & (ablation["delta_cer_vs_full"] < 0)
        ]
        parts += [
            "## 7. Ablation",
            "",
            f"Leave-one-stage-out on `{config.eval.ablation.pipeline}`. A positive delta means "
            "removing the stage made recognition worse, so the stage earns its place.",
            "",
            dataframe_to_markdown(ablation),
            "",
            (
                "Stages whose removal *improved* mean CER on this corpus: "
                + ", ".join(f"`{s}`" for s in hurting["removed_stage"])
                + "."
                if not hurting.empty
                else "Every stage in the pipeline reduced mean CER on this corpus."
            ),
            "",
            "**This table must not be used to pick the pipeline.** These deltas were "
            "measured on the same split the headline numbers come from, so selecting "
            "a configuration on the strength of them would be test-set leakage and "
            "every number above would stop meaning what it claims to mean. The "
            "principled next step is to re-run the comparison on train/val and only "
            "then change the pipeline; `configs/default.yaml` defines "
            "`pipeline_lean` for exactly that experiment.",
            "",
        ]

    if failures is not None and not failures.empty:
        parts += [
            "## 8. Worst individual cases",
            "",
            f"The {len(failures)} highest-CER cases for `{config.eval.failures.method}`, with "
            "side-by-side figures in `results/failures/`.",
            "",
            dataframe_to_markdown(failures),
            "",
        ]

    parts += [
        "## 9. Limitations",
        "",
        "- **Synthetic data.** Pages are rendered, not scanned. Real captures add paper "
        "texture, non-uniform ink, handwriting, stamps, staples, folds and perspective "
        "distortion that this corpus does not contain, so absolute CER here is optimistic.",
        "- **One OCR engine.** All numbers are Tesseract "
        f"`{config.ocr.tesseract.config}`. A different engine would reorder the methods; "
        "preprocessing that helps Tesseract can hurt a learned recogniser trained on "
        "degraded inputs.",
        "- **Degradations are applied one at a time.** Real photographs are blurred *and* "
        "noisy *and* skewed simultaneously.",
        "- **Clean-page cost.** Preprocessing is not free: see the `none` row above for what "
        "the pipeline does to an already-clean scan.",
        "- **Fixed parameters.** Stage parameters were chosen from the train/val split and "
        "not searched exhaustively; a per-degradation parameter sweep would likely improve "
        "every pipeline number.",
        "",
        "## 10. Future work",
        "",
        "The code is structured so that each of these plugs into an existing interface "
        "rather than requiring a rewrite (see `docs/EXTENDING.md`):",
        "",
        "- A document classifier (CNN/ViT) choosing the pipeline per document type.",
        "- Learned OCR (PaddleOCR, TrOCR) behind the same `OCREngine` protocol.",
        "- Field extraction over the segmented regions, supervised by the `.json` "
        "ground truth this generator already writes.",
        "- RAG over policy rules plus an agent loop for claim validation.",
        "- An MCP server exposing `enhance_image`, `run_ocr`, `classify_document`, "
        "`extract_fields`, `search_policy_rules` and `validate_claim` as tools.",
        "",
    ]

    path = config.paths.results_dir / "REPORT.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("wrote filled report -> %s", path)
    return path


def build_report(config: Config) -> list[Path]:
    """Produce every report asset from the results already on disk.

    Args:
        config: Validated configuration.

    Returns:
        The paths of all files written.

    Raises:
        FileNotFoundError: If the results have not been generated yet.
    """
    from doculens.data.corpus import load_degraded_manifest

    if not config.paths.results_long.is_file():
        raise FileNotFoundError(
            f"{config.paths.results_long} not found; run `doculens run` before `doculens report`"
        )
    results = pd.read_csv(config.paths.results_long)
    manifest = load_degraded_manifest(config)

    ablation = (
        pd.read_csv(config.paths.ablation_csv) if config.paths.ablation_csv.is_file() else None
    )
    if ablation is None:
        logger.warning("no ablation.csv found; ablation table and chart will be skipped")
    failures_csv = config.paths.failures_dir / "failures.csv"
    failures = pd.read_csv(failures_csv) if failures_csv.is_file() else None

    written: list[Path] = [write_tables(config, results, ablation)]
    written += generate_charts(config, results, ablation)
    written.append(generate_block_diagram(config))
    written += generate_stage_figures(config, manifest)
    segmentation = generate_segmentation_figure(config, manifest)
    if segmentation is not None:
        written.append(segmentation)
    written.append(write_filled_report(config, results, ablation, failures, manifest))
    logger.info("wrote %d report assets -> %s", len(written), config.paths.results_dir)
    return written
