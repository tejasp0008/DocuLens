"""Chart primitives and the shared visual theme for report figures.

Figures are static PNGs destined for a written report, so there is no hover
layer and no dark mode: one light surface, validated for colour-vision
deficiency, is the whole target.

Colour assignment follows the entity, not its rank -- a method keeps its hue
whichever position it finishes in, so two charts can never swap colours on the
reader. Three of the four categorical slots sit below 3:1 contrast on the light
surface; the relief for that is the table view, and every chart here is written
alongside the CSV and markdown table of the same numbers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CATEGORICAL_SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
"""Fixed categorical hue order (blue, orange, aqua, yellow, magenta, green)."""

DIVERGING_POSITIVE = "#2a78d6"
"""Blue pole: removing the stage made recognition worse (the stage helps)."""

DIVERGING_NEGATIVE = "#e34948"
"""Red pole: removing the stage made recognition better (the stage hurts)."""

BAR_RADIUS_PX = 4.0
BAR_GAP_PX = 2.0
LINE_WIDTH = 2.0
MARKER_SIZE = 8.0


def apply_theme() -> None:
    """Install the shared matplotlib styling for every DocuLens figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK_PRIMARY,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def method_colours(methods: Sequence[str]) -> dict[str, str]:
    """Assign a stable hue to each method in fixed slot order.

    Args:
        methods: Method names in their configured order.

    Returns:
        A mapping of method name to hex colour.
    """
    return {
        method: CATEGORICAL_SLOTS[index % len(CATEGORICAL_SLOTS)]
        for index, method in enumerate(methods)
    }


def _rounded_bar_path(
    ax: plt.Axes, x: float, width: float, height: float, baseline: float = 0.0
) -> MplPath:
    """Build a bar path with rounded corners at the data end only.

    The rounding radius is converted from pixels into separate x and y data
    units so that it renders as a true 4px corner whatever the axis scales are.

    Args:
        ax: Axes the bar will be drawn on.
        x: Left edge of the bar in data units.
        width: Bar width in data units.
        height: Signed bar height in data units.
        baseline: Value the bar is anchored to.

    Returns:
        The bar outline as a matplotlib path.
    """
    top = baseline + height
    try:
        extent = ax.get_window_extent()
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        rx = BAR_RADIUS_PX * (x_hi - x_lo) / max(extent.width, 1.0)
        ry = BAR_RADIUS_PX * (y_hi - y_lo) / max(extent.height, 1.0)
    except (ValueError, AttributeError):  # pragma: no cover - layout not ready
        rx = ry = 0.0
    rx = min(rx, width / 2.0)
    ry = min(ry, abs(height)) if height else 0.0
    sign = 1.0 if height >= 0 else -1.0
    ry *= sign

    vertices = [
        (x, baseline),
        (x, top - ry),
        (x, top),
        (x + rx, top),
        (x + width - rx, top),
        (x + width, top),
        (x + width, top - ry),
        (x + width, baseline),
        (x, baseline),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(vertices, codes)


def draw_rounded_bar(
    ax: plt.Axes, x: float, width: float, height: float, colour: str, baseline: float = 0.0
) -> None:
    """Draw one bar with a rounded data end anchored to the baseline.

    Args:
        ax: Target axes.
        x: Left edge in data units.
        width: Bar width in data units.
        height: Signed bar height in data units.
        colour: Fill colour.
        baseline: Value the bar is anchored to.
    """
    patch = PathPatch(
        _rounded_bar_path(ax, x, width, height, baseline),
        facecolor=colour,
        edgecolor="none",
        linewidth=0,
    )
    ax.add_patch(patch)


def _pixels_to_x_units(ax: plt.Axes, pixels: float) -> float:
    """Convert a pixel distance into x-axis data units.

    Args:
        ax: Axes providing the scale.
        pixels: Distance in pixels.

    Returns:
        The equivalent distance in data units, or ``0.0`` if unavailable.
    """
    try:
        x_lo, x_hi = ax.get_xlim()
        return pixels * (x_hi - x_lo) / max(ax.get_window_extent().width, 1.0)
    except (ValueError, AttributeError):  # pragma: no cover - layout not ready
        return 0.0


def grouped_bar_chart(
    summary: pd.DataFrame,
    methods: Sequence[str],
    value_column: str,
    title: str,
    y_label: str,
    category_column: str = "degradation",
) -> Figure:
    """Draw grouped bars of one metric per category and method.

    Args:
        summary: Aggregated rows with the category, ``method`` and value columns.
        methods: Method names in configured order; sets both the legend order
            and the colour assignment.
        value_column: Column holding the plotted value.
        title: Chart title.
        y_label: Y-axis label.
        category_column: Column holding the category shown on the x axis.

    Returns:
        The finished figure.
    """
    apply_theme()
    categories = list(dict.fromkeys(summary[category_column]))
    colours = method_colours(methods)
    pivot = summary.pivot_table(
        index=category_column, columns="method", values=value_column, aggfunc="mean"
    ).reindex(categories)

    figure, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.set_xlim(-0.5, len(categories) - 0.5)
    top = float(pivot.to_numpy().max()) if pivot.size else 1.0
    ax.set_ylim(0, max(top * 1.18, 1e-3))
    figure.canvas.draw()

    gap = _pixels_to_x_units(ax, BAR_GAP_PX)
    group_width = 0.78
    slot_width = group_width / max(len(methods), 1)
    bar_width = max(slot_width - gap, slot_width * 0.4)

    for method_index, method in enumerate(methods):
        if method not in pivot.columns:
            continue
        for category_index, category in enumerate(categories):
            value = float(pivot.loc[category, method])
            left = category_index - group_width / 2.0 + method_index * slot_width
            draw_rounded_bar(ax, left, bar_width, value, colours[method])

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        plt.Line2D([], [], color=colours[m], linewidth=8, solid_capstyle="round") for m in methods
    ]
    ax.legend(handles, list(methods), loc="upper left", ncol=min(len(methods), 4))
    figure.tight_layout()
    return figure


def severity_line_chart(
    by_level: pd.DataFrame,
    methods: Sequence[str],
    value_column: str,
    title: str,
    y_label: str,
) -> Figure:
    """Draw one small-multiple line panel per degradation: metric vs severity.

    Args:
        by_level: Aggregated rows with ``degradation``, ``level``, ``method``
            and the value column.
        methods: Method names in configured order.
        value_column: Column holding the plotted value.
        title: Figure title.
        y_label: Y-axis label.

    Returns:
        The finished figure.
    """
    apply_theme()
    degradations = [d for d in dict.fromkeys(by_level["degradation"]) if d != "none"]
    colours = method_colours(methods)
    columns = min(len(degradations), 3) or 1
    rows = (len(degradations) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 3.2 * rows), squeeze=False, sharey=True
    )
    flat = [ax for row in axes for ax in row]

    for ax, degradation in zip(flat, degradations):
        subset = by_level[by_level["degradation"] == degradation]
        for method in methods:
            series = subset[subset["method"] == method].sort_values("level")
            if series.empty:
                continue
            ax.plot(
                series["level"],
                series[value_column],
                color=colours[method],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE,
                markeredgecolor=SURFACE,
                markeredgewidth=2.0,
                label=method,
                solid_capstyle="round",
            )
        ax.set_title(degradation)
        ax.set_xlabel("severity level")
        ax.set_xticks(sorted(subset["level"].unique()))
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in flat[len(degradations) :]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel(y_label)

    handles = [
        plt.Line2D(
            [],
            [],
            color=colours[m],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
        )
        for m in methods
    ]
    figure.legend(handles, list(methods), loc="lower center", ncol=min(len(methods), 4))
    figure.suptitle(title, fontsize=13, fontweight="bold", color=INK_PRIMARY)
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    return figure


def ablation_chart(ablation: pd.DataFrame, title: str) -> Figure:
    """Draw the leave-one-stage-out ablation as a diverging horizontal bar chart.

    Sign carries the meaning, so the two poles of the diverging pair are used:
    blue where removing a stage raised CER (the stage was helping) and red
    where removing it lowered CER (the stage was hurting). Every bar is
    directly labelled with its delta, since there are few enough to read.

    Args:
        ablation: Ablation summary with ``removed_stage`` and
            ``delta_cer_vs_full`` columns; the full-pipeline row is dropped.
        title: Chart title.

    Returns:
        The finished figure.
    """
    apply_theme()
    # The unablated reference row has an empty removed_stage, which becomes NaN on
    # a CSV round-trip; both spellings must be dropped.
    removed = ablation["removed_stage"]
    rows = ablation[removed.notna() & (removed.astype(str).str.strip() != "")].copy()
    rows = rows.sort_values("stage_index")
    labels = list(rows["removed_stage"])
    deltas = [float(v) for v in rows["delta_cer_vs_full"]]

    figure, ax = plt.subplots(figsize=(9.0, 0.56 * len(labels) + 2.2))
    span = max((abs(d) for d in deltas), default=1e-3) or 1e-3
    ax.set_xlim(-span * 1.45, span * 1.45)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    figure.canvas.draw()

    for index, delta in enumerate(deltas):
        colour = DIVERGING_POSITIVE if delta >= 0 else DIVERGING_NEGATIVE
        patch = PathPatch(
            _rounded_bar_path_horizontal(ax, index, delta),
            facecolor=colour,
            edgecolor="none",
        )
        ax.add_patch(patch)
        offset = span * 0.04
        ax.text(
            delta + (offset if delta >= 0 else -offset),
            index,
            f"{delta:+.3f}",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=9,
            color=INK_SECONDARY,
        )

    ax.axvline(0, color=BASELINE, linewidth=1.2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("change in mean CER when the stage is removed")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], color=DIVERGING_POSITIVE, linewidth=8, solid_capstyle="round"),
        plt.Line2D([], [], color=DIVERGING_NEGATIVE, linewidth=8, solid_capstyle="round"),
    ]
    ax.legend(
        handles,
        ["removing it raises CER (stage helps)", "removing it lowers CER (stage hurts)"],
        loc="upper right",
    )
    figure.tight_layout()
    return figure


def _rounded_bar_path_horizontal(ax: plt.Axes, y_index: int, value: float) -> MplPath:
    """Build a horizontal bar path with a rounded data end.

    Args:
        ax: Target axes.
        y_index: Row index of the bar.
        value: Signed bar length in data units.

    Returns:
        The bar outline as a matplotlib path.
    """
    height = 0.62
    y0 = y_index - height / 2.0
    try:
        extent = ax.get_window_extent()
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        rx = BAR_RADIUS_PX * (x_hi - x_lo) / max(extent.width, 1.0)
        ry = BAR_RADIUS_PX * abs(y_hi - y_lo) / max(extent.height, 1.0)
    except (ValueError, AttributeError):  # pragma: no cover - layout not ready
        rx = ry = 0.0
    rx = min(rx, abs(value)) if value else 0.0
    ry = min(ry, height / 2.0)
    sign = 1.0 if value >= 0 else -1.0
    rx *= sign
    end = value

    vertices = [
        (0.0, y0),
        (end - rx, y0),
        (end, y0),
        (end, y0 + ry),
        (end, y0 + height - ry),
        (end, y0 + height),
        (end - rx, y0 + height),
        (0.0, y0 + height),
        (0.0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(vertices, codes)


def save_figure(figure: Figure, path: Path, dpi: int) -> Path:
    """Write a figure to disk and close it.

    Args:
        figure: Figure to save.
        path: Destination path; parent directories are created.
        dpi: Output resolution.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    logger.debug("wrote figure %s", path)
    return path
