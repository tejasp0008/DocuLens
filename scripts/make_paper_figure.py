#!/usr/bin/env python
"""Build the mechanism figure used in the IEEE write-up.

The evaluation already produces every chart in ``results/figures``. This script
adds the one figure that is argued rather than measured: *why* a locally
adaptive threshold recovers a shadowed page while a global one destroys it.
It places the degraded page, the illumination profile that defeats a single
threshold, and the two binarisations side by side, and annotates each with the
CER that exact image scored in ``results/results_long.csv`` -- so the picture
and the numbers cannot drift apart.

Everything comes from the library: the corpus image is read from disk, and the
two outputs are produced by running the configured ``pipeline_full`` and
``pipeline_full_otsu`` end to end.

Usage:
    python scripts/make_paper_figure.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doculens.config import load_config  # noqa: E402
from doculens.data.corpus import degraded_filename  # noqa: E402
from doculens.eval.charts import CATEGORICAL_SLOTS  # noqa: E402
from doculens.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

DOC_ID = "doc_002"
DEGRADATION = "lowlight"
LEVEL = 3
ADAPTIVE_METHOD = "pipeline_full"
GLOBAL_METHOD = "pipeline_full_otsu"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#898781"

BACKGROUND_PERCENTILE = 90
"""Per-column percentile taken as the paper background, ignoring the ink."""


def crop_margins(source: Path, destination: Path, pad: int = 10, threshold: int = 250) -> None:
    """Copy an image with its surrounding white margin trimmed.

    The generated charts carry a matplotlib margin that is wasted space in a
    page-limited write-up. Trimming to the ink keeps the figure identical but
    lets it be placed smaller.

    Args:
        source: Image to read.
        destination: Path to write the trimmed copy to.
        pad: Pixels of white to keep around the content.
        threshold: Intensity at or above which a pixel counts as background.

    Raises:
        SystemExit: If the source image is missing or entirely blank.
    """
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"missing figure {source}; run `make report`")
    gray = image if image.ndim == 2 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    content = np.argwhere(gray < threshold)
    if not content.size:
        raise SystemExit(f"figure {source} is blank")
    (top, left), (bottom, right) = content.min(0), content.max(0)
    top = max(0, top - pad)
    left = max(0, left - pad)
    bottom = min(gray.shape[0] - 1, bottom + pad)
    right = min(gray.shape[1] - 1, right + pad)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image[top : bottom + 1, left : right + 1])
    logger.info(
        "cropped %s -> %s (%dx%d)",
        source.name,
        destination,
        right - left + 1,
        bottom - top + 1,
    )


def build_degradation_chart(summary: Path, destination: Path, methods: Sequence[str]) -> None:
    """Redraw the per-degradation CER chart at single-column print size.

    The chart in ``results/figures`` is sized for a screen; scaled into a
    3.4-inch column its tick labels fall to about 2 pt. This redraws the same
    data from the same summary table at the size it will actually be printed,
    so the labels stay legible.

    Args:
        summary: Path of ``summary_by_degradation.csv``.
        destination: Path to write the chart to.
        methods: Method names, in the order they should be plotted.

    Raises:
        SystemExit: If the summary table is missing.
    """
    if not summary.is_file():
        raise SystemExit(f"missing {summary}; run `make run`")
    table = pd.read_csv(summary).pivot(index="degradation", columns="method", values="cer_mean")
    present = [m for m in methods if m in table.columns]
    degradations = list(table.index)
    positions = np.arange(len(degradations), dtype=float)
    width = 0.8 / len(present)

    figure, axis = plt.subplots(figsize=(3.4, 1.62), dpi=400)
    for index, method in enumerate(present):
        offset = (index - (len(present) - 1) / 2) * width
        axis.bar(
            positions + offset,
            table[method].to_numpy(),
            width=width * 0.92,
            color=CATEGORICAL_SLOTS[index % len(CATEGORICAL_SLOTS)],
            label=method,
            linewidth=0,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(degradations, fontsize=6.4, color=INK)
    axis.set_ylabel("mean CER (lower is better)", fontsize=6.4, color=INK)
    axis.tick_params(axis="y", labelsize=6.2, colors=MUTED, length=2)
    axis.tick_params(axis="x", colors=MUTED, length=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
        axis.spines[side].set_linewidth(0.6)
    axis.grid(axis="y", color="#e1e0d9", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(fontsize=5.4, frameon=False, ncol=2, loc="upper left", handlelength=1.1)
    figure.subplots_adjust(left=0.135, right=0.995, top=0.985, bottom=0.115)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=400, facecolor="white")
    plt.close(figure)
    logger.info("drew print-size degradation chart -> %s", destination)


def _cer(results: pd.DataFrame, method: str) -> float:
    """Return the recorded CER for one method on the figure's image.

    Args:
        results: The tidy long-format results table.
        method: Method name.

    Returns:
        The CER of that method on ``DOC_ID``/``DEGRADATION``/``LEVEL``.

    Raises:
        SystemExit: If the evaluation has not been run for that cell.
    """
    rows = results[
        (results["doc_id"] == DOC_ID)
        & (results["degradation"] == DEGRADATION)
        & (results["level"] == LEVEL)
        & (results["method"] == method)
    ]
    if rows.empty:
        raise SystemExit(
            f"no result row for {DOC_ID}/{DEGRADATION}/{LEVEL}/{method}; run `make run`"
        )
    return float(rows.iloc[0]["cer"])


def _panel(axis: plt.Axes, image: np.ndarray, caption: str) -> None:
    """Draw one image panel with a caption below it.

    Args:
        axis: Target axes.
        image: Grayscale image to show.
        caption: Caption text placed under the panel.
    """
    axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.6)
    axis.set_xlabel(caption, fontsize=7.4, color=INK, labelpad=5)


def main() -> int:
    """Render the figure.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("-o", "--out", type=Path, default=Path("paper/figures/mechanism.png"))
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    results = pd.read_csv(config.paths.results_long)

    source = config.paths.degraded_dir / degraded_filename(DOC_ID, DEGRADATION, LEVEL)
    degraded = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if degraded is None:
        raise SystemExit(f"missing corpus image {source}; run `make data`")

    adaptive = config.build_pipeline(ADAPTIVE_METHOD)(degraded)
    global_otsu = config.build_pipeline(GLOBAL_METHOD)(degraded)

    # The single threshold Otsu picks for the whole page, and the local
    # threshold surface the adaptive stage uses instead.
    otsu_threshold, _ = cv2.threshold(degraded, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    adaptive_cfg = next(
        stage.params
        for stage in config.build_pipeline(ADAPTIVE_METHOD).stages
        if stage.name == "adaptive_threshold"
    )
    block = int(adaptive_cfg["block_size"])
    offset = float(adaptive_cfg["c"])
    local_mean = cv2.GaussianBlur(degraded.astype(np.float32), (block, block), 0)

    columns = np.arange(degraded.shape[1])
    background = np.percentile(degraded, BACKGROUND_PERCENTILE, axis=0)
    local_threshold = np.percentile(local_mean - offset, BACKGROUND_PERCENTILE, axis=0)

    figure = plt.figure(figsize=(7.16, 2.34), dpi=400)
    grid = figure.add_gridspec(1, 4, wspace=0.17, left=0.012, right=0.988, top=0.985, bottom=0.145)

    _panel(figure.add_subplot(grid[0, 0]), degraded, "(a) input: lowlight, level 3")

    profile = figure.add_subplot(grid[0, 1])
    profile.plot(columns, background, color=BLUE, linewidth=1.3, label="paper background")
    profile.plot(
        columns, local_threshold, color=GREEN, linewidth=1.3, label="local threshold", zorder=3
    )
    profile.axhline(otsu_threshold, color=ORANGE, linewidth=1.3, label="global Otsu threshold")
    # Where the paper itself is darker than the single global threshold, the
    # background is binarised as ink: this band is panel (c)'s black slab.
    crossing = columns[background < otsu_threshold]
    if crossing.size:
        profile.axvspan(crossing.min(), crossing.max(), color=ORANGE, alpha=0.13, linewidth=0)
        profile.text(
            (crossing.min() + crossing.max()) / 2,
            292,
            f"{crossing.size} columns\nread as ink",
            fontsize=6.2,
            color=ORANGE,
            ha="center",
            va="top",
            linespacing=1.25,
        )
    profile.set_xlim(0, columns.max())
    profile.set_ylim(0, 300)
    profile.set_ylabel("intensity (0-255)", fontsize=7.2, color=INK)
    profile.set_xlabel("(b) illumination across the page", fontsize=7.4, color=INK, labelpad=5)
    profile.tick_params(labelsize=6.2, colors=MUTED, length=2)
    profile.set_xticks([0, 400, 800, 1200])
    for side in ("top", "right"):
        profile.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        profile.spines[side].set_color(MUTED)
        profile.spines[side].set_linewidth(0.6)
    profile.grid(axis="y", color="#e1e0d9", linewidth=0.5)
    profile.set_axisbelow(True)
    profile.legend(fontsize=6.2, frameon=False, loc="lower right", handlelength=1.3, borderpad=0.1)

    _panel(
        figure.add_subplot(grid[0, 2]),
        global_otsu,
        f"(c) global Otsu: CER {_cer(results, GLOBAL_METHOD):.3f}",
    )
    _panel(
        figure.add_subplot(grid[0, 3]),
        adaptive,
        f"(d) adaptive: CER {_cer(results, ADAPTIVE_METHOD):.3f}",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=400, facecolor="white")
    plt.close(figure)

    figures = config.paths.results_dir / "figures"
    crop_margins(figures / "block_diagram.png", args.out.parent / "flow.png")
    build_degradation_chart(
        config.paths.results_dir / "summary_by_degradation.csv",
        args.out.parent / "cer_by_degradation.png",
        config.eval.methods,
    )
    logger.info(
        "otsu threshold=%.1f, background min=%.1f, columns read as ink=%d -> %s",
        otsu_threshold,
        background.min(),
        int(crossing.size),
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
