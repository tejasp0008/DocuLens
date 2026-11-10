"""Connected-component segmentation of label/value field regions.

This is a diagnostic and a foundation, not part of the OCR path used for the
reported numbers: the current pipeline passes the whole page to Tesseract with
``--psm 6``. Region proposals matter for the next step of the project, where a
field extractor needs to know *where* a value sits, not just that the page
contains it. Keeping it here, behind the same config, means that step does not
start from scratch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from doculens.config import SegmentationConfig
from doculens.dip.base import Image, to_grayscale

logger = logging.getLogger(__name__)

_MAX_UINT8 = 255


@dataclass(frozen=True)
class Region:
    """An axis-aligned candidate text region in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    pixel_count: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Return the region as ``(x, y, width, height)``."""
        return (self.x, self.y, self.width, self.height)


def segment_field_regions(image: Image, config: SegmentationConfig) -> list[Region]:
    """Propose text-line regions by merging characters into lines.

    The foreground is isolated with an inverted Otsu threshold and dilated with
    a wide, short kernel so that neighbouring characters merge into one
    component per text line. Components are then filtered by size relative to
    the page, which removes both speckle and full-page artefacts.

    Args:
        image: Grayscale or BGR page image. Not mutated.
        config: Segmentation settings from the YAML config.

    Returns:
        Regions ordered top to bottom, then left to right. Empty if
        segmentation is disabled in the config.
    """
    if not config.enabled:
        return []
    gray = to_grayscale(image)
    height, width = gray.shape[:2]
    _, foreground = cv2.threshold(gray, 0, _MAX_UINT8, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.dilate_kernel_width, config.dilate_kernel_height)
    )
    merged = cv2.dilate(foreground, kernel, iterations=1)

    count, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    min_width = config.min_region_width_frac * width
    min_height = config.min_region_height_frac * height
    max_height = config.max_region_height_frac * height

    regions: list[Region] = []
    for label in range(1, count):  # label 0 is the background
        x, y, w, h, area = (int(v) for v in stats[label])
        if w < min_width or h < min_height or h > max_height:
            continue
        regions.append(Region(x=x, y=y, width=w, height=h, pixel_count=area))
    regions.sort(key=lambda region: (region.y, region.x))
    logger.debug("segmented %d field regions from %d components", len(regions), count - 1)
    return regions


def draw_regions(image: Image, regions: list[Region], thickness: int = 2) -> np.ndarray:
    """Return a BGR copy of ``image`` with region boxes drawn on it.

    Args:
        image: Page image. Not mutated.
        regions: Regions to draw.
        thickness: Rectangle line thickness in pixels.

    Returns:
        A 3-channel annotated image.
    """
    gray = to_grayscale(image)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for region in regions:
        x, y, w, h = region.box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), thickness)
    return canvas


def crop_region(image: Image, region: Region, pad: int = 0) -> np.ndarray:
    """Crop ``region`` out of ``image``, clamped to the image bounds.

    Args:
        image: Page image.
        region: Region to crop.
        pad: Extra pixels to include on every side.

    Returns:
        The cropped sub-image.
    """
    height, width = image.shape[:2]
    x0 = max(0, region.x - pad)
    y0 = max(0, region.y - pad)
    x1 = min(width, region.x + region.width + pad)
    y1 = min(height, region.y + region.height + pad)
    return image[y0:y1, x0:x1]
