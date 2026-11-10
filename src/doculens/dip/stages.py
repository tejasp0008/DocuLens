"""Concrete classical DIP stages used by the DocuLens pipelines.

Every numeric parameter is a required keyword argument supplied from the YAML
config, so no tuning constants are hidden in this module. Stages are pure:
they never mutate their input array and never write to disk.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import cv2
import numpy as np

from doculens.dip.base import (
    BaseStage,
    Image,
    require_odd_positive,
    require_positive,
    to_grayscale,
)

logger = logging.getLogger(__name__)

_MAX_UINT8 = 255
_ANGLE_FOLD_DEG = 45.0
_MIN_FOREGROUND_PIXELS = 50


class GrayscaleStage(BaseStage):
    """Convert a BGR image to single-channel grayscale (no-op if already gray)."""

    name: ClassVar[str] = "grayscale"

    def apply(self, image: Image) -> Image:
        """Return the grayscale version of ``image``."""
        return to_grayscale(image)


class ClaheStage(BaseStage):
    """Contrast Limited Adaptive Histogram Equalisation.

    Boosts local contrast, which is what recovers text from the low-light and
    shadow-gradient degradation. Applied to the grayscale image.
    """

    name: ClassVar[str] = "clahe"

    def __init__(self, clip_limit: float, tile_grid_size: list[int]) -> None:
        """Initialise the stage.

        Args:
            clip_limit: Contrast clipping threshold (OpenCV ``clipLimit``).
            tile_grid_size: Two integers, the tile grid used for equalisation.
        """
        require_positive(self.name, "clip_limit", clip_limit)
        if len(tuple(tile_grid_size)) != 2 or any(int(v) < 1 for v in tile_grid_size):
            raise ValueError(
                f"stage {self.name!r}: 'tile_grid_size' must be two positive integers, "
                f"got {tile_grid_size!r}"
            )
        super().__init__(clip_limit=float(clip_limit), tile_grid_size=list(tile_grid_size))

    def apply(self, image: Image) -> Image:
        """Return the locally contrast-equalised grayscale image."""
        gray = to_grayscale(image)
        grid = (int(self._params["tile_grid_size"][0]), int(self._params["tile_grid_size"][1]))
        clahe = cv2.createCLAHE(clipLimit=self._params["clip_limit"], tileGridSize=grid)
        return clahe.apply(gray)


class MedianBlurStage(BaseStage):
    """Median filter: removes salt-and-pepper style speckle while keeping edges."""

    name: ClassVar[str] = "median_blur"

    def __init__(self, ksize: int) -> None:
        """Initialise the stage.

        Args:
            ksize: Odd positive aperture size.
        """
        require_odd_positive(self.name, "ksize", ksize)
        super().__init__(ksize=int(ksize))

    def apply(self, image: Image) -> Image:
        """Return the median-filtered image."""
        return cv2.medianBlur(image, self._params["ksize"])


class BilateralFilterStage(BaseStage):
    """Edge-preserving smoothing: suppresses Gaussian noise but keeps glyph edges."""

    name: ClassVar[str] = "bilateral_filter"

    def __init__(self, diameter: int, sigma_colour: float, sigma_space: float) -> None:
        """Initialise the stage.

        Args:
            diameter: Neighbourhood diameter in pixels.
            sigma_colour: Filter sigma in the intensity domain.
            sigma_space: Filter sigma in the coordinate domain.
        """
        if not isinstance(diameter, int) or diameter < 1:
            raise ValueError(
                f"stage {self.name!r}: 'diameter' must be a positive integer, got {diameter!r}"
            )
        require_positive(self.name, "sigma_colour", sigma_colour)
        require_positive(self.name, "sigma_space", sigma_space)
        super().__init__(
            diameter=int(diameter),
            sigma_colour=float(sigma_colour),
            sigma_space=float(sigma_space),
        )

    def apply(self, image: Image) -> Image:
        """Return the bilaterally filtered image."""
        return cv2.bilateralFilter(
            image,
            self._params["diameter"],
            self._params["sigma_colour"],
            self._params["sigma_space"],
        )


class NlMeansDenoiseStage(BaseStage):
    """Non-local means denoising.

    Averages over similar patches from across the whole page rather than over a
    local neighbourhood, so it suppresses heavy sensor noise with far less
    stroke damage than a wider median or Gaussian kernel. It is the most
    expensive stage in the project, which is why it is only used where noise
    actually dominates.
    """

    name: ClassVar[str] = "nlmeans_denoise"

    def __init__(self, strength: float, template_window: int, search_window: int) -> None:
        """Initialise the stage.

        Args:
            strength: Filter strength ``h``; higher removes more noise and more detail.
            template_window: Odd patch size used for comparison.
            search_window: Odd window size searched for similar patches.
        """
        require_positive(self.name, "strength", strength)
        require_odd_positive(self.name, "template_window", template_window)
        require_odd_positive(self.name, "search_window", search_window)
        super().__init__(
            strength=float(strength),
            template_window=int(template_window),
            search_window=int(search_window),
        )

    def apply(self, image: Image) -> Image:
        """Return the denoised image."""
        gray = to_grayscale(image)
        return cv2.fastNlMeansDenoising(
            gray,
            None,
            self._params["strength"],
            self._params["template_window"],
            self._params["search_window"],
        )


class OtsuThresholdStage(BaseStage):
    """Global binarisation with an Otsu-selected threshold.

    Produces dark text on a white background. Being global, it is fast and
    clean on evenly lit scans but fails under the shadow gradient, which is
    exactly the contrast drawn out in the results.
    """

    name: ClassVar[str] = "otsu_threshold"

    def apply(self, image: Image) -> Image:
        """Return the Otsu-binarised image."""
        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 0, _MAX_UINT8, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


class AdaptiveThresholdStage(BaseStage):
    """Adaptive Gaussian binarisation: one threshold per local window."""

    name: ClassVar[str] = "adaptive_threshold"

    def __init__(self, block_size: int, c: float) -> None:
        """Initialise the stage.

        Args:
            block_size: Odd positive window size in pixels (OpenCV ``blockSize``).
            c: Constant subtracted from the local weighted mean.
        """
        require_odd_positive(self.name, "block_size", block_size)
        if block_size < 3:
            raise ValueError(f"stage {self.name!r}: 'block_size' must be >= 3, got {block_size!r}")
        super().__init__(block_size=int(block_size), c=float(c))

    def apply(self, image: Image) -> Image:
        """Return the adaptively binarised image."""
        gray = to_grayscale(image)
        return cv2.adaptiveThreshold(
            gray,
            _MAX_UINT8,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self._params["block_size"],
            self._params["c"],
        )


def estimate_skew_angle(image: Image, max_abs_angle_deg: float) -> float:
    """Estimate the skew of a document image in degrees.

    The foreground (dark text) is isolated with an inverted Otsu threshold and
    the minimum-area rectangle of all foreground pixels is measured. OpenCV has
    changed the range returned by ``cv2.minAreaRect`` across versions (``[-90,
    0)`` before 4.5, ``(0, 90]`` from 4.5 onward), so the raw angle is folded
    into ``[-45, 45]``, which is correct under either convention.

    Args:
        image: Grayscale or BGR document image.
        max_abs_angle_deg: Estimates larger than this are treated as unreliable
            and ``0.0`` is returned instead, so a failed estimate degrades to
            "do not rotate" rather than destroying the page.

    Returns:
        The *correction* angle in degrees: the counter-clockwise rotation that
        must be applied to level the page. A page skewed 5 degrees
        counter-clockwise therefore yields ``-5.0``. Returns ``0.0`` when no
        reliable estimate is available.
    """
    gray = to_grayscale(image)
    _, foreground = cv2.threshold(gray, 0, _MAX_UINT8, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(foreground)
    if coords is None or len(coords) < _MIN_FOREGROUND_PIXELS:
        logger.debug("skew estimate skipped: too little foreground")
        return 0.0

    angle = float(cv2.minAreaRect(coords)[-1])
    if angle > _ANGLE_FOLD_DEG:
        angle -= 90.0
    elif angle < -_ANGLE_FOLD_DEG:
        angle += 90.0

    if abs(angle) > max_abs_angle_deg:
        logger.debug(
            "skew estimate %.2f deg exceeds limit %.2f, ignoring", angle, max_abs_angle_deg
        )
        return 0.0
    return angle


def rotate_image(image: Image, angle_deg: float, border_value: int) -> Image:
    """Rotate ``image`` about its centre, keeping the original canvas size.

    Args:
        image: Image to rotate.
        angle_deg: Counter-clockwise rotation in degrees.
        border_value: Fill value for pixels outside the source image.

    Returns:
        The rotated image, same shape as the input.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


class DeskewStage(BaseStage):
    """Estimate page skew and rotate it away, filling the border with paper white."""

    name: ClassVar[str] = "deskew"

    def __init__(self, max_abs_angle_deg: float, border_value: int) -> None:
        """Initialise the stage.

        Args:
            max_abs_angle_deg: Largest skew magnitude trusted as an estimate.
            border_value: Fill value for the exposed border after rotation.
        """
        require_positive(self.name, "max_abs_angle_deg", max_abs_angle_deg)
        if not 0 <= int(border_value) <= _MAX_UINT8:
            raise ValueError(
                f"stage {self.name!r}: 'border_value' must be in [0, 255], got {border_value!r}"
            )
        super().__init__(max_abs_angle_deg=float(max_abs_angle_deg), border_value=int(border_value))

    def apply(self, image: Image) -> Image:
        """Return the deskewed image."""
        angle = estimate_skew_angle(image, self._params["max_abs_angle_deg"])
        if angle == 0.0:
            return image
        return rotate_image(image, angle, self._params["border_value"])


class MorphologyOpenStage(BaseStage):
    """Morphological opening applied to the text foreground.

    Documents here are dark text on light paper, so with ``foreground_is_dark``
    the image is inverted first, opened (erode then dilate), and inverted back.
    That removes isolated dark speckles left by noise and binarisation without
    eating into glyph strokes; applying opening directly to the un-inverted
    image would instead operate on the paper background.
    """

    name: ClassVar[str] = "morphology_open"

    def __init__(
        self,
        kernel_width: int,
        kernel_height: int,
        iterations: int,
        foreground_is_dark: bool = True,
    ) -> None:
        """Initialise the stage.

        Args:
            kernel_width: Structuring-element width in pixels.
            kernel_height: Structuring-element height in pixels.
            iterations: Number of opening iterations.
            foreground_is_dark: Invert before opening (true for dark text on
                light paper, the case for every document in this project).
        """
        for param, value in (
            ("kernel_width", kernel_width),
            ("kernel_height", kernel_height),
            ("iterations", iterations),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"stage {self.name!r}: {param!r} must be a positive integer, got {value!r}"
                )
        super().__init__(
            kernel_width=int(kernel_width),
            kernel_height=int(kernel_height),
            iterations=int(iterations),
            foreground_is_dark=bool(foreground_is_dark),
        )

    def apply(self, image: Image) -> Image:
        """Return the opened image."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self._params["kernel_width"], self._params["kernel_height"])
        )
        working = cv2.bitwise_not(image) if self._params["foreground_is_dark"] else image
        opened = cv2.morphologyEx(
            working, cv2.MORPH_OPEN, kernel, iterations=self._params["iterations"]
        )
        return cv2.bitwise_not(opened) if self._params["foreground_is_dark"] else opened


class PerspectiveCorrectStage(BaseStage):
    """Four-point perspective rectification of the largest quadrilateral found.

    Optional stage, not part of the default pipelines: the synthetic corpus
    contains in-plane skew but no projective distortion, so it is provided for
    the photographed-document case the longer-term system will face.
    """

    name: ClassVar[str] = "perspective_correct"

    def __init__(self, min_area_frac: float, approx_epsilon_frac: float) -> None:
        """Initialise the stage.

        Args:
            min_area_frac: Minimum quadrilateral area as a fraction of the image.
            approx_epsilon_frac: ``cv2.approxPolyDP`` epsilon as a fraction of
                the contour perimeter.
        """
        require_positive(self.name, "min_area_frac", min_area_frac)
        require_positive(self.name, "approx_epsilon_frac", approx_epsilon_frac)
        super().__init__(
            min_area_frac=float(min_area_frac), approx_epsilon_frac=float(approx_epsilon_frac)
        )

    def apply(self, image: Image) -> Image:
        """Return the rectified image, or the input if no quadrilateral is found."""
        gray = to_grayscale(image)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape[:2]
        min_area = self._params["min_area_frac"] * height * width
        quad = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < min_area:
                break
            approx = cv2.approxPolyDP(
                contour, self._params["approx_epsilon_frac"] * cv2.arcLength(contour, True), True
            )
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                break
        if quad is None:
            logger.debug("perspective_correct: no quadrilateral found, passing image through")
            return image

        ordered = _order_quadrilateral(quad)
        target = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(ordered, target)
        return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def _order_quadrilateral(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Args:
        points: Array of shape ``(4, 2)``.

    Returns:
        The same points reordered, shape ``(4, 2)``, dtype ``float32``.
    """
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)
