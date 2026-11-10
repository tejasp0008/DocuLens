"""Registry of controlled document degradations.

Each degradation simulates one real-world capture defect seen in insurance
claim intake (phone photo out of focus, sensor noise in a dim office, a page
fed in crooked, a desk-lamp shadow, aggressive messaging-app recompression).
All of them preserve image shape and ``dtype`` so that a degraded image is a
drop-in replacement for the clean one.

Every function takes an explicit :class:`numpy.random.Generator`, so results
depend only on the seed derived from ``(global seed, doc id, degradation,
level)`` and never on execution order. Severity levels live in the YAML config,
not here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Image = np.ndarray

PAPER_WHITE = 255
"""Fill value for page area exposed by rotation: blank paper, not black."""

_UINT8_MAX = 255


class Degradation(Protocol):
    """Structural type of a degradation function."""

    def __call__(self, image: Image, rng: np.random.Generator, **params: Any) -> Image:
        """Return a degraded copy of ``image`` with the same shape and dtype."""
        ...


def gaussian_blur(image: Image, rng: np.random.Generator, ksize: int) -> Image:
    """Defocus blur, as from a hand-held phone photo that failed to focus.

    Args:
        image: Input image.
        rng: Unused; present for a uniform degradation signature.
        ksize: Odd positive Gaussian kernel size. Larger is more severe.

    Returns:
        The blurred image.

    Raises:
        ValueError: If ``ksize`` is not a positive odd integer.
    """
    del rng
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"blur: 'ksize' must be a positive odd integer, got {ksize!r}")
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def gaussian_noise(image: Image, rng: np.random.Generator, sigma: float) -> Image:
    """Additive zero-mean Gaussian sensor noise, as from a low-light capture.

    Args:
        image: Input image.
        rng: Random generator supplying the noise field.
        sigma: Noise standard deviation in intensity levels (0-255).

    Returns:
        The noisy image, clipped to ``uint8``.

    Raises:
        ValueError: If ``sigma`` is negative.
    """
    if sigma < 0:
        raise ValueError(f"noise: 'sigma' must be >= 0, got {sigma!r}")
    noise = rng.normal(loc=0.0, scale=float(sigma), size=image.shape)
    noisy = image.astype(np.float32) + noise.astype(np.float32)
    return np.clip(noisy, 0, _UINT8_MAX).astype(np.uint8)


def skew(image: Image, rng: np.random.Generator, angle_deg: float) -> Image:
    """In-plane rotation, as from a page fed into a scanner crooked.

    The canvas size is preserved and newly exposed corners are filled with
    paper white, so the result still looks like a scan rather than a photo on
    a black background.

    Args:
        image: Input image.
        rng: Unused; present for a uniform degradation signature.
        angle_deg: Counter-clockwise rotation in degrees.

    Returns:
        The rotated image, same shape as the input.
    """
    del rng
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(angle_deg), 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=PAPER_WHITE,
    )


def lowlight(image: Image, rng: np.random.Generator, min_gain: float, max_gain: float) -> Image:
    """Dim, unevenly lit capture: a horizontal shadow gradient across the page.

    A multiplicative gain ramps linearly from ``min_gain`` at the left edge to
    ``max_gain`` at the right edge. This is the degradation that global
    thresholding cannot handle and that CLAHE plus adaptive thresholding is
    meant to fix, so it is the most informative case in the comparison.

    Args:
        image: Input image.
        rng: Unused; present for a uniform degradation signature.
        min_gain: Brightness multiplier at the shadowed (left) edge.
        max_gain: Brightness multiplier at the lit (right) edge.

    Returns:
        The shadowed image.

    Raises:
        ValueError: If the gains are non-positive or ``min_gain > max_gain``.
    """
    del rng
    if min_gain <= 0 or max_gain <= 0:
        raise ValueError(f"lowlight: gains must be > 0, got {min_gain!r} and {max_gain!r}")
    if min_gain > max_gain:
        raise ValueError(f"lowlight: 'min_gain' ({min_gain}) must be <= 'max_gain' ({max_gain})")
    width = image.shape[1]
    ramp = np.linspace(float(min_gain), float(max_gain), width, dtype=np.float32)
    gain = ramp.reshape(1, width)
    if image.ndim == 3:
        gain = gain[:, :, np.newaxis]
    shadowed = image.astype(np.float32) * gain
    return np.clip(shadowed, 0, _UINT8_MAX).astype(np.uint8)


def jpeg(image: Image, rng: np.random.Generator, quality: int) -> Image:
    """Lossy JPEG recompression, as from a document forwarded over chat apps.

    Args:
        image: Input image.
        rng: Unused; present for a uniform degradation signature.
        quality: JPEG quality in 1-100. Lower is more severe.

    Returns:
        The recompressed image, same shape and dtype as the input.

    Raises:
        ValueError: If ``quality`` is outside 1-100.
        RuntimeError: If OpenCV fails to encode the image.
    """
    del rng
    if not 1 <= int(quality) <= 100:
        raise ValueError(f"jpeg: 'quality' must be in 1-100, got {quality!r}")
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg: cv2.imencode failed")
    flag = cv2.IMREAD_GRAYSCALE if image.ndim == 2 else cv2.IMREAD_COLOR
    decoded = cv2.imdecode(buffer, flag)
    if decoded is None:
        raise RuntimeError("jpeg: cv2.imdecode failed")
    return decoded


DEGRADATION_REGISTRY: dict[str, Degradation] = {
    "blur": gaussian_blur,
    "noise": gaussian_noise,
    "skew": skew,
    "lowlight": lowlight,
    "jpeg": jpeg,
}
"""Mapping of the degradation name used in YAML to its implementing function."""


class UnknownDegradationError(KeyError):
    """Raised when a config references a degradation name that is not registered."""


def available_degradations() -> list[str]:
    """Return the sorted names of all registered degradations."""
    return sorted(DEGRADATION_REGISTRY)


def apply_degradation(
    name: str, image: Image, rng: np.random.Generator, params: Mapping[str, Any]
) -> Image:
    """Apply a registered degradation by name.

    Args:
        name: Registered degradation name.
        image: Input image. Not mutated.
        rng: Random generator for stochastic degradations.
        params: Severity parameters from the config.

    Returns:
        The degraded image, with the same shape and dtype as ``image``.

    Raises:
        UnknownDegradationError: If ``name`` is not registered.
    """
    try:
        function = DEGRADATION_REGISTRY[name]
    except KeyError:
        raise UnknownDegradationError(
            f"unknown degradation {name!r}; registered degradations are {available_degradations()}"
        ) from None
    return function(image, rng, **dict(params))
