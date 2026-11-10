"""Core abstractions for composable image-processing stages.

A *stage* is a callable that maps one image to another image. Stages are the
only unit of image processing in DocuLens: pipelines are ordered lists of
stages, ablation removes stages one at a time, and the per-stage debug images
written by ``save_stages`` are keyed by stage name.

Adding a new operation to the project therefore means writing one subclass of
:class:`BaseStage` and adding one entry to
:data:`doculens.dip.registry.STAGE_REGISTRY` -- nothing else changes.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Protocol, runtime_checkable

import cv2
import numpy as np

Image = np.ndarray
"""An OpenCV-style image: ``uint8``, either ``(H, W)`` or ``(H, W, 3)`` BGR."""


@runtime_checkable
class Stage(Protocol):
    """Structural type implemented by every pipeline stage.

    Implementations must be pure: calling a stage must not mutate its input
    array and must not touch the filesystem.
    """

    name: str

    @property
    def params(self) -> dict[str, Any]:
        """Return the stage's configured parameters (for logging/reporting)."""
        ...

    def __call__(self, image: Image) -> Image:
        """Apply the stage to ``image`` and return a new image."""
        ...


class StageError(RuntimeError):
    """Raised when a stage receives an input it cannot process."""


class BaseStage(abc.ABC):
    """Convenience base class recording parameters and validating inputs.

    Subclasses declare a class-level :attr:`name`, accept their parameters as
    explicit keyword arguments in ``__init__`` (so that an unknown parameter in
    the YAML config raises ``TypeError`` instead of being silently ignored),
    pass those parameters to ``super().__init__`` and implement :meth:`apply`.
    """

    name: ClassVar[str] = "unnamed"

    def __init__(self, **params: Any) -> None:
        self._params: dict[str, Any] = dict(params)

    @property
    def params(self) -> dict[str, Any]:
        """Return a copy of the parameters this stage was constructed with."""
        return dict(self._params)

    @abc.abstractmethod
    def apply(self, image: Image) -> Image:
        """Transform ``image``. Implemented by each concrete stage."""

    def __call__(self, image: Image) -> Image:
        """Validate the input, then delegate to :meth:`apply`."""
        if not isinstance(image, np.ndarray):
            raise StageError(f"stage {self.name!r} expected a numpy array, got {type(image)!r}")
        if image.ndim not in (2, 3):
            raise StageError(f"stage {self.name!r} expected a 2-D or 3-D image, got {image.shape}")
        if image.size == 0:
            raise StageError(f"stage {self.name!r} received an empty image")
        return self.apply(image)

    def __repr__(self) -> str:
        """Return a readable representation including parameters."""
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self._params.items()))
        return f"{type(self).__name__}({args})"


def to_grayscale(image: Image) -> Image:
    """Return a single-channel view of ``image``, converting from BGR if needed.

    Args:
        image: 2-D grayscale or 3-D BGR image.

    Returns:
        A 2-D ``uint8`` image. If ``image`` is already 2-D it is returned
        unchanged (stages never mutate their inputs, so sharing is safe).
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def require_odd_positive(stage: str, param: str, value: int) -> int:
    """Validate that ``value`` is a positive odd integer, else raise.

    Args:
        stage: Stage name, used in the error message.
        param: Parameter name, used in the error message.
        value: Candidate value.

    Returns:
        The validated value.

    Raises:
        ValueError: If ``value`` is not a positive odd integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value % 2 == 0:
        raise ValueError(
            f"stage {stage!r}: parameter {param!r} must be a positive odd integer, got {value!r}"
        )
    return value


def require_positive(stage: str, param: str, value: float) -> float:
    """Validate that ``value`` is a strictly positive number, else raise.

    Args:
        stage: Stage name, used in the error message.
        param: Parameter name, used in the error message.
        value: Candidate value.

    Returns:
        The validated value as a ``float``.

    Raises:
        ValueError: If ``value`` is not strictly positive.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"stage {stage!r}: parameter {param!r} must be a positive number, got {value!r}"
        )
    return float(value)
