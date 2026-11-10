"""OCR engine abstraction.

Evaluation never imports a concrete engine: it is handed an
:class:`OCREngine` instance. Adding PaddleOCR or TrOCR later therefore means
writing one class in this package and selecting it in the config -- the runner,
the metrics and the report are untouched. See ``docs/EXTENDING.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

Image = np.ndarray


@runtime_checkable
class OCREngine(Protocol):
    """Structural type of a text-recognition engine."""

    name: str

    def read(self, image: Image) -> str:
        """Recognise and return all text in ``image`` as a single string.

        Implementations must not mutate ``image`` and must return ``""``
        rather than raising when an image contains no legible text.
        """
        ...


class OCRError(RuntimeError):
    """Raised when an OCR engine is unavailable or fails irrecoverably."""
