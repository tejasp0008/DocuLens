"""Field-extraction interface: a stub only, not implemented in this version.

The OCR evaluation in this project stops at the page transcript. Turning that
transcript into ``{"policy_number": "POL-MH-2023-004821", ...}`` is the next
step, and this protocol is the seam it will plug into.

Everything it needs already exists: the generator writes the structured fields
of every synthetic document to ``<doc_id>.json`` as supervision, and
:func:`doculens.dip.segmentation.segment_field_regions` already proposes the
label and value regions. See ``docs/EXTENDING.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ExtractedField:
    """One field recovered from a document.

    Attributes:
        key: ``snake_case`` field name, matching the generator's JSON keys.
        value: The extracted value as text.
        confidence: Extractor confidence in ``[0, 1]``.
        bbox: Optional ``(x, y, width, height)`` source region in page pixels.
    """

    key: str
    value: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


@runtime_checkable
class FieldExtractor(Protocol):
    """Structural type of a field extractor.

    Implementations will range from regular expressions over the transcript to
    a layout-aware model or an LLM call; the evaluation harness only needs this
    signature.
    """

    name: str

    def extract(self, image: np.ndarray, text: str) -> list[ExtractedField]:
        """Recover structured fields from a page and its transcript.

        Args:
            image: The preprocessed page image.
            text: The OCR transcript of that page.

        Returns:
            The extracted fields.
        """
        ...


class NotImplementedExtractor:
    """Placeholder extractor that documents the interface and refuses to run."""

    name = "not_implemented"

    def extract(self, image: np.ndarray, text: str) -> list[ExtractedField]:
        """Raise :class:`NotImplementedError`.

        Args:
            image: Unused.
            text: Unused.

        Raises:
            NotImplementedError: Always; field extraction is future work.
        """
        raise NotImplementedError(
            "Field extraction is out of scope for this version of DocuLens. "
            "See docs/EXTENDING.md for how to implement FieldExtractor."
        )
