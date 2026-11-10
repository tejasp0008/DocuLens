"""Text normalisation and the CER / WER metrics.

Both metrics are edit-distance ratios computed by :mod:`jiwer` after the same
normalisation is applied to the reference and the hypothesis. Normalisation is
configuration-driven and documented in the README, because CER is meaningless
without saying what was normalised away.
"""

from __future__ import annotations

import logging
import re
import string

import jiwer

from doculens.config import NormalisationConfig

logger = logging.getLogger(__name__)

_WHITESPACE_RUN = re.compile(r"\s+")
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)

EMPTY_REFERENCE_SCORE = 1.0
"""Score returned when the reference is empty but the hypothesis is not."""


def normalise_text(text: str, config: NormalisationConfig) -> str:
    """Normalise ``text`` prior to scoring.

    Steps, applied in this order and each individually switchable in the
    config: lowercase, remove ASCII punctuation, collapse every run of
    whitespace (including newlines) to a single space, strip the ends.
    Collapsing whitespace is what lets the two-column rendering of
    ``Label:        value`` be scored against the ground-truth line
    ``Label: value``.

    Args:
        text: Raw text, typically straight from an OCR engine.
        config: Which normalisation steps to apply.

    Returns:
        The normalised text.
    """
    result = text
    if config.lowercase:
        result = result.lower()
    if config.remove_punctuation:
        result = result.translate(_PUNCTUATION_TABLE)
    if config.collapse_whitespace:
        result = _WHITESPACE_RUN.sub(" ", result)
    if config.strip:
        result = result.strip()
    return result


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute the character error rate between two already-normalised strings.

    CER is ``(substitutions + insertions + deletions) / len(reference)`` over
    characters. It is not capped at 1.0: an OCR engine that hallucinates text
    on a noisy blank region can score above 1.0, and that is reported honestly
    rather than clipped.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted text.

    Returns:
        The character error rate. Identical strings score ``0.0``; an empty
        reference against a non-empty hypothesis scores
        :data:`EMPTY_REFERENCE_SCORE`.
    """
    if reference == hypothesis:
        return 0.0
    if not reference:
        return EMPTY_REFERENCE_SCORE
    return float(jiwer.cer(reference, hypothesis))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute the word error rate between two already-normalised strings.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted text.

    Returns:
        The word error rate, with the same conventions as
        :func:`character_error_rate`.
    """
    if reference == hypothesis:
        return 0.0
    if not reference.split():
        return EMPTY_REFERENCE_SCORE
    return float(jiwer.wer(reference, hypothesis))


def score(reference: str, hypothesis: str, config: NormalisationConfig) -> tuple[float, float]:
    """Normalise both strings and return their CER and WER.

    This is the entry point used by the evaluation runner and the one a future
    agent or MCP tool should call: it takes plain strings and makes no
    assumptions about the filesystem.

    Args:
        reference: Raw ground-truth text.
        hypothesis: Raw predicted text.
        config: Normalisation settings.

    Returns:
        A tuple ``(cer, wer)``.
    """
    reference_norm = normalise_text(reference, config)
    hypothesis_norm = normalise_text(hypothesis, config)
    return (
        character_error_rate(reference_norm, hypothesis_norm),
        word_error_rate(reference_norm, hypothesis_norm),
    )
