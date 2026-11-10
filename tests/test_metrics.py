"""CER / WER conventions and text normalisation."""

from __future__ import annotations

import pytest

from doculens.config import NormalisationConfig
from doculens.eval.metrics import (
    character_error_rate,
    normalise_text,
    score,
    word_error_rate,
)

DEFAULT = NormalisationConfig(
    lowercase=True, collapse_whitespace=True, strip=True, remove_punctuation=False
)


def test_identical_strings_score_zero():
    assert character_error_rate("policy number: pol-1", "policy number: pol-1") == 0.0
    assert word_error_rate("policy number: pol-1", "policy number: pol-1") == 0.0


def test_empty_reference_against_non_empty_hypothesis_scores_one():
    assert character_error_rate("", "anything") == 1.0
    assert word_error_rate("", "anything") == 1.0


def test_two_empty_strings_score_zero():
    assert character_error_rate("", "") == 0.0
    assert word_error_rate("", "") == 0.0


def test_one_wrong_character_in_ten_scores_one_tenth():
    assert character_error_rate("abcdefghij", "abcdefghiX") == pytest.approx(0.1)


def test_one_wrong_word_in_four_scores_one_quarter():
    assert word_error_rate("the quick brown fox", "the quick brown cat") == pytest.approx(0.25)


def test_normalisation_collapses_the_two_column_gap():
    assert normalise_text("Policy Number:        POL-1", DEFAULT) == "policy number: pol-1"


def test_normalisation_collapses_newlines_and_strips():
    assert normalise_text("  a\n\n  b  \n", DEFAULT) == "a b"


def test_lowercasing_can_be_disabled():
    config = NormalisationConfig(
        lowercase=False, collapse_whitespace=True, strip=True, remove_punctuation=False
    )
    assert normalise_text("ABC def", config) == "ABC def"


def test_punctuation_removal_can_be_enabled():
    config = NormalisationConfig(
        lowercase=True, collapse_whitespace=True, strip=True, remove_punctuation=True
    )
    assert normalise_text("Policy-Number: POL/1.", config) == "policynumber pol1"


def test_score_applies_normalisation_to_both_sides():
    cer, wer = score("Policy Number: POL-1", "policy   number:\nPOL-1  ", DEFAULT)
    assert cer == 0.0
    assert wer == 0.0


def test_cer_is_not_capped_at_one_for_hallucinated_text():
    """A hypothesis much longer than the reference may legitimately exceed 1.0."""
    assert character_error_rate("ab", "abcdefghijklmnop") > 1.0
