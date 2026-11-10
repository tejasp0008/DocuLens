"""Tests for corpus naming and seed derivation."""

from __future__ import annotations

import numpy as np
import pytest

from doculens.data.corpus import (
    CLEAN_CONDITION,
    CLEAN_LEVEL,
    degraded_filename,
    derive_variant_rng,
    parse_corpus_filename,
)


def test_degraded_filename_shape() -> None:
    assert degraded_filename("doc_001", "blur", 2) == "doc_001__blur__2.png"


@pytest.mark.parametrize(
    ("doc_id", "degradation", "level"),
    [("doc_001", "blur", 1), ("doc_060", "lowlight", 3), ("doc_007", "jpeg", 2)],
)
def test_parse_round_trips_degraded_filename(doc_id: str, degradation: str, level: int) -> None:
    name = degraded_filename(doc_id, degradation, level)
    assert parse_corpus_filename(name) == (doc_id, degradation, level)


def test_parse_accepts_a_clean_page_name() -> None:
    assert parse_corpus_filename("doc_002.png") == ("doc_002", CLEAN_CONDITION, CLEAN_LEVEL)


def test_parse_ignores_directories() -> None:
    assert parse_corpus_filename("data/degraded/doc_002__noise__3.png") == ("doc_002", "noise", 3)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "doc_002__noise.png",
        "doc_002__noise__x.png",
        "doc_002__noise__3__extra.png",
        "__noise__3.png",
        "doc_002____3.png",
    ],
)
def test_parse_rejects_non_corpus_names(name: str) -> None:
    assert parse_corpus_filename(name) is None


def test_variant_rng_is_deterministic_and_variant_specific() -> None:
    first = derive_variant_rng(42, "doc_001", "blur", 1).integers(0, 2**31, size=4)
    again = derive_variant_rng(42, "doc_001", "blur", 1).integers(0, 2**31, size=4)
    other_level = derive_variant_rng(42, "doc_001", "blur", 2).integers(0, 2**31, size=4)
    other_doc = derive_variant_rng(42, "doc_002", "blur", 1).integers(0, 2**31, size=4)
    other_seed = derive_variant_rng(43, "doc_001", "blur", 1).integers(0, 2**31, size=4)

    assert np.array_equal(first, again)
    for different in (other_level, other_doc, other_seed):
        assert not np.array_equal(first, different)
