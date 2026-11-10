"""Synthetic document generation, ground truth and split assignment."""

from __future__ import annotations

import numpy as np
import pytest

from doculens.data.generator import (
    DocLine,
    assign_splits,
    available_doc_types,
    format_inr,
    generate_document,
    ground_truth_text,
    structured_fields,
)


@pytest.mark.parametrize("doc_type", available_doc_types())
def test_generated_document_has_the_expected_shape(config, doc_type):
    document = generate_document("doc_001", doc_type, seed=1, data=config.data)
    assert document.image.shape == (config.data.canvas.height, config.data.canvas.width)
    assert document.image.dtype == np.uint8


@pytest.mark.parametrize("doc_type", available_doc_types())
def test_generated_document_has_text_and_fields(config, doc_type):
    document = generate_document("doc_001", doc_type, seed=1, data=config.data)
    assert len(document.text.splitlines()) > 10
    assert len(document.fields) > 10
    assert all(value for value in document.fields.values())


@pytest.mark.parametrize("doc_type", available_doc_types())
def test_every_field_value_appears_in_the_ground_truth(config, doc_type):
    document = generate_document("doc_001", doc_type, seed=3, data=config.data)
    for value in document.fields.values():
        assert value in document.text


def test_generation_is_deterministic_for_a_given_seed(config):
    first = generate_document("doc_001", "claim_form", seed=5, data=config.data)
    second = generate_document("doc_001", "claim_form", seed=5, data=config.data)
    assert first.text == second.text
    assert np.array_equal(first.image, second.image)


def test_different_seeds_give_different_documents(config):
    first = generate_document("doc_001", "claim_form", seed=5, data=config.data)
    second = generate_document("doc_001", "claim_form", seed=6, data=config.data)
    assert first.text != second.text


def test_the_page_is_not_blank(config):
    document = generate_document("doc_001", "invoice", seed=2, data=config.data)
    ink_fraction = float((document.image < 128).mean())
    assert 0.005 < ink_fraction < 0.4


def test_unknown_document_type_is_rejected(config):
    with pytest.raises(ValueError, match="unknown doc_type"):
        generate_document("doc_001", "passport", seed=1, data=config.data)


def test_ground_truth_skips_blank_lines():
    lines = [
        DocLine(kind="title", text="A"),
        DocLine(kind="blank"),
        DocLine(kind="title", text="B"),
    ]
    assert ground_truth_text(lines) == "A\nB"


def test_field_lines_render_as_label_colon_value():
    lines = [DocLine(kind="field", label="City", value="Pune", key="city")]
    assert ground_truth_text(lines) == "City: Pune"
    assert structured_fields(lines) == {"city": "Pune"}


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (0, "INR 0.00"),
        (999.5, "INR 999.50"),
        (1234.0, "INR 1,234.00"),
        (123456.5, "INR 1,23,456.50"),
        (12345678.9, "INR 1,23,45,678.90"),
    ],
)
def test_indian_digit_grouping(amount, expected):
    assert format_inr(amount) == expected


def test_splits_cover_every_document_exactly_once(config):
    doc_ids = [f"doc_{i:03d}" for i in range(1, 61)]
    splits = assign_splits(doc_ids, config.data, seed=42)
    assert set(splits) == set(doc_ids)
    assert set(splits.values()) == {"train", "val", "test"}


def test_split_sizes_match_the_configured_fractions(config):
    doc_ids = [f"doc_{i:03d}" for i in range(1, 61)]
    splits = assign_splits(doc_ids, config.data, seed=42)
    counts = {
        name: sum(1 for v in splits.values() if v == name) for name in ("train", "val", "test")
    }
    assert counts["train"] == round(config.data.split.train * 60)
    assert counts["val"] == round(config.data.split.val * 60)
    assert sum(counts.values()) == 60


def test_split_assignment_is_deterministic(config):
    doc_ids = [f"doc_{i:03d}" for i in range(1, 61)]
    assert assign_splits(doc_ids, config.data, 42) == assign_splits(doc_ids, config.data, 42)
