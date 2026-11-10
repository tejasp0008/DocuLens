"""Degradations must preserve image shape and dtype, and be reproducible."""

from __future__ import annotations

import numpy as np
import pytest

from doculens.data.degradations import (
    DEGRADATION_REGISTRY,
    UnknownDegradationError,
    apply_degradation,
    available_degradations,
)

LEVEL_PARAMS = {
    "blur": {"ksize": 5},
    "noise": {"sigma": 25},
    "skew": {"angle_deg": 5.0},
    "lowlight": {"min_gain": 0.4, "max_gain": 1.0},
    "jpeg": {"quality": 30},
}


@pytest.mark.parametrize("name", sorted(DEGRADATION_REGISTRY))
def test_degradation_preserves_shape_and_dtype_grayscale(name, tiny_page):
    rng = np.random.default_rng(0)
    out = apply_degradation(name, tiny_page, rng, LEVEL_PARAMS[name])
    assert out.shape == tiny_page.shape
    assert out.dtype == tiny_page.dtype


@pytest.mark.parametrize("name", sorted(DEGRADATION_REGISTRY))
def test_degradation_preserves_shape_and_dtype_colour(name, tiny_colour_page):
    rng = np.random.default_rng(0)
    out = apply_degradation(name, tiny_colour_page, rng, LEVEL_PARAMS[name])
    assert out.shape == tiny_colour_page.shape
    assert out.dtype == tiny_colour_page.dtype


@pytest.mark.parametrize("name", sorted(DEGRADATION_REGISTRY))
def test_degradation_does_not_mutate_its_input(name, tiny_page):
    original = tiny_page.copy()
    apply_degradation(name, tiny_page, np.random.default_rng(0), LEVEL_PARAMS[name])
    assert np.array_equal(tiny_page, original)


def test_same_seed_gives_identical_noise(tiny_page):
    first = apply_degradation("noise", tiny_page, np.random.default_rng(7), {"sigma": 25})
    second = apply_degradation("noise", tiny_page, np.random.default_rng(7), {"sigma": 25})
    assert np.array_equal(first, second)


def test_different_seeds_give_different_noise(tiny_page):
    first = apply_degradation("noise", tiny_page, np.random.default_rng(1), {"sigma": 25})
    second = apply_degradation("noise", tiny_page, np.random.default_rng(2), {"sigma": 25})
    assert not np.array_equal(first, second)


def test_lowlight_darkens_the_shadowed_edge(tiny_page):
    out = apply_degradation(
        "lowlight", tiny_page, np.random.default_rng(0), {"min_gain": 0.4, "max_gain": 1.0}
    )
    left_mean = float(out[:, :40].mean())
    right_mean = float(out[:, -40:].mean())
    assert left_mean < right_mean


def test_unknown_degradation_is_rejected(tiny_page):
    with pytest.raises(UnknownDegradationError):
        apply_degradation("does_not_exist", tiny_page, np.random.default_rng(0), {})


def test_invalid_parameters_are_rejected(tiny_page):
    with pytest.raises(ValueError):
        apply_degradation("blur", tiny_page, np.random.default_rng(0), {"ksize": 4})
    with pytest.raises(ValueError):
        apply_degradation("jpeg", tiny_page, np.random.default_rng(0), {"quality": 0})


def test_registry_is_not_empty():
    assert set(available_degradations()) == set(DEGRADATION_REGISTRY)
