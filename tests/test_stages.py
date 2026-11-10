"""Individual DIP stages, including the deskew angle convention."""

from __future__ import annotations

import numpy as np
import pytest

from doculens.dip.base import StageError
from doculens.dip.registry import create_stage
from doculens.dip.stages import DeskewStage, estimate_skew_angle, rotate_image

DESKEW_TOLERANCE_DEG = 0.5


@pytest.mark.parametrize("applied_angle", [2.0, 5.0, -5.0, 8.0, -8.0])
def test_deskew_recovers_a_known_rotation(tiny_page, applied_angle):
    """A page rotated by a known angle must come back level within 0.5 degrees."""
    skewed = rotate_image(tiny_page, applied_angle, border_value=255)
    deskewed = DeskewStage(max_abs_angle_deg=15.0, border_value=255)(skewed)
    residual = estimate_skew_angle(deskewed, max_abs_angle_deg=15.0)
    assert abs(residual) < DESKEW_TOLERANCE_DEG


def test_deskew_estimate_matches_the_applied_rotation(tiny_page):
    """The estimate is the correction angle, so it negates the applied rotation."""
    skewed = rotate_image(tiny_page, 5.0, border_value=255)
    assert estimate_skew_angle(skewed, max_abs_angle_deg=15.0) == pytest.approx(
        -5.0, abs=DESKEW_TOLERANCE_DEG
    )


def test_deskew_leaves_a_level_page_alone(tiny_page):
    assert estimate_skew_angle(tiny_page, max_abs_angle_deg=15.0) == pytest.approx(0.0, abs=0.1)


def test_deskew_ignores_an_implausible_estimate():
    """With too little foreground the estimate degrades to 'do not rotate'."""
    blank = np.full((100, 100), 255, dtype=np.uint8)
    assert estimate_skew_angle(blank, max_abs_angle_deg=15.0) == 0.0


def test_grayscale_converts_colour_and_passes_through_gray(tiny_page, tiny_colour_page):
    stage = create_stage("grayscale")
    assert stage(tiny_colour_page).ndim == 2
    assert stage(tiny_page).ndim == 2


def test_thresholding_stages_produce_binary_images(tiny_page):
    for name, params in (
        ("otsu_threshold", {}),
        ("adaptive_threshold", {"block_size": 31, "c": 15}),
    ):
        out = create_stage(name, params)(tiny_page)
        assert set(np.unique(out)).issubset({0, 255}), name


def test_morphology_open_removes_isolated_speckles():
    page = np.full((120, 120), 255, dtype=np.uint8)
    page[60:70, 20:100] = 0  # a text-like bar
    page[20, 20] = 0  # an isolated speckle
    page[30, 80] = 0
    opened = create_stage(
        "morphology_open", {"kernel_width": 2, "kernel_height": 2, "iterations": 1}
    )(page)
    assert opened[20, 20] == 255
    assert opened[30, 80] == 255
    assert opened[65, 60] == 0


def test_stages_do_not_mutate_their_input(tiny_page):
    for name, params in (
        ("clahe", {"clip_limit": 2.0, "tile_grid_size": [8, 8]}),
        ("median_blur", {"ksize": 3}),
        ("bilateral_filter", {"diameter": 7, "sigma_colour": 50.0, "sigma_space": 50.0}),
        ("otsu_threshold", {}),
        ("deskew", {"max_abs_angle_deg": 15.0, "border_value": 255}),
    ):
        original = tiny_page.copy()
        create_stage(name, params)(tiny_page)
        assert np.array_equal(tiny_page, original), name


def test_stage_rejects_a_non_image_input():
    with pytest.raises(StageError):
        create_stage("grayscale")("not an image")


def test_stage_rejects_an_empty_image():
    with pytest.raises(StageError):
        create_stage("grayscale")(np.zeros((0, 0), dtype=np.uint8))


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("median_blur", {"ksize": 4}),
        ("median_blur", {"ksize": -3}),
        ("adaptive_threshold", {"block_size": 30, "c": 15}),
        ("clahe", {"clip_limit": 0.0, "tile_grid_size": [8, 8]}),
        ("morphology_open", {"kernel_width": 0, "kernel_height": 2, "iterations": 1}),
    ],
)
def test_invalid_stage_parameters_are_rejected(name, params):
    with pytest.raises(ValueError):
        create_stage(name, params)
