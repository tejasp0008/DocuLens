"""Shared fixtures for the DocuLens test suite."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from doculens.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@pytest.fixture(scope="session")
def default_config_path() -> Path:
    """Return the path of the default YAML configuration."""
    return DEFAULT_CONFIG


@pytest.fixture(scope="session")
def config() -> Config:
    """Return the validated default configuration."""
    return load_config(DEFAULT_CONFIG)


@pytest.fixture
def tiny_page() -> np.ndarray:
    """Return a small synthetic page: dark horizontal text bars on white paper."""
    image = np.full((240, 320), 255, dtype=np.uint8)
    for y in range(40, 200, 28):
        cv2.rectangle(image, (30, y), (280, y + 9), 30, -1)
    return image


@pytest.fixture
def tiny_colour_page(tiny_page: np.ndarray) -> np.ndarray:
    """Return the same synthetic page as a 3-channel BGR image."""
    return cv2.cvtColor(tiny_page, cv2.COLOR_GRAY2BGR)
