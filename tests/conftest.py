"""Shared pytest fixtures/helpers (hardware-free)."""

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def sample_scene() -> np.ndarray:
    """A synthetic BGR test scene with known shapes."""
    image = np.full((120, 160, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (60, 60), (0, 0, 255), -1)  # red square
    cv2.circle(image, (120, 60), 25, (0, 255, 0), -1)          # green circle
    cv2.rectangle(image, (90, 90), (150, 115), (255, 0, 0), -1)  # blue bar
    return image


@pytest.fixture(scope="session")
def sample_gray(sample_scene) -> np.ndarray:
    return cv2.cvtColor(sample_scene, cv2.COLOR_BGR2GRAY)


@pytest.fixture(scope="session")
def scene_file(tmp_path_factory) -> str:
    """A real PNG on disk for I/O tests."""
    image = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
    path = tmp_path_factory.mktemp("io") / "test.png"
    assert cv2.imwrite(str(path), image)
    return str(path)
