"""Unit tests for image processing (hardware-free)."""
import numpy as np
import pytest

from src.image_processing import processing as P
from src.utils.exceptions import ProcessingError


def make_stripes() -> np.ndarray:
    """High-contrast BGR image with a clear edge."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, 50:] = 255
    return image


class TestFilters:
    def test_filters_preserve_shape(self, sample_scene):
        for fn in (P.blur_gaussian, P.blur_median, P.blur_bilateral):
            assert fn(sample_scene).shape == sample_scene.shape

    def test_gaussian_smooths(self, sample_scene):
        blurred = P.blur_gaussian(sample_scene, 9)
        assert blurred.std() < sample_scene.std()

    def test_even_kernel_rejected(self, sample_scene):
        with pytest.raises(ProcessingError):
            P.blur_gaussian(sample_scene, 4)
        with pytest.raises(ProcessingError):
            P.blur_median(sample_scene, 6)


class TestThreshold:
    def test_threshold_binary_values(self, sample_gray):
        out = P.threshold(sample_gray, 127)
        unique = set(np.unique(out).tolist())
        assert unique <= {0, 255}

    def test_adaptive_threshold(self, sample_gray):
        out = P.adaptive_threshold(sample_gray)
        assert out.dtype == np.uint8
        assert out.ndim == 2


class TestEdge:
    def test_canny_detects_edges(self):
        out = P.canny(make_stripes(), 50, 150)
        # Edge column should be bright, flat areas black
        assert out[:, 49].max() > 0
        assert out[:, 10].max() == 0

    def test_sobel_and_laplacian_shape(self, sample_gray):
        assert P.sobel_x(sample_gray).shape == sample_gray.shape
        assert P.sobel_y(sample_gray).shape == sample_gray.shape
        assert P.sobel_magnitude(sample_gray).shape == sample_gray.shape
        assert P.laplacian(sample_gray).shape == sample_gray.shape

    def test_sharpen_shape(self, sample_scene):
        assert P.sharpen(sample_scene).shape == sample_scene.shape


class TestEnhance:
    def test_brightness_shift(self, sample_scene):
        brighter = P.adjust_brightness(sample_scene, 50)
        assert brighter.mean() > sample_scene.mean()

    def test_brightness_clamps(self):
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        assert P.adjust_brightness(black, -100).max() == 0

    def test_contrast(self, sample_scene):
        high = P.adjust_contrast(sample_scene, 2.0)
        assert high.std() > sample_scene.std()

    def test_invalid_delta(self, sample_scene):
        with pytest.raises(ProcessingError):
            P.adjust_brightness(sample_scene, 9999)


class TestNoise:
    def test_gaussian_noise_changes_pixels(self, sample_scene):
        noisy = P.add_noise(sample_scene, "gaussian", 0.05)
        assert noisy.dtype == np.uint8
        diff = np.abs(noisy.astype(int) - sample_scene.astype(int))
        assert diff.mean() > 0.5 and diff.mean() < 20

    def test_salt_pepper_dtype(self, sample_scene):
        noisy = P.add_noise(sample_scene, "salt_pepper", 0.05)
        assert noisy.dtype == np.uint8

    def test_unknown_noise_type(self, sample_scene):
        with pytest.raises(ProcessingError):
            P.add_noise(sample_scene, "flicker")

    def test_remove_noise_methods(self, sample_scene):
        noisy = P.add_noise(sample_scene, "salt_pepper", 0.05)
        for method in ("median", "bilateral", "gaussian"):
            assert P.remove_noise(noisy, method).shape == sample_scene.shape

    def test_unknown_denoise_method(self, sample_scene):
        with pytest.raises(ProcessingError):
            P.remove_noise(sample_scene, "magic")
