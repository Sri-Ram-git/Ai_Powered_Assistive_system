"""Unit tests for image_utils (hardware-free)."""
from pathlib import Path

import numpy as np
import pytest

from src.image_fundamentals import image_utils as U
from src.utils.exceptions import ImageError


class TestIO:
    def test_read_save_roundtrip(self, scene_file):
        image = U.read_image(scene_file)
        assert image.shape == (50, 60, 3)
        out = scene_file.replace(".png", "_copy.png")
        path = U.save_image(image, out)
        assert Path(path).exists()
        assert U.read_image(out).shape == image.shape

    def test_read_missing_file(self):
        with pytest.raises(ImageError):
            U.read_image("does_not_exist.png")

    def test_save_empty_image(self):
        with pytest.raises(ImageError):
            U.save_image(np.array([]), "empty.png")

    def test_read_grayscale(self, scene_file):
        gray = U.read_image(scene_file, grayscale=True)
        assert gray.ndim == 2


class TestTransforms:
    def test_resize_dims(self, sample_scene):
        out = U.resize(sample_scene, width=80, height=60)
        assert out.shape == (60, 80, 3)

    def test_resize_scale(self, sample_scene):
        out = U.resize(sample_scene, scale=0.5)
        assert out.shape[0] == sample_scene.shape[0] // 2

    def test_resize_requires_dims(self, sample_scene):
        with pytest.raises(ImageError):
            U.resize(sample_scene)

    def test_crop(self, sample_scene):
        out = U.crop(sample_scene, 10, 10, 50, 50)
        assert out.shape == (50, 50, 3)

    def test_crop_out_of_bounds(self, sample_scene):
        with pytest.raises(ImageError):
            U.crop(sample_scene, 100, 100, 1000, 10)

    def test_flip_horizontal(self, sample_scene):
        flipped = U.flip(sample_scene, 1)
        assert np.array_equal(flipped[:, ::-1], sample_scene)

    def test_flip_invalid_code(self, sample_scene):
        with pytest.raises(ImageError):
            U.flip(sample_scene, 5)

    def test_rotate_shape(self, sample_scene):
        assert U.rotate(sample_scene, 30).shape == sample_scene.shape


class TestColour:
    def test_grayscale_channels(self, sample_scene):
        assert U.to_grayscale(sample_scene).ndim == 2

    def test_hsv_roundtrip(self, sample_scene):
        hsv = U.to_hsv(sample_scene)
        assert hsv.shape == sample_scene.shape
        bgr = U.to_bgr_from_hsv(hsv)
        # Round-trip is approximate due to integer conversion
        assert np.abs(bgr.astype(int) - sample_scene.astype(int)).mean() < 5

    def test_rgb_to_bgr(self, sample_scene):
        rgb = U.to_rgb(sample_scene)
        bgr = U.to_bgr(rgb)
        assert np.array_equal(bgr, sample_scene)


class TestInspection:
    def test_image_info(self, sample_scene):
        info = U.image_info(sample_scene)
        assert info["width"] == 160
        assert info["height"] == 120
        assert info["channels"] == 3
        assert info["pixel_count"] == 120 * 160

    def test_pixel_value(self, sample_scene):
        # Red square at (10,10)-(60,60)
        assert U.pixel_value(sample_scene, 30, 30) == (0, 0, 255)

    def test_pixel_value_out_of_bounds(self, sample_scene):
        with pytest.raises(ImageError):
            U.pixel_value(sample_scene, 999, 999)

    def test_image_stats_shape(self, sample_scene):
        stats = U.image_stats(sample_scene)
        assert stats["mean"].shape == (3,)
        assert stats["min"].shape == (3,)

    def test_histogram(self, sample_gray):
        hist, centers = U.histogram(sample_gray)
        assert hist.shape == (256,)
        assert centers.shape == (256,)

    def test_histogram_image(self, sample_scene):
        img = U.histogram_image(sample_scene)
        assert img.shape == (320, 420)
