"""Unit tests for navigation guidance (hardware-free)."""
import numpy as np
import pytest

from src.detection.detector import DetectionResult
from src.navigation.guidance import (
    direction_of,
    distance_estimate,
    nearest_obstacle,
    reference_height,
    scene_cues,
)
from src.ocr.ocr_engine import OcrResult


def _det(label, box, conf=0.9):
    return DetectionResult(label=label, confidence=conf, box=box)


class TestDirection:
    def test_left(self):
        assert direction_of((0, 0, 50, 100), 300) == "left"

    def test_center(self):
        assert direction_of((100, 0, 100, 100), 300) == "ahead"

    def test_right(self):
        assert direction_of((210, 0, 50, 100), 300) == "right"

    def test_zero_width_safe(self):
        assert direction_of((0, 0, 10, 10), 0) == "ahead"


class TestDistance:
    def test_closer_box_means_smaller_distance(self):
        far = distance_estimate((0, 0, 50, 50), 480)
        near = distance_estimate((0, 0, 50, 200), 480)
        assert near < far

    def test_minimum_clamp(self):
        d = distance_estimate((0, 0, 100, 1000), 480)
        assert d >= 0.2

    def test_infinite_on_zero_height(self):
        assert distance_estimate((0, 0, 10, 0), 480) == float("inf")

    def test_vfov_changes_distance(self):
        # A wider FOV -> shorter focal length -> closer estimate.
        narrow = distance_estimate((0, 0, 100, 200), 480, 1.7, vfov_deg=45)
        wide = distance_estimate((0, 0, 100, 200), 480, 1.7, vfov_deg=60)
        assert wide < narrow

    def test_animal_reference_heights(self):
        # New reference heights cover animals (person > dog > cat).
        assert reference_height("person") > reference_height("dog")
        assert reference_height("dog") > reference_height("cat")
        assert reference_height("laptop") < reference_height("person")


class TestNearestObstacle:
    def test_largest_box_wins(self):
        small = _det("person", (0, 0, 10, 10))
        big = _det("car", (0, 0, 100, 200))
        assert nearest_obstacle([small, big], 640) == big

    def test_none_when_empty(self):
        assert nearest_obstacle([], 640) is None


class TestSceneCues:
    def test_person_cue(self):
        person = _det("person", (120, 0, 60, 200))
        cues = scene_cues([person], [], 360, 480)
        assert any("Person ahead" in c for c in cues)

    def test_vehicle_cue(self):
        car = _det("car", (0, 0, 40, 80))
        cues = scene_cues([car], [], 360, 480)
        assert any("Car left" in c for c in cues)

    def test_traffic_signal_priority(self):
        light = _det("traffic light", (160, 0, 30, 60))
        cues = scene_cues([light], [], 480, 480)
        assert any("Traffic light ahead" in c for c in cues)

    def test_crosswalk_from_ocr(self):
        ocr = [OcrResult(text="CROSSWALK", confidence=0.9,
                         box=(0, 0, 100, 20))]
        cues = scene_cues([], ocr, 640, 480)
        assert any("Crosswalk sign ahead" in c for c in cues)

    def test_phone_cue_with_distance(self):
        # A cell phone (an "object" category) must produce a spoken cue
        # with a distance — previously it was detected but never voiced.
        phone = _det("cell phone", (200, 100, 60, 120))
        cues = scene_cues([phone], [], 640, 480)
        assert any("Cell phone ahead" in c for c in cues)
        assert any("metres" in c for c in cues)

    def test_no_cues_for_empty(self):
        assert scene_cues([], [], 640, 480) == []
