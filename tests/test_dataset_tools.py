"""Tests for the vocabulary dataset downloader helpers + classifier."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "scripts")))

from scripts.vocabulary.build_vocabulary import classify  # noqa: E402
from scripts.vocabulary.download_labeled_dataset import (  # noqa: E402
    _write_yolo_label,
    lvis_bbox_to_xyxy,
)


def test_lvis_bbox_conversion():
    assert lvis_bbox_to_xyxy([120, 40, 200, 300]) == [120, 40, 320, 340]
    assert lvis_bbox_to_xyxy([0, 0, 100, 100]) == [0, 0, 100, 100]


def test_write_yolo_label(tmp_path):
    dest = str(tmp_path / "img.txt")
    _write_yolo_label(dest, 2, [("img", [120, 40, 320, 340])], 640, 480)
    line = open(dest, encoding="utf-8").read().strip()
    cls, cx, cy, w, h = line.split()
    assert int(cls) == 2
    assert abs(float(cx) - 0.34375) < 1e-6
    assert abs(float(cy) - 0.395833) < 1e-6
    assert abs(float(w) - 0.3125) < 1e-6
    assert abs(float(h) - 0.625) < 1e-6


def test_write_yolo_label_clamps_out_of_bounds(tmp_path):
    dest = str(tmp_path / "oob.txt")
    _write_yolo_label(dest, 0, [("img", [-50, -20, 9999, 9999])], 640, 480)
    content = open(dest, encoding="utf-8").read().strip()
    assert content  # clamped box still emitted, within [0, 1]


def test_write_yolo_label_skips_zero_box(tmp_path):
    dest = str(tmp_path / "zero.txt")
    _write_yolo_label(dest, 0, [("img", [50, 50, 50, 50])], 640, 480)
    assert open(dest, encoding="utf-8").read() == ""


def test_write_yolo_label_multiple_classes_one_file(tmp_path):
    """All boxes for an image (possibly different classes) share one file."""
    dest = str(tmp_path / "multi.txt")
    _write_yolo_label(
        dest, 0,
        [("img", [0, 0, 320, 480]), ("img", [320, 0, 640, 480])],
        640, 480,
        classes=[0, 3])
    lines = [l.split() for l in open(dest, encoding="utf-8").read().strip()
             .splitlines()]
    assert [int(l[0]) for l in lines] == [0, 3]
    assert len(lines) == 2


def test_classify_tiers():
    assert classify("person") == ("critical", "person")
    assert classify("stop sign")[0] == "critical"
    assert classify("traffic light")[0] == "critical"
    assert classify("dog")[0] == "high"
    assert classify("aerosol_can")[0] == "normal"
    assert classify("painting")[0] == "low"
    assert classify("television")[0] == "normal"
    assert classify("television")[1] == "electronics"


def test_classify_unknown():
    tier, category = classify("gizmo_thing")
    assert tier == "normal"
    assert category == "object"