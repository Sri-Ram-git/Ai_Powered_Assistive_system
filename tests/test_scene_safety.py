"""Tests for SceneContext and the SafetyEngine (deterministic, no models)."""
import pytest

from src.safety import RiskLevel, SafetyEngine
from src.tracking.tracker import TrackedObject
from src.vision.scene_context import (
    SceneContext,
    SceneObject,
    build_scene_context,
)


def _track(label, box, conf=0.9, track_id=0):
    x, y, w, h = box
    return TrackedObject(track_id=track_id, label=label, box=box,
                         confidence=conf, center=(x + w / 2, y + h / 2))


class TestSceneContext:
    def test_build_from_tracks(self):
        tracks = [
            _track("chair", (300, 300, 50, 60), track_id=1),
            _track("person", (100, 200, 60, 150), track_id=2),
        ]
        ctx = build_scene_context(
            tracks=tracks,
            ocr_text=["EXIT"],
            frame_w=640,
            frame_h=480,
        )
        assert len(ctx.objects) == 2
        assert ctx.text == ["EXIT"]
        assert ctx.frame_w == 640
        assert all(o.direction in ("left", "ahead", "right")
                   for o in ctx.objects)
        assert all(o.distance_m is not None for o in ctx.objects)

    def test_nearest_by_distance(self):
        ctx = SceneContext(objects=[
            SceneObject("person", 0.9, (0, 0, 10, 50), "ahead", 5.0),
            SceneObject("chair", 0.8, (0, 0, 10, 50), "ahead", 1.0),
        ])
        assert ctx.nearest.label == "chair"

    def test_nearest_falls_back_to_area(self):
        ctx = SceneContext(objects=[
            SceneObject("person", 0.9, (0, 0, 20, 200), "ahead", None),
            SceneObject("chair", 0.8, (0, 0, 10, 50), "ahead", None),
        ])
        assert ctx.nearest.label == "person"

    def test_hazards(self):
        ctx = SceneContext(objects=[
            SceneObject("chair", 0.8, (0, 0, 10, 50), "ahead", 0.8,
                        category="obstacle"),
            SceneObject("person", 0.9, (0, 0, 10, 50), "ahead", 2.0,
                        category="person"),
        ])
        assert len(ctx.hazards) == 1
        assert ctx.hazards[0].label == "chair"

    def test_to_dict(self):
        ctx = SceneContext(objects=[
            SceneObject("person", 0.9, (0, 0, 10, 50), "ahead", 2.0),
        ])
        d = ctx.to_dict()
        assert d["objects"][0]["label"] == "person"
        assert d["objects"][0]["distance_m"] == 2.0


class TestSafetyEngine:
    def test_no_hazards_is_none(self):
        ctx = SceneContext(objects=[
            SceneObject("person", 0.9, (0, 0, 10, 50), "ahead", 5.0,
                        category="person"),
        ])
        assessment = SafetyEngine().assess(ctx)
        assert assessment.level == RiskLevel.NONE
        assert not assessment.urgent

    def test_near_obstacle_is_high(self):
        ctx = SceneContext(objects=[
            SceneObject("chair", 0.8, (0, 0, 10, 50), "ahead", 0.8,
                        category="obstacle"),
        ])
        assessment = SafetyEngine().assess(ctx)
        assert assessment.level == RiskLevel.HIGH
        assert assessment.urgent
        assert assessment.hazards[0].hazard_type.name == \
            "IMMEDIATE_OBSTACLE"

    def test_near_vehicle_is_collision_risk(self):
        ctx = SceneContext(objects=[
            SceneObject("car", 0.9, (0, 0, 100, 50), "ahead", 2.0,
                        category="vehicle"),
        ])
        assessment = SafetyEngine().assess(ctx)
        assert assessment.level == RiskLevel.HIGH
        assert assessment.hazards[0].hazard_type.name == "COLLISION_RISK"

    def test_medium_proximity(self):
        ctx = SceneContext(objects=[
            SceneObject("chair", 0.8, (0, 0, 10, 50), "ahead", 2.0,
                        category="obstacle"),
        ])
        assessment = SafetyEngine().assess(ctx)
        assert assessment.level == RiskLevel.MEDIUM
        assert not assessment.urgent

    def test_text_crosswalk(self):
        ctx = SceneContext(text=["WALK"])
        assessment = SafetyEngine().assess(ctx)
        assert assessment.level == RiskLevel.MEDIUM
        assert assessment.hazards[0].hazard_type.name == "CROSSWALK"

    def test_stop_sign_text_is_high(self):
        ctx = SceneContext(text=["do not walk"])
        assessment = SafetyEngine().assess(ctx)
        assert assessment.level == RiskLevel.HIGH

    def test_to_dict(self):
        ctx = SceneContext(text=["WALK"])
        d = SafetyEngine().assess(ctx).to_dict()
        assert d["level"] == "medium"
        assert d["hazards"][0]["type"] == "crosswalk"