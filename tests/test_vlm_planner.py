"""Tests for the optional VLM layer and the response planner."""
import pytest

from src.response import (
    Response,
    ResponsePlanner,
    ResponsePriority,
)
from src.safety import RiskLevel
from src.vision.scene_context import SceneContext, SceneObject
from src.vision.vlm import DeterministicVLM, create_vlm


class TestDeterministicVLM:
    def test_describes_objects(self):
        ctx = SceneContext(objects=[
            SceneObject("chair", 0.8, (0, 0, 10, 50), "ahead", 0.9),
        ])
        text = DeterministicVLM().describe(ctx)
        assert "chair" in text
        assert "ahead" in text

    def test_describes_text(self):
        ctx = SceneContext(text=["EXIT"])
        text = DeterministicVLM().describe(ctx)
        assert "EXIT" in text

    def test_empty_scene(self):
        text = DeterministicVLM().describe(SceneContext())
        assert "do not see any objects" in text

    def test_question_appended(self):
        ctx = SceneContext(text=["EXIT"])
        text = DeterministicVLM().describe(ctx, question="What is around me?")
        assert "asked:" in text


class TestRemoteVLM:
    def test_remote_success(self):
        def client(ctx):
            return "You are in a room with an exit to your right."
        engine = create_vlm("remote", client=client)
        ctx = SceneContext(text=["EXIT"])
        result = engine.describe(ctx)
        assert "exit" in result.text.lower()
        assert result.backend == "remote"
        assert not result.fallback

    def test_remote_failure_falls_back_local(self):
        calls = []

        def client(ctx):
            calls.append(1)
            raise RuntimeError("network down")
        engine = create_vlm("remote", client=client)
        ctx = SceneContext(objects=[
            SceneObject("chair", 0.8, (0, 0, 10, 50), "ahead", 0.9),
        ])
        result = engine.describe(ctx)
        assert "chair" in result.text  # deterministic fallback produced it
        assert result.backend == "remote"  # backend label is remote path
        assert calls  # remote was actually attempted

    def test_remote_empty_response_falls_back(self):
        engine = create_vlm("remote", client=lambda ctx: "   ")
        ctx = SceneContext(text=["EXIT"])
        assert "EXIT" in engine.describe(ctx).text


class TestCreateVlm:
    def test_deterministic_default(self):
        assert isinstance(create_vlm().backend, DeterministicVLM)

    def test_remote_without_client_raises(self):
        with pytest.raises(ValueError):
            create_vlm("remote")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            create_vlm("bogus")


class TestResponsePlanner:
    def test_picks_highest_priority(self):
        planner = ResponsePlanner()
        chosen = planner.plan([
            Response("chair ahead", ResponsePriority.INFO),
            Response("STOP", ResponsePriority.URGENT_SAFETY, urgent=True),
        ])
        assert chosen is not None
        assert chosen.text == "STOP"

    def test_dedupes_same_message(self):
        planner = ResponsePlanner()
        first = planner.plan([Response("chair ahead", ResponsePriority.INFO)])
        assert first is not None
        second = planner.plan([Response("chair ahead", ResponsePriority.INFO)])
        assert second is None  # cooldown + dedup

    def test_urgent_bypasses_cooldown(self):
        planner = ResponsePlanner()
        first = planner.plan([Response("chair ahead", ResponsePriority.INFO)])
        assert first is not None
        risk = type("Risk", (), {"urgent": True})()
        second = planner.plan(
            [Response("STOP now", ResponsePriority.URGENT_SAFETY,
                      urgent=True)], risk=risk)
        assert second is not None

    def test_priority_filter(self):
        planner = ResponsePlanner()
        chosen = planner.plan([
            Response("boring detail", ResponsePriority.QUIET),
        ])
        assert chosen is None

    def test_identity_ignores_numbers(self):
        a = Response("chair about 2 metres", ResponsePriority.INFO)
        b = Response("chair about 5 metres", ResponsePriority.INFO)
        assert a.identity() == b.identity()

    def test_reset_clears_history(self):
        planner = ResponsePlanner()
        planner.plan([Response("hi", ResponsePriority.INFO)])
        assert len(planner.history) == 1
        planner.reset()
        assert planner.history == []