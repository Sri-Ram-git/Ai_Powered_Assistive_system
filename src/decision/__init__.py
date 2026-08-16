"""Decision Engine module.

engine: DecisionEngine, evaluate, Decision, FrameSummary — turns
        detections + OCR into prioritised, cooldown-gated spoken phrases.
cue_identity: stable message identity that ignores distance jitter, used
        to dedupe announcements across the monitor and the engine.
"""
from src.decision.engine import (
    Decision,
    DecisionEngine,
    FrameSummary,
    cue_identity,
    evaluate,
)

__all__ = [
    "Decision",
    "DecisionEngine",
    "FrameSummary",
    "cue_identity",
    "evaluate",
]
