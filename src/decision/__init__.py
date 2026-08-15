"""Decision Engine module.

engine: DecisionEngine, evaluate, Decision, FrameSummary — turns
        detections + OCR into prioritised, cooldown-gated spoken phrases.
"""
from src.decision.engine import (
    Decision,
    DecisionEngine,
    FrameSummary,
    evaluate,
)

__all__ = [
    "Decision",
    "DecisionEngine",
    "FrameSummary",
    "evaluate",
]
