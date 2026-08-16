"""Distance and end-to-end assistive evaluation metrics.

Distance metrics reuse the calibration error model (MAE / RMSE /
relative error).  Assistive metrics evaluate the *behaviour* of the
decision/guidance system against an expected behaviour label:

    expectation   — what the system SHOULD say: a short phrase keyword
                    (e.g. "obstacle", "person", "traffic", "text").
    produced      — the phrase the system actually spoke.
    key           — keyword(s) that satisfy the expectation.

This is deliberately simple and deterministic; it measures whether the
right *kind* of guidance was produced, not exact wording.  The dataset
size limits statistical power — reports must state this.
"""
from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass
class AssistiveCase:
    """One end-to-end guidance evaluation case."""

    scenario: str
    expected_keyword: str          # e.g. "obstacle"
    produced_text: str             # what the system said
    produced_keywords: Sequence[str] = ()
    response_latency_ms: float = 0.0
    missed_object: bool = False
    false_warning: bool = False


def evaluate_assistive(cases: Sequence[AssistiveCase]) -> Dict[str, float]:
    """Compute end-to-end assistive metrics over a case list."""
    if not cases:
        return {
            "cases": 0, "correct_guidance": 0.0, "incorrect_guidance": 0.0,
            "missed_object": 0.0, "false_warning": 0.0,
            "response_latency_ms": 0.0, "accuracy": 0.0,
        }

    correct = sum(1 for c in cases if _is_correct(c))
    missed = sum(1 for c in cases if c.missed_object)
    false_warn = sum(1 for c in cases if c.false_warning)
    latencies = [c.response_latency_ms for c in cases]

    return {
        "cases": len(cases),
        "correct_guidance": correct,
        "incorrect_guidance": len(cases) - correct,
        "missed_object": missed,
        "false_warning": false_warn,
        "response_latency_ms": sum(latencies) / len(latencies),
        "accuracy": correct / len(cases),
    }


def _is_correct(case: AssistiveCase) -> bool:
    text = case.produced_text.lower()
    if case.expected_keyword in text:
        return True
    return any(k.lower() in text for k in case.produced_keywords)