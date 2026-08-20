"""Response Planner — decides WHAT to say, WHEN to say it, and HOW
urgently.

Every module (decision engine, tracking monitor, safety engine, STT, VLM)
may *propose* a response.  The planner is the single point of arbitration:

1. Priority — safety-related messages win.
2. Deduplication — the same message is never repeated verbatim.
3. Cooldown — a global minimum gap between spoken responses.

The existing DecisionEngine is preserved and evolved; the planner layers
arbitration on top of it (and on top of safety/VLM/STT proposals).
"""
import re
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional



class ResponsePriority(IntEnum):
    URGENT_SAFETY = 0
    SAFETY = 1
    COMMAND = 2
    NAVIGATION = 3
    INFO = 4
    QUIET = 5


@dataclass
class Response:
    """A single proposed response."""

    text: str
    priority: ResponsePriority = ResponsePriority.INFO
    source: str = "planner"         # which module proposed it
    urgent: bool = False

    def identity(self) -> str:
        """Stable identity ignoring numbers/urgency words (dedup key)."""
        lowered = re.sub(r"\b\d+(\.\d+)?\b", "#", (self.text or "").lower())
        return " ".join(lowered.split())


@dataclass
class PlannerConfig:
    """Tuning knobs for the response planner."""

    cooldown_seconds: float = 2.5
    min_priority: int = int(ResponsePriority.QUIET)  # skip worse than this
    dedupe: bool = True


class ResponsePlanner:
    """Arbitrates proposed responses into a single spoken message."""

    def __init__(self, config: Optional[PlannerConfig] = None) -> None:
        self._cfg = config or PlannerConfig()
        self._last_identity: Optional[str] = None
        self._last_time: float = 0.0
        self._history: List[str] = []

    def plan(
        self,
        proposals: List[Response],
        risk: Optional[object] = None,
        now: Optional[float] = None,
    ) -> Optional[Response]:
        """Pick the single best response from the proposals.

        Args:
            proposals: Candidate responses from any module.
            risk: Optional SafetyEngine RiskAssessment — urgent safety
                hazards override the cooldown.
            now: Current time (defaults to monotonic clock).

        Returns:
            The chosen Response, or None when nothing is due.
        """
        if now is None:
            now = time.monotonic()

        urgent = bool(risk is not None and getattr(risk, "urgent", False))

        # Rank by priority; keep only proposals worth saying.
        eligible = [p for p in proposals
                    if p.priority < self._cfg.min_priority]
        if not eligible:
            return None
        eligible.sort(key=lambda p: (p.priority, p.urgent))

        best = eligible[0]

        # Deduplication: never repeat the same message verbatim.
        if self._cfg.dedupe and best.identity() == self._last_identity:
            return None

        # Cooldown: safety/urgent responses bypass the gap.
        elapsed = now - self._last_time
        if not urgent and elapsed < self._cfg.cooldown_seconds:
            return None

        # First response is always allowed.
        if self._last_identity is None or urgent or \
                elapsed >= self._cfg.cooldown_seconds:
            self._last_identity = best.identity()
            self._last_time = now
            self._history.append(best.text)
            return best
        return None

    def reset(self) -> None:
        self._last_identity = None
        self._last_time = 0.0
        self._history.clear()

    @property
    def history(self) -> List[str]:
        return list(self._history)