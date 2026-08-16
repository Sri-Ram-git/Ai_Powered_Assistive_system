"""Rule-based decision engine.

Combines object detections, OCR text, and navigation cues into a
prioritised list of spoken phrases.  Applies a cooldown so the user is
not bombarded with the same message every frame.

Design: the engine is *stateless* in its core logic (`evaluate` is a
pure function of a frame summary); the `DecisionEngine` wrapper adds the
temporal state (cooldowns, last spoken text) on top.
"""
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from src.detection.detector import DetectionResult
from src.navigation.guidance import nearest_obstacle, scene_cues
from src.ocr.ocr_engine import OcrResult
from src.utils.logger import setup_logger

_logger = setup_logger("DecisionEngine")

# Priority ordering: 0 = highest.
_PRIORITIES = {
    "traffic light": 0,
    "stop sign": 1,
    "crosswalk": 2,
    "do not walk": 2,
    "text": 3,
    "person": 3,
    "obstacle": 5,
}

# Vehicle labels map to the vehicle priority tier.
_VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle",
                   "train", "airplane", "boat"}

# Trailing distance words stripped when building a cue identity.
_DISTANCE_SUFFIX = re.compile(
    r",?\s*(?:about \d+\.?\d* metres?|far away|very close|close|near)\s*$",
    re.IGNORECASE,
)


def cue_identity(text: str) -> str:
    """Stable identity for a cue, ignoring the numeric distance.

    "Person ahead, about 5 metres" and "Person ahead, about 6 metres"
    are the *same* message; jitter between frames must not cause the
    engine to re-speak it.  Bounding-box noise changes the distance by
    ±1m continuously, which was why the voice repeated itself.
    """
    lowered = _DISTANCE_SUFFIX.sub("", text or "").strip().lower()
    return lowered


@dataclass
class Decision:
    """A single spoken decision for the current frame."""

    text: str
    priority: int
    source: str  # 'detection' | 'ocr' | 'navigation'


@dataclass
class FrameSummary:
    """Snapshot of one processed frame, ready for decisioning."""

    detections: List[DetectionResult] = None
    ocr_items: List[OcrResult] = None
    frame_w: int = 640
    frame_h: int = 480
    read_ocr_text: bool = False
    max_ocr_chars: int = 80


def evaluate(summary: FrameSummary) -> List[Decision]:
    """Turn a frame summary into prioritised decisions.

    Args:
        summary: Detections, OCR results, and frame geometry.

    Returns:
        Decisions sorted by priority (0 = highest first).
    """
    detections = summary.detections or []
    ocr_items = summary.ocr_items or []
    frame_w = summary.frame_w or 1
    frame_h = summary.frame_h or 480

    cues = scene_cues(detections, ocr_items, frame_w, frame_h)
    obstacle = nearest_obstacle(detections, frame_w)

    decisions: List[Decision] = []

    for cue in cues:
        priority = _priority_for_cue(cue)
        decisions.append(Decision(
            text=cue,
            priority=priority,
            source=_source_for_cue(cue),
        ))

    if obstacle is not None and obstacle.category == "obstacle":
        decisions.append(Decision(
            text=f"Obstacle ahead",
            priority=_PRIORITIES["obstacle"],
            source="detection",
        ))

    if summary.read_ocr_text and ocr_items:
        text = " ".join(r.text for r in ocr_items)
        text = text.strip()
        if text:
            text = text[:summary.max_ocr_chars]
            decisions.append(Decision(
                text=f"Text says, {text}",
                priority=_PRIORITIES["text"],
                source="ocr",
            ))

    decisions.sort(key=lambda d: d.priority)
    return decisions


class DecisionEngine:
    """Stateful wrapper adding cooldown + de-duplication to `evaluate`."""

    def __init__(
        self,
        cooldown_seconds: float = 4.0,
        min_priority: int = 5,
        read_ocr_text: bool = False,
        max_ocr_chars: int = 80,
    ) -> None:
        """Configure the engine.

        Args:
            cooldown_seconds: Minimum gap between spoken phrases.
            min_priority: Only decisions with priority <= this are spoken.
            read_ocr_text: Speak the recognised OCR text aloud.
            max_ocr_chars: Cap on how many OCR characters are spoken.
        """
        self._cooldown = float(cooldown_seconds)
        self._min_priority = int(min_priority)
        self._read_ocr_text = bool(read_ocr_text)
        self._max_ocr_chars = int(max_ocr_chars)
        self._last_spoken: Optional[str] = None
        self._last_identity: Optional[str] = None
        self._last_time: float = 0.0

    def decide(
        self,
        summary: FrameSummary,
        now: Optional[float] = None,
        already_spoken: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        """Return the next phrase to speak, or None.

        Args:
            summary: Frame summary to evaluate.
            now: Current time in seconds (defaults to wall clock).
            already_spoken: Identities already announced this frame by
                another source (e.g. the tracking monitor).  The engine
                skips a decision whose identity was just announced, so
                one object is never narrated twice per frame.

        Returns:
            The highest-priority phrase if it is due (cooldown elapsed,
            identity changed, and not just spoken), otherwise None.
        """
        import time

        if now is None:
            now = time.monotonic()

        decisions = evaluate(_with_read_ocr(summary, self))
        if not decisions:
            self._last_spoken = None
            self._last_identity = None
            return None

        top = decisions[0]
        if top.priority > self._min_priority:
            return None

        identity = cue_identity(top.text)
        if already_spoken:
            for spoken in already_spoken:
                if cue_identity(spoken) == identity:
                    return None

        elapsed = now - self._last_time
        # First utterance is always allowed.
        if self._last_identity is None:
            self._last_spoken = top.text
            self._last_identity = identity
            self._last_time = now
            _logger.info("Decision: %s", top.text)
            return top.text

        # Global rate limit: nothing spoken within the cooldown window.
        if elapsed < self._cooldown:
            return None

        # The same *message* is not repeated (reset() re-enables it).
        # Identity comparison ignores distance jitter: "about 5 metres"
        # vs "about 6 metres" is still "Person ahead".
        if identity == self._last_identity:
            return None

        self._last_spoken = top.text
        self._last_identity = identity
        self._last_time = now
        _logger.info("Decision: %s", top.text)
        return top.text

    def reset(self) -> None:
        """Clear cooldown state (e.g. when switching scenes)."""
        self._last_spoken = None
        self._last_identity = None
        self._last_time = 0.0

    def set_read_ocr_text(self, enabled: bool) -> None:
        """Enable/disable speaking recognised text aloud."""
        self._read_ocr_text = bool(enabled)

    @property
    def read_ocr_text(self) -> bool:
        """Whether recognised text is spoken aloud."""
        return self._read_ocr_text


def _with_read_ocr(summary: FrameSummary, engine: "DecisionEngine") -> FrameSummary:
    """Return a summary with OCR-reading configured from the engine."""
    summary.read_ocr_text = engine._read_ocr_text
    summary.max_ocr_chars = engine._max_ocr_chars
    return summary


def _priority_for_cue(cue: str) -> int:
    lowered = cue.lower()
    for key, priority in _PRIORITIES.items():
        if key in lowered:
            return priority
    for label in _VEHICLE_LABELS:
        if label in lowered:
            return 4
    return 5


def _source_for_cue(cue: str) -> str:
    lowered = cue.lower()
    if any(k in lowered for k in ("crosswalk", "walk")):
        return "ocr"
    if any(k in lowered for k in ("sign", "light")):
        return "detection"
    return "navigation"
