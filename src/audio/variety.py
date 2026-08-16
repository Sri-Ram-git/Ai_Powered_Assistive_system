"""Speech variety — reduce repetitive phrasing in spoken announcements.

The engine produces cues like "Person left, about 5 metres".  When the
same cue repeats (dedupe cooldown passes), the SpeechVariety re-renders
it using synonym variants so consecutive announcements sound different
instead of a fixed loop:

    "Person left, about 5 metres"
    "Person on your left, around 5 metres"
    "Person to your left, roughly 5 metres"

Words not covered by the synonym map are left untouched, so the meaning
never changes — only the wording.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

_DIRECTION_SYNONYMS: Dict[str, List[str]] = {
    "left": ["left", "on your left", "to your left"],
    "right": ["right", "on your right", "to your right"],
    "ahead": ["ahead", "in front", "straight ahead", "in front of you"],
    "front": ["ahead", "in front", "straight ahead", "in front of you"],
    "behind": ["behind", "behind you"],
}

_DISTANCE_RE = re.compile(
    r"\b(about|around|roughly|approximately|approx)\s+"
    r"(\d+(?:\.\d+)?)\s+metres?\b", re.IGNORECASE)
_DISTANCE_VARIANTS = ["about", "around", "roughly"]

_DIRECTION_RE = re.compile(
    r"\b(on your left|to your left|on your right|to your right|"
    r"straight ahead|in front of you|in front|behind you|"
    r"left|right|ahead|behind|front)\b", re.IGNORECASE)


class SpeechVariety:
    """Rotates synonym variants per cue so repeats are not word-identical."""

    def __init__(self, variants_per_phrase: int = 3) -> None:
        self._variants_per_phrase = max(1, variants_per_phrase)
        self._index: Dict[str, int] = {}
        self._last_spoken: Optional[str] = None

    def render(self, text: str, direction: Optional[str] = None) -> str:
        """Return a (possibly varied) rendering of the cue text."""
        base = (text or "").strip()
        if not base:
            return base

        # Rotate the distance wording ("about 5 metres" -> "around 5 metres").
        def _dist(match: re.Match) -> str:
            value = match.group(2)
            variant = _DISTANCE_VARIANTS[
                self._idx_for(base, key="dist") % len(_DISTANCE_VARIANTS)]
            return f"{variant} {value} metres"

        varied = _DISTANCE_RE.sub(_dist, base)

        # Rotate a direction word/phrase if present.
        direction_variants = _DIRECTION_SYNONYMS.get(
            (direction or "").lower())
        if direction_variants:
            key = direction.lower()
            chosen = direction_variants[
                self._idx_for(base, key="dir:" + key) %
                len(direction_variants)]
            varied = re.sub(
                r"\b(on your left|to your left|on your right|"
                r"to your right|straight ahead|in front of you|"
                r"in front|behind you|left|right|ahead|behind|front)\b",
                chosen, varied, count=1, flags=re.IGNORECASE)
        return varied

    def _idx_for(self, base: str, key: str = "") -> int:
        k = key or base
        self._index[k] = self._index.get(k, 0) + 1
        return self._index[k]

    def spoke(self, text: str) -> None:
        self._last_spoken = text