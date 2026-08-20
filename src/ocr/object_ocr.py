"""Object-aware OCR: results, validation, variant selection, per-track
store, and the trigger policy.

This module has no I/O or threads of its own — the pipeline hosts the
worker and the UI reads the store.  Everything here is pure and testable.

Key ideas:

    * ``ObjectOcrResult`` — one finished OCR attempt on one object.
    * ``TrackOcrStore``   — per-track results with temporal voting, so a
      noisy one-frame read never replaces a stable result, and expiry so
      results of disappeared objects are dropped.
    * ``OcrTrigger``      — decides *when* an object deserves re-OCR
      (new / moved / stale / user), with a per-track cooldown.
    * ``validate_text``   — rejects garbage before it reaches the UI/TTS.
    * variant selection   — a few cheap preprocessings are tried and the
      best OCR result is kept (short-circuit on high confidence).
"""
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.ocr.ocr_engine import OcrResult
from src.ocr.preprocess import preprocess

# Candidate preprocessings evaluated per accepted ROI (ordered, cheapest
# first).  The worker stops early when one yields high-confidence text.
DEFAULT_VARIANTS: List[str] = ["none", "contrast", "adaptive", "sharpen"]

_WS = re.compile(r"\s+")
_ALNUM = re.compile(r"[^A-Za-z0-9\s.,;:'\"!?()\-/&%$#@*+=<>\[\]{}]")
_REPEATED = re.compile(r"^(.)\1{3,}$", re.IGNORECASE)
_LONG_RUN = re.compile(r"[^\w\s]{6,}")
_VOWELS = "aeiouyAEIOUY"


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

@dataclass
class ObjectOcrResult:
    """One finished OCR attempt on one object (or a manual request)."""

    track_id: Optional[int] = None
    label: Optional[str] = None
    text: str = ""
    confidence: float = 0.0
    raw_text: str = ""
    roi_box: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2)
    variant: str = "none"
    scale: float = 1.0
    timestamp: float = 0.0
    latency_ms: float = 0.0
    status: str = "ok"      # ok | empty | no_text | timeout | error
    trigger: str = "new"    # new | moved | stale | user | manual
    source: str = "object"  # object | manual

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


@dataclass
class TrackOcrEntry:
    """Stable per-track OCR state for display + history."""

    track_id: int
    label: str
    text: str
    confidence: float
    raw_text: str = ""
    roi_box: Optional[Tuple[int, int, int, int]] = None
    variant: str = "none"
    timestamp: float = 0.0
    latency_ms: float = 0.0
    source: str = "object"
    stable: bool = False
    votes: Deque[str] = field(default_factory=deque)
    _pending: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Text validation
# ----------------------------------------------------------------------

def normalize_text(text: Optional[str]) -> str:
    """Collapse whitespace and strip (display form)."""
    if not text:
        return ""
    return _WS.sub(" ", text).strip()


def _token_plausible(token: str) -> bool:
    """Whether a single OCR token looks like a real word.

    Catches the "random junk" the OCR net emits on noisy/low-res ROIs:
    consonant-only soup (``KZ8XQP3``), vowel-less clusters (``TTWW4P``),
    and digit-heavy fragments.  Numbers alone are kept (prices, codes);
    a digit buried in the middle of a short word is treated as noise
    (``C0LA``), while trailing codes (``2024``, ``X1``) still pass.
    """
    if not token:
        return False
    if token.isdigit():
        return True  # numbers are meaningful (prices, codes)
    letters = [c for c in token if c.isalpha()]
    if not letters:
        return False
    lower = "".join(letters).lower()
    if not any(c in _VOWELS for c in lower):
        return False
    max_run = run = 0
    for c in lower:
        if c in _VOWELS:
            run = 0
        else:
            run += 1
            max_run = max(max_run, run)
    if max_run > 5:  # "strengths" = 4; anything longer is soup
        return False
    if len(letters) / len(token) < 0.5:
        return False
    if len(token) >= 4 and any(ch.isdigit() for ch in token[1:-1]):
        return False
    return True


def is_garbage(text: str, min_chars: int = 2) -> bool:
    """Whether an OCR string is unusable (empty/symbols/noise)."""
    clean = normalize_text(text)
    if len(clean) < min_chars:
        return True
    if _REPEATED.match(clean.replace(" ", "")):
        return True
    if _LONG_RUN.search(clean):
        return True
    alnum = re.sub(r"\s", "", clean)
    good = sum(1 for ch in alnum if ch.isalnum())
    if alnum and good / len(alnum) < 0.6:
        return True
    tokens = [t for t in clean.split() if any(c.isalnum() for c in t)]
    if tokens and not all(_token_plausible(t) for t in tokens):
        return True
    return False


def validate_text(text: str, min_chars: int = 2) -> Optional[str]:
    """Return a cleaned usable text, or None for garbage."""
    clean = normalize_text(text)
    if is_garbage(clean, min_chars=min_chars):
        return None
    return clean


# ----------------------------------------------------------------------
# Text quality tiers
# ----------------------------------------------------------------------

QUALITY_HIGH_CONF = 0.75
QUALITY_MIN_CONF = 0.45


def text_quality(text: str, confidence: float) -> str:
    """Classify a validated text as 'high' | 'medium' | 'low'.

    ``low`` text must never be shown or spoken: the worker reports it as
    ``status="unclear"`` so the UI can say "Text unclear" instead of
    confidently reading wrong characters.  ``medium`` is plausible but
    not certain (kept for display, never auto-spoken).
    """
    clean = normalize_text(text)
    if len(clean) < 2:
        return "low"
    if confidence >= QUALITY_HIGH_CONF:
        return "high"
    if confidence >= QUALITY_MIN_CONF:
        return "medium"
    return "low"


# ----------------------------------------------------------------------
# Variant selection
# ----------------------------------------------------------------------

def best_variant(
    items: Sequence[Tuple[str, List[OcrResult], float]],
) -> Tuple[str, List[OcrResult]]:
    """Pick the best OCR result among preprocessing variants.

    Args:
        items: (variant_name, ocr_results, latency_ms) candidates.

    Returns:
        (variant_name, ocr_results) of the winner.
    """
    best_variant_name = "none"
    best_items: List[OcrResult] = []
    best_score = -1.0
    for name, results, _latency in items:
        if not results:
            continue
        text = " ".join(r.text for r in results).strip()
        if not text:
            continue
        # Confidence weighted by text length (longer consistent lines
        # beat a short high-conf fragment).
        total_len = sum(len(r.text) for r in results) or 1
        conf = sum(r.confidence * len(r.text) for r in results) / total_len
        score = conf + (len(text) / 1000.0)
        if score > best_score:
            best_score = score
            best_variant_name = name
            best_items = results
    return best_variant_name, best_items


def run_variants(
    engine,
    image: np.ndarray,
    variants: Sequence[str],
    stop_confidence: float = 0.92,
    min_confidence: float = 0.0,
    rot_fallback: bool = True,
    fallback_variants: Optional[Sequence[str]] = None,
) -> Tuple[str, List[OcrResult], float]:
    """Run OCR over candidate preprocessings; return the best.

    The requested ``variants`` run first (fast path, low latency).  When
    none yield usable text, a ``fallback_variants`` chain is tried —
    these run only on failure, so the common path is unaffected:

    * ``rotate90`` / ``rotate270`` — book covers, spines, and screens
      often hold vertical or "inverted" text the detector only reads
      after rotation;
    * ``otsu`` / ``denoise`` / ``gray_norm`` — thin-stroke or low-contrast
      text (serifs like Times New Roman, noisy video) that survive better
      under auto-thresholding / mild denoising / brightness normalisation
      than under CLAHE or adaptive thresholding.

    Args:
        engine: Object with ``read_text(image) -> List[OcrResult]``.
        image: ROI image (BGR).
        variants: Preprocessing names to try, in order.
        stop_confidence: Skip remaining variants when a result already
            reaches this confidence.
        min_confidence: Drop candidate lines below this confidence.
        rot_fallback: Try the fallback chain when nothing usable found.
        fallback_variants: Names tried on total failure; defaults to
            the rotation + clean-text helpers above.

    Returns:
        (best_variant, best_results, total_latency_ms).
    """
    started = time.monotonic()
    candidates: List[Tuple[str, List[OcrResult], float]] = []
    names = list(variants or ["none"])
    if rot_fallback and not names:
        names = ["none"]
    for name in names:
        try:
            processed = preprocess(image, name)
            results = engine.read_text(processed)
        except Exception:
            results = []
        results = [r for r in results if r.confidence >= min_confidence]
        latency = (time.monotonic() - started) * 1000.0
        candidates.append((name, results, latency))
        if results:
            text = " ".join(r.text for r in results).strip()
            avg = sum(r.confidence for r in results) / len(results)
            if text and avg >= stop_confidence:
                break
    variant, items = best_variant(candidates)
    if rot_fallback and not items:
        fallback = list(fallback_variants or
                        ("rotate90", "rotate270", "otsu", "denoise",
                         "gray_norm"))
        for name in fallback:
            try:
                processed = preprocess(image, name)
                results = engine.read_text(processed)
            except Exception:
                results = []
            results = [r for r in results if r.confidence >= min_confidence]
            latency = (time.monotonic() - started) * 1000.0
            candidates.append((name, results, latency))
            if results:
                break
        variant, items = best_variant(candidates)
    total_latency = (time.monotonic() - started) * 1000.0
    return variant, items, total_latency


def combine_results(items: List[OcrResult]) -> Tuple[str, float]:
    """Join OcrResult lines into text + length-weighted confidence."""
    if not items:
        return "", 0.0
    text = " ".join(r.text for r in items).strip()
    total_len = sum(len(r.text) for r in items) or 1
    conf = sum(r.confidence * len(r.text) for r in items) / total_len
    return text, conf


# ----------------------------------------------------------------------
# Track OCR store (voting + expiry + history)
# ----------------------------------------------------------------------

@dataclass
class TrackOcrStore:
    """Per-track OCR results with temporal voting and expiry."""

    confirm_votes: int = 2
    history_max: int = 20

    _tracks: Dict[int, TrackOcrEntry] = field(default_factory=dict)
    _order: Deque[int] = field(default_factory=deque)

    def update(self, result: ObjectOcrResult) -> Optional[TrackOcrEntry]:
        """Adopt a finished OCR result into per-track voting.

        A new text is only adopted once it has been seen
        ``confirm_votes`` consecutive times, so transient OCR noise
        ("COC4 C0LA") never replaces the stable result ("COCA COLA").
        """
        if result.track_id is None:
            return None
        text = validate_text(result.text)
        entry = self._tracks.get(result.track_id)
        if entry is None:
            if text is None:
                return None
            entry = TrackOcrEntry(
                track_id=result.track_id,
                label=result.label or "",
                text=text,
                confidence=result.confidence,
                raw_text=normalize_text(result.raw_text) or text,
                roi_box=result.roi_box,
                variant=result.variant,
                timestamp=result.timestamp or time.time(),
                latency_ms=result.latency_ms,
                source=result.source,
                stable=True,
            )
            entry.votes.append(text)
            self._tracks[result.track_id] = entry
            self._touch(result.track_id)
            return entry

        entry.label = result.label or entry.label
        if text is None:
            return entry  # garbage read: keep the stable result
        if text == entry.text:
            entry.votes.append(text)
            while len(entry.votes) > self.confirm_votes * 3:
                entry.votes.popleft()
            return entry

        # New text: require confirm_votes consecutive identical reads.
        entry._pending.append(text)
        if len(entry._pending) >= self.confirm_votes and \
                all(t == text for t in entry._pending):
            entry.text = text
            entry.confidence = result.confidence
            entry.raw_text = normalize_text(result.raw_text) or text
            entry.roi_box = result.roi_box
            entry.variant = result.variant
            entry.timestamp = result.timestamp or time.time()
            entry.latency_ms = result.latency_ms
            entry.votes.append(text)
            entry._pending = []
            self._touch(result.track_id)
        else:
            # Update timestamp so the entry is not re-OCR'd for staleness,
            # but keep the stable text.
            entry.timestamp = result.timestamp or time.time()
        return entry

    def _touch(self, track_id: int) -> None:
        if track_id in self._order:
            self._order.remove(track_id)
        self._order.append(track_id)
        while len(self._order) > self.history_max:
            oldest = self._order.popleft()
            self._tracks.pop(oldest, None)

    def for_track(self, track_id: int) -> Optional[TrackOcrEntry]:
        return self._tracks.get(track_id)

    def latest(self) -> Optional[TrackOcrEntry]:
        """The most recently updated entry (regardless of track)."""
        best: Optional[TrackOcrEntry] = None
        for entry in self._tracks.values():
            if best is None or entry.timestamp > best.timestamp:
                best = entry
        return best

    def texts(self) -> List[Tuple[int, str, str, float]]:
        """All (track_id, label, text, confidence) with usable text."""
        out = []
        for entry in self._tracks.values():
            if entry.text:
                out.append((entry.track_id, entry.label, entry.text,
                            entry.confidence))
        out.sort(key=lambda t: t[3], reverse=True)
        return out

    def history(self) -> List[TrackOcrEntry]:
        """Entries newest-first (bounded by history_max)."""
        entries = list(self._tracks.values())
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries

    def expire(
        self,
        max_age: float,
        alive_track_ids: Optional[Iterable[int]] = None,
        now: Optional[float] = None,
    ) -> None:
        """Drop entries for objects no longer present or too old."""
        now = now if now is not None else time.time()
        alive = set(alive_track_ids or [])
        stale = []
        for track_id, entry in self._tracks.items():
            if track_id not in alive and now - entry.timestamp > max_age:
                stale.append(track_id)
        for track_id in stale:
            self._tracks.pop(track_id, None)
            if track_id in self._order:
                self._order.remove(track_id)

    def clear(self) -> None:
        self._tracks.clear()
        self._order.clear()


# ----------------------------------------------------------------------
# Trigger policy
# ----------------------------------------------------------------------

@dataclass
class OcrTrigger:
    """Decides when a tracked object deserves OCR."""

    cooldown_s: float = 3.0
    stale_after_s: float = 5.0
    move_px: int = 40

    _last: Dict[int, Dict] = field(default_factory=dict)

    def decide(
        self,
        track_id: int,
        label: str,
        box: Tuple[int, int, int, int],
        now: Optional[float] = None,
    ) -> Optional[str]:
        """Return a trigger reason ('new'|'moved'|'stale') or None.

        Args:
            track_id: Tracker identity.
            label: Object label (unused here; eligibility is upstream).
            box: Track box (x, y, w, h).
            now: Monotonic time.

        Returns:
            Trigger reason, or None when the object should not be OCR'd.
        """
        now = now if now is not None else time.monotonic()
        prev = self._last.get(track_id)
        if prev is None:
            self._last[track_id] = {
                "time": now, "box": box, "reason": "new",
            }
            return "new"

        elapsed = now - prev.get("time", 0.0)
        if elapsed < self.cooldown_s:
            return None

        px, py, pw, ph = box
        ox, oy, ow, oh = prev["box"]
        moved = (abs(px - ox) >= self.move_px or
                 abs(py - oy) >= self.move_px)
        if moved:
            self._last[track_id] = {"time": now, "box": box, "reason": "moved"}
            return "moved"

        if elapsed >= self.stale_after_s:
            self._last[track_id] = {"time": now, "box": box, "reason": "stale"}
            return "stale"

        return None

    def touched(self, track_id: int, now: Optional[float] = None) -> None:
        """Refresh the trigger clock (e.g. after a manual OCR)."""
        now = now if now is not None else time.monotonic()
        prev = self._last.get(track_id)
        if prev is not None:
            prev["time"] = now

    def drop(self, track_id: int) -> None:
        self._last.pop(track_id, None)

    def prune(self, alive: Iterable[int]) -> None:
        """Forget trigger state for tracks no longer present."""
        alive = set(alive)
        for track_id in list(self._last):
            if track_id not in alive:
                self._last.pop(track_id, None)

    def reset(self) -> None:
        self._last.clear()


def rank_targets(
    tracks: Sequence,
    rank_of,
    area_of=None,
) -> List:
    """Order eligible tracks by priority tier, then area, then confidence.

    Args:
        tracks: Sequence of track objects with ``label`` and
            ``confidence`` and (optionally) ``area``.
        rank_of: Callable(label) -> int priority rank.
        area_of: Optional callable(track) -> float area; defaults to
            the track's ``.area`` attribute.

    Returns:
        Tracks sorted best-first.
    """
    def _area(track) -> float:
        if area_of is not None:
            return float(area_of(track))
        return float(getattr(track, "area", 0.0))

    return sorted(
        tracks,
        key=lambda t: (rank_of(getattr(t, "label", None)),
                       _area(t),
                       float(getattr(t, "confidence", 0.0))),
        reverse=True,
    )