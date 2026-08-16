"""Object vocabulary manifest — a 1000+ word list of things the assistive
system can name.

Every word in the manifest is a *real* object category that exists in a
labelled image dataset (LVIS v1 = 1203 categories, OpenImages = 601
classes, COCO-80), so "a word with labelled images" is literally true for
each entry.

Runtime use:
    vocab = ObjectVocabulary.load()
    vocab.tier_for("person")      -> "critical"
    vocab.display_word("aerosol_can")  -> "aerosol can"
    vocab.resolve("person")       -> VocabularyEntry

Tier meaning (drives how urgently a word is announced/speech tier):
    critical  immediate safety — spoken first without delay
    high      important — announced promptly
    normal    routine everyday objects
    low       background clutter — announced rarely or never

The manifest also carries a small *phrase template* pool per tier so the
same object can be announced in varied wording instead of repeating one
fixed string.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

try:
    import yaml
except Exception:  # pragma: no cover - environment without pyyaml
    yaml = None  # type: ignore[assignment]

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MANIFEST = os.path.join(
    PROJECT_ROOT, "data", "vocabulary", "object_vocabulary.yaml")

TIERS: Sequence[str] = ("critical", "high", "normal", "low")

# One canonical variant per tier; builders may extend these pools.
DEFAULT_TEMPLATES: Dict[str, List[str]] = {
    "critical": [
        "{word} in the way!",
        "Careful, {word} ahead!",
        "Stop, {word} in front!",
    ],
    "high": [
        "{word} ahead",
        "{word} on your path",
        "There's {word} up ahead",
    ],
    "normal": [
        "{word} seen",
        "{word} in view",
        "I can see {word}",
    ],
    "low": [
        "{word} nearby",
        "{word} in the background",
    ],
}


@dataclass
class VocabularyEntry:
    """A single word in the vocabulary with its labels/dataset ids."""

    word: str
    tier: str = "normal"
    category: str = "object"
    aliases: List[str] = field(default_factory=list)
    coco_id: Optional[int] = None
    lvis_id: Optional[int] = None
    openimages_id: Optional[str] = None

    @property
    def display_word(self) -> str:
        """Canonical display form (lowercase, underscores to spaces)."""
        return (self.word or "").strip().lower().replace("_", " ").strip()

    def normalize_key(self) -> str:
        return normalize_word(self.word)


def normalize_word(word: str) -> str:
    """Normalise a label to a canonical lookup key."""
    return " ".join((word or "").strip().lower().replace("_", " ").split())


class ObjectVocabulary:
    """Loaded vocabulary manifest with resolution + tier helpers."""

    def __init__(self, entries: Sequence[VocabularyEntry]) -> None:
        self._entries: Dict[str, VocabularyEntry] = {}
        for entry in entries:
            self._entries[entry.normalize_key()] = entry
        self._alias_map: Dict[str, str] = {}
        for entry in entries:
            for alias in entry.aliases:
                self._alias_map[normalize_word(alias)] = entry.normalize_key()

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ObjectVocabulary":
        """Load the manifest from YAML (committed copy or override path)."""
        path = path or DEFAULT_MANIFEST
        if yaml is None:
            raise RuntimeError("PyYAML is required to load the vocabulary")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        entries = [VocabularyEntry(**{k: v for k, v in e.items() if v is not None})
                   for e in data.get("words", [])]
        return cls(entries)

    @property
    def words(self) -> List[str]:
        """All canonical words in the manifest."""
        return sorted(self._entries.keys())

    @property
    def size(self) -> int:
        return len(self._entries)

    def resolve(self, word: str) -> Optional[VocabularyEntry]:
        """Resolve a raw label to its vocabulary entry (or None)."""
        key = normalize_word(word)
        if key in self._entries:
            return self._entries[key]
        target = self._alias_map.get(key)
        if target:
            return self._entries[target]
        return None

    def is_known(self, word: str) -> bool:
        return self.resolve(word) is not None

    def tier_for(self, word: str) -> Optional[str]:
        """Announcement tier for a label, or None when unknown."""
        entry = self.resolve(word)
        return entry.tier if entry else None

    def display_word(self, word: str) -> str:
        """Canonical display word for a label (pass-through when unknown)."""
        entry = self.resolve(word)
        return entry.display_word if entry else (word or "").strip()

    def category_for(self, word: str) -> Optional[str]:
        entry = self.resolve(word)
        return entry.category if entry else None

    def template(self, tier: str, word: str, rng: Optional[random.Random] = None) -> str:
        """A varied phrase template for a word at a tier."""
        entry = self.resolve(word)
        display = entry.display_word if entry else (word or "").strip()
        pool = list(DEFAULT_TEMPLATES.get(tier, DEFAULT_TEMPLATES["normal"]))
        chosen = (rng or random).choice(pool)
        return chosen.format(word=display)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {t: 0 for t in TIERS}
        for entry in self._entries.values():
            out[entry.tier] = out.get(entry.tier, 0) + 1
        return out


def validate(entries: Sequence[VocabularyEntry],
             min_words: int = 1000) -> List[str]:
    """Structural checks. Returns a list of problems (empty = valid)."""
    problems: List[str] = []
    if len(entries) < min_words:
        problems.append(
            f"manifest has {len(entries)} words, need >= {min_words}")
    keys: Set[str] = set()
    for entry in entries:
        key = entry.normalize_key()
        if not key:
            problems.append(f"empty word: {entry!r}")
            continue
        if key in keys:
            problems.append(f"duplicate word: {key}")
        keys.add(key)
        if entry.tier not in TIERS:
            problems.append(f"bad tier {entry.tier!r} for {key}")
    return problems
