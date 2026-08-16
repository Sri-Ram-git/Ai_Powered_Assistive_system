"""Object vocabulary package — 1000+ word list of nameable objects.

See :mod:`src.vocabulary.manifest` for the runtime API.
"""
from src.vocabulary.manifest import (  # noqa: F401
    DEFAULT_MANIFEST,
    DEFAULT_TEMPLATES,
    ObjectVocabulary,
    TIERS,
    VocabularyEntry,
    normalize_word,
    validate,
)

__all__ = [
    "DEFAULT_MANIFEST",
    "DEFAULT_TEMPLATES",
    "ObjectVocabulary",
    "TIERS",
    "VocabularyEntry",
    "normalize_word",
    "validate",
]
