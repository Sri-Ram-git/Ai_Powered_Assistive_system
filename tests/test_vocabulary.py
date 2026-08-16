"""Tests for the object vocabulary manifest + runtime module."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "scripts")))

from src.vocabulary import (  # noqa: E402
    DEFAULT_MANIFEST,
    ObjectVocabulary,
    normalize_word,
    validate,
)


@pytest.fixture(scope="module")
def vocab() -> ObjectVocabulary:
    assert os.path.exists(DEFAULT_MANIFEST), "manifest must be built"
    return ObjectVocabulary.load()


def test_manifest_has_at_least_1000_words(vocab):
    assert vocab.size >= 1000, f"only {vocab.size} words"


def test_manifest_validation_clean():
    assert validate(ObjectVocabulary.load()._entries.values()) == []


def test_manifest_has_words_txt():
    words_txt = os.path.join(os.path.dirname(DEFAULT_MANIFEST), "words.txt")
    assert os.path.exists(words_txt)
    words = [l for l in open(words_txt, encoding="utf-8") if l.strip()]
    assert len(words) >= 1000


def test_normalize_word():
    assert normalize_word("aerosol_can") == "aerosol can"
    assert normalize_word("  Person ") == "person"
    assert normalize_word("Television") == "television"


@pytest.mark.parametrize("word,expected", [
    ("person", "critical"),
    ("car", "critical"),
    ("stop sign", "critical"),
    ("dog", "high"),
    ("chair", "high"),
    ("painting", "low"),
    ("aerosol can", "normal"),
    ("cell phone", "normal"),
])
def test_tier_for(vocab, word, expected):
    assert vocab.tier_for(word) == expected


def test_unknown_word(vocab):
    assert vocab.tier_for("zzz_unknown_thing") is None
    assert vocab.is_known("person")
    assert not vocab.is_known("zzz_unknown_thing")


def test_display_word(vocab):
    assert vocab.display_word("aerosol_can") == "aerosol can"
    assert vocab.display_word("Television") == "television"
    assert vocab.display_word("zzz_unknown_thing") == "zzz_unknown_thing"


def test_resolve_alias(vocab):
    # LVIS stores synonyms; "truck" is a synonym/name of many truck words.
    entry = vocab.resolve("person")
    assert entry is not None
    assert entry.tier == "critical"


def test_template_contains_word(vocab):
    phrase = vocab.template("high", "person")
    assert "person" in phrase


def test_counts_match_size(vocab):
    counts = vocab.counts()
    assert sum(counts.values()) == vocab.size


def test_every_entry_has_valid_tier(vocab):
    from src.vocabulary import TIERS
    for entry in vocab._entries.values():
        assert entry.tier in TIERS