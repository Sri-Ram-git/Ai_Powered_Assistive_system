"""Tests for speech variety (repetitive-phrase reduction)."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

from src.audio.variety import SpeechVariety  # noqa: E402


def test_repeat_cue_produces_variants():
    sv = SpeechVariety()
    cue = "Person left, about 5 metres"
    variants = {sv.render(cue) for _ in range(3)}
    assert len(variants) >= 2
    assert "5 metres" in next(iter(variants))


def test_variants_preserve_meaning():
    sv = SpeechVariety()
    cue = "Person left, about 5 metres"
    for _ in range(3):
        out = sv.render(cue)
        assert "Person" in out and "5 metres" in out


def test_different_cues_not_mixed():
    sv = SpeechVariety()
    a = sv.render("Person left, about 5 metres")
    b = sv.render("Car right, about 3 metres")
    assert a != b


def test_empty_text_unchanged():
    sv = SpeechVariety()
    assert sv.render("") == ""
    assert sv.render("   ") == ""


def test_plain_text_unchanged():
    sv = SpeechVariety()
    out = sv.render("Stop, obstacle in front!")
    assert out == "Stop, obstacle in front!"