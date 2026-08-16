"""Tests for the speech input module (STT, commands, command parser)."""
import pytest

from src.speech.command_parser import CommandParser
from src.speech.commands import Command, CommandRegistry
from src.speech.stt import KeywordSTT, create_stt


class TestCommandRegistry:
    def test_lookup_phrase(self):
        reg = CommandRegistry()
        assert reg.lookup_phrase("read the text") == Command.READ_TEXT
        assert reg.lookup_phrase("PLEASE READ THE TEXT NOW") == \
            Command.READ_TEXT
        assert reg.lookup_phrase("unknown command") is None

    def test_help_text_mentions_commands(self):
        reg = CommandRegistry()
        text = reg.help_text()
        assert "read the text" in text.lower()
        assert "what do you see" in text.lower()

    def test_get_by_id(self):
        reg = CommandRegistry()
        spec = reg.get(Command.HELP)
        assert spec.command == Command.HELP


class TestCommandParser:
    def test_parses_known_commands(self):
        parser = CommandParser()
        for phrase, expected in [
            ("read the text", Command.READ_TEXT),
            ("what do you see", Command.WHAT_DO_YOU_SEE),
            ("describe the scene", Command.DESCRIBE_SCENE),
            ("repeat", Command.REPEAT),
            ("stop speaking", Command.STOP_SPEAKING),
            ("start ocr", Command.START_OCR),
            ("stop ocr", Command.STOP_OCR),
            ("enable navigation", Command.ENABLE_NAVIGATION),
            ("disable navigation", Command.DISABLE_NAVIGATION),
            ("help", Command.HELP),
        ]:
            parsed = parser.parse(phrase)
            assert parsed.command == expected, phrase

    def test_unknown_returns_none(self):
        parsed = CommandParser().parse("tell me a joke")
        assert parsed.command is None
        assert parsed.confidence == 0.0

    def test_empty(self):
        assert CommandParser().parse("").command is None
        assert CommandParser().parse(None).command is None


class TestKeywordSTT:
    def test_parse(self):
        stt = KeywordSTT()
        assert stt.parse("stop speaking").command == Command.STOP_SPEAKING

    def test_transcribe_passthrough(self):
        stt = KeywordSTT()
        assert stt.transcribe("help") == "help"

    def test_create_stt_default(self):
        assert isinstance(create_stt(), KeywordSTT)

    def test_create_stt_unknown_backend(self):
        with pytest.raises(ValueError):
            create_stt("bogus")

    def test_help_text(self):
        assert "read the text" in KeywordSTT().help_text()