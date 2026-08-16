"""Command definitions for the speech interface.

The set is intentionally small and deterministic so a lightweight STT
backend (keyword matching or a tiny local model) can recognise every
command without an LLM.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence


class Command(Enum):
    """Every supported voice command."""

    READ_TEXT = "read_text"
    WHAT_DO_YOU_SEE = "what_do_you_see"
    DESCRIBE_SCENE = "describe_scene"
    REPEAT = "repeat"
    STOP_SPEAKING = "stop_speaking"
    START_OCR = "start_ocr"
    STOP_OCR = "stop_ocr"
    ENABLE_NAVIGATION = "enable_navigation"
    DISABLE_NAVIGATION = "disable_navigation"
    HELP = "help"


@dataclass(frozen=True)
class CommandSpec:
    """One command: its id, user phrases, and a handler label."""

    command: Command
    phrases: Sequence[str]     # spoken alternatives, lower-case
    description: str


COMMAND_SPECS: List[CommandSpec] = [
    CommandSpec(Command.READ_TEXT, [
        "read the text", "read text", "read the sign",
    ], "Read the most recently detected text aloud"),
    CommandSpec(Command.WHAT_DO_YOU_SEE, [
        "what do you see", "what do you see now", "what's in front of me",
        "what is in front of me", "what can you see",
    ], "Describe the objects currently detected"),
    CommandSpec(Command.DESCRIBE_SCENE, [
        "describe the scene", "describe what you see",
    ], "Give a higher-level description of the scene"),
    CommandSpec(Command.REPEAT, [
        "repeat", "say that again", "repeat that",
    ], "Repeat the last spoken message"),
    CommandSpec(Command.STOP_SPEAKING, [
        "stop speaking", "stop talking", "shut up", "be quiet",
    ], "Silence speech output"),
    CommandSpec(Command.START_OCR, [
        "start ocr", "enable text recognition", "turn on text recognition",
    ], "Enable OCR / text reading"),
    CommandSpec(Command.STOP_OCR, [
        "stop ocr", "disable text recognition", "turn off text recognition",
    ], "Disable OCR / text reading"),
    CommandSpec(Command.ENABLE_NAVIGATION, [
        "enable navigation", "turn on navigation",
    ], "Enable navigation guidance"),
    CommandSpec(Command.DISABLE_NAVIGATION, [
        "disable navigation", "turn off navigation",
    ], "Disable navigation guidance"),
    CommandSpec(Command.HELP, [
        "help", "what can you do", "help menu",
    ], "List the available commands"),
]


class CommandRegistry:
    """Holds every command spec and answers lookups by id/phrase."""

    def __init__(self, specs: Sequence[CommandSpec] = COMMAND_SPECS) -> None:
        self._specs = list(specs)
        self._by_id: Dict[Command, CommandSpec] = {
            s.command: s for s in self._specs
        }
        self._phrases: Dict[str, Command] = {}
        for s in self._specs:
            for phrase in s.phrases:
                self._phrases[phrase] = s.command

    @property
    def specs(self) -> List[CommandSpec]:
        return list(self._specs)

    def get(self, command: Command) -> CommandSpec:
        return self._by_id[command]

    def lookup_phrase(self, text: str):
        """Return the Command whose phrases appear in ``text`` (or None).

        Phrases are matched case-insensitively as substring matches so
        an STT transcript like "please read the text" still resolves.
        """
        lowered = (text or "").lower()
        for phrase, command in self._phrases.items():
            if phrase in lowered:
                return command
        return None

    def help_text(self) -> str:
        lines = [f"Say \"{s.phrases[0]}\" — {s.description}"
                 for s in self._specs]
        return "Voice commands. " + " ".join(lines)