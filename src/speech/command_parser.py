"""Deterministic command parser.

Converts a raw text transcript into a ``ParsedCommand``.  Kept fully
deterministic (keyword matching against the command registry) so voice
control never depends on an LLM.  Unknown input yields a ``None``
command with the raw text retained for logging.
"""
from dataclasses import dataclass
from typing import Optional

from src.speech.commands import Command, CommandRegistry


@dataclass
class ParsedCommand:
    """The outcome of parsing one transcript."""

    command: Optional[Command]   # None when unrecognised
    text: str                    # original transcript
    confidence: float            # 1.0 for exact phrase match, 0.0 otherwise


class CommandParser:
    """Parses transcripts into commands using the registry."""

    def __init__(self, registry: Optional[CommandRegistry] = None) -> None:
        self._registry = registry or CommandRegistry()

    def parse(self, text: str) -> ParsedCommand:
        """Parse a transcript into a ParsedCommand.

        Args:
            text: The raw STT transcript (may be empty or None).

        Returns:
            ParsedCommand with ``command`` None if unrecognised.
        """
        if not text or not text.strip():
            return ParsedCommand(command=None, text=text or "", confidence=0.0)
        command = self._registry.lookup_phrase(text)
        confidence = 1.0 if command is not None else 0.0
        return ParsedCommand(command=command, text=text, confidence=confidence)


def parse_command(text: str) -> ParsedCommand:
    """Convenience: parse one transcript with a default parser."""
    return CommandParser().parse(text)