"""Product modes — user-facing behaviours of the assistive device.

Five modes, each a different emphasis of the same underlying pipeline:

    object      — object detection + proximity guidance (default)
    reading     — OCR-focused: read text aloud as it appears
    navigation  — maximise navigation guidance, reduce chatter
    scene       — higher-level scene description (VLM/deterministic)
    voice       — voice-command-first: quiet otherwise, always listening

Modes are **behavioural knobs** over the shared engine; they never
recompile or restart the pipeline.  The response planner, safety engine,
and camera always stay active (safety must not be mode-dependent).
"""
from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class ModeBehavior:
    """Per-mode tuning applied to the pipeline on switch."""

    name: str
    label: str
    description: str
    # Knobs (each maps to a PipelineConfig field).
    ocr_enabled: bool = True
    navigation_enabled: bool = True
    announce_objects: bool = True        # object proximity chatter
    announce_text: bool = True           # read recognised text aloud
    scene_describe: bool = False         # describe whole scene
    quiet: bool = False                  # minimal speaking


MODES: Dict[str, ModeBehavior] = {
    "object": ModeBehavior(
        "object", "Object", "Detect objects and guide by proximity.",
        ocr_enabled=True, navigation_enabled=True,
        announce_objects=True, announce_text=False),
    "reading": ModeBehavior(
        "reading", "Reading", "Focus on text: read it aloud as found.",
        ocr_enabled=True, navigation_enabled=False,
        announce_objects=False, announce_text=True),
    "navigation": ModeBehavior(
        "navigation", "Navigation", "Concentrate on safe navigation.",
        ocr_enabled=False, navigation_enabled=True,
        announce_objects=True, announce_text=False),
    "scene": ModeBehavior(
        "scene", "Scene", "Describe the environment at a higher level.",
        ocr_enabled=True, navigation_enabled=False,
        announce_objects=False, announce_text=False,
        scene_describe=True, quiet=True),
    "voice": ModeBehavior(
        "voice", "Voice", "Command-first: stay quiet, respond on demand.",
        ocr_enabled=False, navigation_enabled=False,
        announce_objects=False, announce_text=False, quiet=True),
}


class ModeManager:
    """Resolves a mode name into behaviour and applies it to config."""

    def __init__(self, config) -> None:
        self._cfg = config

    @property
    def valid_modes(self) -> Iterable[str]:
        return tuple(MODES.keys())

    def get(self, name: str) -> ModeBehavior:
        name = (name or "object").strip().lower()
        if name not in MODES:
            raise ValueError(
                f"unknown mode {name!r}; expected one of {tuple(MODES)}")
        return MODES[name]

    def apply(self, name: str) -> ModeBehavior:
        """Apply the mode's knobs to the pipeline config and return it."""
        behavior = self.get(name)
        self._cfg.mode = behavior.name
        self._cfg.ocr_enabled = behavior.ocr_enabled
        self._cfg.navigation_enabled = behavior.navigation_enabled
        return behavior


def validate_mode(name: str) -> str:
    """Return a canonical mode name or raise ValueError."""
    name = (name or "").strip().lower()
    if name not in MODES:
        raise ValueError(
            f"unknown mode {name!r}; expected one of {tuple(MODES)}")
    return name