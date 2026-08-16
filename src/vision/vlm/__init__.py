"""Optional Vision-Language Model (VLM) enhancement.

The VLM is strictly an OPTIONAL description layer.  It consumes the
deterministic SceneContext and produces natural-language descriptions /
answers.  It MUST NOT control safety-critical decisions (those live in
the safety engine, which never touches an LLM).

Backends:
    ``DeterministicVLM`` — offline fallback: renders the SceneContext as
        a plain, grammatically simple description.  Always available.
    ``RemoteVLM`` — optional cloud VLM via a pluggable client callable.
        Falls back to DeterministicVLM when the API is unavailable, the
        model times out, or the response is malformed.

The core application continues operating offline using the deterministic
backend — cloud processing is opt-in and never required.
"""
from src.vision.vlm.vlm_engine import (
    DeterministicVLM,
    RemoteVLM,
    VLMEngine,
    VLMResult,
    create_vlm,
)

__all__ = [
    "DeterministicVLM",
    "RemoteVLM",
    "VLMEngine",
    "VLMResult",
    "create_vlm",
]