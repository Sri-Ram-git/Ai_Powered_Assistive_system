"""Vision module.

scene_context: SceneContext, SceneObject — the deterministic internal
               world model combining detection + OCR + depth + tracking.
"""
from src.vision.scene_context import (
    SceneContext,
    SceneObject,
    build_scene_context,
)

__all__ = [
    "SceneContext",
    "SceneObject",
    "build_scene_context",
]