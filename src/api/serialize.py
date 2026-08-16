"""API serialization helpers — turn internal objects into public JSON.

Only non-sensitive configuration is ever exposed.  Model paths, API
keys, and any secrets must never appear here.
"""
from typing import Any, Dict


def public_config(cfg: Any) -> Dict[str, Any]:
    """Effective pipeline config minus secrets / internal fields."""
    fields = (
        "camera_id", "camera_resolution", "detect_every", "ocr_every",
        "mode", "metrics", "depth_enabled", "depth_backend",
        "planner_cooldown",
    )
    out: Dict[str, Any] = {}
    for name in fields:
        if hasattr(cfg, name):
            out[name] = getattr(cfg, name)
    return out