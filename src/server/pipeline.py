"""Vision pipeline server — thin Flask-facing wrapper over the core
engine (``src.core.pipeline.AsyncVisionPipeline``).

Keeps the same public interface as before (PipelineConfig, PipelineServer,
latest_jpeg, state_snapshot) so the dashboard and existing tests keep
working, while the actual pipeline now runs asynchronously in the core
engine (grab + detect + OCR worker threads).

Architecture:
    src.core.AsyncVisionPipeline (no Flask dependency)
        ├─ grab thread   → camera → FrameManager → annotated JPEG
        ├─ detect thread → YOLO → IoU tracker → decision/monitor
        └─ OCR worker    → OcrWorker (latest OCR result, non-blocking)

The camera factory is resolved *dynamically* through this module's
``Camera`` name so tests can monkeypatch ``src.server.pipeline.Camera``
to inject a stub camera (existing behaviour preserved).
"""
from typing import Optional, Tuple

from src.camera.camera import Camera as Camera  # noqa: F401  (patch target)
from src.core.config import PipelineConfig  # noqa: F401  (re-export)
from src.core.pipeline import (  # noqa: F401  (kept for tests)
    AsyncVisionPipeline,
    _track_direction,
    _track_distance_m,
)


def _camera_factory(camera_id: int = 0,
                    resolution: Tuple[int, int] = (1280, 720),
                    **kwargs):
    """Build a camera using whatever ``Camera`` currently points at.

    Indirect access through the module global lets tests patch
    ``src.server.pipeline.Camera`` and have the change take effect.
    """
    return Camera(camera_id=camera_id, resolution=resolution)


class PipelineServer(AsyncVisionPipeline):
    """AsyncVisionPipeline pre-bound to the server's camera factory."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        super().__init__(config=config, camera_factory=_camera_factory)


__all__ = ["PipelineConfig", "PipelineServer"]