"""Web Dashboard module.

pipeline: PipelineServer, PipelineConfig — background vision pipeline
          (camera → detect → track → OCR → decision → guidance).
app:      Flask app + the dashboard page (/video_feed, /api/state, /).
"""
from src.server.pipeline import (
    PipelineConfig,
    PipelineServer,
)

__all__ = [
    "PipelineConfig",
    "PipelineServer",
]
