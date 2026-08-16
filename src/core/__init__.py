"""Core Vision Engine package.

This package holds the asynchronous processing pipeline that is the
heart of the product: a grab thread feeds a shared frame manager, and
independent worker threads consume the latest frame for object
detection, OCR, and (optionally) depth.  Results are published as the
"latest" values so no slow stage ever blocks camera capture, tracking,
the UI, or speech.

The core engine has **no Flask dependency** — the web dashboard and the
desktop app are thin consumers of this package (see ``src.server``).
"""
from src.core.config import PipelineConfig
from src.core.frame_manager import FrameManager
from src.core.pipeline import AsyncVisionPipeline
from src.core.results import LatestResults

__all__ = [
    "AsyncVisionPipeline",
    "FrameManager",
    "LatestResults",
    "PipelineConfig",
]