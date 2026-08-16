"""Depth estimation result container.

A depth map (normalised to [0, 1]) plus per-object representative depth
extracted for each bounding box (median depth inside the box is the
default — robust to box-edge contamination).
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class DepthResult:
    """The output of a depth-estimation stage.

    Attributes:
        map: Normalised depth map, same HxW as the input frame.
            Values are *relative* (closer = lower) unless the backend
            provides absolute depth; backends must document this.
        per_box_depth: Mapping of a stable box id -> representative depth.
            Depth units follow the map's convention.
        backend: Name of the backend that produced this result.
        latency_ms: Inference latency of the depth stage.
    """

    map: np.ndarray
    per_box_depth: Dict[int, float]
    backend: str = "unknown"
    latency_ms: float = 0.0

    def box_depth(self, box: Tuple[int, int, int, int]) -> Optional[float]:
        """Median depth inside a box (x, y, w, h).

        Returns None when the box is empty or the map is not sized to
        the frame.
        """
        if self.map is None or self.map.size == 0:
            return None
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return None
        hh, ww = self.map.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(ww, x + w), min(hh, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        region = self.map[y1:y2, x1:x2]
        if region.size == 0:
            return None
        return float(np.median(region))