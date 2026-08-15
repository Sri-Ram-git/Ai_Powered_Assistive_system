"""Environment Guidance module.

guidance: stateless helpers that turn detections + OCR into navigation
          cues (direction, distance, crosswalk / traffic-signal warnings).
"""
from src.navigation.guidance import (
    direction_of,
    distance_estimate,
    distance_phrase,
    nearest_obstacle,
    reference_height,
    scene_cues,
)

__all__ = [
    "direction_of",
    "distance_estimate",
    "distance_phrase",
    "nearest_obstacle",
    "reference_height",
    "scene_cues",
]
