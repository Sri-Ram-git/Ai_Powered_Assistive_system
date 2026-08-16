"""Safety reasoning — deterministic, LLM-independent.

The safety engine turns a SceneContext into a risk assessment.  It is a
pure function of the world model and MUST never depend on an LLM or on
natural-language generation: the camera → perception → safety-engine →
safe/unsafe decision path is the hard guarantee of the product.

Risk model:
    A risk assessment contains the highest risk level found plus the
    individual hazards that produced it.  Risk levels are ordered:
    NONE < LOW < MEDIUM < HIGH.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from src.vision.scene_context import SceneContext, SceneObject

_NEAR_DISTANCE_M = 1.5          # within this -> HIGH for an obstacle
_WARN_DISTANCE_M = 3.0          # within this -> MEDIUM for an obstacle
_VEHICLE_NEAR_M = 3.0           # vehicles within this -> HIGH
_VEHICLE_WARN_M = 6.0           # vehicles within this -> MEDIUM
_APPROACH_SPEED_M_S = 0.5       # approaching faster than this -> HIGH


class RiskLevel(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class HazardType(IntEnum):
    IMMEDIATE_OBSTACLE = 1
    PROXIMITY = 2
    DANGEROUS_DIRECTION = 3
    MOVING_OBJECT = 4
    TRAFFIC_SIGNAL = 5
    STOP_SIGN = 6
    CROSSWALK = 7
    COLLISION_RISK = 8


@dataclass
class Hazard:
    """One concrete hazard the engine identified."""

    hazard_type: HazardType
    label: str
    distance_m: Optional[float]
    direction: str
    risk: RiskLevel
    track_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "type": self.hazard_type.name.lower(),
            "label": self.label,
            "distance_m": self.distance_m,
            "direction": self.direction,
            "risk": self.risk.name.lower(),
            "track_id": self.track_id,
        }


@dataclass
class RiskAssessment:
    """The safety engine's verdict for one frame."""

    level: RiskLevel
    hazards: List[Hazard]

    @property
    def urgent(self) -> bool:
        """Whether the user should stop / be warned immediately."""
        return self.level >= RiskLevel.HIGH

    def to_dict(self) -> dict:
        return {
            "level": self.level.name.lower(),
            "urgent": self.urgent,
            "hazards": [h.to_dict() for h in self.hazards],
        }


class SafetyEngine:
    """Deterministic risk assessment from a SceneContext."""

    def __init__(
        self,
        near_distance_m: float = _NEAR_DISTANCE_M,
        warn_distance_m: float = _WARN_DISTANCE_M,
        vehicle_near_m: float = _VEHICLE_NEAR_M,
        vehicle_warn_m: float = _VEHICLE_WARN_M,
        approach_speed_m_s: float = _APPROACH_SPEED_M_S,
    ) -> None:
        """Configure risk thresholds (all in metres / m/s)."""
        self._near = float(near_distance_m)
        self._warn = float(warn_distance_m)
        self._vehicle_near = float(vehicle_near_m)
        self._vehicle_warn = float(vehicle_warn_m)
        self._approach = float(approach_speed_m_s)

    def assess(self, context: SceneContext) -> RiskAssessment:
        """Compute the risk assessment for a scene snapshot.

        Args:
            context: The current SceneContext.

        Returns:
            RiskAssessment with the highest risk level found.
        """
        hazards: List[Hazard] = []
        level = RiskLevel.NONE

        for obj in context.objects:
            h = self._hazard_for(obj)
            if h is not None:
                hazards.append(h)
                level = max(level, h.risk)

        # Text-based hazards: crosswalk / stop-sign text.
        for h in self._text_hazards(context.text):
            hazards.append(h)
            level = max(level, h.risk)

        return RiskAssessment(level=level, hazards=hazards)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hazard_for(self, obj: SceneObject) -> Optional[Hazard]:
        dist = obj.distance_m
        direction = obj.direction

        if obj.category == "vehicle":
            if dist is not None and dist <= self._vehicle_near:
                return Hazard(HazardType.COLLISION_RISK, obj.label, dist,
                              direction, RiskLevel.HIGH, obj.track_id)
            if dist is not None and dist <= self._vehicle_warn:
                return Hazard(HazardType.PROXIMITY, obj.label, dist,
                              direction, RiskLevel.MEDIUM, obj.track_id)
            if obj.velocity is not None and obj.velocity > self._approach:
                return Hazard(HazardType.MOVING_OBJECT, obj.label, dist,
                              direction, RiskLevel.HIGH, obj.track_id)
            return None

        if obj.category == "obstacle":
            if dist is not None and dist <= self._near:
                return Hazard(HazardType.IMMEDIATE_OBSTACLE, obj.label,
                              dist, direction, RiskLevel.HIGH, obj.track_id)
            if dist is not None and dist <= self._warn:
                return Hazard(HazardType.PROXIMITY, obj.label, dist,
                              direction, RiskLevel.MEDIUM, obj.track_id)
            return None

        if obj.category == "traffic signal":
            return Hazard(HazardType.TRAFFIC_SIGNAL, obj.label, dist,
                          direction, RiskLevel.MEDIUM, obj.track_id)

        if obj.category == "stop sign":
            return Hazard(HazardType.STOP_SIGN, obj.label, dist,
                          direction, RiskLevel.MEDIUM, obj.track_id)

        if dist is not None and dist <= self._near and direction == "ahead":
            return Hazard(HazardType.IMMEDIATE_OBSTACLE, obj.label, dist,
                          direction, RiskLevel.HIGH, obj.track_id)

        return None

    def _text_hazards(self, text: List[str]) -> List[Hazard]:
        hazards: List[Hazard] = []
        joined = " ".join(text).lower()
        if any(k in joined for k in ("crosswalk", "walk", "pedestrian")):
            hazards.append(Hazard(HazardType.CROSSWALK, "crosswalk",
                                  None, "ahead", RiskLevel.MEDIUM))
        if any(k in joined for k in ("don't walk", "do not walk",
                                     "dont walk")):
            hazards.append(Hazard(HazardType.CROSSWALK, "do not walk",
                                  None, "ahead", RiskLevel.HIGH))
        return hazards