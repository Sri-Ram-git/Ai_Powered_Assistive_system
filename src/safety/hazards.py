"""Hazard helpers — convenience wrappers over the safety engine's
hazard model for use by the response planner and UI."""
from src.safety.risk import (  # noqa: F401
    Hazard,
    HazardType,
    RiskAssessment,
    RiskLevel,
)

# Human-friendly phrasing for each hazard type (used by the response
# planner; NOT safety-critical — the decision is always RiskAssessment).
HAZARD_PHRASES = {
    HazardType.IMMEDIATE_OBSTACLE: "obstacle immediately ahead",
    HazardType.PROXIMITY: "object nearby",
    HazardType.DANGEROUS_DIRECTION: "hazard in your path",
    HazardType.MOVING_OBJECT: "moving object approaching",
    HazardType.TRAFFIC_SIGNAL: "traffic signal",
    HazardType.STOP_SIGN: "stop sign",
    HazardType.CROSSWALK: "crosswalk",
    HazardType.COLLISION_RISK: "collision risk",
}