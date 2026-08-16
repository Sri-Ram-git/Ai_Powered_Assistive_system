"""Safety module — deterministic, LLM-independent safety reasoning.

safety_engine: SafetyEngine — turns a SceneContext into a RiskAssessment.
risk:         RiskLevel, Hazard, HazardType, RiskAssessment.
hazards:      hazard phrasing helpers (non-critical).

Guarantee: the safety-critical path is
    camera → perception → safety engine → safe/unsafe decision
and never involves an LLM.
"""
from src.safety.risk import (
    Hazard,
    HazardType,
    RiskAssessment,
    RiskLevel,
)
from src.safety.safety_engine import SafetyEngine

__all__ = [
    "Hazard",
    "HazardType",
    "RiskAssessment",
    "RiskLevel",
    "SafetyEngine",
]