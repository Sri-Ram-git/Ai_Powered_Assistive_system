"""Risk model — re-exports the risk/hazard types used by the safety
engine so consumers can import them from one place."""
from src.safety.safety_engine import (  # noqa: F401
    Hazard,
    HazardType,
    RiskAssessment,
    RiskLevel,
)