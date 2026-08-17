"""Object → OCR eligibility policy.

Decides *which* detected objects are worth inspecting for text, using a
configurable YAML (``configs/ocr_policy.yaml``) that a developer/user can
edit without touching source code.

Only labels that actually exist in the detector's class list are accepted
— the policy never invents YOLO classes.  A label absent from the file
falls back to ``default_tier``.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from src.detection.detector import COCO_NAMES
from src.utils.logger import setup_logger

_logger = setup_logger("OcrPolicy")

_TIER_RANK = {"high": 3, "medium": 2, "low": 1}


@dataclass
class OcrPolicy:
    """Eligibility tiers keyed by object label (lower rank = higher prio)."""

    high: List[str] = field(default_factory=list)
    medium: List[str] = field(default_factory=list)
    low: List[str] = field(default_factory=list)
    disabled: List[str] = field(default_factory=list)
    default_tier: str = "medium"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def defaults(cls) -> "OcrPolicy":
        """Built-in policy (used when no YAML is configured)."""
        return cls(
            high=["book", "bottle", "laptop", "cell phone", "tv",
                  "stop sign"],
            medium=["cup", "backpack", "handbag", "suitcase",
                    "keyboard", "remote", "vase", "clock"],
            low=[],
            disabled=["person", "chair", "dining table", "potted plant",
                      "couch", "bed"],
        )

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "OcrPolicy":
        """Load a policy from YAML, validated against real COCO classes.

        Args:
            path: YAML path.  None or missing file → defaults.

        Returns:
            A policy whose lists contain only supported COCO labels.
        """
        policy = cls.defaults()
        if not path:
            return policy
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            _logger.info("OCR policy %s not found — using defaults", path)
            return policy

        section = data.get("ocr_policy", {}) or {}
        supported = set(COCO_NAMES)

        def _clean(raw: List[str], tier: str) -> List[str]:
            clean: List[str] = []
            for label in raw or []:
                label = str(label).strip().lower()
                if tier != "disabled" and label in policy.disabled:
                    _logger.warning(
                        "ocr_policy: %s listed in %s is also disabled - "
                        "ignoring", label, tier)
                    continue
                if label not in supported:
                    _logger.warning(
                        "ocr_policy: '%s' is not a supported detector "
                        "class - ignored", label)
                    continue
                clean.append(label)
            return clean

        policy.high = _clean(section.get("high_priority", []), "high")
        policy.medium = _clean(section.get("medium_priority", []), "medium")
        policy.low = _clean(section.get("low_priority", []), "low")
        policy.disabled = _clean(section.get("disabled", []), "disabled")
        policy.default_tier = str(
            section.get("default_tier", policy.default_tier))
        if policy.default_tier not in _TIER_RANK:
            policy.default_tier = "medium"
        return policy

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def tier_for(self, label: Optional[str]) -> str:
        """The tier for an object label ('high'|'medium'|'low')."""
        if not label:
            return "none"
        label = str(label).strip().lower()
        if label in self.disabled:
            return "none"
        if label in self.high:
            return "high"
        if label in self.medium:
            return "medium"
        if label in self.low:
            return "low"
        return self.default_tier

    def is_eligible(self, label: Optional[str]) -> bool:
        """Whether a label is worth OCR (high/medium/low, not disabled)."""
        return self.tier_for(label) in _TIER_RANK

    def rank(self, label: Optional[str]) -> int:
        """Priority rank for ordering targets (higher = better)."""
        return _TIER_RANK.get(self.tier_for(label), 0)

    def eligible_labels(self) -> List[str]:
        """All labels the policy would consider for OCR."""
        return list(self.high) + list(self.medium) + list(self.low)

    def to_dict(self) -> Dict:
        return {
            "high_priority": list(self.high),
            "medium_priority": list(self.medium),
            "low_priority": list(self.low),
            "disabled": list(self.disabled),
            "default_tier": self.default_tier,
        }