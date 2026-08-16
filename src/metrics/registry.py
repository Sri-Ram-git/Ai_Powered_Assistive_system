"""Prometheus-style metrics for the assistive vision system.

A tiny, dependency-free metrics registry that the pipeline, workers, and
API can increment/observe, then expose as Prometheus text format at
``GET /api/metrics`` (or any consumer).  No external scraping library
required; keeps the device footprint small.
"""
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional


class MetricsRegistry:
    """Thread-safe counters, gauges, and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._histograms: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        self._started = time.time()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def inc(self, name: str, labels: Optional[Dict[str, str]] = None,
            amount: int = 1) -> None:
        """Increment a counter (optionally per label set)."""
        key = _key(labels)
        with self._lock:
            self._counters[name][key] += amount

    def set(self, name: str, value: float,
            labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge to a value (optionally per label set)."""
        key = _key(labels)
        with self._lock:
            self._gauges[name][key] = float(value)

    def observe(self, name: str, value: float) -> None:
        """Record a latency sample (count/sum/min/max)."""
        with self._lock:
            d = self._histograms[name]
            d["count"] += 1
            d["sum"] += value
            if not d.get("seen"):
                d["min"] = value
                d["max"] = value
                d["seen"] = True
            else:
                d["min"] = min(d["min"], value)
                d["max"] = max(d["max"], value)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Prometheus text exposition format."""
        lines: List[str] = []
        with self._lock:
            for name, by_key in sorted(self._counters.items()):
                for key, value in sorted(by_key.items()):
                    lines.append(f"{name}{key} {value}")
            for name, by_key in sorted(self._gauges.items()):
                for key, value in sorted(by_key.items()):
                    lines.append(f"{name}{key} {value:.4g}")
            for name, d in sorted(self._histograms.items()):
                lines.append(f"{name}_count {int(d['count'])}")
                lines.append(f"{name}_sum {d['sum']:.4g}")
                if d.get("seen"):
                    lines.append(f"{name}_min {d['min']:.4g}")
                    lines.append(f"{name}_max {d['max']:.4g}")
        lines.append(
            f"process_uptime_seconds {time.time() - self._started:.1f}")
        return "\n".join(lines) + "\n"

    @property
    def started_at(self) -> float:
        return self._started


def _key(labels: Optional[Dict[str, str]]) -> str:
    """Serialise a label dict into the Prometheus label suffix."""
    if not labels:
        return ""
    parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + parts + "}"