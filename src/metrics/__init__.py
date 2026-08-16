"""Observability — Prometheus-style metrics for the assistive device.

    src.metrics.registry.MetricsRegistry -> counters/gauges/histograms
"""
from src.metrics.registry import MetricsRegistry

__all__ = ["MetricsRegistry"]