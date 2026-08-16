"""HTTP JSON API layer for the assistive vision system.

Completely decoupled from the dashboard UI — anything that speaks JSON
can drive the device: web dashboards, mobile clients, scripts, tests.

    src.api.routes    -> create_api(pipeline): JSON blueprint with /api/*
    src.api.serialize -> JSON serialization (public config only, no secrets)
"""
from src.api.routes import create_api

__all__ = ["create_api"]