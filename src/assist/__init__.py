"""Assistive Vision App module.

assist_app: end-to-end integration app wiring
            Camera → Detection → OCR → Decision → Speech.
"""
from src.assist.assist_app import main

__all__ = ["main"]
