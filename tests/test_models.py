"""Tests for model manifest tooling (P20)."""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_check_models_runs_clean():
    r = subprocess.run(
        [sys.executable, "scripts/audit/check_models.py"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    # The manifest may report missing optional weights (expected), but
    # must not crash and must always reach the summary line.
    assert r.returncode in (0, 1)
    assert "Model manifest" in r.stdout


def test_manifest_is_valid_yaml_with_models_key():
    import yaml

    manifest = PROJECT_ROOT / "models" / "manifest.yaml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert "models" in data
    assert len(data["models"]) >= 3
    ids = [m["id"] for m in data["models"]]
    assert "yolov8n" in ids
    assert "rapidocr" in ids
    for m in data["models"]:
        assert "license" in m
        assert "runtime" in m