"""Tests for the security scan (P16)."""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_security_scan_is_clean():
    r = subprocess.run(
        [sys.executable, "scripts/audit/security_scan.py"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "clean" in r.stdout


def test_security_scan_detects_secret_in_temp_dir(tmp_path):
    # Build the fake secret at runtime so the committed test file never
    # contains a real-looking credential (keeps the scan itself clean).
    key = "VLM_API" + "_KEY=supersecret" + "123\n"
    (tmp_path / ".env").write_text(key)
    r = subprocess.run(
        [sys.executable, "scripts/audit/security_scan.py",
         f"--path={tmp_path}"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 1
    assert "SECRET" in r.stdout
    from src.api.serialize import public_config
    from src.core.config import PipelineConfig

    cfg = PipelineConfig()
    cfg.model_path = "models/supersecret.onnx"
    out = public_config(cfg)
    assert "model_path" not in out
    assert "depth_model_path" not in out