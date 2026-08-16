"""Verify the model manifest against what's actually on disk.

Reads models/manifest.yaml and reports:
    * which declared model files are present / missing;
    * whether sha256 checksums match when populated;
    * any model on disk that is not declared (ad-hoc weight, flag it).

Usage:
    python scripts/audit/check_models.py
"""
import hashlib
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "models" / "manifest.yaml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print("No models/manifest.yaml — run model setup first.")
        return 1

    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    declared_files = set()
    problems = 0

    for entry in data.get("models", []):
        rel = entry.get("file")
        declared_files.add(rel)
        if not rel:
            # Wheel-bundled or downloaded-at-runtime model (e.g. RapidOCR).
            print(f"[ok]   {entry['id']}: {entry.get('task')} "
                  f"({entry.get('runtime')})")
            continue
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"[miss] {entry['id']}: {rel} NOT PRESENT "
                  f"(expected; weights are git-ignored)")
            continue
        print(f"[ok]   {entry['id']}: {rel} "
              f"({path.stat().st_size / 1e6:.1f} MB)")
        declared_files.add(rel)
        sha = entry.get("sha256")
        if sha:
            actual = _sha256(path)
            status = "MATCH" if actual == sha else "MISMATCH"
            if status == "MISMATCH":
                problems += 1
            print(f"       sha256: {status}")

    # Flag undeclared weights on disk.
    for candidate in (PROJECT_ROOT / "models").glob("*.onnx"):
        rel = f"models/{candidate.name}"
        if rel not in declared_files:
            print(f"[warn] {rel} on disk but NOT declared in manifest")
            problems += 1

    if problems:
        print(f"\n{problems} problem(s) — review before shipping.")
        return 1
    print("\nModel manifest OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())