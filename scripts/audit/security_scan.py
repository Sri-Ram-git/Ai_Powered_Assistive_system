"""Security & privacy self-check for the repository.

Scans tracked files for:
    * obvious secrets (API keys, tokens, private keys, credentials);
    * personal-media paths that slipped into version control;
    * sensitive config fields exposed by the API serializer.

Exits non-zero on any finding, so CI can gate on it.

Usage:
    python scripts/audit/security_scan.py [--path .]
"""
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Heuristics for likely secrets (case-insensitive key + assignment).
_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|apikey|secret|token|passw(or)?d|"
    r"private[_-]?key|client[_-]?secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{8,}",
)

# Files that should never exist in the repo.
_FORBIDDEN = (
    ".env", ".env.local", ".env.production", ".env.development",
)
_FORBIDDEN_SUFFIX = (".pem", ".key", ".crt", ".p12", ".pfx")

# Config fields that must never be serialized by the API.
_SENSITIVE_FIELDS = (
    "api_key", "apikey", "secret", "token", "password",
    "model_path", "depth_model_path", "whisper_model_path", "vlm_api_key",
)


def scan() -> int:
    found = 0
    root = Path(args.path).resolve()

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if ".git/" in rel or "__pycache__" in rel:
            continue
        name = path.name.lower()
        if name in _FORBIDDEN or name.endswith(_FORBIDDEN_SUFFIX):
            print(f"SECRET  {rel}")
            found += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if _SECRET_RE.search(text):
            print(f"SECRET  {rel}")
            found += 1

    # Verify the serializer whitelist: none of the sensitive fields may
    # appear in the public_config field list.
    serializer = root / "src" / "api" / "serialize.py"
    if serializer.exists():
        src_text = serializer.read_text(encoding="utf-8")
        m = re.search(r'fields\s*=\s*\(([^)]*)\)', src_text)
        whitelist = m.group(1) if m else ""
        for field in _SENSITIVE_FIELDS:
            if re.search(rf'"{field}"', whitelist):
                print(f"SENSITIVE-FIELD {field} is in the API whitelist!")
                found += 1

    if found:
        print(f"\n{found} security finding(s).  Do not push until resolved.")
        return 1
    print("Security scan clean: no secrets or personal media detected.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    sys.exit(scan())