# CI/CD

## CI (GitHub Actions)

`.github/workflows/ci.yml` runs on every push/PR to `master`, `main`, and
`feature/**` branches.

**Test job** (matrix: Python 3.11, 3.12, 3.13, Ubuntu):

1. `pip install -r requirements.txt pytest pytest-cov`
2. `python -m pytest -q` — full suite with coverage gate
   (`.coveragerc` enforces ≥80%, fails the build otherwise)
3. `python scripts/audit/security_scan.py` — secrets/personal-media scan
4. Headless import check — verifies the core/API/UI layers import
   without camera hardware or a display

**Lint job**: `ruff check src tests scripts performance`.

### Why the CI runs headless

The device normally drives a camera and display.  CI runners have
neither, so:

* tests use stub cameras / synthetic frames exclusively;
* the `OPENCV_VIDEOIO_PRIORITY_MSMF=0` env disables the Windows-only
  MSMF backend on the Linux runner (no-op, defensive);
* the model weights (`models/*.onnx`) are git-ignored; tests either
  fake the detector or skip gracefully when weights are absent.

## CD (publishing)

No automatic CD pipeline is wired yet — deliberate.  Publishing a
prerelease device build to end users has compliance implications
(privacy review, model licensing).  When ready:

1. Tag a version (`git tag vX.Y.Z`) — existing convention is `v1.0`.
2. Add a release workflow that builds a wheel/sdist and attaches it to
   the GitHub Release.
3. Add a smoke test step on the tagged artifact (boot headless, hit
   `/api/health`, expect `ok`).

## Local equivalents

Everything CI does, you can run locally:

    python -m pytest -q                      # tests + coverage gate
    python scripts/audit/security_scan.py    # security self-check
    ruff check src tests scripts performance # lint (if installed)
    python -m pip install ruff               # to enable ruff locally