# Productisation — Phase Index

This directory tracks the phased work turning the audited VERSION 1.0
MVP into a credible, production-oriented assistive-vision prototype.

Each phase is committed separately on `feature/ai-productization`
(local git log = the phase record).  Nothing is pushed to the remote
until the branch is accepted.

## Phase map

| # | Deliverable | Status | Key artifact |
|---|---|---|---|
| 0 | Baseline audit + performance baseline | done | `baseline.md` |
| 1–2 | Async pipeline + non-blocking OCR | done | `phases_01_02_async_pipeline_ocr.md` |
| 3 | Distance calibration | done | `src/navigation/calibration.py` |
| 4 | AI evaluation system | done | `model_evaluation.md`, `evaluation/` |
| 5 | Speech input | done | `src/speech/` |
| 6 | Optional depth | done | `src/depth/` |
| 7–8 | Scene context + safety engine | done | `src/safety/` |
| 9 | Optional VLM (offline fallback) | done | `src/vision/vlm/` |
| 10 | Response planner | done | `src/response/` |
| 11 | Benchmark suite | done | `performance/`, `performance.md` |
| 12 | Model optimisation (honest analysis) | done | `scripts/optimize_model.py` |
| 13 | API / core / UI split | done | `src/api/`, `src/ui/` |
| 14 | Professional dark dashboard | done | `src/ui/templates/dashboard.html` |
| 15 | Observability | done | `observability.md`, `/api/metrics` |
| 16 | Security & privacy | done | `security_privacy.md`, `scripts/audit/security_scan.py` |
| 17 | Testing (coverage ≥80%) | done | `pytest.ini`, `.coveragerc` |
| 18 | CI/CD | done | `.github/workflows/ci.yml`, `ci_cd.md` |
| 19 | Containerization | done | `Dockerfile`, `containerization.md` |
| 20 | Model management | done | `models/manifest.yaml`, `scripts/audit/check_models.py` |
| 21 | Product modes | done | `modes.md`, `src/modes/` |
| 22 | Offline-first design | done | `offline_first.md` |
| 23 | Mobile path | done | `mobile_path.md` |
| 24 | Cloud / DevOps architecture | done | `cloud_devops.md` |
| 25 | Documentation + README overhaul | done | `README.md` |

## Verification

Each phase keeps the suite green.  Current state:

    python -m pytest -q            # 284 tests, coverage ≥80%
    python scripts/audit/security_scan.py   # clean
    python scripts/audit/check_models.py    # manifest consistent
    python performance/benchmarks/run_all.py # per-stage latency report

## Branch / delivery

Work lives on `feature/ai-productization` (never pushed).  When
accepted: merge to `master`, bump VERSION, tag, and push — see
`ci_cd.md` for the release checklist.