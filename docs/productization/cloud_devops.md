# Cloud / DevOps Architecture

## Positioning (honest)

The device is **offline-first and private** (P22, P16).  A cloud
architecture exists only for *fleet* scenarios — managing many devices
in assisted-living facilities, testing, or supervised deployments — not
for the core assistance loop.

## Reference topology

    [device 1] ─┐
    [device 2] ─┼─ HTTPS ──> [edge gateway] ──> [central services]
    [device N] ─┘                │
                                 ├─ telemetry ingest (/api/metrics)
                                 ├─ config management (modes, calibration)
                                 └─ optional model/update serving

### Per device (unchanged)
* Flask app exposing `/api/*` + `/video_feed` on `127.0.0.1`.
* Prometheus-style `/api/metrics` for health + latency.

### Edge gateway
* TLS termination + auth (reverse proxy).
* Device registry, heartbeat, and per-device config push.
* Never stores camera frames; forwards only metrics/commands.

### Central services
* Prometheus (+ optionally Grafana) scraping `/api/metrics`.
* Git-based config/module repository (the repo itself).
* Release pipeline: build container (P19) → smoke test → roll out.

## Telemetry contract

Scrape `/api/metrics` (Prometheus text) every 15 s.  The set is small
and private: FPS, per-stage latency histograms, frame counters.  **No
frames, no text content, no personal data** leave the device (see
observability.md for the full metric list).

## Deployment patterns

| Pattern | When |
|---------|------|
| **Single device, standalone** | default; the repo's Docker image + local run |
| **Bulk fleet via compose** | one host per device, or `docker-compose` per site |
| **Kubernetes** | only at large scale; prefer edge gateway + per-device pods |

Keep each device's state local (SQLite/config files) — the pipeline is
stateless enough to restart cleanly on config change.

## DevOps principles

1. **CI gates everything** (P18): tests, coverage ≥80%, security scan,
   lint — before any tag/release.
2. **Reproducible builds**: `requirements.txt` + Docker image + pinned
   model manifest (`models/manifest.yaml`, checksums).
3. **Secrets never in repo**: `.env` / keys are git-ignored; injected at
   deploy time.
4. **Rollbacks are cheap**: containers are immutable; tag + rollback by
   image tag.
5. **Observability before rollout**: `/api/metrics` scrape + health
   endpoint must pass before a device is considered "deployed".

## Not yet implemented (roadmap)

* Central device registry / heartbeat service.
* Config push channel (mode + calibration overrides).
* Release workflow with tagged artifacts (documented in ci_cd.md).

These are intentionally *not* built until a real fleet deployment
justifies them — building them now would be speculative.