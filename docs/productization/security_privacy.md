# Security & Privacy

This is an **assistive device** that records and processes **camera
frames in a private home/office environment**.  Privacy is a first-class
requirement, not an afterthought.

## Design principles

1. **Offline-first.**  The core pipeline — camera, YOLO, OCR, tracking,
   navigation, safety engine, speech, response planner — runs entirely
   on-device with no network access.  No cloud dependency, no phone-home.
2. **Nothing personal is stored or uploaded.**  Frames live in memory
   only; the annotated JPEG is served over the local dashboard and is
   never persisted.  Personal media is explicitly git-ignored.
3. **Deterministic safety path.**  Safety-critical decisions are made by
   the SafetyEngine (pure code, no LLM).  An optional VLM may *describe*
   scenes but can never gate a safety decision, and it falls back to a
   fully offline deterministic describer when unreachable.
4. **Least privilege in config.**  The API exposes only whitelisted,
   non-sensitive configuration (`src/api/serialize.py`).  Model paths,
   keys, and credentials are never emitted.

## What is protected / never committed

| Item                        | Guard                                                       |
|-----------------------------|-------------------------------------------------------------|
| Camera frames / personal media | git-ignored (`assets/*`); in-memory only                |
| Model weights               | git-ignored (`models/*.onnx`); re-downloaded via config     |
| API keys / tokens / certs   | git-ignored (`.env`, `*.pem`, `secrets/*`)                  |
| Local config overrides      | git-ignored (`*.local.yaml`)                                |
| Recordings / screenshots    | git-ignored; keep only synthetic placeholders               |

`.env.example` documents what a *real* deployment would need; the actual
`.env` must never be committed.

## Threat model (short)

* **Local attacker on the same machine**: the dashboard binds to
  `127.0.0.1` by default and is not exposed externally; bind to a real
  interface only behind an authenticated reverse proxy.
* **Remote attacker**: no inbound ports are opened by default; the only
  listeners are local (dashboard + API).
* **Cloud misuse**: the optional remote VLM never receives frames unless
  explicitly configured with a key — and a privacy review must approve
  enabling it (see below).

## Checklist before enabling any cloud service

- [ ] Confirm the cloud provider never trains on or stores uploaded
      frames.
- [ ] Confirm data is encrypted in transit (HTTPS/TLS).
- [ ] Confirm frames are deleted promptly server-side.
- [ ] Confirm the local fallback keeps the device fully functional
      offline.
- [ ] Add a privacy notice for end users covering what is sent and why.

## Running the security self-check

A dedicated scan verifies the repo never contains secrets or personal
media and that the API exposes no sensitive fields.  See
`scripts/audit/security_scan.py` and the CI workflow (P18) that runs it
on every change.

## Operator guidance

* Run the dashboard on `127.0.0.1` (the default).  If you must expose
  it, put it behind HTTPS with authentication.
* Keep `*.local.yaml` and `.env` out of version control (they are).
* Rotate any cloud keys in `.env`; never paste them into the repo.
* Review `models/manifest.yaml` (P20) before adding new weights; prefer
  official, verifiable sources.