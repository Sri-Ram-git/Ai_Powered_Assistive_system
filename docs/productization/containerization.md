# Containerization

## Why a container

The device normally runs on its own hardware (a Raspberry Pi-class board
or laptop) with a physical camera.  A container is useful for:

* **Headless smoke tests / CI** — boot the image, hit `/api/health`,
  expect `ok`.
* **Deployment** of the dashboard + JSON API on a host, with the camera
  passed through or a network/virtual camera used.
* **Reproducible environments** for evaluation tooling.

## Build & run

    docker build -t assistive-vision .
    docker run --rm -p 5000:5000 assistive-vision

    # or with compose (mounts models/, pass-through camera commented out)
    docker-compose up --build

The dashboard is at `http://localhost:5000`; JSON API under `/api/*`.

## Camera handling

A container cannot "see" a USB camera by default.  Three options:

1. **Pass the device through** (Linux host): uncomment `devices:` in
   `docker-compose.yml` to bind `/dev/video0`.
2. **Network/virtual camera**: set `camera.id` and a stream URL in
   `configs/assist_config.yaml`.
3. **No camera**: run headless for API/evaluation work only — the
   pipeline starts, reports an error state, and `/api/metrics` still
   serves.

## GPU note (honest)

On-device inference is **CPU** (OpenCV DNN + RapidOCR on onnxruntime
CPU).  The image deliberately has **no CUDA toolchain**: it is smaller
and portable.  If you later switch the detector to a GPU ORT session,
add a `runtime: nvidia` compose section and a CUDA-enabled base image —
document the size increase in a PR.

## What the image excludes

`.dockerignore` keeps the image lean: git metadata, docs, performance
suites, tools, test caches, and (critically) **model weights** and
**any personal media** (`assets/*`) are not baked in.  Weights are
mounted from `./models` (git-ignored) at runtime.

## Health check

    docker run -d --name av -p 5000:5000 assistive-vision
    curl -s http://localhost:5000/api/health   # -> {"status":"ok", ...}
    docker stop av && docker rm av