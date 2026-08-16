# Observability

## What is exposed

The device exposes **Prometheus-style metrics** over a single text
endpoint so operators can monitor a live deployment without any external
scraping library on the device:

    GET /api/metrics

Content-type is `text/plain; version=0.0.4` (Prometheus exposition
format).  The registry is dependency-free
(`src/metrics/registry.py`).

## Metrics

| Metric                  | Type      | Meaning                              |
|-------------------------|-----------|--------------------------------------|
| `camera_fps`            | gauge     | Actual camera read rate (Hz)         |
| `yolo_latency_ms`       | histogram | Detection stage latency (count/sum/min/max) |
| `ocr_latency_ms`        | histogram | OCR worker latency                   |
| `depth_latency_ms`      | histogram | Depth stage latency                  |
| `frames_processed`      | counter   | Frames through the detect loop       |
| `detections_found`      | counter   | Objects detected (per frame)         |
| `process_uptime_seconds`| gauge     | Process uptime                       |

Latency histograms are minimal (count/sum/min/max) — enough to detect
regressions without the weight of full histograms.  These are published
from the detect loop each tick, so a scrape always reflects the current
device state.

## How to turn it on/off

The `metrics` flag in `configs/assist_config.yaml` (`app.metrics`, default
on) gates the registry.  When disabled, `/api/metrics` returns 404.

## Typical usage

    # live tail on the device
    curl -s http://127.0.0.1:5000/api/metrics

    # in Prometheus
    scrape_configs:
      - job_name: assistive_vision
        metrics_path: /api/metrics
        static_configs:
          - targets: ["device.local:5000"]

    # quick latency sanity check
    curl -s http://127.0.0.1:5000/api/metrics | grep yolo_latency_ms

## What is deliberately NOT exposed

* Camera frames (binary; use `/video_feed`).
* Model weights, calibration data, or any personal media.
* API keys or cloud credentials (the serialization layer in
  `src/api/serialize.py` whitelists fields — nothing else is emitted).
* Full per-frame detail — the registry keeps only aggregate values to
  protect privacy and keep the device footprint small.