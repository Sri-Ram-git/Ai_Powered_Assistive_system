# Camera Module

## Overview

Provides a clean, safe abstraction over OpenCV's `VideoCapture` for
webcam initialisation, frame acquisition, and camera management.

## Architecture

```
┌──────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  CameraManager   │────▶│       Camera         │────▶│   CameraUtils   │
│  (discovery)      │     │  (frame capture)     │     │  (screenshots,  │
│  (selection)      │     │  (FPS tracking)      │     │   recording,    │
└──────────────────┘     │  (context manager)   │     │   overlays)     │
                          └──────────────────────┘     └─────────────────┘
```

## Files

| File | Responsibility |
|---|---|
| `camera.py` | Core `Camera` class — init, start, read, stop, context manager |
| `camera_manager.py` | `CameraManager` + `CameraInfo` dataclass — discovery, selection |
| `camera_utils.py` | `take_screenshot`, `record_video`, `draw_fps`, `draw_timestamp`, `show_feed` |
| `camera_test.py` | Interactive CLI test script |

## Classes / Functions

### `Camera` (camera.py)

| Method | Input | Output | Description |
|---|---|---|---|
| `__init__` | camera_id, resolution, fps, backend | — | Configure camera parameters |
| `start()` | — | — | Open the camera device |
| `read()` | — | `np.ndarray` (BGR) | Grab next frame |
| `stop()` | — | — | Release camera |
| `set_resolution(w, h)` | int, int | — | Change resolution at runtime |

**Properties:** `is_running`, `resolution`, `camera_id`, `actual_fps`, `frame_count`

**Context manager:** `with Camera(...) as cam:` — auto start/stop.

### `CameraManager` (camera_manager.py)

| Method | Input | Output | Description |
|---|---|---|---|
| `list_cameras(max_cameras)` | int | `List[CameraInfo]` | Probe device indices |
| `select_camera(camera_id)` | int or None | int | Validate / auto-select camera ID |

### `CameraInfo` dataclass

| Field | Type | Description |
|---|---|---|
| `id` | int | Device index |
| `name` | str | Human-readable label |
| `resolution` | `(int, int) or None` | Detected resolution |
| `backend` | str | OpenCV backend name |

### Utilities (camera_utils.py)

| Function | Input | Output | Description |
|---|---|---|---|
| `take_screenshot(frame, save_dir)` | `np.ndarray`, str | str (path) | Save frame as PNG |
| `record_video(camera, output_dir, duration, fps)` | Camera, str, int, float | str (path) | Record N-second video |
| `draw_fps(frame, fps)` | `np.ndarray`, float | `np.ndarray` | Overlay FPS |
| `draw_timestamp(frame)` | `np.ndarray` | `np.ndarray` | Overlay timestamp |
| `show_feed(camera, window_name, process_frame, on_key)` | Camera, str, callable, callable | — | Live display loop |

## Dependencies

- Python 3.11+
- OpenCV (`cv2`)
- NumPy

## Limitations

- Camera index probing is heuristic — some cameras may not appear if their
  drivers require a non-DShow backend on Windows.
- `cv2.CAP_DSHOW` is Windows-specific. On Linux, change to `cv2.CAP_V4L2`.
- Resolution requests are best-effort; the camera may supply a nearby
  supported resolution, not the exact one requested.

## Future Extensions

- Camera property get/set (exposure, autofocus, white balance, gain)
- Multi-camera synchronisation
- Hot-plug detection (polling USB device events)
- RTSP / IP camera URL support
