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
| `camera_utils.py` | `take_screenshot`, `record_video`, `draw_fps`, `draw_timestamp`, `show_feed` + display helpers |
| `hud.py` | `HUD` class — minimal professional overlay (PIL + TTF fonts) |
| `camera_test.py` | Interactive CLI test script (fullscreen, auto high-res) |

## Classes / Functions

### `Camera` (camera.py)

| Method | Input | Output | Description |
|---|---|---|---|
| `__init__` | camera_id, resolution, fps, backend, mirror | — | Configure camera parameters |
| `start()` | — | — | Open the camera device |
| `read()` | — | `np.ndarray` (BGR) | Grab next frame |
| `stop()` | — | — | Release camera |
| `set_resolution(w, h)` | int, int | — | Change resolution at runtime |
| `set_mirror(enabled)` | bool | — | Toggle horizontal flip at runtime |

**Properties:** `is_running`, `resolution`, `camera_id`, `actual_fps`, `frame_count`, `mirror`

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
| `record_video(camera, output_dir, duration, fps)` | Camera, str, int, float | str (path) | Record N-second video (blocking) |
| `VideoRecorder(camera, output_dir, duration, fps)` | Camera, str, int, float | class | Record in a background thread; `latest_frame` keeps UI live |
| `draw_fps(frame, fps)` | `np.ndarray`, float | `np.ndarray` | Overlay FPS |
| `draw_timestamp(frame)` | `np.ndarray` | `np.ndarray` | Overlay timestamp |
| `show_feed(camera, window_name, process_frame, on_key)` | Camera, str, callable, callable | — | Live display loop |
| `get_screen_size()` | — | `(int, int)` | Primary display resolution |
| `open_fullscreen_window(name)` | str | — | Borderless fullscreen OpenCV window |
| `scale_to_fit(frame, w, h)` | `np.ndarray`, int, int | `np.ndarray` | Aspect-preserving letterbox scale (INTER_CUBIC upscale) |
| `auto_select_resolution(camera)` | Camera | `(int, int)` | Pick best supported resolution |

### `VideoRecorder`

| Member | Type | Description |
|---|---|---|
| `start()` | method → str | Begin background recording, return target path |
| `stop()` | method | Stop early and flush the video file |
| `is_recording` | property (bool) | True while the background thread writes frames |
| `latest_frame` | property (`np.ndarray` or None) | Most recent captured frame for live UI display |
| `saved_path` | property (str or None) | Output file path once recording finishes |

### HUD overlay (hud.py)

The HUD renders a minimal professional UI using Pillow with TrueType
fonts (Segoe UI on Windows, fallback to Arial / DejaVu Sans):

- **Top bar** — `ASSISTIVE VISION` title, camera/resolution/FPS meta
  (FPS colour-coded green ≥ 24, orange ≥ 12, red below), mode chip
- **Bottom bar** — keyboard hints + transient status message
- Rounded semi-transparent panels with accent borders
- Antialiased text with subtle shadows; pure presentation layer

| Method | Input | Output | Description |
|---|---|---|---|
| `tick(fps)` | float | — | Record rendered frame + FPS sample |
| `render(frame, camera, mode, status)` | `np.ndarray`, Camera, str, str | `np.ndarray` | Draw full HUD on the frame |
| `show_toast(message, duration)` | str, float | — | Transient bottom notification (auto-fade) |
| `set_recording(active)` | bool | — | Show/hide red REC pill below the top bar |
| `reset()` | — | — | Clear counters/history |
| `font_family` | — | str | Resolved font family name |

## Dependencies

- Python 3.11+
- OpenCV (`cv2`)
- NumPy
- Pillow (professional HUD text rendering)

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
