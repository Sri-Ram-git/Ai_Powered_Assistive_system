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
| `show_toast(message, duration)` | str, float | — | Transient notification (auto-fade) |
| `set_recording(active)` | bool | — | Show/hide red REC pill below the menu bar |
| `widget_rect(widget, w, h)` | str, int, int | `(x, y, w, h)` | Current rectangle of a draggable widget |
| `hit_test(x, y, w, h)` | int, int, int, int | str or None | Widget under a point ('top'/'bottom') |
| `set_widget_pos(widget, x, y, w, h)` | str, int, int, int, int | — | Move a widget, clamped to the canvas |
| `reset()` | — | — | Clear counters/history/positions |
| `font_family` | — | str | Resolved font family name |

### Draggable widgets

The menu bar and dashboard are floating widgets. Wire OpenCV's mouse
callback to the HUD for drag-and-drop repositioning:

```python
def on_mouse(event, x, y, flags, hud):
    if event == cv2.EVENT_LBUTTONDOWN:
        widget = hud.hit_test(x, y, canvas_w, canvas_h)
        if widget:
            rx, ry, _, _ = hud.widget_rect(widget, canvas_w, canvas_h)
            drag["widget"], drag["offset"] = widget, (x - rx, y - ry)
    elif event == cv2.EVENT_MOUSEMOVE and drag["widget"]:
        ox, oy = drag["offset"]
        hud.set_widget_pos(drag["widget"], x - ox, y - oy, canvas_w, canvas_h)
    elif event == cv2.EVENT_LBUTTONUP:
        drag["widget"] = None

cv2.setMouseCallback(window, on_mouse, hud)
```

The REC pill follows the menu bar; toasts stack above the dashboard.

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
