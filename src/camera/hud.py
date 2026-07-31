import time
from collections import deque
from typing import Deque, Iterable, Optional, Tuple

import cv2
import numpy as np

BGRColor = Tuple[int, int, int]

# ----------------------------------------------------------------------
# Styling constants
# ----------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_BG = (28, 28, 32)           # dark panel background
COLOR_ACCENT = (0, 215, 255)      # amber accent
COLOR_FPS_OK = (80, 220, 120)     # green
COLOR_FPS_LOW = (60, 140, 255)    # orange
COLOR_FPS_BAD = (60, 60, 255)     # red
COLOR_TEXT = (235, 235, 235)      # near-white
COLOR_TEXT_DIM = (170, 170, 170)  # grey
COLOR_OUTLINE = (0, 0, 0)         # text outline / shadow


def _outlined_text(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.6,
    color: BGRColor = COLOR_TEXT,
    thickness: int = 2,
    outline: int = 3,
) -> None:
    """Draw text with a black outline for readability on any background."""
    x, y = origin
    cv2.putText(
        frame, text, (x, y), FONT, scale, COLOR_OUTLINE, thickness + outline,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA,
    )


def _blend_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    alpha: float = 0.55,
) -> None:
    """Overlay a semi-transparent dark rounded rectangle on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BG, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


class HUD:
    """Renders a professional overlay (HUD) on camera frames.

    The HUD is a pure presentation layer: it never touches the camera
    or the processing pipeline.  It tracks its own frame counter and
    FPS history so it can draw live graphs.

    Usage:
        hud = HUD()
        while camera.is_running:
            frame = camera.read()
            hud.tick(camera.actual_fps)
            display = hud.render(frame, camera=camera, mode="RAW")
            cv2.imshow("Feed", display)
    """

    def __init__(self, fps_history_len: int = 90) -> None:
        self._frame_count = 0
        self._start_time = time.time()
        self._fps_history: Deque[float] = deque(maxlen=fps_history_len)
        self._last_fps: float = 0.0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def tick(self, fps: float) -> None:
        """Record one rendered frame and its FPS value."""
        self._frame_count += 1
        self._last_fps = fps
        self._fps_history.append(fps)

    def reset(self) -> None:
        """Reset internal counters (e.g. when reopening a feed)."""
        self._frame_count = 0
        self._start_time = time.time()
        self._fps_history.clear()
        self._last_fps = 0.0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        *,
        camera=None,
        mode: str = "RAW",
        status: str = "",
    ) -> np.ndarray:
        """Draw the full HUD onto a copy of the frame.

        Args:
            frame: Input BGR frame.
            camera: Optional object exposing ``camera_id`` and
                ``resolution`` (the src.camera.Camera instance).
            mode: Name of the active processing mode.
            status: Optional transient status message.

        Returns:
            A copy of the frame with the HUD rendered on it.
        """
        display = frame.copy()

        self._draw_info_panel(display, camera=camera, mode=mode)
        self._draw_fps_graph(display)
        self._draw_status_bar(display, status=status)

        return display

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _draw_info_panel(self, frame: np.ndarray, camera, mode: str) -> None:
        h, w = frame.shape[:2]
        x, y, pad = 12, 12, 12
        line_h = 26
        n_lines = 5
        panel_w = 260
        panel_h = n_lines * line_h + pad * 2

        _blend_panel(frame, x, y, panel_w, panel_h)

        camera_label = "N/A"
        res_label = "N/A"
        if camera is not None:
            camera_label = f"Camera {camera.camera_id}"
            if hasattr(camera, "resolution"):
                res_label = f"{camera.resolution[0]}x{camera.resolution[1]}"

        uptime = time.time() - self._start_time
        fps_color = self._fps_color(self._last_fps)

        cx = x + pad
        cy = y + pad + line_h

        _outlined_text(
            frame, "ASSISTIVE VISION", (cx, cy - 10), 0.62, COLOR_ACCENT, 2, 3,
        )

        rows = [
            ("Source", camera_label),
            ("Resolution", res_label),
            ("FPS", f"{self._last_fps:5.1f}"),
            ("Frames", f"{self._frame_count}"),
            ("Uptime", f"{int(uptime)}s"),
        ]
        for i, (label, value) in enumerate(rows):
            row_y = cy + 14 + i * line_h
            _outlined_text(frame, label, (cx, row_y), 0.5, COLOR_TEXT_DIM, 1, 2)
            value_color = fps_color if label == "FPS" else COLOR_TEXT
            _outlined_text(
                frame, value, (cx + 110, row_y), 0.55, value_color, 1, 2,
            )

        mode_x = x + panel_w + 10
        _outlined_text(frame, f"MODE  {mode}", (mode_x, y + 24), 0.55, COLOR_ACCENT, 1, 2)

    def _draw_fps_graph(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        graph_w = min(220, w - 24)
        graph_h = 44
        graph_x = w - graph_w - 12
        graph_y = 12

        _blend_panel(frame, graph_x, graph_y, graph_w, graph_h)

        data = list(self._fps_history)
        if len(data) < 2:
            _outlined_text(
                frame, "fps", (graph_x + 8, graph_y + graph_h - 8),
                0.5, COLOR_TEXT_DIM, 1, 2,
            )
            return

        max_val = max(max(data), 1.0)
        n = len(data)
        step_x = (graph_w - 16) / (n - 1)
        points = [
            (int(graph_x + 8 + i * step_x),
             int(graph_y + graph_h - 8 - (data[i] / max_val) * (graph_h - 16)))
            for i in range(n)
        ]

        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, COLOR_FPS_OK, 2, cv2.LINE_AA)

        avg = sum(data) / n
        avg_y = int(graph_y + graph_h - 8 - (avg / max_val) * (graph_h - 16))
        cv2.line(
            frame,
            (graph_x + 8, avg_y),
            (graph_x + graph_w - 8, avg_y),
            COLOR_FPS_OK,
            1,
            cv2.LINE_AA,
        )
        _outlined_text(
            frame, f"avg {avg:4.1f}", (graph_x + 8, graph_y + graph_h - 8),
            0.5, COLOR_FPS_OK, 1, 2,
        )

    def _draw_status_bar(self, frame: np.ndarray, status: str) -> None:
        h, w = frame.shape[:2]
        bar_h = 34
        bar_y = h - bar_h
        pad = 12

        _blend_panel(frame, 0, bar_y, w, bar_h)

        hints = "[S] Screenshot   [R] Record   [Q] Quit"
        _outlined_text(
            frame, hints, (pad, bar_y + 22), 0.5, COLOR_TEXT_DIM, 1, 2,
        )

        if status:
            text_w = int(cv2.getTextSize(
                status, FONT, 0.55, 2,
            )[0][0])
            _outlined_text(
                frame, status, (w - text_w - pad, bar_y + 22),
                0.55, COLOR_ACCENT, 2, 3,
            )

    @staticmethod
    def _fps_color(fps: float) -> BGRColor:
        if fps >= 24.0:
            return COLOR_FPS_OK
        if fps >= 12.0:
            return COLOR_FPS_LOW
        return COLOR_FPS_BAD


def annotate(
    frame: np.ndarray,
    labels: Iterable[Tuple[int, int, str]] = (),
) -> np.ndarray:
    """Draw small labeled markers on a frame (e.g. detected objects).

    Args:
        frame: Input BGR frame.
        labels: Iterable of ``(x, y, text)`` tuples.

    Returns:
        A copy of the frame with markers drawn.
    """
    display = frame.copy()
    for x, y, text in labels:
        cv2.circle(display, (x, y), 6, COLOR_ACCENT, -1, cv2.LINE_AA)
        _outlined_text(display, text, (x + 10, y + 6), 0.5, COLOR_TEXT, 1, 2)
    return display
