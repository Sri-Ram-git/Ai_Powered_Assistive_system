import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    _PIL_AVAILABLE = False

# ----------------------------------------------------------------------
# Colours (BGR for cv2, converted to RGB for PIL)
# Monochrome design: black panels, white text, light borders.
# ----------------------------------------------------------------------

COLOR_PANEL = (0, 0, 0)
COLOR_BORDER = (90, 90, 96)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_DIM = (200, 200, 206)
COLOR_OUTLINE = (0, 0, 0)

_PANEL_RGB = (0, 0, 0)
_BORDER_RGB = (90, 90, 96)
_TEXT_RGB = (255, 255, 255)
_DIM_RGB = (200, 200, 206)


# ----------------------------------------------------------------------
# Font management
# ----------------------------------------------------------------------

_FONT_CANDIDATES = [
    {
        "family": "Segoe UI",
        "regular": "C:/Windows/Fonts/segoeui.ttf",
        "semibold": "C:/Windows/Fonts/seguisb.ttf",
        "bold": "C:/Windows/Fonts/segoeuib.ttf",
        "light": "C:/Windows/Fonts/segoeuil.ttf",
    },
    {
        "family": "Arial",
        "regular": "C:/Windows/Fonts/arial.ttf",
        "semibold": "C:/Windows/Fonts/arialbd.ttf",
        "bold": "C:/Windows/Fonts/arialbd.ttf",
        "light": "C:/Windows/Fonts/arial.ttf",
    },
    {
        "family": "DejaVu Sans",
        "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "semibold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "light": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    },
]


class FontManager:
    """Locates a professional TTF font and caches loaded variants.

    Searches well-known font paths across Windows, Linux, and macOS.
    Falls back to the first family that provides a "regular" face.
    """

    _WEIGHTS = ("regular", "semibold", "bold", "light")

    def __init__(self) -> None:
        self._path: Dict[str, str] = {}
        self._family = "Sans"
        self._select_family()
        self._cache: Dict[Tuple[int, str], ImageFont.FreeTypeFont] = {}

    def _select_family(self) -> None:
        for candidate in _FONT_CANDIDATES:
            regular = candidate.get("regular", "")
            if Path(regular).is_file():
                self._family = candidate["family"]
                self._path = {w: candidate.get(w, regular) for w in self._WEIGHTS}
                return
        self._family = "Default"

    @property
    def family(self) -> str:
        return self._family

    def font(self, size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
        """Return a cached PIL font of the given size and weight."""
        key = (size, weight)
        if key not in self._cache:
            path = self._path.get(weight) or self._path.get("regular", "")
            if path:
                self._cache[key] = ImageFont.truetype(path, size)
            else:
                self._cache[key] = ImageFont.load_default(size)
        return self._cache[key]


# ----------------------------------------------------------------------
# Canvas — draws text, rounded panels, and lines onto a BGR frame
# ----------------------------------------------------------------------

class Canvas:
    """High-level 2D drawing surface backed by PIL (antialiased text).

    Accepts BGR frames (OpenCV convention) and returns a BGR frame,
    so it slots cleanly into the existing pipeline.
    """

    def __init__(self, frame_bgr: np.ndarray, fonts: FontManager) -> None:
        self.width = frame_bgr.shape[1]
        self.height = frame_bgr.shape[0]
        self._fonts = fonts
        self._image = Image.fromarray(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )
        self._draw = ImageDraw.Draw(self._image, "RGBA")

    # -- text ---------------------------------------------------------

    def text(
        self,
        x: int,
        y: int,
        text: str,
        size: int = 14,
        color: Tuple[int, int, int] = _TEXT_RGB,
        weight: str = "regular",
        anchor: str = "lt",
        alpha: int = 255,
    ) -> None:
        """Draw antialiased text. Anchor is a PIL anchor (e.g. 'lt', 'lm')."""
        self._draw.text(
            (x, y),
            text,
            font=self._fonts.font(size, weight),
            fill=(*color, alpha),
            anchor=anchor,
        )

    def label(
        self,
        x: int,
        y: int,
        text: str,
        size: int = 14,
        weight: str = "regular",
        color: Tuple[int, int, int] = _TEXT_RGB,
        anchor: str = "lt",
    ) -> None:
        """Draw text without shadow (clean look on white panels)."""
        self.text(x, y, text, size, color, weight, anchor)

    # -- shapes -------------------------------------------------------

    def panel(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        radius: int = 18,
        fill: Tuple[int, int, int] = _PANEL_RGB,
        alpha: int = 230,
        outline: Optional[Tuple[int, int, int]] = _BORDER_RGB,
        outline_alpha: int = 255,
        outline_width: int = 1,
    ) -> None:
        """Draw a translucent white rounded rectangle."""
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        fill_rgba = (*fill, alpha)
        outline_rgba = (
            (*outline, outline_alpha) if outline is not None else None
        )
        self._draw.rounded_rectangle(
            (x, y, x + w, y + h),
            radius=radius,
            fill=fill_rgba,
            outline=outline_rgba,
            width=outline_width,
        )

    def line(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        width: int = 2,
    ) -> None:
        self._draw.line((x1, y1, x2, y2), fill=color, width=width)

    def circle(
        self,
        cx: int,
        cy: int,
        radius: int,
        fill: Tuple[int, int, int],
        alpha: int = 255,
        outline: Optional[Tuple[int, int, int]] = None,
        outline_alpha: int = 255,
        outline_width: int = 1,
    ) -> None:
        """Draw a filled (optionally outlined) circle."""
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        fill_rgba = (*fill, alpha)
        outline_rgba = (
            (*outline, outline_alpha) if outline is not None else None
        )
        self._draw.ellipse(
            bbox, fill=fill_rgba, outline=outline_rgba, width=outline_width,
        )

    # -- text measurement ---------------------------------------------

    def text_width(self, text: str, size: int, weight: str = "regular") -> int:
        return int(
            self._draw.textbbox(
                (0, 0), text, font=self._fonts.font(size, weight)
            )[2]
        )

    # -- output -------------------------------------------------------

    def to_bgr(self) -> np.ndarray:
        return cv2.cvtColor(np.array(self._image), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------
# HUD — minimal monochrome overlay
# ----------------------------------------------------------------------

class HUD:
    """Renders a short, pill-shaped, all-white HUD on camera frames.

    Layout (designed for fullscreen / high resolution):

        ┌──────────────────────────────────────────────────────────────┐
        │  ASSISTIVE VISION   Camera 0 | 1920x1080 | 30.0 FPS   [MODE] │
        │                                                              │
        │                       (live frame)                           │
        │                                                              │
        │  S Screenshot  R Record  Q Quit         <status message>     │
        └──────────────────────────────────────────────────────────────┘

    Every bar is a short white pill with dark text — no accent colours.
    The HUD is a pure presentation layer: it never touches the camera
    or the processing pipeline.
    """

    BAR_HEIGHT = 34
    MARGIN = 10
    RADIUS = 17

    def __init__(self, fps_history_len: int = 30) -> None:
        self._fonts = FontManager()
        self._frame_count = 0
        self._start_time = time.time()
        self._fps_samples: Deque[float] = deque(maxlen=fps_history_len)
        self._last_fps: float = 0.0
        self._toast: Optional[Tuple[str, float, float]] = None
        self._recording: bool = False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def tick(self, fps: float) -> None:
        """Record one rendered frame and its FPS value."""
        self._frame_count += 1
        self._last_fps = fps
        self._fps_samples.append(fps)

    def reset(self) -> None:
        self._frame_count = 0
        self._start_time = time.time()
        self._fps_samples.clear()
        self._last_fps = 0.0
        self._toast = None
        self._recording = False

    def show_toast(self, message: str, duration: float = 2.5) -> None:
        """Display a transient toast notification near the bottom.

        Args:
            message: Text to show (e.g. "Screenshot saved: file.png").
            duration: How long the toast stays visible, in seconds.
        """
        self._toast = (message, time.time(), duration)

    def set_recording(self, active: bool) -> None:
        """Toggle the red 'REC' indicator overlay.

        Args:
            active: True while a recording is in progress.
        """
        self._recording = bool(active)

    @property
    def font_family(self) -> str:
        return self._fonts.family

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        *,
        camera=None,
        mode: str = "LIVE",
        status: str = "",
    ) -> np.ndarray:
        """Draw the HUD onto a copy of the frame.

        Args:
            frame: Input BGR frame.
            camera: Optional object exposing ``camera_id`` and
                ``resolution`` (the src.camera.Camera instance).
            mode: Name of the active processing mode.
            status: Optional transient status message (bottom right).

        Returns:
            A copy of the frame with the HUD rendered on it.
        """
        canvas = Canvas(frame.copy(), self._fonts)
        self._draw_top_bar(canvas, camera=camera, mode=mode)
        self._draw_bottom_bar(canvas, status=status)
        self._draw_recording(canvas)
        self._draw_toast(canvas)
        return canvas.to_bgr()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _draw_top_bar(
        self, canvas: Canvas, camera, mode: str
    ) -> None:
        m = self.MARGIN
        title = "ASSISTIVE VISION"
        title_w = canvas.text_width(title, 15, "semibold")

        cam_label = "N/A"
        res_label = "N/A"
        if camera is not None:
            cam_label = f"Camera {camera.camera_id}"
            if hasattr(camera, "resolution"):
                res_label = f"{camera.resolution[0]}x{camera.resolution[1]}"

        meta = f"{cam_label}  |  {res_label}  |  FPS"
        meta_w = canvas.text_width(meta, 12, "regular")
        fps_w = canvas.text_width(f"{self.avg_fps:.1f}", 12, "semibold")

        chip_text = f"  {mode}  "
        chip_w = canvas.text_width(chip_text, 12, "semibold") + 10

        gap = 24
        pad = 22
        panel_w = pad + title_w + gap + meta_w + 6 + fps_w + gap + chip_w + pad

        cy = m + self.BAR_HEIGHT // 2
        canvas.panel(m, m, panel_w, self.BAR_HEIGHT, radius=self.RADIUS)

        x = m + pad
        canvas.label(x, cy, title, 15, "semibold", _TEXT_RGB, "lm")
        x += title_w + gap
        canvas.label(x, cy, meta, 12, "regular", _DIM_RGB, "lm")
        x += meta_w + 6
        canvas.label(x, cy, f"{self.avg_fps:.1f}", 12, "semibold", _TEXT_RGB, "lm")
        x += fps_w + gap
        canvas.panel(
            x, m + 6, chip_w, self.BAR_HEIGHT - 12,
            radius=(self.BAR_HEIGHT - 12) // 2,
        )
        canvas.label(x + chip_w // 2, cy, chip_text, 12, "semibold",
                     _TEXT_RGB, "mm")

    def _draw_bottom_bar(self, canvas: Canvas, status: str) -> None:
        m = self.MARGIN
        h = canvas.height

        pad = 22
        gap = 28
        hints = [
            ("S", "Screenshot"),
            ("R", "Record"),
            ("Q", "Quit"),
        ]
        content_w = 0
        for key, action in hints:
            content_w += canvas.text_width(key, 13, "semibold")
            content_w += 6 + canvas.text_width(action, 13) + gap
        if status:
            content_w += canvas.text_width(status, 13, "semibold") + 28
        panel_w = pad + content_w + pad

        cy = h - m - self.BAR_HEIGHT // 2
        canvas.panel(m, h - m - self.BAR_HEIGHT, panel_w, self.BAR_HEIGHT,
                     radius=self.RADIUS)

        x = m + pad
        for key, action in hints:
            canvas.label(x, cy, key, 13, "semibold", _TEXT_RGB, "lm")
            x += canvas.text_width(key, 13, "semibold") + 6
            canvas.label(x, cy, action, 13, "regular", _DIM_RGB, "lm")
            x += canvas.text_width(action, 13) + gap

        if status:
            canvas.label(x, cy, status, 13, "semibold", _TEXT_RGB, "lm")

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _draw_recording(self, canvas: Canvas) -> None:
        """Draw a red 'REC' pill below the top bar while recording."""
        if not self._recording:
            return

        text = "REC"
        w = 74
        h = 26
        x = (canvas.width - w) // 2
        y = self.MARGIN + self.BAR_HEIGHT + 12

        canvas.panel(x, y, w, h, radius=h // 2)
        canvas.circle(x + 20, y + h // 2, 5, fill=(255, 60, 60))
        canvas.label(x + 34, y + h // 2, text, 13, "semibold", _TEXT_RGB, "lm")

    def _draw_toast(self, canvas: Canvas) -> None:
        """Draw a transient notification above the bottom bar."""
        if self._toast is None:
            return

        message, t0, duration = self._toast
        elapsed = time.time() - t0
        if elapsed > duration:
            self._toast = None
            return

        fade = max(0.0, 1.0 - (elapsed / duration))
        alpha = int(120 + 120 * fade)

        w = canvas.text_width(message, 14, "semibold") + 36
        h = 34
        x = (canvas.width - w) // 2
        y = canvas.height - self.MARGIN - self.BAR_HEIGHT - 12 - h

        canvas.panel(x, y, w, h, radius=h // 2, alpha=alpha)
        canvas.label(
            x + w // 2, y + h // 2, message, 14, "semibold",
            _TEXT_RGB, "mm",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def avg_fps(self) -> float:
        if not self._fps_samples:
            return 0.0
        return sum(self._fps_samples) / len(self._fps_samples)


def annotate(
    frame: np.ndarray,
    labels: Iterable[Tuple[int, int, str]] = (),
) -> np.ndarray:
    """Draw small labelled markers on a frame (e.g. detected objects)."""
    fonts = FontManager()
    canvas = Canvas(frame.copy(), fonts)
    for x, y, text in labels:
        canvas.line(x - 8, y, x - 2, y, _TEXT_RGB, 2)
        canvas.line(x + 2, y, x + 8, y, _TEXT_RGB, 2)
        canvas.line(x, y - 8, x, y - 2, _TEXT_RGB, 2)
        canvas.line(x, y + 2, x, y + 8, _TEXT_RGB, 2)
        canvas.panel(x + 6, y - 34, canvas.text_width(text, 13, "semibold") + 16, 26,
                     radius=13, alpha=225)
        canvas.label(x + 14, y - 21, text, 13, "semibold", _TEXT_RGB, "lm")
    return canvas.to_bgr()
