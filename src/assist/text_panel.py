"""Side text panel for the assistive vision app.

Sits on the right edge of the display while the camera feed stays on the
left, and shows:

    * the most recently recognised text (from the highest-priority
      text-bearing object — a book, bottle label, laptop screen...);
    * the recognition history (newest first), bounded by config;
    * clickable buttons: READ ALOUD, READ NOW, COPY, CLEAR;
    * a status footer with the OCR latency / variant / pending state
      (plus worker counters in debug mode).

The panel is pure presentation: it reads the pipeline's track-OCR store
through simple accessors and reports button hits; the app owns the
action dispatch (speech, clipboard, clearing).

Mouse handling: ``render()`` records the button rectangles each frame,
and ``hit_test(x, y)`` maps a display-space click to a button name.  The
app wires this to an OpenCV mouse callback; ``on_motion(x, y)`` drives a
hover highlight.
"""
from typing import Dict, List, Optional

import numpy as np

from src.camera.hud import Canvas, FontManager

# Panel colours (BGR panel, RGB text — matches the monochrome HUD).
_PANEL_RGB = (12, 12, 14)
_BORDER_RGB = (70, 70, 76)
_TEXT_RGB = (255, 255, 255)
_DIM_RGB = (190, 190, 198)
_ACCENT_RGB = (150, 210, 255)

_BUTTON_DEFS = (
    ("read", "READ"),
    ("now", "NOW"),
    ("copy", "COPY"),
    ("clear", "CLEAR"),
)


class TextPanel:
    """Right-edge text panel with clickable buttons."""

    def __init__(self, panel_frac: float = 0.28, pad: int = 14) -> None:
        """Configure the panel.

        Args:
            panel_frac: Fraction of display width taken by the panel
                (camera keeps the remaining ~72%).
            pad: Inner padding in pixels.
        """
        self.panel_frac = float(panel_frac)
        self.pad = int(pad)
        self._fonts = FontManager()
        self._buttons: Dict[str, tuple] = {}
        self._hover: Optional[str] = None

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def hit_test(self, x: int, y: int) -> Optional[str]:
        """Map a display-space point to a button name (or None)."""
        for name, (bx, by, bw, bh) in self._buttons.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return name
        return None

    def on_motion(self, x: int, y: int) -> None:
        """Update the hover highlight from a mouse-move event."""
        hover = self.hit_test(x, y)
        if hover != self._hover:
            self._hover = hover

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        *,
        latest: Optional[Dict] = None,
        history: Optional[List[Dict]] = None,
        stats: Optional[Dict] = None,
        busy: bool = False,
        debug: bool = False,
    ) -> np.ndarray:
        """Draw the panel onto the right edge of ``frame``.

        Args:
            frame: Full display BGR frame (screen-sized).
            latest: ``pipe.latest_track_ocr()`` (may be None).
            history: ``pipe.track_ocr_history()`` list of dicts.
            stats: ``pipe.ocr_stats()`` worker counters (debug only).
            busy: Whether the OCR worker is currently processing.
            debug: Show worker counters under the status footer.

        Returns:
            The frame with the panel rendered (buttons recorded for
            hit-testing).
        """
        canvas = Canvas(frame.copy(), self._fonts)
        w, h = canvas.width, canvas.height

        x0 = max(self.pad, int(w * (1.0 - self.panel_frac)))
        pw = w - x0 - self.pad
        if pw < 120:
            return frame  # display too narrow for a panel

        self._buttons = {}

        # Panel background.
        canvas.panel(x0, self.pad, pw, h - 2 * self.pad,
                     radius=18, fill=_PANEL_RGB, outline=_BORDER_RGB)

        cursor = x0 + self.pad
        top = self.pad + self.pad

        # Header.
        canvas.label(cursor, top, "TEXT", 16, "semibold", _TEXT_RGB)
        top += 24

        # Buttons.
        btn_w = (pw - 2 * self.pad - 3 * 6) // 4
        btn_h = 30
        for i, (name, label) in enumerate(_BUTTON_DEFS):
            bx = x0 + self.pad + i * (btn_w + 6)
            by = top
            hover = (name == self._hover)
            canvas.panel(bx, by, btn_w, btn_h, radius=8,
                         fill=(60, 70, 84) if hover else (38, 40, 46),
                         outline=(150, 210, 255) if hover else _BORDER_RGB)
            canvas.label(bx + btn_w // 2, by + btn_h // 2, label, 11,
                         "semibold", _ACCENT_RGB if hover else _TEXT_RGB, "mm")
            self._buttons[name] = (bx, by, btn_w, btn_h)
        top += btn_h + 14

        # Latest recognised text.
        canvas.label(cursor, top, "RECOGNISED", 11, "semibold", _DIM_RGB)
        top += 18
        if latest and latest.get("text"):
            label = latest.get("label") or "object"
            canvas.label(cursor, top, f"#{latest.get('track_id')} {label}",
                         11, "regular", _DIM_RGB)
            top += 16
            for line in self._wrap(canvas, latest["text"], pw - 2 * self.pad,
                                   size=15):
                canvas.label(cursor, top, line, 15, "semibold", _TEXT_RGB)
                top += 20
            conf = latest.get("confidence", 0.0)
            variant = latest.get("variant", "")
            meta = f"conf {conf:.2f}" + (f" | {variant}" if variant else "")
            canvas.label(cursor, top, meta, 10, "regular", _DIM_RGB)
            top += 18
        else:
            canvas.label(cursor, top, "No text recognised yet -", 12,
                         "regular", _DIM_RGB)
            canvas.label(cursor, top + 16,
                         "look at a book or bottle with a label.", 12,
                         "regular", _DIM_RGB)
            top += 34
        top += 8

        # History.
        canvas.line(x0 + self.pad, top, x0 + pw - self.pad, top,
                    _BORDER_RGB, 1)
        top += 10
        canvas.label(cursor, top, "HISTORY", 11, "semibold", _DIM_RGB)
        top += 18
        for entry in (history or [])[:self._history_lines(
                h, top, 20)]:
            entry_label = entry.get("label") or "object"
            header = f"#{entry.get('track_id')} {entry_label}"
            canvas.label(cursor, top, header, 10, "regular", _DIM_RGB)
            top += 13
            for line in self._wrap(canvas, entry.get("text", ""),
                                   pw - 2 * self.pad, size=12):
                canvas.label(cursor, top, line, 12, "regular", _TEXT_RGB)
                top += 15
            top += 4

        # Status footer (pinned to the bottom of the panel).
        fy = h - self.pad - 30
        dot = (120, 200, 120) if busy else _BORDER_RGB
        canvas.circle(x0 + self.pad + 4, fy, 4, fill=dot)
        status = "reading..." if busy else "idle"
        latency = ""
        if latest:
            latency = f"{latest.get('latency_ms', 0.0):.0f} ms"
        canvas.label(x0 + self.pad + 16, fy,
                     f"OCR {status}" + (f"  |  {latency}" if latency else ""),
                     10, "regular", _DIM_RGB)
        if debug and stats:
            extra = (f"runs {stats.get('runs', 0)}"
                     f"  no-text {stats.get('no_text', 0)}"
                     f"  timeouts {stats.get('timeouts', 0)}"
                     f"  replaced {stats.get('replaced', 0)}"
                     f"  pending {int(stats.get('pending', False))}")
            canvas.label(x0 + self.pad, fy - 16, extra, 10, "regular",
                         _DIM_RGB)

        return canvas.to_bgr()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wrap(self, canvas: Canvas, text: str, max_w: int,
              size: int = 12) -> List[str]:
        """Wrap text to fit the panel width (simple word wrap)."""
        if not text:
            return [""]
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if canvas.text_width(candidate, size) > max_w and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    def _history_lines(self, h: int, top: int, line_h: int) -> int:
        available = h - self.pad - 60 - top
        return max(1, int(available / line_h))