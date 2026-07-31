import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.exceptions import RecordingError
from src.utils.logger import setup_logger

_logger = setup_logger("CameraUtils")


def take_screenshot(
    frame: np.ndarray,
    save_dir: str = "assets/screenshots",
) -> str:
    """Save a single frame as a timestamped PNG screenshot.

    Args:
        frame: BGR image array to save.
        save_dir: Directory where the screenshot will be stored.

    Returns:
        Absolute path to the saved screenshot file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    full_path = str(save_path / filename)
    cv2.imwrite(full_path, frame)
    _logger.info("Screenshot saved: %s", full_path)
    return full_path


def record_video(
    camera,
    output_dir: str = "assets/sample_videos",
    duration: int = 10,
    fps: float = 20.0,
) -> str:
    """Record a video from the camera for a fixed duration.

    Args:
        camera: A Camera instance that is already running.
        output_dir: Directory where the video file will be saved.
        duration: Recording length in seconds.
        fps: Frames per second for the output video.

    Returns:
        Absolute path to the saved video file.

    Raises:
        RecordingError: If the VideoWriter cannot be created.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.avi"
    full_path = str(out_path / filename)

    width, height = camera.resolution
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(full_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RecordingError(f"Could not create video writer at {full_path}")

    start = time.time()
    frame_count = 0
    while time.time() - start < duration:
        try:
            frame = camera.read()
            writer.write(frame)
            frame_count += 1
        except Exception:
            _logger.warning("Frame dropped during recording", exc_info=True)

    writer.release()
    _logger.info("Recording saved: %s (%d frames, %ds)", full_path, frame_count, duration)
    return full_path


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay the current FPS value on the top-left corner.

    Args:
        frame: Input BGR image.
        fps: Frames-per-second value to display.

    Returns:
        Copy of the frame with FPS text rendered on it.
    """
    display = frame.copy()
    cv2.putText(
        display,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    return display


def draw_timestamp(frame: np.ndarray) -> np.ndarray:
    """Overlay the current system timestamp on the frame.

    Args:
        frame: Input BGR image.

    Returns:
        Copy of the frame with timestamp text rendered on it.
    """
    display = frame.copy()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        display,
        timestamp,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return display


def show_feed(
    camera,
    window_name: str = "Camera Feed",
    process_frame: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    on_key: Optional[Callable[[int], None]] = None,
) -> None:
    """Open a window and display the live camera feed.

    The loop runs until 'q' is pressed or the camera stops.
    An optional ``process_frame`` callback can transform each frame
    before display. An optional ``on_key`` callback receives the
    waitKey result for custom key bindings.

    Args:
        camera: A running Camera instance.
        window_name: Title of the display window.
        process_frame: Callable that receives a BGR frame and returns
            a (possibly modified) BGR frame.  If None, a no-op overlay
            of FPS and timestamp is applied.
        on_key: Callable that receives the key code (int) each frame.
    """
    cv2.namedWindow(window_name)

    try:
        while camera.is_running:
            frame = camera.read()

            if process_frame is not None:
                frame = process_frame(frame)
            else:
                frame = draw_fps(frame, camera.actual_fps)
                frame = draw_timestamp(frame)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if on_key is not None:
                on_key(key)
            if key == ord("q"):
                _logger.info("Feed terminated by user (q key).")
                break
    except KeyboardInterrupt:
        _logger.info("Feed interrupted.")
    finally:
        cv2.destroyWindow(window_name)


# ----------------------------------------------------------------------
# Display helpers (fullscreen, scaling, resolution selection)
# ----------------------------------------------------------------------

DEFAULT_RESOLUTIONS: List[Tuple[int, int]] = [
    (1920, 1080),
    (1280, 720),
    (640, 480),
    (320, 240),
]


def get_screen_size() -> Tuple[int, int]:
    """Return the primary display resolution in pixels.

    Uses Win32 API on Windows and falls back to 1920x1080 elsewhere.
    """
    try:
        import ctypes

        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def open_fullscreen_window(window_name: str) -> None:
    """Open a borderless fullscreen display window.

    Args:
        window_name: Title used to identify the window.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )


def scale_to_fit(
    frame: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Scale a frame to fit within (width, height) while keeping aspect
    ratio, centred on a black canvas.

    Args:
        frame: Input BGR image.
        width: Target canvas width in pixels.
        height: Target canvas height in pixels.

    Returns:
        Canvas of (height, width) with the frame letterboxed inside.
    """
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_CUBIC
    )
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def auto_select_resolution(
    camera,
    preferred: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[int, int]:
    """Select the best supported resolution for a running camera.

    Tries each preferred resolution in order and keeps the first that
    the camera accepts at >= 90% of the requested size.

    Args:
        camera: A started Camera instance.
        preferred: Ordered list of (width, height) to try.

    Returns:
        The (width, height) actually selected.
    """
    candidates = preferred or DEFAULT_RESOLUTIONS
    for width, height in candidates:
        camera.set_resolution(width, height)
        actual_w, actual_h = camera.resolution
        if actual_w >= width * 0.9 and actual_h >= height * 0.9:
            _logger.info("Selected resolution %dx%d", actual_w, actual_h)
            return actual_w, actual_h
    _logger.warning(
        "No preferred resolution matched; keeping %dx%d",
        *camera.resolution,
    )
    return camera.resolution


class VideoRecorder:
    """Records camera frames to a video file in a background thread.

    While recording, the latest captured frame is exposed via
    ``latest_frame`` so the UI can keep displaying the feed without
    blocking the main loop.

    Usage:
        recorder = VideoRecorder(camera, duration=5)
        recorder.start()
        ...
        while recorder.is_recording:
            frame = recorder.latest_frame   # display this
            cv2.imshow("Feed", frame)
        path = recorder.saved_path
    """

    def __init__(
        self,
        camera,
        output_dir: str = "assets/sample_videos",
        duration: int = 5,
        fps: float = 20.0,
    ) -> None:
        """Configure the recorder.

        Args:
            camera: A running Camera instance (source of frames).
            output_dir: Directory for the output video file.
            duration: Recording length in seconds.
            fps: Frames per second for the output video.
        """
        self._camera = camera
        self._output_dir = output_dir
        self._duration = duration
        self._fps = fps
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[np.ndarray] = None
        self._saved_path: Optional[str] = None
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        return self._latest

    @property
    def saved_path(self) -> Optional[str]:
        return self._saved_path

    def start(self) -> str:
        """Begin recording in a background thread.

        Returns:
            Path where the video file will be written.

        Raises:
            RecordingError: If a recording is already in progress.
        """
        if self._is_recording:
            raise RecordingError("A recording is already in progress.")

        out_path = Path(self._output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.avi"
        full_path = str(out_path / filename)

        self._saved_path = full_path
        self._is_recording = True
        self._latest = None
        self._thread = threading.Thread(
            target=self._run, name="video-recorder", daemon=True,
        )
        self._thread.start()
        _logger.info("Recording started -> %s", full_path)
        return full_path

    def stop(self) -> None:
        """Stop recording early and wait for the writer to flush."""
        self._is_recording = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        width, height = self._camera.resolution
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(
            self._saved_path, fourcc, self._fps, (width, height)
        )

        start = time.time()
        try:
            while self._is_recording and time.time() - start < self._duration:
                try:
                    frame = self._camera.read()
                    writer.write(frame)
                    self._latest = frame
                except Exception:
                    _logger.warning(
                        "Frame dropped during recording", exc_info=True,
                    )
        finally:
            writer.release()
            self._is_recording = False
            _logger.info("Recording saved: %s", self._saved_path)
