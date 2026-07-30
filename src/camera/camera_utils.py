import time
from pathlib import Path
from typing import Callable, Optional

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
