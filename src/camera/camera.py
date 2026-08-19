import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.exceptions import (
    CameraAccessError,
    CameraNotFoundError,
    FrameGrabError,
    InvalidResolutionError,
)
from src.utils.logger import setup_logger


class Camera:
    """Core camera class for webcam initialization and frame acquisition.

    Wraps OpenCV's VideoCapture with a clean interface, FPS tracking,
    and safe resource management via context manager support.

    Usage:
        with Camera(camera_id=0, resolution=(640, 480)) as cam:
            while True:
                frame = cam.read()
                cv2.imshow("Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    """

    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        backend: int = cv2.CAP_DSHOW,
        mirror: bool = False,
        rotate: int = 0,
    ) -> None:
        """Configure camera parameters without starting the capture.

        The camera now captures RAW sensor frames: ``mirror`` defaults to
        False so the vision pipeline (YOLO/OCR) always receives text in
        its true reading orientation.  Front-camera preview mirroring is
        a *display* concern and is applied by the app layer, never here.

        Args:
            camera_id: Device index (0 for built-in, 1+ for external).
            resolution: Desired (width, height) in pixels.
            fps: Target frames per second.
            backend: OpenCV backend (default CAP_DSHOW for Windows).
            mirror: Horizontally flip every frame (mirror mode).  Kept
                for callers that want a mirrored feed; the assistive app
                leaves it off so OCR sees real text orientation.
            rotate: Physical sensor orientation in degrees (0/90/180/270)
                applied to every frame exactly once, so portrait-mounted
                cameras produce upright vision frames.
        """
        self._camera_id = camera_id
        self._resolution = resolution
        self._target_fps = fps
        self._backend = backend
        self._mirror = mirror
        self._rotate = rotate % 360
        self._rotate_code = self._rotation_code(self._rotate)
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._actual_fps: float = 0.0
        self._frame_count: int = 0
        self._fps_start_time: Optional[float] = None
        self._logger = setup_logger(f"Camera[{camera_id}]")

    @staticmethod
    def _rotation_code(degrees: int) -> int:
        if degrees % 360 in (90, 270):
            return {
                90: cv2.ROTATE_90_CLOCKWISE,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }[degrees % 360]
        if degrees % 360 == 180:
            return cv2.ROTATE_180
        return -1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @property
    def camera_id(self) -> int:
        return self._camera_id

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def mirror(self) -> bool:
        """Whether the feed is horizontally flipped (mirror mode)."""
        return self._mirror

    @property
    def rotate(self) -> int:
        """Physical sensor rotation applied to every frame (degrees)."""
        return self._rotate

    def set_mirror(self, enabled: bool) -> None:
        """Toggle mirror mode at runtime.

        Args:
            enabled: True to horizontally flip frames.
        """
        self._mirror = bool(enabled)
        self._logger.info("Mirror mode %s", "enabled" if self._mirror else "disabled")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize the camera and begin capturing frames.

        Raises:
            CameraNotFoundError: If the camera cannot be opened.
        """
        self._cap = cv2.VideoCapture(self._camera_id, self._backend)

        if not self._cap or not self._cap.isOpened():
            raise CameraNotFoundError(
                f"Camera {self._camera_id} not found or cannot be opened."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self._rotate_code in (cv2.ROTATE_90_CLOCKWISE,
                                 cv2.ROTATE_90_COUNTERCLOCKWISE):
            actual_w, actual_h = actual_h, actual_w
        self._resolution = (actual_w, actual_h)

        self._is_running = True
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._logger.info(
            "Camera started | id=%d res=%dx%d",
            self._camera_id, actual_w, actual_h,
        )

    def read(self) -> np.ndarray:
        """Grab the next frame from the camera.

        Returns:
            Frame as a BGR numpy array.

        Raises:
            CameraAccessError: If camera is not running.
            FrameGrabError: If reading the frame fails.
        """
        if not self._is_running or self._cap is None:
            raise CameraAccessError(
                "Camera is not running. Call start() first."
            )

        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise FrameGrabError(
                f"Failed to grab frame from camera {self._camera_id}."
            )

        self._frame_count += 1

        # Geometry is corrected ONCE here, for every consumer (YOLO,
        # tracking, OCR): mirror (opt-in) then physical sensor rotation.
        if self._mirror:
            frame = cv2.flip(frame, 1)
        if self._rotate_code >= 0:
            frame = cv2.rotate(frame, self._rotate_code)

        if self._frame_count % 30 == 0:
            elapsed = time.time() - self._fps_start_time
            if elapsed > 0:
                self._actual_fps = self._frame_count / elapsed

        return frame

    def stop(self) -> None:
        """Release the camera and clean up resources."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_running = False
        elapsed = (
            time.time() - self._fps_start_time
            if self._fps_start_time
            else 0.0
        )
        self._logger.info(
            "Camera stopped | frames=%d duration=%.2fs",
            self._frame_count, elapsed,
        )

    def set_resolution(self, width: int, height: int) -> None:
        """Change the camera resolution at runtime.

        Args:
            width: Desired frame width in pixels.
            height: Desired frame height in pixels.

        Raises:
            CameraAccessError: If camera is not initialized.
            InvalidResolutionError: If width or height are non-positive.
        """
        if width <= 0 or height <= 0:
            raise InvalidResolutionError(
                f"Invalid resolution: {width}x{height}. "
                "Width and height must be positive."
            )
        if self._cap is None:
            raise CameraAccessError("Camera is not initialized.")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution = (actual_w, actual_h)
        self._logger.info("Resolution changed to %dx%d", actual_w, actual_h)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, *exc_args) -> bool:
        self.stop()
        return False

    def __del__(self) -> None:
        self.stop()
