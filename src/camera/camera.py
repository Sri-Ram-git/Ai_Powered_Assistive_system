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
    ) -> None:
        """Configure camera parameters without starting the capture.

        Args:
            camera_id: Device index (0 for built-in, 1+ for external).
            resolution: Desired (width, height) in pixels.
            fps: Target frames per second.
            backend: OpenCV backend (default CAP_DSHOW for Windows).
        """
        self._camera_id = camera_id
        self._resolution = resolution
        self._target_fps = fps
        self._backend = backend
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._actual_fps: float = 0.0
        self._frame_count: int = 0
        self._fps_start_time: Optional[float] = None
        self._logger = setup_logger(f"Camera[{camera_id}]")

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

        if self._frame_count % 30 == 0:
            elapsed = time.time() - self._fps_start_time
            if elapsed > 0:
                self._actual_fps = self._frame_count / elapsed

        return frame

    def stop(self) -> None:
        """Release the camera and clean up resources."""
        if self._cap is not None:
            self._cap.release()
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
        if self._cap is None:
            raise CameraAccessError("Camera is not initialized.")
        if width <= 0 or height <= 0:
            raise InvalidResolutionError(
                f"Invalid resolution: {width}x{height}. "
                "Width and height must be positive."
            )

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
