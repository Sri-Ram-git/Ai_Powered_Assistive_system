from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2

from src.utils.logger import setup_logger


@dataclass
class CameraInfo:
    """Describes a detected camera device."""

    id: int
    name: str = ""
    resolution: Optional[Tuple[int, int]] = None
    backend: str = "DShow"


class CameraManager:
    """Discovers, lists, and selects available camera devices.

    Usage:
        mgr = CameraManager()
        cameras = mgr.list_cameras()
        cam_id = mgr.select_camera(0)
    """

    COMMON_RESOLUTIONS: List[Tuple[int, int]] = [
        (1920, 1080),
        (1280, 720),
        (640, 480),
        (320, 240),
    ]

    def __init__(self) -> None:
        self._logger = setup_logger("CameraManager")
        self._cameras: List[CameraInfo] = []

    def list_cameras(self, max_cameras: int = 10) -> List[CameraInfo]:
        """Enumerate available cameras by probing device indices.

        Each index from 0 to max_cameras - 1 is tested; only
        successfully opened devices are reported.

        Args:
            max_cameras: Maximum number of indices to probe.

        Returns:
            List of CameraInfo for each detected camera.
        """
        self._cameras.clear()

        for idx in range(max_cameras):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                res = (w, h) if w > 0 and h > 0 else None
                self._cameras.append(
                    CameraInfo(id=idx, name=f"Camera {idx}", resolution=res)
                )
                cap.release()

        self._logger.info("Found %d camera(s)", len(self._cameras))
        return list(self._cameras)

    @property
    def available_cameras(self) -> List[CameraInfo]:
        """Return the last enumerated list of cameras."""
        return list(self._cameras)

    def select_camera(self, camera_id: Optional[int] = None) -> int:
        """Select a camera ID.

        If camera_id is None, returns the first available camera.
        If camera_id is specified, validates it exists.

        Args:
            camera_id: Desired camera index, or None for auto-select.

        Returns:
            Validated camera device index.

        Raises:
            RuntimeError: If no cameras are available.
            ValueError: If the requested camera_id is not found.
        """
        if not self._cameras:
            self.list_cameras()
        if not self._cameras:
            raise RuntimeError("No cameras are available on this system.")

        if camera_id is None:
            return self._cameras[0].id

        for cam in self._cameras:
            if cam.id == camera_id:
                return cam.id

        raise ValueError(
            f"Camera {camera_id} not found. "
            f"Available: {[c.id for c in self._cameras]}"
        )
