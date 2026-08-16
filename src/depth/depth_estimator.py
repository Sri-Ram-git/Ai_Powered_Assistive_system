"""Depth estimator backends.

The base ``DepthEstimator`` extracts per-box depth from a depth map.  Two
backends are provided:

* ``SyntheticDepthBackend`` — no model.  Produces a normalised depth map
  where depth increases with image height (a plausible indoor
  approximation), used for tests and as an offline fallback.
* ``DepthAnythingV2Backend`` — wraps a Depth Anything V2 ONNX export.
  Optional: instantiated only when the model file exists.

``create_depth_estimator`` picks a backend from configuration.  The
whole module is optional and never required by the core pipeline.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.depth.depth_result import DepthResult
from src.utils.logger import setup_logger

_logger = setup_logger("DepthEstimator")


class DepthEstimator:
    """Base class: runs a backend and extracts per-box depth."""

    backend_name: str = "base"

    def estimate(
        self,
        frame: np.ndarray,
        boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> DepthResult:
        """Produce a DepthResult for a frame.

        Args:
            frame: BGR image (HxWx3 or HxW).
            boxes: Optional list of (x, y, w, h) boxes to extract
                per-box depth for.  If None, no per-box depths set.

        Returns:
            DepthResult with a normalised depth map and per-box depths.
        """
        depth_map = self._infer(frame)
        per_box: dict = {}
        result = DepthResult(map=depth_map, per_box_depth=per_box,
                             backend=self.backend_name)
        if boxes:
            for i, box in enumerate(boxes):
                d = result.box_depth(box)
                if d is not None:
                    per_box[i] = d
        return result

    def _infer(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover


class SyntheticDepthBackend(DepthEstimator):
    """Model-free depth: depth grows with vertical position.

    Deterministic and fast; useful for tests, offline fallback, and as a
    baseline to compare a real depth model against.
    """

    backend_name = "synthetic"

    def _infer(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        # Row -> normalised depth (near at top of frame).
        depth = np.linspace(0.0, 1.0, h, dtype=np.float32)
        return np.tile(depth[:, None], (1, w))


class DepthAnythingV2Backend(DepthEstimator):
    """Depth Anything V2 via ONNX (optional, model file must exist).

    The model is downloaded separately (never committed to git) and its
    metadata recorded in ``models/manifest.yaml``.  The default ONNX
    export from the official repo expects a 518x518 input and outputs a
    relative depth map; we resize the frame, run the net, and resize the
    map back.

    Only instantiate when the model file exists — importing this class
    must not require onnxruntime.
    """

    backend_name = "depth_anything_v2"

    def __init__(self, model_path: str, input_size: int = 518) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Depth model not found: {model_path}")
        import cv2
        import onnxruntime as ort

        self._input_size = int(input_size)
        self._session = ort.InferenceSession(str(path),
                                             providers=["CPUExecutionProvider"])
        self._cv2 = cv2
        _logger.info("DepthAnythingV2 loaded (%s)", path.name)

    def _infer(self, frame: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        size = self._input_size
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (size, size))
        blob = cv2.dnn.blobFromImage(resized, 1.0 / 255.0, (size, size),
                                     swapRB=True, crop=False)
        blob = np.transpose(blob, (0, 2, 3, 1))
        inputs = {self._session.get_inputs()[0].name: blob}
        out = self._session.run(None, inputs)[0]
        depth = np.squeeze(out)
        # Normalise to [0, 1].
        dmin, dmax = float(depth.min()), float(depth.max())
        if dmax - dmin > 1e-6:
            depth = (depth - dmin) / (dmax - dmin)
        return cv2.resize(depth.astype(np.float32), (w, h),
                          interpolation=cv2.INTER_LINEAR)


def create_depth_estimator(
    backend: str = "synthetic",
    model_path: Optional[str] = None,
    input_size: int = 518,
) -> DepthEstimator:
    """Create a depth estimator from configuration.

    Args:
        backend: "synthetic" (default, no model) or "depth_anything_v2".
        model_path: Path to the ONNX model (required for V2).
        input_size: Model input size (V2 default 518).

    Returns:
        A DepthEstimator instance.

    Raises:
        ValueError: For an unknown backend.
        FileNotFoundError: For V2 without an existing model file.
    """
    if backend == "synthetic":
        return SyntheticDepthBackend()
    if backend == "depth_anything_v2":
        if not model_path:
            raise ValueError("depth_anything_v2 requires model_path")
        return DepthAnythingV2Backend(model_path, input_size=input_size)
    raise ValueError(f"Unknown depth backend: {backend}")