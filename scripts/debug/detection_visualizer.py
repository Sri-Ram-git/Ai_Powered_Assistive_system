"""Detection diagnostic visualiser (Phase 3).

Shows, side by side, exactly what the model sees and what it decides:

    left   original frame
    right  the letterboxed 640x640 model input (as BGR for display)

Detections are drawn on the original frame with label, confidence, box,
centre dot and area, so a missing object can be traced to either
preprocessing (is it in the model input at all?) or postprocessing
(confidence / NMS / coordinate scaling).

Usage:
    python scripts/debug/detection_visualizer.py --camera 0            # live
    python scripts/debug/detection_visualizer.py --image path.png      # one file
    python scripts/debug/detection_visualizer.py --image a.png b.png   # several

Keys (live mode):
    s   save the current annotated view to assets/debug/
    q   quit
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.detection import YoloDetector  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("DetectionVisualizer")
SAVE_DIR = PROJECT_ROOT / "assets" / "debug"


def _make_side_by_side(frame, blob_input_bgr):
    """Concatenate original + model input for one diagnostic window."""
    h, w = frame.shape[:2]
    resized = cv2.resize(blob_input_bgr, (w, h),
                         interpolation=cv2.INTER_NEAREST)
    return np.hstack([frame, resized])


def _annotate(frame, detections):
    out = frame.copy()
    for d in detections:
        x, y, w, h = d.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = d.center
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        text = (f"{d.label} {d.confidence:.2f} "
                f"area={int(d.area)}")
        label_y = y - 8 if y - 8 > 10 else y + h + 18
        cv2.putText(out, text, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return out


def _model_input_bgr(detector, frame):
    """Re-derive the letterboxed model input as a BGR display image."""
    blob, _, pad_x, pad_y = detector._letterbox(frame)
    # blob is NCHW float [0,1]; pull out the 640x640 RGB image.
    rgb = np.transpose(blob[0], (1, 2, 0))
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _inspect(detector, frame):
    """Return (annotated_view, model_input_view, detections)."""
    detections = detector.detect(frame)
    model_view = _model_input_bgr(detector, frame)
    annotated = _annotate(frame, detections)
    side = _make_side_by_side(annotated, model_view)
    return side, detections


def run_live(camera_id: int, model_path: str, conf: float) -> None:
    from src.camera import Camera

    detector = YoloDetector(model_path, conf_threshold=conf)
    window = "Detection visualiser (left: frame | right: model input)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    with Camera(camera_id=camera_id, resolution=(1280, 720)) as cam:
        print(f"Camera {camera_id} | {cam.resolution}")
        while True:
            frame = cam.read()
            t0 = time.monotonic()
            view, detections = _inspect(detector, frame)
            ms = (time.monotonic() - t0) * 1000.0
            cv2.putText(view, f"infer {ms:.0f}ms  {len(detections)} det",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            cv2.imshow(window, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                SAVE_DIR.mkdir(parents=True, exist_ok=True)
                path = SAVE_DIR / f"vis_{int(time.time())}.png"
                cv2.imwrite(str(path), view)
                print(f"Saved {path}")
    cv2.destroyAllWindows()


def run_images(paths, model_path: str, conf: float,
               save_dir: "Path | None" = None) -> None:
    detector = YoloDetector(model_path, conf_threshold=conf)
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"SKIP (unreadable): {path}")
            continue
        view, detections = _inspect(detector, img)
        print(f"\n{path}: {len(detections)} detection(s)")
        for d in detections:
            print(f"  {d.label:<12} conf={d.confidence:.2f} "
                  f"box={d.box} center={tuple(round(v,1) for v in d.center)} "
                  f"area={int(d.area)}")
        if save_dir is not None:
            out = Path(save_dir) / f"{Path(path).stem}_vis.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), view)
            print(f"  -> saved {out}")
        else:
            cv2.imshow("Detection visualiser", view)
            cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                        help="Live camera index (e.g. 0)")
    parser.add_argument("--image", nargs="+", default=None,
                        help="Image file(s) to inspect")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "yolov8s.onnx"))
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--save-dir", default=None,
                        help="Write annotated images here instead of showing")
    args = parser.parse_args()

    model_path = str(Path(args.model))
    if not Path(model_path).is_absolute():
        model_path = str(PROJECT_ROOT / model_path)

    if args.camera is not None:
        run_live(args.camera, model_path, args.conf)
    elif args.image:
        run_images([Path(p) for p in args.image], model_path, args.conf,
                   save_dir=args.save_dir)
    else:
        parser.error("provide --camera or --image")


if __name__ == "__main__":
    main()