"""Camera distance-calibration tool.

Measures and validates the monocular pinhole distance estimator against
ground truth.  Two workflows:

1. Offline CSV mode (no camera):
       python tools/calibrate_camera.py --csv data/calibration.csv
   reads a CSV of (label, box_height_px, frame_height_px, real_distance_m)
   and reports MAE / RMSE / relative error + a fitted vertical FOV.

2. Live mode (needs a camera + a tape measure):
       python tools/calibrate_camera.py --live
   detects objects at a distance you set, prompts you to enter the real
   distance, and appends a measured sample to the CSV.

The fitted FOV and per-class reference heights can then be written into
``configs/assist_config.yaml`` (navigation.vertical_fov / reference_heights).

CSV format:
    label,box_height_px,frame_height_px,real_distance_m
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.navigation.calibration import (  # noqa: E402
    CalibratedDistanceEstimator,
    DistanceSample,
    compute_metrics,
)

_DEFAULT_CSV = PROJECT_ROOT / "data" / "calibration.csv"


def read_csv(path: Path) -> list:
    """Load DistanceSample rows from a CSV file (skips '#' comments)."""
    samples = []
    with open(path, newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.lstrip().startswith("#") is False]
        for row in csv.DictReader(lines):
            samples.append(DistanceSample(
                label=str(row["label"]).strip(),
                box_height_px=float(row["box_height_px"]),
                frame_height_px=float(row["frame_height_px"]),
                real_distance_m=float(row["real_distance_m"]),
                reference_height_m=(
                    float(row["reference_height_m"])
                    if row.get("reference_height_m") else None
                ),
            ))
    return samples


def write_csv(path: Path, samples: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "label", "box_height_px", "frame_height_px",
            "real_distance_m", "reference_height_m",
        ])
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "label": s.label,
                "box_height_px": s.box_height_px,
                "frame_height_px": s.frame_height_px,
                "real_distance_m": s.real_distance_m,
                "reference_height_m": s.reference_height_m or "",
            })


def run_csv(path: Path) -> None:
    samples = read_csv(path)
    if not samples:
        print(f"No samples in {path} — nothing to calibrate.")
        sys.exit(1)
    estimator = CalibratedDistanceEstimator()
    result = estimator.calibrate(samples)
    print(f"Calibration dataset: {path}")
    print(f"  samples            : {result.samples}")
    print(f"  fitted VFOV        : {result.calibrated_vfov_deg:.1f} deg")
    print(f"  MAE                : {result.mae_m:.2f} m")
    print(f"  RMSE               : {result.rmse_m:.2f} m")
    print(f"  mean relative error: {result.relative_error * 100:.1f}%")
    for label, metrics in sorted(result.per_label.items()):
        print(f"  {label:<12} MAE={metrics['mae_m']:.2f}m "
              f"RMSE={metrics['rmse_m']:.2f}m "
              f"rel={metrics['relative_error']*100:.1f}%")
    print("\nCopy into configs/assist_config.yaml:")
    print(f"navigation:\n  vertical_fov: {result.calibrated_vfov_deg:.1f}")


def run_live(csv_path: Path, camera_id: int = 0) -> None:
    """Live calibration: detect an object, ask the user for its real
    distance, record a sample.  Requires a camera and a known-size
    reference (reference heights come from guidance defaults).
    """
    import cv2

    from src.camera import Camera
    from src.detection import YoloDetector

    samples = read_csv(csv_path) if csv_path.exists() else []

    cam = Camera(camera_id=camera_id, resolution=(1280, 720))
    cam.start()
    detector = YoloDetector("models/yolov8n.onnx", conf_threshold=0.5)
    window = "Calibration — press SPACE to record, Q to quit"
    cv2.namedWindow(window)

    print("Point the camera at an object at a known distance.")
    print("Press SPACE to record the currently detected object, Q to quit.")
    try:
        while True:
            frame = cam.read()
            detections = detector.detect(frame)
            for d in detections:
                x, y, w, h = d.box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{d.label} {d.confidence:.2f}",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and detections:
                d = max(detections, key=lambda x: x.area)
                real = input(
                    f"Real distance to '{d.label}' (metres): "
                ).strip()
                try:
                    real_m = float(real)
                except ValueError:
                    print("Ignored — not a number.")
                    continue
                samples.append(DistanceSample(
                    label=d.label,
                    box_height_px=float(d.box[3]),
                    frame_height_px=float(frame.shape[0]),
                    real_distance_m=real_m,
                ))
                write_csv(csv_path, samples)
                print(f"Recorded {d.label} @ {real_m} m "
                      f"(box_h={d.box[3]}px) -> {csv_path}")
    finally:
        cv2.destroyAllWindows()
        cam.stop()

    if samples:
        run_csv(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(_DEFAULT_CSV))
    parser.add_argument("--live", action="store_true",
                        help="Live calibration with a camera")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.csv)
    if args.live:
        run_live(path, camera_id=args.camera)
    else:
        run_csv(path)


if __name__ == "__main__":
    main()