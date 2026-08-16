"""Detection quality metrics (Phases 14-15).

Computes precision / recall / F1 / FP / FN and per-class average
precision from real detections vs. ground-truth labels in
``evaluation/object_detection/``.

Structure (create images + labels as you collect real scenes):

    evaluation/object_detection/
        images/            *.jpg (raw camera frames)
        annotations/       *.json  [{ "image": "x.jpg",
                                      "objects": [ { "label": "person",
                                                     "box": [x, y, w, h] } ] }]
        results/           metrics.json (written here)

A detection matches a ground-truth box of the same class when IoU >= 0.5.
AP is computed by ranking detections by confidence and integrating the
precision/recall curve (11-point).  If no ground truth exists the tool
reports that honestly instead of inventing numbers.

Usage:
    python scripts/benchmark/object_detection_metrics.py [--conf 0.40]
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from src.detection import YoloDetector  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "evaluation" / "object_detection"
IMAGES_DIR = EVAL_DIR / "images"
ANNOT_DIR = EVAL_DIR / "annotations"
RESULTS_DIR = EVAL_DIR / "results"


def _iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax1 + aw, bx1 + bw)
    inter_y2 = min(ay1 + ah, by1 + bh)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    a_area = aw * ah
    b_area = bw * bh
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _load_annotations() -> Dict[str, List[Dict]]:
    """Map image filename -> list of {label, box} ground truths."""
    gt: Dict[str, List[Dict]] = {}
    if not ANNOT_DIR.exists():
        return gt
    for path in sorted(ANNOT_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            items = json.load(fh)
        for item in items:
            gt.setdefault(item["image"], []).extend(
                {"label": o["label"], "box": tuple(o["box"])}
                for o in item["objects"])
    return gt


def _evaluate(image_path: Path, gt_boxes: List[Dict],
              detector: YoloDetector,
              conf: float) -> Tuple[List[Dict], int]:
    """Return (detections ranked by conf, number of GT boxes)."""
    image = cv2.imread(str(image_path))
    if image is None:
        return [], len(gt_boxes)
    dets = detector.detect(image)
    return [
        {"label": d.label, "confidence": d.confidence, "box": d.box}
        for d in dets
    ], len(gt_boxes)


def _match_and_score(dets: List[Dict], gts: List[Dict],
                     cls: str) -> Dict:
    """Count TP/FP/FN + AP for one class at IoU 0.5."""
    gt_boxes = [g for g in gts if g["label"] == cls]
    cls_dets = [d for d in dets if d["label"] == cls]
    cls_dets.sort(key=lambda d: d["confidence"], reverse=True)

    used = [False] * len(gt_boxes)
    matches = []  # (is_tp, confidence) in rank order
    for det in cls_dets:
        best_idx, best_iou = None, 0.5
        for gi, gb in enumerate(gt_boxes):
            if used[gi]:
                continue
            iou = _iou(det["box"], gb["box"])
            if iou >= best_iou:
                best_iou = iou
                best_idx = gi
        if best_idx is not None:
            used[best_idx] = True
            matches.append((True, det["confidence"]))
        else:
            matches.append((False, det["confidence"]))

    tp = sum(1 for is_tp, _ in matches if is_tp)
    fp = sum(1 for is_tp, _ in matches if not is_tp)
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    ap = _average_precision(matches, len(gt_boxes))

    return {
        "gt": len(gt_boxes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "ap": round(ap, 3),
    }


def _average_precision(matches: List[Tuple[bool, float]],
                       n_gt: int) -> float:
    """11-point interpolated AP."""
    if n_gt == 0:
        return 0.0
    tp_running = 0
    precisions = []
    recalls = []
    for is_tp, _ in matches:
        if is_tp:
            tp_running += 1
        precisions.append(tp_running / (len(precisions) + 1))
        recalls.append(tp_running / n_gt)

    if not recalls:
        return 0.0

    # Interpolate precision at 11 equally spaced recall levels.
    ap = 0.0
    for r in (i / 10.0 for i in range(11)):
        best = 0.0
        for p, rec in zip(precisions, recalls):
            if rec >= r:
                best = max(best, p)
        ap += best / 11.0
    return ap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "yolov8s.onnx"))
    args = parser.parse_args()

    model_path = str(Path(args.model))
    if not Path(model_path).is_absolute():
        model_path = str(PROJECT_ROOT / model_path)

    gt = _load_annotations()
    images = sorted(IMAGES_DIR.glob("*.jpg")) if IMAGES_DIR.exists() else []

    if not images or not gt:
        print("No images or ground truth found under "
              f"{EVAL_DIR}.")
        print("Per-phase rule: never invent accuracy.  Add images/ and "
              "annotations/ with real labelled scenes, then re-run.")
        return

    detector = YoloDetector(model_path, conf_threshold=args.conf)
    print(f"Model {Path(model_path).name} @ conf {args.conf} | "
          f"{len(images)} image(s)")

    # Run detection once per image; accumulate GT and detections.
    all_dets: List[Dict] = []
    all_gts: List[Dict] = []
    for image in images:
        dets, n_gt = _evaluate(image, gt.get(image.name, []),
                               detector, args.conf)
        all_dets.extend(dets)
        all_gts.extend(gt.get(image.name, []))

    results: Dict = {}
    agg_tp = agg_fp = agg_fn = 0
    classes = sorted({g["label"] for g in all_gts}
                     | {d["label"] for d in all_dets})
    for cls in classes:
        score = _match_and_score(all_dets, all_gts, cls)
        results[cls] = score
        agg_tp += score["tp"]
        agg_fp += score["fp"]
        agg_fn += score["fn"]

    precision = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else 0.0
    recall = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    mAP = sum(s["ap"] for s in results.values()) / len(results) if results else 0.0

    report = {
        "model": Path(model_path).name,
        "conf": args.conf,
        "images": len(images),
        "total_gt_boxes": len(all_gts),
        "aggregate": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "mAP@0.5": round(mAP, 3),
        },
        "per_class": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()