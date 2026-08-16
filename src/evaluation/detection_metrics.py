"""Object detection metrics: precision, recall, mAP@50, mAP@50:95.

Reference boxes and predicted boxes are axis-aligned rectangles with a
class label.  Standard COCO-style IoU matching is used:

* a prediction matches a ground-truth box when IoU >= threshold AND the
  class label matches (a matching pair is consumed once, greedily by
  descending confidence);
* mAP@50 = mean over classes of AP at IoU 0.50;
* mAP@50:95 = mean of AP computed at IoU thresholds 0.50..0.95 (step
  0.05), averaged over classes.

Inputs are simple dataclasses so the metrics work with any detector
output without importing the runtime.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class Box:
    """An axis-aligned box in image coordinates."""

    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # (x, y, w, h)


def _iou(a: Tuple[int, int, int, int],
         b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match(
    predictions: List[Box],
    ground_truth: List[Box],
    iou_threshold: float,
) -> Tuple[Dict[int, bool], int, int, int]:
    """Greedy confidence-ranked matching.

    Returns:
        (matched, tp, fp, total_gt) where ``matched`` maps prediction
        index -> True/False, tp/fp are counts, total_gt is the number of
        ground-truth boxes.
    """
    order = sorted(range(len(predictions)),
                   key=lambda i: predictions[i].confidence, reverse=True)
    gt_used = [False] * len(ground_truth)
    matched: Dict[int, bool] = {}
    tp = 0
    for i in order:
        pred = predictions[i]
        best_j, best_iou = -1, iou_threshold
        for j, gt in enumerate(ground_truth):
            if gt_used[j] or gt.label != pred.label:
                continue
            iou = _iou(pred.box, gt.box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            gt_used[best_j] = True
            matched[i] = True
            tp += 1
        else:
            matched[i] = False
    fp = len(predictions) - tp
    return matched, tp, fp, len(ground_truth)


def _ap_at_iou(
    predictions: List[Box],
    ground_truth: List[Box],
    iou_threshold: float,
    label: str,
) -> float:
    """Average precision for a single class at a single IoU threshold."""
    preds = [p for p in predictions if p.label == label]
    gts = [g for g in ground_truth if g.label == label]
    if not gts:
        # No ground truth for this class: a perfect detector has AP 1.0,
        # anything that fires a false positive has AP 0.0.
        return 1.0 if not preds else 0.0

    matched, _, _, _ = _match(preds, gts, iou_threshold)
    order = sorted(range(len(preds)),
                   key=lambda i: preds[i].confidence, reverse=True)

    tp_cum, fp_cum = 0, 0
    recalls: List[float] = []
    precisions: List[float] = []
    for i in order:
        if matched.get(i, False):
            tp_cum += 1
        else:
            fp_cum += 1
        recalls.append(tp_cum / len(gts))
        precisions.append(tp_cum / (tp_cum + fp_cum) if tp_cum + fp_cum else 0.0)

    # 11-point interpolation.
    ap = 0.0
    for t in [x / 10 for x in range(0, 11)]:
        pr = max([p for r, p in zip(recalls, precisions) if r >= t],
                 default=0.0)
        ap += pr / 11.0
    return ap


def mean_average_precision(
    predictions: List[Box],
    ground_truth: List[Box],
    iou_thresholds: Sequence[float] = (0.50,),
) -> float:
    """Mean AP over classes and the given IoU thresholds."""
    labels = {g.label for g in ground_truth} | {p.label for p in predictions}
    if not labels:
        return 0.0
    aps = [_ap_at_iou(predictions, ground_truth, t, label)
           for t in iou_thresholds for label in labels]
    return sum(aps) / len(aps) if aps else 0.0


def evaluate_detections(
    predictions: List[Box],
    ground_truth: List[Box],
) -> Dict[str, float]:
    """Compute precision, recall, mAP@50 and mAP@50:95 for one image."""
    matched, tp, fp, total_gt = _match(predictions, ground_truth, 0.50)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    map50 = mean_average_precision(predictions, ground_truth, (0.50,))
    map50_95 = mean_average_precision(
        predictions, ground_truth, [x / 100 for x in range(50, 100, 5)]
    )
    return {
        "precision": precision,
        "recall": recall,
        "mAP@50": map50,
        "mAP@50:95": map50_95,
        "false_positives": fp,
        "false_negatives": total_gt - tp,
        "true_positives": tp,
    }