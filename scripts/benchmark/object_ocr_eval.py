"""Object-aware OCR evaluation: dataset + benchmark.

Generates a small reproducible synthetic dataset of *object-like* regions
(a bottle label, a book cover, a laptop screen, a cup, a sign...) and
runs the real object-aware OCR path against it:

    extract_roi -> text-presence gate -> preprocessing variants -> best
    result -> combine

reports per-sample recognised text, character-level accuracy vs ground
truth, latency, and per-variant statistics, and writes the images +
ground truth to ``assets/ocr_eval/`` for inspection.

Usage:
    python scripts/benchmark/object_ocr_eval.py [--out assets/ocr_eval]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.object_ocr import combine_results, run_variants  # noqa: E402
from src.ocr.roi import extract_roi  # noqa: E402
from src.ocr.text_presence import has_text  # noqa: E402

# (name, text, kind, box_w, box_h, contrast)
#   kind      -> rendering style (label / screen / sign / lowlight)
#   box_w/h   -> size of the text-bearing region (tests smart upscaling)
#   contrast  -> 'high' | 'low' (low = gray on gray, tests variants)
DATASET = [
    ("bottle_label_large", "COCA COLA", "label", 320, 72, "high"),
    ("bottle_label_small", "COCA COLA", "label", 96, 24, "high"),
    ("book_cover", "THE ART OF WAR", "label", 300, 80, "high"),
    ("laptop_screen", "WELCOME TO THE DEMO", "screen", 420, 96, "high"),
    ("sign_exit", "EXIT", "sign", 200, 72, "high"),
    ("sign_do_not_enter", "DO NOT ENTER", "sign", 320, 72, "high"),
    ("cup_coffee", "FRESH BREW", "label", 180, 48, "high"),
    ("lowlight_label", "SALT AND PEPPER", "label", 280, 56, "low"),
    ("lowlight_screen", "PRESS START", "screen", 240, 56, "low"),
    ("tiny_laptop", "HELLO", "screen", 64, 18, "high"),
    ("tiny_sign", "STOP", "sign", 48, 24, "high"),
    ("remote_control", "MENU", "label", 200, 40, "high"),
]


def _font_scale_for(width: int) -> float:
    return max(0.5, width / 220.0)


def _render_text(text: str, kind: str, box_w: int, box_h: int,
                 contrast: str, seed: int) -> np.ndarray:
    """Render a text region in an object-like frame, deterministically."""
    rng = np.random.default_rng(seed)

    if kind == "screen":
        bg, fg = (20, 20, 24), (230, 235, 240)
    elif kind == "sign":
        bg, fg = (20, 120, 30), (255, 255, 255)
    else:  # label
        if contrast == "low":
            bg, fg = (128, 128, 128), (100, 100, 100)
        else:
            bg, fg = (245, 245, 245), (20, 20, 20)

    img = np.full((box_h, box_w, 3), bg, dtype=np.uint8)
    scale = _font_scale_for(box_w)
    thickness = 2 if box_h >= 40 else 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                  scale, thickness)
    x = max(2, (box_w - tw) // 2)
    y = max(th, (box_h + th) // 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, thickness, cv2.LINE_AA)

    # Mild sensor-like noise (deterministic).
    noise = rng.integers(0, 12, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def _place_in_frame(text_region: np.ndarray, seed: int) -> np.ndarray:
    """Wrap a text region in a 640x480 'scene' as one object box."""
    rng = np.random.default_rng(seed)
    th, tw = text_region.shape[:2]
    frame = np.full((480, 640, 3), (90, 90, 95), dtype=np.uint8)
    x = int(rng.integers(40, 640 - tw - 40))
    y = int(rng.integers(40, 480 - th - 40))
    frame[y:y + th, x:x + tw] = text_region
    return frame, (x, y, tw, th)


def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance (no external deps)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,                 # deletion
                curr[j - 1] + 1,             # insertion
                prev[j - 1] + (ca != cb),    # substitution
            ))
        prev = curr
    return prev[-1]


def char_accuracy(gt: str, rec: str) -> float:
    if not gt:
        return 0.0 if rec else 1.0
    if not rec:
        return 0.0
    dist = levenshtein(gt.lower(), rec.lower())
    return max(0.0, 1.0 - dist / max(len(gt), 1))


def _canonical(text: str) -> str:
    return " ".join(text.strip().split()).upper()


def evaluate(engine, sample, variants) -> dict:
    name, gt, kind, box_w, box_h, contrast = sample
    region = _render_text(gt, kind, box_w, box_h, contrast, seed=hash(name) % 1000)
    frame, box = _place_in_frame(region, seed=len(name))
    truth = _canonical(gt)

    roi = extract_roi(frame, box, padding=0.1)
    entry = {"name": name, "truth": truth, "kind": kind, "contrast": contrast,
             "box": box, "roi_rejected": roi is None,
             "presence": None, "result": "", "accuracy": 0.0,
             "latency_ms": 0.0, "variant": "", "status": "error"}

    if roi is None:
        entry["status"] = "roi_rejected"
        return entry

    present = has_text(roi.image)
    entry["presence"] = present

    started = time.monotonic()
    try:
        variant, items, _latency = run_variants(engine, roi.image, variants,
                                                stop_confidence=0.92)
        text, _conf = combine_results(items)
        entry["variant"] = variant
        entry["latency_ms"] = (time.monotonic() - started) * 1000.0
        entry["result"] = _canonical(text)
        entry["accuracy"] = char_accuracy(truth, entry["result"])
        entry["status"] = "ok" if entry["accuracy"] >= 0.9 else \
            ("partial" if entry["accuracy"] > 0.0 else "miss")
    except Exception as exc:  # pragma: no cover - env dependent
        entry["latency_ms"] = (time.monotonic() - started) * 1000.0
        entry["status"] = f"error:{exc}"
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="assets/ocr_eval")
    parser.add_argument("--variants", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(PROJECT_ROOT) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.ocr import OcrEngine
    from src.ocr.object_ocr import DEFAULT_VARIANTS

    variants = DEFAULT_VARIANTS[:max(1, min(4, args.variants))]
    print(f"OpenCV: {cv2.__version__} | variants: {variants}")
    print("Loading OCR engine ...")
    engine = OcrEngine(min_confidence=0.3)

    # Save the dataset images + ground truth for inspection.
    gt_rows = []
    for sample in DATASET:
        name, gt, kind, box_w, box_h, contrast = sample
        region = _render_text(gt, kind, box_w, box_h, contrast,
                              seed=hash(name) % 1000)
        frame, box = _place_in_frame(region, seed=len(name))
        cv2.imwrite(str(out_dir / f"{name}.png"), frame)
        gt_rows.append({"name": name, "text": gt, "kind": kind,
                        "contrast": contrast, "box": list(box)})
    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as fh:
        json.dump(gt_rows, fh, indent=2)
    print(f"Dataset written to {out_dir}/ ({len(DATASET)} images)")

    rows = [evaluate(engine, sample, variants) for sample in DATASET]

    print("\n=== Per-sample results ===")
    for row in rows:
        print(f"  {row['name']:<22} acc={row['accuracy']:.2f} "
              f"lat={row['latency_ms']:>6.0f}ms var={row['variant']:<8} "
              f"status={row['status']:<7} "
              f"got={row['result'][:28]!r}")

    ok = [r for r in rows if r["status"] == "ok"]
    partial = [r for r in rows if r["status"] == "partial"]
    missed = [r for r in rows if r["status"] == "miss"]
    avg_acc = float(np.mean([r["accuracy"] for r in rows]))
    avg_lat = float(np.mean([r["latency_ms"] for r in rows]))
    rejected = [r for r in rows if r["status"] == "roi_rejected"]
    rejected_presence = [r for r in rows if r["presence"] is False]

    print("\n=== Summary ===")
    print(f"  samples           : {len(rows)}")
    print(f"  exact (>=0.9 acc) : {len(ok)}")
    print(f"  partial           : {len(partial)}")
    print(f"  missed            : {len(missed)}")
    print(f"  roi rejected      : {len(rejected)}")
    print(f"  presence skipped  : {len(rejected_presence)}")
    print(f"  mean char acc     : {avg_acc:.3f}")
    print(f"  mean latency      : {avg_lat:.0f} ms")

    by_contrast = {}
    for row in rows:
        by_contrast.setdefault(row["contrast"], []).append(row["accuracy"])
    print("\n  accuracy by contrast:")
    for k, v in by_contrast.items():
        print(f"    {k:<8} {float(np.mean(v)):.3f}")

    by_kind = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row["accuracy"])
    print("  accuracy by kind:")
    for k, v in by_kind.items():
        print(f"    {k:<8} {float(np.mean(v)):.3f}")

    # Variant win statistics.
    wins = {}
    for row in rows:
        if row["status"] in ("ok", "partial") and row["variant"]:
            wins[row["variant"]] = wins.get(row["variant"], 0) + 1
    print("  winning variant:")
    for k, v in sorted(wins.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {k:<9} {v}")

    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": {
            "samples": len(rows),
            "exact": len(ok), "partial": len(partial), "missed": len(missed),
            "mean_char_acc": round(avg_acc, 3),
            "mean_latency_ms": round(avg_lat, 1),
        }, "rows": rows}, fh, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()