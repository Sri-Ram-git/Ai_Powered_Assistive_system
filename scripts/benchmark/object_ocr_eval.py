"""Object-aware OCR evaluation: dataset + benchmark.

Generates a reproducible synthetic dataset of *object-like* regions (bottle
labels, book covers, laptop screens, medicine packs, room-number plates,
signs...) with varied degradations (rotation, perspective, blur, lighting,
small sizes, fonts) and runs the real object-aware OCR path against it:

    extract_roi -> text-presence gate -> preprocessing variants -> best
    result -> combine

Reports per-sample recognised text, CER / WER / exact-match vs ground
truth, latency, per-variant statistics, and reading-order violations, and
writes the images + ground truth to ``assets/ocr_eval/`` for inspection.

All samples are seeded deterministically (``zlib.crc32`` — ``hash()`` is
randomised per process), so a run is reproducible byte-for-byte.

Usage:
    python scripts/benchmark/object_ocr_eval.py [--out assets/ocr_eval]
        [--per-text 6] [--limit 0]
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ocr_metrics import (  # noqa: E402
    aggregate_ocr_metrics,
    character_error_rate,
    exact_match,
    word_error_rate,
)
from src.ocr.object_ocr import combine_results, run_variants  # noqa: E402
from src.ocr.roi import extract_roi  # noqa: E402
from src.ocr.text_presence import has_text  # noqa: E402

# (name, text, kind, box_w, box_h, contrast)
#   kind      -> rendering style (label / screen / sign / lowlight)
#   box_w/h   -> size of the text-bearing region (tests smart upscaling)
#   contrast  -> 'high' | 'low' (low = gray on gray, tests variants)
#   font      -> cv2.FONT_HERSHEY_* index
#   lines     -> render as multiple centred lines when text contains "\n"
BASE_SAMPLES = [
    ("bottle_label_large", "COCA COLA", "label", 320, 72, "high", 0, 0),
    ("bottle_label_small", "COCA COLA", "label", 96, 24, "high", 0, 0),
    ("book_cover", "THE ART OF WAR", "label", 300, 80, "high", 0, 0),
    ("laptop_screen", "WELCOME TO THE DEMO", "screen", 420, 96, "high", 0, 0),
    ("sign_exit", "EXIT", "sign", 200, 72, "high", 0, 0),
    ("sign_do_not_enter", "DO NOT ENTER", "sign", 320, 72, "high", 0, 0),
    ("cup_coffee", "FRESH BREW", "label", 180, 48, "high", 0, 0),
    ("lowlight_label", "SALT AND PEPPER", "label", 280, 56, "low", 0, 0),
    ("lowlight_screen", "PRESS START", "screen", 240, 56, "low", 0, 0),
    ("tiny_laptop", "HELLO", "screen", 64, 18, "high", 0, 0),
    ("tiny_sign", "STOP", "sign", 48, 24, "high", 0, 0),
    ("remote_control", "MENU", "label", 200, 40, "high", 0, 0),
]

# Extra content categories that matter to the user: products, medicine,
# room numbers, book titles, screens, signs, door labels, and multi-line
# reading-order samples.  Each is rendered with several deterministic
# degradations to reach 100+ samples.
EXTRA_TEXTS = [
    ("vitamin_c", "VITAMIN C 500MG", "label", "high"),
    ("paracetamol", "PARACETAMOL 250 MG", "label", "high"),
    ("room_204", "ROOM 204", "sign", "high"),
    ("exit_3", "EXIT 3", "sign", "high"),
    ("no_smoking", "NO SMOKING", "sign", "high"),
    ("emergency", "EMERGENCY", "sign", "high"),
    ("isbn", "ISBN 978-0-13-468599-1", "label", "high"),
    ("band_aid", "BAND-AID", "label", "high"),
    ("paint", "MATT EMULSION 5L", "label", "high"),
    ("screen_error", "CONNECTION LOST", "screen", "high"),
    ("screen_charge", "CHARGING 82%", "screen", "high"),
    ("shampoo", "ANTI-DANDRUFF", "label", "low"),
    ("juice", "ORANGE JUICE 1L", "label", "low"),
    ("book_title", "THE GREAT GATSBY", "label", "high"),
    ("order_do_not_walk", "DO NOT\nWALK", "sign", "high"),
    ("order_phone", "CALL\nNOW", "sign", "high"),
]


def _stable_seed(name: str, salt: int = 0) -> int:
    """Deterministic seed (hash() is randomised per process)."""
    return (zlib.crc32(name.encode("utf-8")) + salt) % 1000


def _font_scale_for(width: int) -> float:
    return max(0.5, width / 220.0)


def _font(index: int) -> int:
    return (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX)[index % 4]


# ----------------------------------------------------------------------
# Real typography rendering (Rule #6/#16/#17): Times New Roman, Arial,
# Calibri, Consolas (digital-clock-ish) with regular / bold / italic.
# Falls back to OpenCV Hershey when the Windows font files are absent.
# ----------------------------------------------------------------------
_WIN_FONTS = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
_PIL_FONT_FILES = {
    "times": "times.ttf",
    "times_bold": "timesbd.ttf",
    "times_italic": "timesi.ttf",
    "arial": "arial.ttf",
    "arial_bold": "arialbd.ttf",
    "arial_italic": "ariali.ttf",
    "calibri": "calibri.ttf",
    "calibri_bold": "calibrib.ttf",
    "consolas": "consola.ttf",
    "consolas_bold": "consolab.ttf",
}


def _pil_render(text: str, font_name: str, size: int, bg, fg) -> np.ndarray:
    """Render ``text`` with a real truetype font; returns BGR image."""
    from PIL import Image, ImageDraw, ImageFont

    rel = _PIL_FONT_FILES[font_name]
    font_path = _WIN_FONTS / rel
    try:
        font = ImageFont.truetype(str(font_path), size)
    except Exception:
        font = ImageFont.load_default(size)
    # Measure the rendered box (bbox ignores ascenders/descenders, so use
    # the textbbox to size the canvas properly).
    tmp = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = max(6, size // 4)
    img = Image.new("RGB", (tw + 2 * pad, th + 2 * pad), bg)
    d = ImageDraw.Draw(img)
    d.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=fg)
    arr = np.asarray(img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _resolve_colors(kind: str, contrast: str) -> Tuple[Tuple, Tuple]:
    if kind == "screen":
        return (20, 20, 24), (230, 235, 240)
    if kind == "sign":
        return (20, 120, 30), (255, 255, 255)
    if contrast == "low":
        return (128, 128, 128), (100, 100, 100)
    return (245, 245, 245), (20, 20, 20)


def _render_text(text: str, kind: str, box_w: int, box_h: int,
                 contrast: str, seed: int, font_idx: int = 0,
                 font: Optional[str] = None,
                 mirrored: bool = False) -> np.ndarray:
    """Render a (possibly multi-line) text region deterministically."""
    rng = np.random.default_rng(seed)
    bg, fg = _resolve_colors(kind, contrast)

    if font is not None:
        # Real typography: size scales with the box height, so the same
        # word is tested at small / medium / large pixel heights.
        size = max(10, min(48, int(box_h * 0.62)))
        img = _pil_render(text.replace("\n", " "), font, size, bg, fg)
        if mirrored:  # simulated front-camera selfie mirror
            img = cv2.flip(img, 1)
        noise = rng.integers(0, 10, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        return img

    img = np.full((box_h, box_w, 3), bg, dtype=np.uint8)
    font_obj = _font(font_idx)
    scale = _font_scale_for(box_w)
    thickness = 2 if box_h >= 40 else 1

    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font_obj, scale, thickness)
        heights.append(th)
        widths.append(tw)
    line_h = max(heights) + max(2, int(scale * 6))
    total_h = sum(heights) + line_h * (len(lines) - 1)
    y0 = max(heights[0], (box_h + total_h) // 2)
    y = y0
    for i, line in enumerate(lines):
        x = max(2, (box_w - widths[i]) // 2)
        cv2.putText(img, line, (x, y), font_obj, scale, fg, thickness,
                    cv2.LINE_AA)
        y += line_h

    noise = rng.integers(0, 12, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def _box_after_transform(wh, transform, is_perspective) -> Tuple[int, int, int, int]:
    """Content bounding box after warping the region corners."""
    w, h = wh
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]],
                       dtype=np.float32)
    if is_perspective:
        ones = np.ones((4, 1), dtype=np.float32)
        pts = transform @ np.hstack([corners, ones]).T
        pts = pts[:2] / pts[2]
        xs, ys = pts[0], pts[1]
    else:
        pts = transform @ np.hstack([corners, np.ones((4, 1))]).T
        xs, ys = pts[0], pts[1]
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()), float(ys.max())
    return (int(round(x0)), int(round(y0)),
            int(round(x1 - x0)), int(round(y1 - y0)))


def _augment(region: np.ndarray, seed: int):
    """Apply rotation + perspective + blur + lighting (deterministic).

    Returns (augmented_region, content_box) where content_box tightly
    bounds the warped text so the downstream ROI stays accurate.
    """
    rng = np.random.default_rng(seed)
    h, w = region.shape[:2]
    box = (0, 0, w, h)

    # Rotation (-12..12 deg) around the centre, canvas kept fixed size.
    angle = float(rng.uniform(-12, 12))
    if abs(angle) >= 1.0:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        region = cv2.warpAffine(region, m, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        box = _box_after_transform((w, h), m, False)

    # Mild perspective skew (keeps the text legible but off-axis).
    if rng.random() < 0.5:
        s = float(rng.uniform(0.02, 0.06))
        dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        src = np.float32([
            [0 + w * s, 0], [w - w * s, 0],
            [w * s, h], [w - w * s, h],
        ])
        m = cv2.getPerspectiveTransform(src, dst)
        region = cv2.warpPerspective(region, m, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        box = _box_after_transform((w, h), m, True)

    # Blur.
    k = int(rng.integers(0, 3))
    if k > 0:
        region = cv2.GaussianBlur(region, (2 * k + 1, 2 * k + 1), 0)

    # Lighting (dim/brighten).
    gain = float(rng.uniform(0.7, 1.3))
    if abs(gain - 1.0) > 0.05:
        region = np.clip(region.astype(np.float32) * gain, 0, 255
                         ).astype(np.uint8)

    return region, box


FONT_TEXTS = [
    # (name, text, kind, contrast) — Rules #16/#17: real typography.
    ("time_12_45", "12:45 PM", "label", "high"),
    ("time_10_30", "10:30 PM", "label", "high"),
    ("time_12_am", "12:00 AM", "label", "high"),
    ("time_23_59", "23:59", "label", "high"),
    ("time_08_15", "08:15", "label", "high"),
    ("time_12_pm", "12:00 PM", "label", "high"),
    ("emergency_exit", "EMERGENCY EXIT", "sign", "high"),
    ("coca_cola", "COCA COLA", "label", "high"),
    ("room_204", "ROOM 204", "sign", "high"),
    ("medicine", "MEDICINE 500MG", "label", "high"),
    ("art_of_war", "THE ART OF WAR", "label", "high"),
    ("no_smoking", "NO SMOKING", "sign", "high"),
]

# Times New Roman (serif), Arial / Calibri (sans-serif), Consolas
# (monospace / digital-clock-ish) × regular / bold / italic.
_FONTS = ["times", "times_bold", "times_italic", "arial", "arial_bold",
          "calibri", "consolas"]
_SIZES = [16, 28, 44]  # small / medium / large pixel heights


def _generate_dataset(per_text: int, font_per_text: int = 3) -> List[tuple]:
    """BASE_SAMPLES + degraded EXTRA_TEXTS + real-font + mirror samples."""
    samples: List[tuple] = []
    for base in BASE_SAMPLES:
        name, text, kind, bw, bh, contrast, fidx, aug = base
        samples.append((name, text, kind, bw, bh, contrast, fidx, aug,
                        None, False))
    for name, text, kind, contrast in EXTRA_TEXTS:
        multi = "\n" in text
        for i in range(per_text):
            font_idx = i % 4
            if multi:
                box_w = 320
                box_h = 72
            else:
                box_w = int(160 + (i * 37) % 200)
                box_h = int(32 + (i * 13) % 40)
            samples.append((f"{name}_{i}", text, kind, box_w, box_h,
                            contrast, font_idx, i, None, False))
    for name, text, kind, contrast in FONT_TEXTS:
        for i, fname in enumerate(_FONTS):
            size = _SIZES[i % len(_SIZES)]
            samples.append((f"font_{name}_{fname}", text, kind, 320, size * 2,
                            contrast, i, i, fname, False))
    # Simulated front-camera mirroring (the OLD broken path): the SAME
    # text is horizontally flipped before OCR.  These are expected to
    # fail — they quantify why OCR must receive unmirrored frames.
    for name, text, kind, contrast in FONT_TEXTS[:4]:
        samples.append((f"mirrored_{name}", text, kind, 320, 64, contrast,
                        0, 0, "arial", True))
    return samples


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
    name, gt, kind, box_w, box_h, contrast, font_idx, aug_seed, font, \
        mirrored = sample
    truth = _canonical(gt)
    region = _render_text(gt, kind, box_w, box_h, contrast,
                          seed=_stable_seed(name), font_idx=font_idx,
                          font=font, mirrored=mirrored)
    region, cbox = _augment(region, seed=_stable_seed(name, 1) + aug_seed)
    frame, box = _place_in_frame(region, cbox, seed=_stable_seed(name, 2))

    entry = {"name": name, "truth": truth, "kind": kind, "contrast": contrast,
             "box": list(box), "roi_rejected": False,
             "presence": None, "result": "", "accuracy": 0.0,
             "cer": 1.0, "wer": 1.0, "exact": 0,
             "latency_ms": 0.0, "variant": "", "status": "error",
             "order_violation": False, "font": font, "mirrored": mirrored}

    roi = extract_roi(frame, box, padding=0.1)
    if roi is None:
        entry["roi_rejected"] = True
        entry["status"] = "roi_rejected"
        return entry

    present = has_text(roi.image)
    entry["presence"] = present
    if not present:
        entry["status"] = "presence_gate"
        return entry

    started = time.monotonic()
    try:
        variant, items, _latency = run_variants(engine, roi.image, variants,
                                                stop_confidence=0.92)
        text, _conf = combine_results(items)
        entry["variant"] = variant
        entry["latency_ms"] = (time.monotonic() - started) * 1000.0
        entry["result"] = _canonical(text)
        entry["accuracy"] = char_accuracy(truth, entry["result"])
        entry["cer"] = character_error_rate(truth, entry["result"])
        entry["wer"] = word_error_rate(truth, entry["result"])
        entry["exact"] = exact_match(truth, entry["result"])
        # Reading-order violation: same words, wrong order.
        if truth and entry["result"]:
            if (sorted(truth.split()) == sorted(entry["result"].split())
                    and truth != entry["result"]):
                entry["order_violation"] = True
        entry["status"] = "ok" if entry["accuracy"] >= 0.9 else \
            ("partial" if entry["accuracy"] > 0.0 else "miss")
    except Exception as exc:  # pragma: no cover - env dependent
        entry["latency_ms"] = (time.monotonic() - started) * 1000.0
        entry["status"] = f"error:{exc}"
    return entry


def _place_in_frame(region: np.ndarray, box: Tuple[int, int, int, int],
                    seed: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Wrap a warped text region in a 640x480 'scene' as one object box."""
    rng = np.random.default_rng(seed)
    th, tw = region.shape[:2]
    frame = np.full((480, 640, 3), (90, 90, 95), dtype=np.uint8)
    x = int(rng.integers(40, 640 - tw - 40))
    y = int(rng.integers(40, 480 - th - 40))
    frame[y:y + th, x:x + tw] = region
    bx, by, bw, bh = box
    return frame, (x + bx, y + by, bw, bh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="assets/ocr_eval")
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--per-text", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0,
                        help="max samples to evaluate (0 = all)")
    args = parser.parse_args()

    out_dir = Path(PROJECT_ROOT) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.ocr import OcrEngine
    from src.ocr.object_ocr import DEFAULT_VARIANTS

    variants = DEFAULT_VARIANTS[:max(1, min(4, args.variants))]
    samples = _generate_dataset(args.per_text)
    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"OpenCV: {cv2.__version__} | variants: {variants} "
          f"| samples: {len(samples)}")
    print("Loading OCR engine ...")
    engine = OcrEngine(min_confidence=0.3)

    # Save the dataset images + ground truth for inspection.
    gt_rows = []
    for sample in samples:
        name, gt, kind, box_w, box_h, contrast, font_idx, aug_seed, font, \
            mirrored = sample
        region = _render_text(gt, kind, box_w, box_h, contrast,
                              seed=_stable_seed(name), font_idx=font_idx,
                              font=font, mirrored=mirrored)
        region, cbox = _augment(region, seed=_stable_seed(name, 1) + aug_seed)
        frame, box = _place_in_frame(region, cbox, seed=_stable_seed(name, 2))
        cv2.imwrite(str(out_dir / f"{name}.png"), frame)
        gt_rows.append({"name": name, "text": gt, "kind": kind,
                        "contrast": contrast, "box": list(box),
                        "font": font, "mirrored": mirrored})
    with open(out_dir / "ground_truth.json", "w", encoding="utf-8") as fh:
        json.dump(gt_rows, fh, indent=2)
    print(f"Dataset written to {out_dir}/ ({len(samples)} images)")

    rows = [evaluate(engine, sample, variants) for sample in samples]

    print("\n=== Per-sample results ===")
    for row in rows:
        flag = " ORDER!" if row["order_violation"] else ""
        tag = ""
        if row["font"]:
            tag = f" [{row['font']}]"
        if row["mirrored"]:
            tag += " [MIRRORED]"
        print(f"  {row['name']:<24} acc={row['accuracy']:.2f} "
              f"cer={row['cer']:.2f} wer={row['wer']:.2f} "
              f"lat={row['latency_ms']:>6.0f}ms var={row['variant']:<8} "
              f"status={row['status']:<7} got={row['result'][:26]!r}"
              f"{tag}{flag}")

    ok = [r for r in rows if r["status"] == "ok"]
    partial = [r for r in rows if r["status"] == "partial"]
    missed = [r for r in rows if r["status"] == "miss"]
    evaluated = [r for r in rows if r["status"] not in ("roi_rejected",
                                                        "presence_gate",
                                                        "error")]
    mirrored_rows = [r for r in rows if r["mirrored"]]
    unmirrored = [r for r in evaluated if not r["mirrored"]]
    avg_acc = float(np.mean([r["accuracy"] for r in unmirrored])) \
        if unmirrored else 0.0
    agg = aggregate_ocr_metrics([r["truth"] for r in unmirrored],
                                [r["result"] for r in unmirrored]) \
        if unmirrored else {"cer": 0.0, "wer": 0.0, "exact_match": 0.0,
                            "detection_success": 0.0}
    avg_lat = float(np.mean([r["latency_ms"] for r in evaluated])) \
        if evaluated else 0.0
    p95_lat = float(np.percentile([r["latency_ms"] for r in evaluated], 95)) \
        if evaluated else 0.0
    rejected = [r for r in rows if r["status"] == "roi_rejected"]
    gated = [r for r in rows if r["status"] == "presence_gate"]
    order_viol = [r for r in rows if r["order_violation"]]

    print("\n=== Summary ===")
    print(f"  samples           : {len(rows)}")
    print(f"  evaluated         : {len(evaluated)}")
    print(f"  exact (>=0.9 acc) : {len(ok)}")
    print(f"  partial           : {len(partial)}")
    print(f"  missed            : {len(missed)}")
    print(f"  roi rejected      : {len(rejected)}")
    print(f"  presence gated    : {len(gated)}")
    print(f"  order violations  : {len(order_viol)}")
    print(f"  mirrored samples  : {len(mirrored_rows)} (expected to fail)")
    print(f"  mean char acc     : {avg_acc:.3f}")
    print(f"  mean CER          : {agg['cer']:.3f}")
    print(f"  mean WER          : {agg['wer']:.3f}")
    print(f"  exact match rate  : {agg['exact_match']:.3f}")
    print(f"  detection success : {agg['detection_success']:.3f}")
    print(f"  mean latency      : {avg_lat:.0f} ms")
    print(f"  p95 latency       : {p95_lat:.0f} ms")

    by_contrast = {}
    by_kind = {}
    for row in evaluated:
        by_contrast.setdefault(row["contrast"], []).append(row["accuracy"])
        by_kind.setdefault(row["kind"], []).append(row["accuracy"])
    print("\n  accuracy by contrast:")
    for k, v in by_contrast.items():
        print(f"    {k:<8} {float(np.mean(v)):.3f}")
    print("  accuracy by kind:")
    for k, v in by_kind.items():
        print(f"    {k:<8} {float(np.mean(v)):.3f}")

    wins = {}
    for row in evaluated:
        if row["status"] in ("ok", "partial") and row["variant"]:
            wins[row["variant"]] = wins.get(row["variant"], 0) + 1
    print("  winning variant:")
    for k, v in sorted(wins.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {k:<9} {v}")

    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": {
            "samples": len(rows),
            "evaluated": len(evaluated),
            "exact": len(ok), "partial": len(partial), "missed": len(missed),
            "roi_rejected": len(rejected),
            "presence_gated": len(gated),
            "order_violations": len(order_viol),
            "mean_char_acc": round(avg_acc, 3),
            "cer": round(agg["cer"], 3),
            "wer": round(agg["wer"], 3),
            "exact_match": round(agg["exact_match"], 3),
            "detection_success": round(agg["detection_success"], 3),
            "mean_latency_ms": round(avg_lat, 1),
            "p95_latency_ms": round(p95_lat, 1),
        }, "rows": rows}, fh, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()