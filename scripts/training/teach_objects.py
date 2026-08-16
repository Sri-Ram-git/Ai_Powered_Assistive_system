"""Capture + label your OWN objects from the webcam into a training set.

Records frames, lets you draw a bounding box with the mouse, assigns a
class name, and writes standard YOLO-format labels plus a COCO-style
annotation file — the same format produced by the dataset downloader, so
your custom objects can be mixed in with (or trained alongside) the big
labelled datasets.

Output layout (default `data/datasets/custom/`):
    images/            saved frames (custom_0001.jpg, ...)
    labels/            one YOLO .txt per image (normalised boxes)
    dataset.yaml       YOLO data file (names, nc)
    annotations.json   COCO-style annotations
    classes.txt        your class names (one per line)

Controls:
    camera view:   '1'..'9' pick class   'c' capture frame to label
    annotator:     drag mouse to draw box  's' save  'r' reset box
                   'b' back to camera  'q' quit

Example:
    python scripts/training/teach_objects.py --classes person,mug,toaster

    Then: put your mug in front of the camera, press 2, press c, drag a
    box around the mug, press s.  Repeat for other views/objects.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Sequence

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "..", ".."))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "datasets", "custom")

WINDOW = "Teach objects"
_boxes: List[tuple] = []  # current drag box (x0, y0, x1, y1)


def _on_mouse(event, x, y, flags, param) -> None:
    state = param
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing"] = True
        state["origin"] = (x, y)
        state["box"] = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state.get("drawing"):
        ox, oy = state["origin"]
        state["box"] = (min(ox, x), min(oy, y), max(ox, x), max(oy, y))
    elif event == cv2.EVENT_LBUTTONUP:
        state["drawing"] = False
        ox, oy = state["origin"]
        state["box"] = (min(ox, x), min(oy, y), max(ox, x), max(oy, y))


def _draw_hud(frame: np.ndarray, class_name: str, class_id: int,
              count: int, captured: bool) -> np.ndarray:
    cv2.putText(frame, f"class {class_id}: {class_name}  | saved: {count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "1-9 pick class | c capture | q quit",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if captured:
        cv2.putText(frame, "s save | r reset box | b back",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return frame


def _save_sample(out: str, frame: np.ndarray, box: tuple,
                 class_id: int, counter: int) -> str:
    """Save an image + YOLO label. Returns the image filename."""
    img_path = os.path.join(out, "images", f"custom_{counter:05d}.jpg")
    lab_path = os.path.join(out, "labels", f"custom_{counter:05d}.txt")
    cv2.imwrite(img_path, frame)
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, min(x0, w - 1)); x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1)); y1 = max(0, min(y1, h - 1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box has zero size")
    cx = (x0 + x1) / 2 / w
    cy = (y0 + y1) / 2 / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    with open(lab_path, "w", encoding="utf-8") as fh:
        fh.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return os.path.basename(img_path)


def _load_state(out: str) -> tuple:
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    counter = 0
    existing = [f for f in os.listdir(os.path.join(out, "images"))
                if f.startswith("custom_") and f.endswith(".jpg")]
    if existing:
        nums = [int(f[7:12]) for f in existing if f[7:12].isdigit()]
        counter = max(nums) + 1 if nums else 0
    return counter


def _write_metadata(out: str, classes: List[str], images: List[dict]) -> None:
    with open(os.path.join(out, "classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(classes) + "\n")
    with open(os.path.join(out, "dataset.yaml"), "w", encoding="utf-8") as fh:
        fh.write("names:\n")
        for i, name in enumerate(classes):
            fh.write(f"  {i}: {name}\n")
        fh.write("nc: " + str(len(classes)) + "\n")
    with open(os.path.join(out, "annotations.json"), "w", encoding="utf-8") as fh:
        json.dump({"classes": classes, "images": images}, fh, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--classes", required=True,
                    help="comma-separated class names (index = class id)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if len(classes) > 9:
        print("ERROR: max 9 classes in one session (keys 1-9). "
              "Run again for more.")
        return 1
    if not classes:
        print("ERROR: --classes is required")
        return 1

    counter = _load_state(args.out)
    images: List[dict] = []
    ann_path = os.path.join(args.out, "annotations.json")
    if os.path.exists(ann_path):
        try:
            with open(ann_path, "r", encoding="utf-8") as fh:
                images = json.load(fh).get("images", [])
        except Exception:
            images = []

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return 1

    class_id = 0
    captured: Optional[np.ndarray] = None
    box: Optional[tuple] = None
    state = {"drawing": False, "origin": None, "box": None}
    saved_this_session = 0
    start = time.time()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, _on_mouse, state)

    try:
        while True:
            if captured is None:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = _draw_hud(frame, classes[class_id], class_id,
                                  saved_this_session, False)
                cv2.imshow(WINDOW, frame)
            else:
                vis = captured.copy()
                if state.get("box"):
                    x0, y0, x1, y1 = state["box"]
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
                vis = _draw_hud(vis, classes[class_id], class_id,
                                saved_this_session, True)
                cv2.imshow(WINDOW, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if captured is None:
                if ord("1") <= key <= ord("1") + len(classes) - 1:
                    class_id = key - ord("1")
                elif key == ord("c"):
                    ok, frame = cap.read()
                    if ok:
                        captured = frame.copy()
                        box = None
                        state = {"drawing": False, "origin": None, "box": None}
                        cv2.setMouseCallback(WINDOW, _on_mouse, state)
            else:
                if key == ord("s") and state.get("box"):
                    x0, y0, x1, y1 = state["box"]
                    if x1 > x0 and y1 > y0:
                        try:
                            fname = _save_sample(
                                args.out, captured, state["box"],
                                class_id, counter)
                        except ValueError:
                            fname = None
                        if fname:
                            images.append({
                                "file_name": fname, "class_id": class_id,
                                "class_name": classes[class_id],
                                "box": list(state["box"]),
                                "width": int(captured.shape[1]),
                                "height": int(captured.shape[0]),
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
                            counter += 1
                            saved_this_session += 1
                elif key == ord("r"):
                    state = {"drawing": False, "origin": None, "box": None}
                    cv2.setMouseCallback(WINDOW, _on_mouse, state)
                elif key == ord("b"):
                    captured = None
                    state = {"drawing": False, "origin": None, "box": None}
                    cv2.setMouseCallback(WINDOW, _on_mouse, state)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    _write_metadata(args.out, classes, images)
    elapsed = time.time() - start
    print(f"Session: {saved_this_session} saved this run "
          f"({len(images)} total in dataset)")
    print(f"Output: {args.out}  (elapsed {elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())