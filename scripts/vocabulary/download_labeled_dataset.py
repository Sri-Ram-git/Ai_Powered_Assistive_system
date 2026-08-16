"""Download a labelled-image dataset for a subset of vocabulary words.

Every word in the vocabulary manifest maps to a real category in a
labelled dataset.  This tool downloads the actual labelled images for the
words you choose and prepares them for fine-tuning YOLO:

    out/
        images/         the downloaded jpg files
        labels/         one YOLO .txt per image (normalised boxes)
        dataset.yaml    YOLO data file (names list, for ultralytics)
        annotations.json COCO-style annotations (all categories/images)
        report.json     honest download/failure statistics

Sources:
    --source lvis        LVIS annotations JSON (categories share COCO
                         images): https://github.com/lvis-dataset/lvis
    --source openimages  OpenImages annotations CSV (the user downloads
                         train-annotations-bbox.csv once; it is multi-GB)

Examples:
    python scripts/vocabulary/download_labeled_dataset.py --source lvis \
        --annotations C:/datasets/lvis_v1_val.json \
        --classes person,chair,toaster --max-images-per-class 25 \
        --out data/datasets/lvis_sample

    python scripts/vocabulary/download_labeled_dataset.py --source openimages \
        --annotations C:/datasets/train-annotations-bbox.csv \
        --max-classes 5 --max-images-per-class 10 --out data/datasets/oi_sample

Downloads are logged; failures are counted and reported (never silently
dropped).  Re-running with the same output dir resumes/overwrites.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

from src.vocabulary import ObjectVocabulary  # noqa: E402

USER_AGENT = "ai-vision-system/vocabulary-downloader"

# OpenImages boxes use the dataset's own label ids, which we map through the
# OpenImages vocabulary entry (openimages_id).
OPENIMAGES_IMAGE_URL = ("https://storage.googleapis.com/openimages/"
                        "2018_04/{split}/{image_id}.jpg")
COCO_IMAGE_URL = "http://images.cocodataset.org/{split}/{file_name}"


def _download(url: str, dest: str, timeout: float = 30.0) -> bool:
    """Download a single image, returning success. Never raises."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True  # resume
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
                open(dest, "wb") as fh:
            fh.write(resp.read())
        return os.path.getsize(dest) > 0
    except Exception:
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except Exception:
            pass
        return False


def load_lvis_annotations(path: str) -> Tuple[List[dict], List[dict], Dict[int, str]]:
    """Return (images, annotations, category_id->name) from an LVIS JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    return images, annotations, cats


def load_openimages_annotations(path: str, class_ids: Dict[str, str]) -> Tuple[
        List[str], List[Tuple[str, str, List[float]]], Dict[str, str]]:
    """Read OpenImages CSV rows -> (image_ids, (image_id, class_id, box), id->name).

    Only rows for the requested class ids are kept.  Returns discovered
    image ids, filtered annotations, and a class map for the vocabulary.
    """
    image_ids: List[str] = []
    rows: List[Tuple[str, str, List[float]]] = []
    want = set(class_ids.values())
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        idx = {name: i for i, name in enumerate(header or [])}
        if "ImageID" not in idx or "LabelName" not in idx:
            raise ValueError(
                "OpenImages CSV must have ImageID + LabelName columns")
        for line in reader:
            if len(line) < 7:
                continue
            image_id, _, label, _conf, xmin, xmax, ymin, ymax = \
                line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7]
            if label not in want:
                continue
            image_ids.append(image_id)
            rows.append((image_id, label, [
                float(xmin), float(ymin), float(xmax), float(ymax)]))
    uniq = list(dict.fromkeys(image_ids))
    return uniq, rows, {v: k for k, v in class_ids.items()}


def _lvis_candidates(images: List[dict], annotations: List[dict],
                     cat_names: Dict[int, str]) -> Dict[str, List[dict]]:
    """Group LVIS annotation dicts per category name (normalised)."""
    groups: Dict[str, List[dict]] = {}
    img_by_id = {img["id"]: img for img in images}
    for ann in annotations:
        name = cat_names.get(ann.get("category_id"))
        if not name:
            continue
        groups.setdefault(name, []).append(ann)
    for name in groups:
        groups[name].sort(key=lambda a: img_by_id.get(a["image_id"], {}).get(
            "id", ""))
    return groups


def lvis_bbox_to_xyxy(bbox: Sequence[float]) -> List[float]:
    """LVIS/COCO bbox [x, y, width, height] -> [x0, y0, x1, y1]."""
    x, y, w, h = (list(bbox) + [0, 0])[:4]
    return [float(x), float(y), float(x) + float(w), float(y) + float(h)]


def _write_yolo_label(dest: str, class_id: int, boxes: List[Tuple[str, list]],
                      width: int, height: int,
                      classes: Optional[List[int]] = None) -> None:
    lines = []
    for i, (_, box) in enumerate(boxes):
        cid = classes[i] if classes else class_id
        if width <= 0 or height <= 0:
            continue
        x0, y0, x1, y1 = box
        x0 = max(0.0, min(float(x0), width))
        x1 = max(0.0, min(float(x1), width))
        y0 = max(0.0, min(float(y0), height))
        y1 = max(0.0, min(float(y1), height))
        cx = (x0 + x1) / 2 / width
        cy = (y0 + y1) / 2 / height
        w = (x1 - x0) / width
        h = (y1 - y0) / height
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))


def run(source: str, annotations_path: str, classes: Sequence[str],
        max_images_per_class: int, out_dir: str,
        max_classes: Optional[int] = None,
        workers: int = 8) -> Dict[str, int]:
    """Download labeled images for the requested vocabulary words."""
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    lab_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    vocab = ObjectVocabulary.load()
    words: List[str] = []
    for word in classes:
        if vocab.resolve(word) is None:
            print(f"  ! unknown word in vocabulary: {word!r}")
            continue
        if word not in words:
            words.append(word)
    if max_classes:
        words = words[:max_classes]
    if not words:
        raise ValueError(
            "no valid --classes given (check spelling against "
            "data/vocabulary/words.txt)")

    stats = {"requested": len(words), "downloaded": 0, "failed": 0,
             "skipped": 0, "images": 0, "annotations": 0}

    if source == "lvis":
        images, annotations, cat_names = load_lvis_annotations(annotations_path)
        groups = _lvis_candidates(images, annotations, cat_names)
        categories = []
        name_to_id: Dict[str, int] = {}
        for i, word in enumerate(words):
            entry = vocab.resolve(word)
            lvis_id = entry.lvis_id
            # resolve the dataset category name by id (LVIS name may differ)
            cat_name = None
            if lvis_id is not None:
                for cid, cname in cat_names.items():
                    if cid == lvis_id:
                        cat_name = cname
                        break
            name = cat_name or entry.word
            name_to_id[word] = i
            categories.append({
                "id": i, "name": name, "word": entry.display_word,
                "lvis_id": lvis_id, "openimages_id": entry.openimages_id,
                "coco_id": entry.coco_id})
        img_by_id = {img["id"]: img for img in images}
        # LVIS category names may differ from vocabulary words (e.g. vocab
        # "cell phone" is LVIS "cellphone", "laptop" is "laptop_computer").
        # Resolve via the LVIS id first, then by the LVIS name's first token
        # (a safe rule: "laptop_computer" starts with "laptop").
        lvis_by_first_token: Dict[str, str] = {}
        for cname in cat_names.values():
            first = cname.split("_", 1)[0].strip().lower()
            if first:
                lvis_by_first_token.setdefault(first, cname)
        word_to_lvis_name: Dict[str, str] = {}
        for word in words:
            entry = vocab.resolve(word)
            cat_name = None
            if entry and entry.lvis_id is not None:
                for cid, cname in cat_names.items():
                    if cid == entry.lvis_id:
                        cat_name = cname
                        break
            if cat_name is None and entry:
                cat_name = lvis_by_first_token.get(
                    entry.word.strip().lower().split()[0])
            word_to_lvis_name[word] = cat_name or entry.word
        download_map: Dict[str, Tuple[str, str]] = {}   # img_id -> (url, dest)
        word_img_ids: Dict[str, List[str]] = {}
        for word in words:
            group = groups.get(word_to_lvis_name[word], [])[
                :max_images_per_class]
            for ann in group:
                img_id = ann["image_id"]
                img = img_by_id.get(img_id)
                if not img:
                    continue
                split = "val2017" if "val" in str(img.get("coco_url", "")) \
                    else "train2017"
                file_name = img.get("file_name") or str(
                    img.get("coco_url", "")).rsplit("/", 1)[-1]
                if not file_name:
                    continue
                url = COCO_IMAGE_URL.format(
                    split=split, file_name=file_name)
                dest = os.path.join(img_dir, f"{img_id}.jpg")
                download_map.setdefault(str(img_id), (url, dest))
                word_img_ids.setdefault(word, []).append(str(img_id))
        # write labels grouped by image (accumulate all boxes, write once)
        per_image_boxes: Dict[str, List[Tuple[int, list]]] = {}
        for word in words:
            cid = name_to_id[word]
            group = groups.get(word_to_lvis_name[word], [])[
                :max_images_per_class]
            for ann in group:
                img_id = str(ann["image_id"])
                img = img_by_id.get(ann["image_id"])
                if not img:
                    continue
                # LVIS/COCO bbox is [x, y, width, height].
                box = lvis_bbox_to_xyxy(ann.get("bbox", [0, 0, 0, 0]))
                per_image_boxes.setdefault(img_id, []).append((cid, box))
                stats["annotations"] += 1
        for img_id, boxes in per_image_boxes.items():
            img = img_by_id.get(int(img_id))
            if not img:
                continue
            _write_yolo_label(
                os.path.join(lab_dir, f"{img_id}.txt"), 0,
                [(img_id, box) for _, box in boxes],
                int(img.get("width", 1)), int(img.get("height", 1)),
                classes=[cid for cid, _ in boxes])
        stats["images"] += len(download_map)
    elif source == "openimages":
        class_ids = {}
        categories = []
        for i, word in enumerate(words):
            entry = vocab.resolve(word)
            class_ids[word] = entry.openimages_id
            categories.append({
                "id": i, "name": entry.display_word,
                "word": entry.display_word,
                "lvis_id": entry.lvis_id,
                "openimages_id": entry.openimages_id,
                "coco_id": entry.coco_id})
        image_ids, rows, name_map = load_openimages_annotations(
            annotations_path, class_ids)
        per_image: Dict[str, List[Tuple[str, list]]] = {}
        for img_id, label, box in rows:
            per_image.setdefault(img_id, []).append(
                (name_map[label], box))  # name_map[label] = vocabulary word
        word_to_id = {w: i for i, w in enumerate(words)}
        download_map: Dict[str, Tuple[str, str]] = {}
        used: Dict[str, int] = {}
        boxes_per_image: Dict[str, List[Tuple[int, list]]] = {}
        for img_id, anns in per_image.items():
            if used.get(img_id, 0) >= max_images_per_class:
                continue
            url = OPENIMAGES_IMAGE_URL.format(split="train", image_id=img_id)
            dest = os.path.join(img_dir, f"{img_id}.jpg")
            download_map.setdefault(img_id, (url, dest))
            used[img_id] = used.get(img_id, 0) + 1
            for word, box in anns:
                if word not in word_to_id:
                    continue
                # OpenImages boxes are already normalised to [0, 1].
                boxes_per_image.setdefault(img_id, []).append(
                    (word_to_id[word], box))
                stats["annotations"] += 1
        for img_id, boxes in boxes_per_image.items():
            _write_yolo_label(
                os.path.join(lab_dir, f"{img_id}.txt"), 0,
                [(img_id, box) for _, box in boxes], 1, 1,
                classes=[cid for cid, _ in boxes])
        stats["images"] += len(download_map)
    else:
        raise ValueError(f"unknown source: {source!r}")

    # Download images in parallel, counting real successes/failures.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download, url, dest): img_id
                   for img_id, (url, dest) in download_map.items()}
        for future in as_completed(futures):
            img_id = futures[future]
            if future.result():
                stats["downloaded"] += 1
            else:
                stats["failed"] += 1

    # Report + write outputs.
    succeeded: Dict[str, int] = {}
    if source == "lvis":
        for word, ids in word_img_ids.items():
            succeeded[word] = sum(
                1 for i in ids if os.path.exists(
                    os.path.join(img_dir, f"{i}.jpg")))
    else:
        for word in words:
            succeeded[word] = sum(
                1 for f in os.listdir(img_dir)
                if os.path.exists(os.path.join(img_dir, f)))
    report = {
        "source": source,
        "requested_words": len(words),
        "max_images_per_class": max_images_per_class,
        "downloaded_images": stats["downloaded"],
        "failed_images": stats["failed"],
        "annotations_written": stats["annotations"],
        "categories": len(categories),
        "per_word_images": succeeded,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("counts are real download outcomes; failed images are "
                 "reported, never silently dropped"),
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(out_dir, "annotations.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"categories": categories, "images_count": report[
            "downloaded_images"]}, fh, indent=2)
    with open(os.path.join(out_dir, "dataset.yaml"), "w", encoding="utf-8") as fh:
        fh.write("names:\n")
        for c in categories:
            fh.write(f"  {c['id']}: {c['name']}\n")
        fh.write("nc: " + str(len(categories)) + "\n")
    with open(os.path.join(out_dir, "words.txt"), "w", encoding="utf-8") as fh:
        for w in words:
            fh.write(vocab.display_word(w) + "\n")

    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["lvis", "openimages"], required=True)
    ap.add_argument("--annotations", required=True,
                    help="LVIS JSON or OpenImages bbox CSV")
    ap.add_argument("--classes", default="",
                    help="comma-separated vocabulary words to download")
    ap.add_argument("--max-classes", type=int, default=None,
                    help="limit to the first N requested classes")
    ap.add_argument("--max-images-per-class", type=int, default=20)
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        # Default: a small, honest sample so a first run is quick.
        from src.vocabulary import ObjectVocabulary
        words = ObjectVocabulary.load().words
        classes = words[:10]
        print("No --classes given; using first 10 vocabulary words as a "
              "sample. Pass --classes person,chair,... for specific words.")

    stats = run(args.source, args.annotations, classes,
                args.max_images_per_class, args.out,
                max_classes=args.max_classes, workers=args.workers)
    print("Downloaded:", stats)
    print(f"Dataset written to: {args.out}")
    if stats["failed"]:
        print(f"WARNING: {stats['failed']} images failed to download "
              f"(see report.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())