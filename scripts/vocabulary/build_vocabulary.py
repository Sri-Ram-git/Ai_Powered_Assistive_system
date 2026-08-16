"""Build the 1000+ word object vocabulary manifest.

Merges the real class lists of three *labelled image* datasets — LVIS v1
(1203 categories), OpenImages (601 boxable classes), COCO-80 — into one
de-duplicated vocabulary where every word has labelled images available
in at least one dataset.

Outputs (committed):
    data/vocabulary/object_vocabulary.yaml   manifest (runtime source)
    data/vocabulary/words.txt                plain one-word-per-line list

Usage:
    python scripts/vocabulary/build_vocabulary.py [--sources DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

from src.detection.detector import COCO_NAMES  # noqa: E402
from src.vocabulary.manifest import (  # noqa: E402
    PROJECT_ROOT,
    VocabularyEntry,
    normalize_word,
    validate,
)

DEFAULT_SOURCES = os.path.join(PROJECT_ROOT, "data", "vocabulary", "sources")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "data", "vocabulary")

# ---------------------------------------------------------------------------
# Tier + category rules (heuristic, auditable in this file).
#
# A word is critical if any of its *tokens* is in CRITICAL_TOKENS or its
# full normalized name is in CRITICAL_PHRASES; else high, else low, else
# normal.  Rules deliberately err towards "normal" so safety tiers are
# conservative (never over-announce).
# ---------------------------------------------------------------------------

CRITICAL_TOKENS: set = {
    "person", "people", "pedestrian", "woman", "man", "child", "baby",
    "crowd", "car", "bus", "truck", "pickup", "taxi", "van", "lorry",
    "ambulance", "police", "firetruck", "garbage", "tractor", "forklift",
    "bulldozer", "excavator", "crane", "motorcycle", "moped", "scooter",
    "bicycle", "bike", "tricycle", "skateboard", "train", "tram",
    "subway", "trolley", "airplane", "aeroplane", "helicopter",
    "airliner", "boat", "ship", "ferry", "sailboat", "stair",
    "staircase", "stairway", "steps", "curb", "hydrant", "stroller",
}

CRITICAL_PHRASES: set = {
    "stop sign", "traffic light", "traffic signal", "crosswalk",
    "speed bump", "guardrail", "handrail", "barrier", "traffic cone",
    "roadblock", "wheelchair", "walking stick", "crutch",
}

HIGH_TOKENS: set = {
    "dog", "cat", "horse", "cow", "sheep", "goat", "pig", "chicken",
    "duck", "goose", "turkey", "rabbit", "deer", "elephant", "bear",
    "tiger", "lion", "zebra", "giraffe", "monkey", "fox", "wolf",
    "bird", "fish", "chair", "stool", "bench", "couch", "sofa",
    "table", "desk", "bed", "wardrobe", "dresser", "cabinet",
    "refrigerator", "freezer", "oven", "stove", "microwave", "toaster",
    "dishwasher", "washer", "dryer", "sink", "bathtub", "toilet",
    "door", "gate", "fence", "wall", "pillar", "column", "pole",
    "tree", "stump", "lamp",
}

HIGH_PHRASES: set = {
    "fire extinguisher", "fire alarm", "potted plant", "fireplace",
    "coffee maker", "lawn mower", "power drill", "curb",
}

LOW_TOKENS: set = {
    "picture", "painting", "poster", "photograph", "banner", "flag",
    "curtain", "drape", "rug", "carpet", "pillow", "cushion",
    "blanket", "quilt", "vase", "candle", "chandelier", "sculpture",
    "figurine", "ornament", "garland", "wreath", "tapestry", "mirror",
    "frame", "doormat", "towel", "soap", "toothbrush", "comb",
    "cosmetic", "jewelry", "necklace", "earring", "watch", "pen",
    "pencil", "eraser", "book", "magazine", "newspaper",
}

LOW_PHRASES: set = {"stained glass", "christmas tree", "christmas ornament"}

CATEGORY_TOKENS: Dict[str, set] = {
    "vehicle": {
        "car", "bus", "truck", "van", "pickup", "taxi", "lorry",
        "ambulance", "police", "firetruck", "tractor", "forklift",
        "bulldozer", "excavator", "crane", "motorcycle", "moped",
        "scooter", "bicycle", "bike", "tricycle", "skateboard", "train",
        "tram", "subway", "trolley", "airplane", "helicopter", "boat",
        "ship", "ferry", "sailboat", "wheelchair", "stroller",
    },
    "animal": {
        "dog", "cat", "horse", "cow", "sheep", "goat", "pig", "chicken",
        "duck", "goose", "turkey", "rabbit", "deer", "elephant", "bear",
        "tiger", "lion", "zebra", "giraffe", "monkey", "fox", "wolf",
        "bird", "fish",
    },
    "person": {
        "person", "people", "pedestrian", "woman", "man", "child",
        "baby", "crowd",
    },
    "furniture": {
        "chair", "stool", "bench", "couch", "sofa", "table", "desk",
        "bed", "wardrobe", "dresser", "cabinet", "bookcase", "shelf",
        "mattress", "pillow", "cushion", "cradle", "cage",
    },
    "appliance": {
        "refrigerator", "freezer", "oven", "stove", "microwave",
        "toaster", "dishwasher", "washer", "dryer", "sink", "bathtub",
        "toilet", "kettle", "blender", "coffee maker", "vacuum",
        "hair drier", "fan",
    },
    "electronics": {
        "television", "tv", "computer", "laptop", "phone", "camera",
        "keyboard", "mouse", "monitor", "printer", "speaker",
        "headphones", "remote", "radio", "clock", "tablet", "projector",
    },
    "clothing": {
        "shirt", "pants", "jeans", "dress", "coat", "jacket",
        "sweater", "t-shirt", "scarf", "hat", "cap", "shoe", "boot",
        "sock", "glove", "belt", "underwear", "swimsuit", "apron",
        "helmet", "suitcase", "backpack", "handbag", "umbrella",
        "headband", "bracelet", "watch",
    },
    "food": {
        "apple", "banana", "orange", "carrot", "broccoli", "bread",
        "cheese", "pizza", "sandwich", "burger", "cake", "cookie",
        "egg", "milk", "juice", "water", "soup", "salad", "rice",
        "pasta", "donut", "hot dog", "pretzel", "muffin", "cupcake",
    },
    "drink": {
        "bottle", "can", "cup", "glass", "mug", "teapot", "juice",
        "wine", "beer", "soda", "coffee", "tea",
    },
    "kitchen": {
        "pot", "pan", "plate", "bowl", "knife", "fork", "spoon",
        "cutting board", "spatula", "ladle", "tray",
    },
    "building": {
        "door", "window", "wall", "roof", "floor", "ceiling",
        "stair", "staircase", "gate", "fence", "pillar", "column",
        "pole", "post", "brick", "concrete", "building", "house",
        "chimney", "balcony", "porch",
    },
    "outdoor": {
        "tree", "plant", "flower", "grass", "bush", "rock", "stone",
        "curb", "hydrant", "sign", "street", "lamp", "fountain",
        "playground", "pool", "beach", "mountain", "snow",
    },
    "tool": {
        "hammer", "screwdriver", "wrench", "pliers", "saw", "drill",
        "screw", "nail", "tape", "scissors", "chain", "rope",
    },
    "sport": {
        "ball", "bat", "racket", "glove", "helmet", "skateboard",
        "surfboard", "skis", "snowboard", "frisbee", "kite",
        "sports ball", "tennis racket", "baseball bat",
    },
    "plant": {
        "tree", "plant", "flower", "bush", "grass", "leaf", "flower",
    },
    "container": {
        "box", "bag", "bottle", "jar", "crate", "basket", "bucket",
        "bin", "trash", "basket",
    },
    "stationery": {
        "pen", "pencil", "eraser", "ruler", "notebook", "paper",
        "envelope", "marker", "book", "magazine", "newspaper",
    },
    "toy": {"toy", "doll", "ball", "kite", "puzzle", "teddy", "lego"},
    "body": {
        "hand", "arm", "leg", "head", "eye", "ear", "nose", "mouth",
        "finger", "foot", "face",
    },
    "medical": {
        "wheelchair", "walker", "crutch", "stethoscope", "syringe",
        "bandage", "medicine",
    },
    "music": {
        "guitar", "piano", "drum", "violin", "trumpet", "flute",
        "microphone", "organ", "cello",
    },
}

DEFAULT_CATEGORY = "object"


def _token_match(tokens: Sequence[str], keyword_set: set) -> bool:
    return any(tok in keyword_set for tok in tokens)


def classify(word: str) -> Tuple[str, str]:
    """Return (tier, category) for a normalized word."""
    norm = normalize_word(word)
    tokens = norm.split()
    # Critical first (safety), then high, then low.
    if norm in CRITICAL_PHRASES or _token_match(tokens, CRITICAL_TOKENS):
        tier = "critical"
    elif norm in HIGH_PHRASES or _token_match(tokens, HIGH_TOKENS):
        tier = "high"
    elif norm in LOW_PHRASES or _token_match(tokens, LOW_TOKENS):
        tier = "low"
    else:
        tier = "normal"
    category = DEFAULT_CATEGORY
    for cat, kw in CATEGORY_TOKENS.items():
        if _token_match(tokens, kw):
            category = cat
            break
    return tier, category


def load_lvis(path: str) -> Dict[str, dict]:
    """LVIS v1 categories JSON -> {norm_key: info}."""
    out: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        cats = json.load(fh)
    for c in cats:
        name = (c.get("name") or "").strip()
        if not name:
            continue
        key = normalize_word(name)
        out[key] = {
            "word": name,
            "lvis_id": c.get("id"),
            "aliases": list(c.get("synonyms") or []),
        }
    return out


def load_openimages(path: str) -> Dict[str, dict]:
    """OpenImages boxable classes CSV -> {norm_key: info}."""
    out: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or len(row) < 2:
                continue
            mid, name = row[0].strip(), row[1].strip()
            key = normalize_word(name)
            out[key] = {"word": name, "openimages_id": mid}
    return out


def load_coco() -> Dict[str, dict]:
    """COCO-80 names from the detector -> {norm_key: info}."""
    out: Dict[str, dict] = {}
    for idx, name in enumerate(COCO_NAMES):
        key = normalize_word(name)
        out[key] = {"word": name, "coco_id": idx}
    return out


def merge(*sources: Dict[str, dict]) -> List[VocabularyEntry]:
    """Union sources by normalized key, filling in ids/aliases."""
    merged: Dict[str, dict] = {}
    order: List[str] = []
    for source in sources:
        for key, info in source.items():
            if key not in merged:
                merged[key] = {"word": info["word"], "aliases": [],
                               "lvis_id": None, "openimages_id": None,
                               "coco_id": None}
                order.append(key)
            m = merged[key]
            m["aliases"] = sorted(set(m["aliases"]) | set(info.get("aliases") or []))
            for field in ("lvis_id", "openimages_id", "coco_id"):
                if info.get(field) is not None and m[field] is None:
                    m[field] = info[field]
    entries: List[VocabularyEntry] = []
    for key in order:
        info = merged[key]
        tier, category = classify(info["word"])
        entries.append(VocabularyEntry(
            word=info["word"],
            tier=tier,
            category=category,
            aliases=info["aliases"],
            coco_id=info["coco_id"],
            lvis_id=info["lvis_id"],
            openimages_id=info["openimages_id"],
        ))
    entries.sort(key=lambda e: e.normalize_key())
    return entries


def emit_yaml(entries: List[VocabularyEntry], path: str) -> None:
    counts: Dict[str, int] = {}
    for e in entries:
        counts[e.tier] = counts.get(e.tier, 0) + 1
    header = [
        "# Object vocabulary manifest (AUTO-GENERATED — edit the builder,",
        "# not this file).  Every word is a real category that exists in a",
        "# labelled image dataset, so each word has labelled images available.",
        "#",
        "# Regenerate:  python scripts/vocabulary/build_vocabulary.py",
        "# Runtime API:  src/vocabulary (ObjectVocabulary.load()).",
        "#",
        f"version: 1",
        f"word_count: {len(entries)}",
        "tiers:",
        f"  critical: {counts.get('critical', 0)}",
        f"  high: {counts.get('high', 0)}",
        f"  normal: {counts.get('normal', 0)}",
        f"  low: {counts.get('low', 0)}",
        "sources:",
        '  lvis: "LVIS v1 — 1203 categories (labelled images on COCO images)"',
        '  openimages: "OpenImages — 601 boxable classes (labelled images)"',
        '  coco: "COCO-80 — the model labels (labelled images)"',
        "words:",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        for e in entries:
            row = OrderedDict()
            row["word"] = e.word
            row["tier"] = e.tier
            row["category"] = e.category
            if e.aliases:
                row["aliases"] = e.aliases
            if e.coco_id is not None:
                row["coco_id"] = e.coco_id
            if e.lvis_id is not None:
                row["lvis_id"] = e.lvis_id
            if e.openimages_id is not None:
                row["openimages_id"] = e.openimages_id
            fh.write("  - {" + ", ".join(
                f"{k}: {json.dumps(v, ensure_ascii=False)}"
                for k, v in row.items()) + "}\n")


def emit_words(entries: List[VocabularyEntry], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(e.display_word + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", default=DEFAULT_SOURCES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    lvis_path = os.path.join(args.sources, "lvis_v1_categories.json")
    oi_path = os.path.join(args.sources, "openimages_boxable_classes.csv")
    if not os.path.exists(lvis_path) or not os.path.exists(oi_path):
        print(f"ERROR: class lists missing under {args.sources}")
        return 1

    lvis = load_lvis(lvis_path)
    oi = load_openimages(oi_path)
    coco = load_coco()
    entries = merge(coco, oi, lvis)

    problems = validate(entries, min_words=1000)
    if problems:
        print("VALIDATION PROBLEMS:")
        for p in problems[:50]:
            print("  -", p)
        return 1

    os.makedirs(args.out, exist_ok=True)
    emit_yaml(entries, os.path.join(args.out, "object_vocabulary.yaml"))
    emit_words(entries, os.path.join(args.out, "words.txt"))

    counts: Dict[str, int] = {}
    for e in entries:
        counts[e.tier] = counts.get(e.tier, 0) + 1
    print(f"Wrote {len(entries)} words "
          f"(critical={counts.get('critical', 0)}, "
          f"high={counts.get('high', 0)}, "
          f"normal={counts.get('normal', 0)}, "
          f"low={counts.get('low', 0)})")
    print(f"  manifest: {os.path.join(args.out, 'object_vocabulary.yaml')}")
    print(f"  word list: {os.path.join(args.out, 'words.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())