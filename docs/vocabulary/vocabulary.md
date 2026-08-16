# Object vocabulary — 1551 words with labelled images

The assistive system can only name what its model detects.  The stock
YOLOv8s model knows 80 COCO classes, so a real room produces the same
few labels over and over ("person", "chair", "tv", ...) and speech
repeats them.  To move past that, the vocabulary manifest gives the
system a large, auditable word list — and, crucially, every word is a
real category that exists in a **labelled image dataset**, so labelled
training images actually exist for each word.

## What is it

| File | Purpose |
|---|---|
| `data/vocabulary/object_vocabulary.yaml` | the runtime manifest (1551 words) |
| `data/vocabulary/words.txt` | plain one-word-per-line list |
| `data/vocabulary/sources/lvis_v1_categories.json` | LVIS v1 categories (1203) |
| `data/vocabulary/sources/openimages_boxable_classes.csv` | OpenImages boxable classes (601) |
| `src/vocabulary/` | runtime module (`ObjectVocabulary.load()`) |
| `scripts/vocabulary/build_vocabulary.py` | regenerates the manifest |

## Sources (real labelled-image datasets)

| Dataset | Categories | Where the labelled images live |
|---|---|---|
| **LVIS v1** | 1203 | instance segmentation labels on COCO images |
| **OpenImages** | 601 boxable | bounding-box labels, ~16M boxes |
| **COCO-80** | 80 | the labels the current YOLO model was trained on |

The manifest is the de-duplicated **union** (after normalisation):
**1551 unique words**, every one mapped to its `lvis_id` /
`openimages_id` / `coco_id` where present.

## Tiers

Tiers decide how urgently a word is announced (and its speech priority):

| Tier | Count | Meaning |
|---|---|---|
| `critical` | 56 | immediate safety — spoken first, no delay (vehicles, person, stop sign, stairs, hydrant, …) |
| `high` | 112 | announced promptly (animals, large furniture/appliances, doors, poles, trees, …) |
| `normal` | 1333 | routine everyday objects |
| `low` | 50 | background decor, announced rarely (paintings, rugs, vases, …) |

Rules are **explicit and auditable** in
`scripts/vocabulary/build_vocabulary.py` (`CRITICAL_TOKENS`,
`HIGH_TOKENS`, `LOW_TOKENS`, …).  They deliberately err towards
`normal`, so safety tiers are conservative — never over-announced.

## Runtime use

```python
from src.vocabulary import ObjectVocabulary
vocab = ObjectVocabulary.load()
vocab.tier_for("person")         # "critical"
vocab.display_word("aerosol_can") # "aerosol can"
vocab.size                       # 1551
```

The desktop app uses the tier to raise speech priority (a detected
"car"/"person" is spoken as `CRITICAL`) and uses phrase variety
(`src/audio/variety.py`) so a repeated cue is not one fixed sentence:
"Person left, about 5 metres" → "Person left, around 5 metres" →
"Person left, roughly 5 metres".

## Rebuilding

Class lists are committed under `data/vocabulary/sources/`, so the
manifest rebuilds fully offline:

```
python scripts/vocabulary/build_vocabulary.py
```

The builder validates (≥ 1000 words, unique keys, valid tiers) and
refuses to write an invalid manifest.

## Honest limits

- Tiers are **heuristics** from keyword rules, not learned behaviour.
  Tune the token sets in the builder and rebuild.
- Recognising 1551 words requires a model that *knows* those words.  The
  current YOLOv8s still only outputs its 80 COCO labels; the dataset
  downloader + teach tool (see `training.md`) produce the labelled images
  needed to fine-tune a model with a much larger class list.