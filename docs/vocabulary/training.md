# Labelled-image dataset + fine-tuning path

The vocabulary lists 1551 words that *have* labelled images.  This page
shows how to actually get those images and turn them into a model that
recognises many more objects than the stock 80-class YOLOv8s.

Two complementary tools:

1. **`scripts/vocabulary/download_labeled_dataset.py`** — download
   real labelled images for chosen vocabulary words (from LVIS or
   OpenImages) into YOLO-ready format.
2. **`scripts/training/teach_objects.py`** — capture and label *your own*
   objects from the webcam (same output format), so your personal items
   can be trained in too.

Both write the same layout, so you can mix downloaded + self-labelled
data into one training set.

## 1. Download labelled images for vocabulary words

### From LVIS (recommended)

LVIS categories share COCO images.  Download the LVIS annotations once:

```
# from https://github.com/lvis-dataset/lvis (val json is ~24 MB)
# (used here only to map category -> images; images come from COCO)
```

Then:

```
python scripts/vocabulary/download_labeled_dataset.py \
    --source lvis \
    --annotations C:/datasets/lvis_v1_val.json \
    --classes person,chair,toaster,mug \
    --max-images-per-class 25 \
    --out data/datasets/lvis_sample
```

### From OpenImages

OpenImages boxes are in the (multi-GB) annotations CSV, which you
download once from the OpenImages download page
(`train-annotations-bbox.csv`).  The class list (already committed) maps
each vocabulary word to its `/m/...` label id.

```
python scripts/vocabulary/download_labeled_dataset.py \
    --source openimages \
    --annotations C:/datasets/train-annotations-bbox.csv \
    --classes person,chair \
    --max-images-per-class 20 \
    --out data/datasets/oi_sample
```

### Output

```
out/
    images/          downloaded jpg files
    labels/          one YOLO .txt per image (normalised boxes)
    dataset.yaml     YOLO data file (names, nc) — for ultralytics
    annotations.json COCO-style categories + counts
    report.json      honest download/failure statistics
```

Downloads are parallel; failures are counted in `report.json` (never
silently dropped), and re-running resumes.

## 2. Teach your own objects

```
python scripts/training/teach_objects.py --classes person,mug,toaster
```

Controls: `1`-`9` pick class, `c` capture frame, drag mouse to draw the
box, `s` save, `r` reset box, `b` back to camera, `q` quit.

Writes the same `images/ + labels/ + dataset.yaml + annotations.json`
layout into `data/datasets/custom/`.  Classes are appended across runs.

## 3. Mix and fine-tune

```
# merge downloaded + custom folders into one dataset dir (images/,
# labels/, dataset.yaml with the union class list), then e.g.:

# ultralytics
pip install ultralytics
yolo train model=yolov8s.pt data=data/datasets/my_dataset.yaml epochs=100

# export to the ONNX format this project loads
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640
```

Point `models/<model_path>` in `configs/assist_config.yaml` at the
exported ONNX.  Each class in `dataset.yaml` then maps to a word in the
vocabulary (build the manifest with the extra class names added, or the
word list updated via the builder).

## Honest guidance

- **More classes ≠ better accuracy.**  A 1000-class model is harder to
  train than a 20-class one.  Start from the pretrained YOLOv8 weights
  (never from scratch), keep `max_images_per_class` high (50+) for the
  classes you actually care about, and validate with the evaluation
  tooling before trusting any new accuracy claim.
- The downloader downloads **real** images and reports real failure
  counts — it never fabricates labels.
- Fine-tuning has not been run yet; per the perception evaluation policy,
  do it only once you have a real labelled set worth training on.