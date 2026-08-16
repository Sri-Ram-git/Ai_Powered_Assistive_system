"""Run the full AI evaluation pipeline.

Loads a small, honest evaluation dataset (synthetic today) and writes a
metrics report to evaluation/results/.  Run with:

    python evaluation/scripts/run_evaluation.py

The report explicitly flags small-sample datasets so no statistically
meaningful claims are accidentally made.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.assistive_metrics import (  # noqa: E402
    AssistiveCase,
    evaluate_assistive,
)
from src.evaluation.detection_metrics import Box, evaluate_detections  # noqa: E402
from src.evaluation.ocr_metrics import (  # noqa: E402
    character_error_rate,
    text_detection_success,
    word_error_rate,
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def _avg(metric: str, images: list) -> float:
    vals = [im["metrics"][metric] for im in images]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    # --- Load the small synthetic dataset ---------------------------
    dataset = json.loads(
        (PROJECT_ROOT / "evaluation" / "datasets" / "synthetic.json")
        .read_text(encoding="utf-8")
    )
    images = dataset["images"]
    n_images = len(images)
    print(f"Evaluation dataset: {dataset['name']} "
          f"({n_images} image(s))")

    # --- Object detection -------------------------------------------
    det_results = []
    for im in images:
        preds = [Box(label=p["label"], confidence=p["confidence"],
                     box=tuple(p["box"]))
                 for p in im.get("predictions", [])]
        gts = [Box(label=g["label"], confidence=1.0,
                   box=tuple(g["box"]))
               for g in im.get("ground_truth", [])]
        det_results.append({"image": im["id"],
                            "metrics": evaluate_detections(preds, gts)})

    det_summary = {
        "precision": _avg("precision", det_results),
        "recall": _avg("recall", det_results),
        "mAP@50": _avg("mAP@50", det_results),
        "mAP@50:95": _avg("mAP@50:95", det_results),
        "false_positives": sum(im["metrics"]["false_positives"]
                               for im in det_results),
        "false_negatives": sum(im["metrics"]["false_negatives"]
                               for im in det_results),
    }

    # --- OCR --------------------------------------------------------
    cer, wer, dss = [], [], []
    for im in images:
        refs = im.get("reference_text", [])
        hyps = im.get("recognised_text", [])
        joined_ref = " ".join(refs)
        joined_hyp = " ".join(hyps)
        cer.append(character_error_rate(joined_ref, joined_hyp))
        wer.append(word_error_rate(joined_ref, joined_hyp))
        dss.append(text_detection_success(refs, hyps))

    ocr_summary = {
        "character_error_rate": sum(cer) / len(cer) if cer else 0.0,
        "word_error_rate": sum(wer) / len(wer) if wer else 0.0,
        "detection_success_rate": sum(dss) / len(dss) if dss else 0.0,
    }

    # --- End-to-end assistive --------------------------------------
    cases = [AssistiveCase(**c) for c in dataset.get("assistive_cases", [])]
    assistive = evaluate_assistive(cases)

    report = {
        "dataset": dataset["name"],
        "n_images": n_images,
        "statistical_caveat": (
            f"Only {n_images} image(s) — too small for statistically "
            "meaningful claims; treat as a smoke evaluation."
        ),
        "object_detection": det_summary,
        "ocr": ocr_summary,
        "assistive": assistive,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nObject detection (mean over {n_images} image(s)):")
    for k, v in det_summary.items():
        print(f"  {k:<16} {v:.3f}")
    print("\nOCR:")
    for k, v in ocr_summary.items():
        print(f"  {k:<24} {v:.3f}")
    print("\nAssistive (end-to-end):")
    for k, v in assistive.items():
        print(f"  {k:<20} {v}")
    print(f"\nReport written to {out}")
    print(f"\nCAVEAT: {report['statistical_caveat']}")


if __name__ == "__main__":
    main()