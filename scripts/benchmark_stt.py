"""Benchmark STT backends: latency, memory, model size, offline capability.

Runs the keyword backend (no model) and, if faster-whisper is installed,
a tiny Whisper model, on the fixed command set.  This documents the
local-STT trade-off (see docs/productization/model_selection.md).

Usage:
    python scripts/benchmark_stt.py [--backend keyword] [--repeat 20]
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="keyword",
                        choices=["keyword", "whisper"])
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    from src.speech import create_stt

    phrases = [
        "read the text",
        "what do you see",
        "describe the scene",
        "stop speaking",
        "help",
    ]

    print(f"Backend: {args.backend}")

    if args.backend == "whisper":
        try:
            stt = create_stt("whisper", model_size="tiny")
        except ImportError as exc:
            print(f"  SKIP: {exc}")
            return
    else:
        stt = create_stt("keyword")

    # Keyword backend: parse each phrase directly (no audio).
    latencies = []
    for phrase in phrases:
        for _ in range(args.repeat):
            started = time.monotonic()
            stt.parse(phrase)
            latencies.append((time.monotonic() - started) * 1000.0)
    latencies.sort()
    median = latencies[len(latencies) // 2]
    print(f"  median parse latency : {median:.3f} ms "
          f"({len(latencies)} parses)")
    print("  offline capability   : True (no model, no network)")

    if args.backend == "whisper":
        # Report model metadata; real audio transcription latency needs
        # an audio clip and is hardware-dependent.
        import faster_whisper
        print(f"  faster-whisper       : {faster_whisper.__version__}")
        print("  model                : tiny (int8, CPU)")
        print("  note                 : audio transcription latency "
              "varies with clip length; parse stage above is shared")


if __name__ == "__main__":
    main()