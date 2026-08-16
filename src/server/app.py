"""Composition root for the assistive vision web app.

Wires the three architectural layers together:

    src.core.AsyncVisionPipeline   -> engine (no HTTP/UI knowledge)
    src.api                        -> JSON API blueprint (/api/*)
    src.ui                         -> dashboard blueprint (page + video)

Usage:
    python src/server/app.py [--config configs/assist_config.yaml]
                             [--port 5000] [--host 127.0.0.1]
"""
import argparse
import sys
from pathlib import Path

from flask import Flask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import create_api  # noqa: E402
from src.server.pipeline import PipelineConfig, PipelineServer  # noqa: E402
from src.ui import create_ui  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("WebApp")


def create_app(pipeline: PipelineServer) -> Flask:
    """Build the Flask app from the core/API/UI layers."""
    app = Flask(__name__, static_folder=None)  # UI blueprint owns /static

    # JSON API layer (decoupled from presentation).
    app.register_blueprint(create_api(pipeline))

    # Presentation layer (fresh blueprint bound to this pipeline).
    app.register_blueprint(create_ui(pipeline))
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistive Vision Web App")
    parser.add_argument("--config", default=str(
        PROJECT_ROOT / "configs" / "assist_config.yaml"))
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    pipeline = PipelineServer(cfg)

    # Wire TTS so the device can speak through the dashboard too.
    try:
        from src.audio import SpeechOutput
        tts = SpeechOutput()
    except Exception as exc:  # pragma: no cover - env dependent
        _logger.warning("TTS unavailable: %s", exc)
        tts = None

    pipeline.start(speech_callback=tts.speak if tts else None)

    app = create_app(pipeline)
    print(f"\nAssistive Vision dashboard:  "
          f"http://{args.host}:{args.port}\n")
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if tts is not None:
            tts.shutdown()


if __name__ == "__main__":
    main()