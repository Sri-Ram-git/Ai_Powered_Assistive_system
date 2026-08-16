"""Web user interface for the assistive vision system.

Pure presentation layer.  It knows nothing about the engine or the API
contract — it renders the dashboard page and serves static assets, and
talks to the backend exclusively over the JSON API (/api/*).

    src.ui.templates.dashboard.html
    src.ui.static.dashboard.css / dashboard.js
"""
import time
from pathlib import Path

from flask import Blueprint, Response, render_template

UI_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = str(UI_DIR / "templates")
STATIC_DIR = str(UI_DIR / "static")


def create_ui(pipeline) -> Blueprint:
    """Build a fresh presentation blueprint bound to the pipeline.

    A new blueprint per call keeps the composition root repeatable
    (tests create many apps without blueprint re-registration errors).
    """
    ui = Blueprint(
        "ui", __name__,
        template_folder=TEMPLATE_DIR,
        static_folder=STATIC_DIR,
        static_url_path="/static",
    )

    @ui.get("/")
    def index() -> str:
        return render_template("dashboard.html")

    @ui.get("/video_feed")
    def video_feed() -> Response:
        """MJPEG stream of the annotated camera feed (presentation)."""

        def generate():
            while True:
                jpeg = pipeline.latest_jpeg
                if jpeg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg +
                           b"\r\n")
                    time.sleep(1.0 / 30.0)
                else:
                    time.sleep(0.05)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return ui


__all__ = ["create_ui"]