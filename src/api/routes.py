"""JSON API blueprint for the assistive vision system.

Decoupled from the dashboard UI: any client (web, mobile, script) can
consume the same contract.  Endpoints:

    GET  /api/health       -> status + uptime + latency
    GET  /api/state        -> current detections/guidance/OCR/FPS
    GET  /api/config       -> effective pipeline config (no secrets)
    POST /api/command      -> speak a voice-command string (parsed)
    POST /api/mode         -> switch product mode ("object"|"reading"|...)

All responses are JSON; the UI and future API clients share this.
"""
from flask import Blueprint, Response, jsonify, request

API_NAME = "api"


def create_api(pipeline) -> Blueprint:
    """Build the /api blueprint bound to the given PipelineServer."""
    api = Blueprint(API_NAME, __name__, url_prefix="/api")

    @api.get("/health")
    def health() -> Response:
        st = pipeline.state_snapshot()
        return jsonify({
            "status": "error" if st.get("error") else "ok",
            "uptime_s": round(st.get("uptime_s", 0.0), 1),
            "latency_ms": round(st.get("latency_ms", 0.0), 1),
            "fps": round(st.get("fps", 0.0), 1),
            "mode": st.get("mode", "object"),
        })

    @api.get("/state")
    def state() -> Response:
        return jsonify(pipeline.state_snapshot())

    @api.get("/config")
    def config() -> Response:
        from src.api.serialize import public_config

        return jsonify(public_config(pipeline.config))

    @api.post("/command")
    def command() -> Response:
        body = request.get_json(silent=True) or {}
        text = str(body.get("text", "")).strip()
        if not text:
            return jsonify({"ok": False, "error": "empty command"}), 400
        try:
            from src.speech.command_parser import parse_command

            parsed = parse_command(text)
            handled = pipeline.handle_command(parsed)
            return jsonify({
                "ok": handled,
                "command": parsed.command.value if parsed.command else None,
            })
        except Exception as exc:  # pragma: no cover - defensive
            return jsonify({"ok": False, "error": str(exc)}), 500

    @api.post("/mode")
    def mode() -> Response:
        body = request.get_json(silent=True) or {}
        name = str(body.get("mode", "")).strip()
        if not name:
            return jsonify({"ok": False, "error": "missing mode"}), 400
        try:
            pipeline.set_mode(name)
            return jsonify({"ok": True, "mode": name})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    return api