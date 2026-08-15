"""Flask app exposing the vision pipeline to the web dashboard.

Endpoints:
    GET /             -> the dashboard page
    GET /video_feed   -> MJPEG stream of the annotated camera feed
    GET /api/state    -> JSON state (detections, distances, guidance,
                         FPS, latency)

Usage:
    python src/server/app.py [--config configs/assist_config.yaml] [--port 5000]
"""
import argparse
import sys
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.server.pipeline import PipelineConfig, PipelineServer  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

_logger = setup_logger("WebApp")

_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Assistive Vision — Live</title>
<style>
  :root {
    --bg:#050505; --surface:#131313; --panel:#1c1b1b; --border:#ffffff1a;
    --text:#e5e2e1; --dim:#bbc9cf; --accent:#00d1ff; --accent-dim:#a4e6ff;
    --safe:#4ade80; --hazard:#ef4444; --mono:'JetBrains Mono',monospace;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); min-height:100vh;
         font-family:'Inter',sans-serif; display:flex; flex-direction:column; }
  nav { display:flex; justify-content:space-between; align-items:center;
        padding:0 24px; height:64px; border-bottom:1px solid var(--border);
        background:rgba(19,19,19,.8); backdrop-filter:blur(20px);
        flex-shrink:0; }
  nav .brand { font-weight:700; font-size:24px; color:var(--accent-dim); }
  nav .brand small { color:var(--dim); font-size:14px; font-weight:400; margin-left:16px;}
  .pills { display:flex; gap:20px; align-items:center; }
  .pill { display:flex; align-items:center; gap:8px; font-size:13px; color:var(--dim);}
  .dot { width:9px; height:9px; border-radius:50%; }
  .dot.green { background:var(--safe); animation:pulse 1.6s infinite; }
  .dot.cyan  { background:var(--accent); animation:pulse 1.6s infinite; }
  @keyframes pulse { 50% { opacity:.35; } }
  main { flex:1; display:flex; gap:24px; padding:24px; max-width:1920px;
         width:100%; margin:0 auto; overflow:hidden; }
  .feed-wrap { flex:3; position:relative; border-radius:12px; overflow:hidden;
               background:#000; border:1px solid var(--border); min-height:0; }
  .feed { width:100%; height:100%; object-fit:contain; display:block; }
  .rec { position:absolute; top:18px; left:18px; display:flex; align-items:center;
         gap:8px; background:rgba(0,0,0,.55); backdrop-filter:blur(8px);
         padding:6px 12px; border-radius:8px; font-family:var(--mono);
         font-size:13px; }
  .rec .r { width:11px; height:11px; border-radius:50%; background:var(--hazard);
            animation:pulse 1.2s infinite; }
  aside { flex:1; display:flex; flex-direction:column; gap:24px; min-width:320px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:12px;
          padding:24px; }
  .card h2 { font-family:var(--mono); font-size:13px; letter-spacing:.08em;
             text-transform:uppercase; color:var(--accent); margin-bottom:12px;
             display:flex; align-items:center; gap:8px; }
  .guidance { font-size:24px; font-weight:600; line-height:1.3; }
  .guidance em { color:var(--accent); font-style:normal; }
  .speech { display:flex; align-items:center; gap:10px; margin-top:20px;
            color:var(--safe); }
  .bars { display:flex; gap:4px; align-items:center; height:22px; }
  .bars span { width:5px; background:var(--safe); border-radius:2px;
               animation:bounce 1.1s infinite; }
  .bars span:nth-child(1){height:7px;animation-delay:.1s}
  .bars span:nth-child(2){height:16px;animation-delay:.2s}
  .bars span:nth-child(3){height:22px;animation-delay:.3s}
  .bars span:nth-child(4){height:12px;animation-delay:.4s}
  @keyframes bounce { 0%,100%{transform:scaleY(.5)} 50%{transform:scaleY(1.1)} }
  .list { display:flex; flex-direction:column; gap:10px; overflow-y:auto;
          max-height:380px; }
  .item { display:flex; justify-content:space-between; align-items:center;
          padding:12px 14px; border-radius:8px; background:#ffffff08;
          border:1px solid var(--border); }
  .item .l { display:flex; align-items:center; gap:10px; }
  .item .conf { font-family:var(--mono); font-size:14px; }
  .item .dist { font-family:var(--mono); font-size:13px; color:var(--dim); }
  .tag { font-family:var(--mono); font-size:12px; padding:2px 8px; border-radius:6px;
         border:1px solid var(--accent); color:var(--accent-dim); }
  footer { flex-shrink:0; display:flex; justify-content:space-between;
           align-items:center; padding:0 24px; height:40px; font-size:12px;
           color:var(--dim); border-top:1px solid var(--border);
           font-family:var(--mono); }
  .spacer{height:0;}
</style>
</head>
<body>
<nav>
  <div class="brand">Assistive Vision <small>AI Vision Assistant</small></div>
  <div class="pills">
    <div class="pill"><span class="dot green"></span><span id="sysStatus">System Ready</span></div>
    <div class="pill"><span class="dot cyan"></span><span id="camStatus">Camera Active</span></div>
  </div>
</nav>
<main>
  <section class="feed-wrap">
    <img id="feed" class="feed" src="/video_feed" alt="camera feed"/>
    <div class="rec"><span class="r"></span><span id="recTime">REC 00:00:00</span></div>
  </section>
  <aside>
    <div class="card">
      <h2>&#x1F9E0; AI Guidance</h2>
      <p class="guidance" id="guidance">Scanning the environment...</p>
      <div class="speech" id="speechRow" style="visibility:hidden">
        <span>&#x1F50A;</span><div class="bars"><span></span><span></span><span></span><span></span></div>
      </div>
    </div>
    <div class="card" style="flex:1; display:flex; flex-direction:column;">
      <h2>&#x1F441; Current Detections</h2>
      <div class="list" id="detList"></div>
    </div>
  </aside>
</main>
<footer>
  <span>&copy; 2026 Assistive Vision AI</span>
  <span>FPS: <b id="fps">--</b> &nbsp; Res: <b id="res">--</b> &nbsp; AI: <b id="aiStat" style="color:var(--safe)">Active</b> &nbsp; Latency: <b id="latency">--</b></span>
</footer>
<script>
const $ = id => document.getElementById(id);
const recStart = Date.now();
setInterval(() => {
  const s = Math.floor((Date.now() - recStart) / 1000);
  const h = String(Math.floor(s/3600)).padStart(2,'0');
  const m = String(Math.floor(s%3600/60)).padStart(2,'0');
  const ss = String(s%60).padStart(2,'0');
  $('recTime').textContent = `REC ${h}:${m}:${ss}`;
}, 1000);

async function refresh() {
  try {
    const r = await fetch('/api/state');
    const s = await r.json();
    $('fps').textContent = (s.fps ?? 0).toFixed(1);
    $('res').textContent = (s.resolution||[]).join('x') || '--';
    $('latency').textContent = (s.latency_ms ?? 0).toFixed(0) + 'ms';
    if (s.error) { $('sysStatus').textContent='Error'; $('camStatus').textContent=s.error; }

    const guidance = $('guidance');
    if (s.guidance) {
      guidance.innerHTML = 'Detected <em>' + s.guidance + '</em>';
      $('speechRow').style.visibility = 'visible';
    } else {
      guidance.textContent = 'Scanning the environment...';
      $('speechRow').style.visibility = 'hidden';
    }

    const list = $('detList');
    list.innerHTML = '';
    if (s.detections && s.detections.length) {
      for (const d of s.detections) {
        const item = document.createElement('div');
        item.className = 'item';
        item.innerHTML = `
          <div class="l">
            <span class="tag">#${d.track_id}</span>
            <span>${d.label}</span>
            <span class="dist">${d.direction}</span>
          </div>
          <div style="text-align:right">
            <div class="conf">${(d.confidence*100).toFixed(0)}%</div>
            <div class="dist">${d.distance.toFixed(1)} m</div>
          </div>`;
        list.appendChild(item);
      }
    } else {
      list.innerHTML = '<div style="color:var(--dim)">No objects detected.</div>';
    }
  } catch (e) { /* feed/state temporarily unavailable */ }
}
setInterval(refresh, 700);
refresh();
</script>
</body>
</html>
"""


def create_app(pipeline: PipelineServer) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return render_template_string(_PAGE)

    @app.route("/video_feed")
    def video_feed() -> Response:
        def generate():
            while True:
                jpeg = pipeline.latest_jpeg
                if jpeg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg +
                           b"\r\n")
                    # Rate-limit to ~30 fps even if the client drains
                    # faster; avoids flooding the network for no gain.
                    time.sleep(1.0 / 30.0)
                else:
                    time.sleep(0.05)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/state")
    def api_state() -> Response:
        return jsonify(pipeline.state_snapshot())

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistive Vision Web UI")
    parser.add_argument("--config", default=str(
        PROJECT_ROOT / "configs" / "assist_config.yaml"))
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    pipeline = PipelineServer(cfg)

    # Wire TTS so the dashboard speaks too.
    try:
        from src.audio import SpeechOutput
        tts = SpeechOutput()
    except Exception as exc:  # pragma: no cover - env dependent
        _logger.warning("TTS unavailable: %s", exc)
        tts = None

    pipeline.start(speech_callback=tts.speak if tts else None)

    app = create_app(pipeline)
    print(f"\nAssistive Vision dashboard:  http://{args.host}:{args.port}\n")
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
