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
    $('mode').textContent = (s.mode || 'object');
    if (s.error) { $('sysStatus').textContent='Error'; $('camStatus').textContent=s.error; }

    const guidance = $('guidance');
    if (s.risk && s.risk === 'high') {
      guidance.className = 'guidance hazard';
      guidance.textContent = s.guidance || 'Hazard ahead — stop';
      $('speechRow').style.visibility = 'visible';
    } else if (s.guidance) {
      guidance.className = 'guidance';
      guidance.innerHTML = 'Detected <em>' + s.guidance + '</em>';
      $('speechRow').style.visibility = 'visible';
    } else {
      guidance.className = 'guidance';
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

async function setMode(name) {
  try {
    await fetch('/api/mode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode: name}),
    });
  } catch (e) { /* transient */ }
}