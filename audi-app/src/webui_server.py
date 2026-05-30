"""
Pi Audio Guard — Web UI

Minimal Flask web server serving a touch-friendly interface for a
small Pi display. Shows live status, big control buttons, and
alarm history.
"""

import logging
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

logger = logging.getLogger("audio_guard.webui")

# Path to HTML template
HERE = Path(__file__).parent
TEMPLATE_PATH = HERE / ".." / "webui" / "index.html"


class WebUI:
    """Flask-based touch UI for the audio guard system."""

    def __init__(self, config: dict):
        web_cfg = config.get("web", {})
        self.host = web_cfg.get("host", "0.0.0.0")
        self.port = web_cfg.get("port", 80)

        # References to other subsystems (set externally)
        self.recorder = None
        self.storage = None
        self.detector = None
        self.gpio = None

        self._app = Flask(__name__)
        self._register_routes()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(
            target=lambda: self._app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True,
            ),
            daemon=True,
        )
        self._thread.start()
        logger.info("Web UI started on http://%s:%s", self.host, self.port)

    def stop(self):
        logger.info("Web UI stopped")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _register_routes(self):
        app = self._app

        @app.route("/")
        def index():
            """Serve the main touch UI."""
            try:
                html = TEMPLATE_PATH.read_text(encoding="utf-8")
            except FileNotFoundError:
                html = self._fallback_html()
            return render_template_string(html)

        @app.route("/api/status")
        def api_status():
            """Full system status as JSON."""
            data = {
                "recorder": self.recorder and self.recorder.status or {},
                "storage": self.storage and self.storage.status or {},
                "detector": self.detector and self.detector.status or {},
                "gpio": self.gpio and self.gpio.status() or {},
            }
            return jsonify(data)

        @app.route("/api/alarm_history")
        def api_alarm_history():
            """Return recent alarm history from detector."""
            if not self.detector:
                return jsonify({"history": []})
            last = self.detector.last_inference
            return jsonify(
                {
                    "last_inference": last,
                    "total_detections": self.detector.status.get(
                        "total_detections", 0
                    ),
                    "alarms_triggered": self.detector.status.get(
                        "alarms_triggered", 0
                    ),
                    "smoothed": self.detector.smoothed_predictions,
                }
            )

        @app.route("/api/force_inference", methods=["POST"])
        def api_force_inference():
            """Manually trigger a detection cycle."""
            if not self.detector:
                return jsonify({"error": "Detector not running"}), 503
            result = self.detector.force_inference()
            return jsonify({"result": result})

        @app.route("/api/threshold_profile", methods=["POST"])
        def api_threshold_profile():
            """Switch the active detector threshold profile."""
            if not self.detector:
                return jsonify({"error": "Detector not running"}), 503
            payload = request.get_json(silent=True) or {}
            profile = str(payload.get("profile", ""))
            if not profile:
                return jsonify({"error": "profile is required"}), 400
            try:
                status = self.detector.set_threshold_profile(profile)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            return jsonify({"status": "profile_updated", "detector": status})

        @app.route("/api/clear_alarm", methods=["POST"])
        def api_clear_alarm():
            """Manually clear GPIO alarm from UI."""
            if self.gpio:
                self.gpio.clear_alarm()
                return jsonify({"status": "cleared"})
            return jsonify({"error": "GPIO not available"}), 503

        @app.route("/api/restart_recorder", methods=["POST"])
        def api_restart_recorder():
            """Restart the audio recorder."""
            if not self.recorder:
                return jsonify({"error": "Recorder not available"}), 503
            try:
                self.recorder.recorder.stop()
                self.recorder.recorder.start()
                return jsonify({"status": "restarted"})
            except Exception as e:
                logger.error("Restart failed: %s", e)
                return jsonify({"error": str(e)}), 500

        @app.route("/api/audio_level")
        def api_audio_level():
            """Live RMS audio level (0.0–1.0) for VU meter."""
            import math

            if not self.recorder:
                return jsonify({"rms": 0.0, "peak": 0.0, "db": -120.0})
            r = self.recorder.recorder
            rms = r.get_rms_level()
            db = 20.0 * math.log10(max(float(rms), 1e-10))
            return jsonify(
                {
                    "rms": round(rms, 4),
                    "peak": min(1.0, rms * 3.0),
                    "db": round(db, 1),
                }
            )

        @app.route("/api/alert_history_persistent")
        def api_alert_history_persistent():
            """Return persistent alert history from JSON file."""
            if not self.detector:
                return jsonify({"history": [], "total": 0})
            entries = self.detector.alert_history.read_recent(100)
            return jsonify(
                {
                    "history": entries,
                    "total": self.detector.alert_history.count,
                }
            )

        @app.route("/api/toggle_recording", methods=["POST"])
        def api_toggle_recording():
            """Toggle recording on/off."""
            if not self.recorder:
                return jsonify({"error": "Not available"}), 503
            recording = self.recorder.recorder.toggle_start_stop()
            return jsonify({"recording": recording})

        @app.route("/api/start_recording", methods=["POST"])
        def api_start_recording():
            """Start recording immediately. Also cancels any active pause."""
            if not self.recorder:
                return jsonify({"error": "Not available"}), 503
            r = self.recorder.recorder
            # Cancel pause if active
            if r.is_paused:
                r.resume()
            # Start if stopped
            if r.is_force_stopped or not r.is_recording:
                r._force_stopped = False
                if not r.is_recording:
                    r.start()
            return jsonify({"recording": True})

        @app.route("/api/stop_recording", methods=["POST"])
        def api_stop_recording():
            """Stop recording (finishes current segment first)."""
            if not self.recorder:
                return jsonify({"error": "Not available"}), 503
            self.recorder.recorder._force_stopped = True
            return jsonify({"recording": False})

        @app.route("/api/pause_recording", methods=["POST"])
        def api_pause_recording():
            """Pause recording for 5 minutes."""
            if not self.recorder:
                return jsonify({"error": "Not available"}), 503
            self.recorder.recorder.pause(300)
            return jsonify({"paused": True, "duration_seconds": 300})

        @app.route("/api/test_alert", methods=["POST"])
        def api_test_alert():
            """Trigger a test YES alert to verify the full alert chain."""
            import time

            result = {
                "timestamp": time.time(),
                "alert_id": f"test_{int(time.time())}",
                "state": "YES",
                "alert_level": "RED_ALERT",
                "yes_confidence": 0.95,
                "threshold_yes": self.detector.threshold_yes
                if self.detector
                else 0.70,
                "test_alert": True,
                "drone_color": "RED",
                "red_confidence": 0.95,
                "blue_confidence": 0.05,
            }
            if self.detector:
                # Persist to alert history
                self.detector.alert_history.append(result)
                # Save pre/post audio snapshots in background
                import threading

                threading.Thread(
                    target=self.detector._save_snapshot_and_alert,
                    args=(result,),
                    daemon=True,
                ).start()
            if self.gpio:
                self.gpio.trigger_alarm(result["alert_level"])
            logger.warning("TEST ALERT TRIGGERED (simulated YES detection)")
            return jsonify({"status": "alert_triggered", "result": result})

        @app.route("/api/label_alert", methods=["POST"])
        def api_label_alert():
            """Attach an operator label to an alert history entry."""
            if not self.detector:
                return jsonify({"error": "Detector not running"}), 503
            payload = request.get_json(silent=True) or {}
            alert_id = str(payload.get("alert_id", ""))
            label = str(payload.get("label", ""))
            if not alert_id or not label:
                return jsonify({"error": "alert_id and label are required"}), 400
            try:
                updated = self.detector.alert_history.label_alert(alert_id, label)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            if updated is None:
                return jsonify({"error": "alert not found"}), 404
            return jsonify({"status": "labeled", "alert": updated})

        @app.route("/api/system")
        def api_system():
            """System health: CPU temp, memory, uptime, load."""
            from system_info import all_stats

            return jsonify(all_stats())

    # ------------------------------------------------------------------
    # Fallback inline template
    # ------------------------------------------------------------------

    def _fallback_html(self) -> str:
        """Inline fallback if template file is missing."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no">
<title>Pi Audio Guard</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #111; color: #eee; padding: 16px; }
h1 { font-size: 1.4rem; margin-bottom: 12px; color: #0af; }
.status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px; }
.status-card { background: #1a1a2e; border-radius: 10px; padding: 12px; }
.status-card .label { font-size: 0.7rem; color: #888; text-transform: uppercase; }
.status-card .value { font-size: 1.1rem; font-weight: 600; margin-top: 4px; }
.btn-group { display: flex; gap: 12px; margin-top: 16px; }
.btn { flex: 1; padding: 20px; border: none; border-radius: 14px; font-size: 1.1rem;
  font-weight: 700; color: #fff; cursor: pointer; touch-action: manipulation; }
.btn-primary { background: #0a7; }
.btn-danger { background: #c33; }
.btn-warning { background: #c83; }
.btn:active { transform: scale(0.95); opacity: 0.8; }
.alarm-box { margin-top: 16px; padding: 16px; border-radius: 12px; text-align: center; }
.alarm-box.active { background: #3a1111; border: 2px solid #c33; }
.alarm-box.inactive { background: #1a1a2e; }
.alarm-label { font-size: 1rem; margin-bottom: 8px; }
.alarm-label.alert { color: #f44; font-weight: 700; }
.prob-bar { margin: 4px 0; display: flex; align-items: center; }
.prob-bar .label { width: 100px; font-size: 0.8rem; }
.prob-bar .track { flex: 1; height: 16px; background: #222; border-radius: 8px; overflow: hidden; }
.prob-bar .fill { height: 100%; border-radius: 8px; transition: width 0.3s; }
.footer { margin-top: 24px; font-size: 0.7rem; color: #555; text-align: center; }
</style>
</head>
<body>
<h1>Pi Audio Guard</h1>
<div class="status-grid" id="statusGrid"></div>
<div class="alarm-box inactive" id="alarmBox">
  <div class="alarm-label">Status: Monitoring</div>
  <div id="topPrediction"></div>
</div>
<div id="probBars"></div>
<div class="btn-group">
  <button class="btn btn-primary" onclick="forceInference()">Force Scan</button>
  <button class="btn btn-danger" onclick="clearAlarm()">Silence Alarm</button>
</div>
<div class="footer" id="footer"></div>
<script>
function update() {
  fetch('/api/status').then(r=>r.json()).then(d=>{
    const g = document.getElementById('statusGrid');
    const rec = d.recorder || {};
    const st = d.storage || {};
    const det = d.detector || {};
    const gp = d.gpio || {};
    g.innerHTML = `
      <div class="status-card"><div class="label">Recording</div>
        <div class="value" style="color:${rec.running?'#0a7':'#c33'}">${rec.running?'Active':'Stopped'}</div></div>
      <div class="status-card"><div class="label">Storage</div>
        <div class="value">${st.used_gb||0} / ${st.max_size_gb||32} GB</div></div>
      <div class="status-card"><div class="label">Detections</div>
        <div class="value">${det.alarms_triggered||0}</div></div>
      <div class="status-card"><div class="label">GPIO Alarm</div>
        <div class="value" style="color:${gp.alarming?'#f44':'#0a7'}">${gp.alarming?'ACTIVE':'Idle'}</div></div>
    `;
    const box = document.getElementById('alarmBox');
    box.className = 'alarm-box ' + (gp.alarming ? 'active' : 'inactive');
    document.getElementById('topPrediction').textContent = '';
  });
  fetch('/api/alarm_history').then(r=>r.json()).then(d=>{
    const pb = document.getElementById('probBars');
    const s = d.smoothed || {};
    const colors = ['#f44','#fa0','#fc0','#0af','#0f7','#a0f'];
    const entries = Object.entries(s).sort((a,b)=>b[1]-a[1]);
    pb.innerHTML = entries.map(([l,p],i)=>`
      <div class="prob-bar">
        <div class="label">${l}</div>
        <div class="track"><div class="fill" style="width:${(p*100).toFixed(0)}%;background:${colors[i%colors.length]}"></div></div>
        <span style="margin-left:8px;font-size:0.8rem;width:40px">${(p*100).toFixed(0)}%</span>
      </div>
    `).join('');
    const top = entries[0];
    if (top && top[1] > (d.last_inference?.threshold || 0.7)) {
      document.getElementById('topPrediction').innerHTML =
        '<strong>DETECTED: ' + top[0] + '</strong>';
    }
  });
  document.getElementById('footer').textContent =
    'Updated: ' + new Date().toLocaleTimeString();
}
function forceInference() {
  fetch('/api/force_inference',{method:'POST'}).then(()=>update());
}
function clearAlarm() {
  fetch('/api/clear_alarm',{method:'POST'}).then(()=>update());
}
setInterval(update, 2000);
update();
</script>
</body>
</html>"""
