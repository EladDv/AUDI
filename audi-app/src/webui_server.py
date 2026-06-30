"""
AUDI Type A — Web UI

Minimal Flask web server serving a touch-friendly interface for a
small Pi display. Shows live status, big control buttons, and
alarm history.
"""

import logging
import math
import os
import threading
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file

logger = logging.getLogger("audi.webui")

# Path to HTML template
HERE = Path(__file__).parent
TEMPLATE_PATH = HERE / ".." / "webui" / "index.html"
VU_DB_FLOOR = -120.0
VU_DB_CEILING = 0.0


def rms_to_dbfs(rms: float, floor_db: float = VU_DB_FLOOR) -> float:
    """Convert linear full-scale RMS to dBFS for the VU display."""
    try:
        value = float(rms)
    except (TypeError, ValueError):
        return floor_db
    if value <= 0.0 or not math.isfinite(value):
        return floor_db
    value = min(value, 1.0)
    return max(floor_db, 20.0 * math.log10(value))


def dbfs_to_vu_percent(db: float) -> float:
    """Map dBFS to a 0-100 VU bar where 0 dBFS is full scale."""
    try:
        value = float(db)
    except (TypeError, ValueError):
        value = VU_DB_FLOOR
    if not math.isfinite(value):
        value = VU_DB_FLOOR
    value = max(VU_DB_FLOOR, min(VU_DB_CEILING, value))
    return ((value - VU_DB_FLOOR) / (VU_DB_CEILING - VU_DB_FLOOR)) * 100.0


def _cpu_temp() -> float:
    """Try to read CPU temperature from common Linux paths."""
    paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
    ]
    for p in paths:
        try:
            with open(p) as f:
                return int(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError):
            continue
    return 0.0


def _memory() -> dict:
    """Read memory info from /proc/meminfo."""
    mem = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    mem["total_kb"] = int(line.split()[1])
                elif "MemAvailable" in line:
                    mem["available_kb"] = int(line.split()[1])
    except OSError:
        pass
    total = mem.get("total_kb", 0)
    avail = mem.get("available_kb", 0)
    used_pct = round((1 - avail / total) * 100, 1) if total > 0 else 0
    return {"used_pct": used_pct, "total_kb": total, "available_kb": avail}


def _load() -> dict:
    try:
        a, b, c = os.getloadavg()
        return {"1min": round(a, 2), "5min": round(b, 2), "15min": round(c, 2)}
    except OSError:
        return {"1min": 0, "5min": 0, "15min": 0}


def _uptime() -> tuple[float, str]:
    try:
        with open("/proc/uptime") as f:
            secs = float(f.read().split()[0])
    except OSError:
        secs = 0
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    return secs, f"{h}h {m}m"


def _disk() -> dict:
    try:
        stat = os.statvfs("/")
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bavail * stat.f_frsize
        used_pct = round((1 - free / total) * 100, 1) if total > 0 else 0
        return {"used_pct": used_pct, "free_gb": round(free / 1e9, 1)}
    except OSError:
        return {"used_pct": 0, "free_gb": 0}


def system_stats() -> dict:
    uptime_secs, uptime_str = _uptime()
    return {
        "cpu_temperature_c": _cpu_temp(),
        "memory": _memory(),
        "load": _load(),
        "uptime_seconds": uptime_secs,
        "uptime_str": uptime_str,
        "disk": _disk(),
    }


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
        self.doa = None
        self.gpio = None

        self._app = Flask(__name__)
        self._register_routes()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _compact_mel(mel: np.ndarray, max_columns: int) -> np.ndarray:
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim != 2:
            raise ValueError(f"Expected 2D mel array, got {mel.shape}")
        if mel.shape[1] > max_columns:
            edges = np.linspace(0, mel.shape[1], max_columns + 1).astype(int)
            mel = np.stack(
                [
                    mel[:, edges[i] : max(edges[i] + 1, edges[i + 1])].mean(
                        axis=1
                    )
                    for i in range(max_columns)
                ],
                axis=1,
            )
        return mel

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
            if not TEMPLATE_PATH.exists():
                return "Web UI template missing", 500
            return send_file(TEMPLATE_PATH)

        @app.route("/api/status")
        def api_status():
            """Full system status as JSON."""
            data = {
                "recorder": self.recorder and self.recorder.status or {},
                "storage": self.storage and self.storage.status or {},
                "detector": self.detector and self.detector.status or {},
                "doa": self.doa and self.doa.status or {},
                "gpio": self.gpio and self.gpio.status() or {},
            }
            return jsonify(data)

        @app.route("/api/doa")
        def api_doa():
            """Return the latest DOA estimate. Estimation runs only on detections."""
            if not self.doa:
                return jsonify({"enabled": False, "last_result": {}})
            return jsonify(self.doa.status)

        @app.route("/api/doa_profile", methods=["POST"])
        def api_doa_profile():
            """Switch the active DOA profile without restarting the app."""
            if not self.doa:
                return jsonify({"error": "DOA estimator not running"}), 503
            payload = request.get_json(silent=True) or {}
            profile = str(payload.get("profile", ""))
            if not profile:
                return jsonify({"error": "profile is required"}), 400
            try:
                status = self.doa.set_profile(profile)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            return jsonify({"status": "doa_profile_updated", "doa": status})

        @app.route("/api/doa_channel", methods=["POST"])
        def api_doa_channel():
            """Enable or disable one captured channel for DOA."""
            if not self.doa:
                return jsonify({"error": "DOA estimator not running"}), 503
            payload = request.get_json(silent=True) or {}
            if "channel_index" not in payload or "enabled" not in payload:
                return jsonify({"error": "channel_index and enabled are required"}), 400
            try:
                status = self.doa.set_channel_enabled(
                    int(payload["channel_index"]),
                    bool(payload["enabled"]),
                )
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            return jsonify({"status": "doa_channel_updated", "doa": status})

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

        @app.route("/api/alert_routing", methods=["POST"])
        def api_alert_routing():
            """Update which color classes are allowed to trigger alerts."""
            if not self.detector:
                return jsonify({"error": "Detector not running"}), 503
            payload = request.get_json(silent=True) or {}
            status = self.detector.set_alert_routing(
                alert_on_blue=payload.get("alert_on_blue")
                if "alert_on_blue" in payload
                else None,
                alert_on_unknown=payload.get("alert_on_unknown")
                if "alert_on_unknown" in payload
                else None,
            )
            return jsonify({"status": "alert_routing_updated", "detector": status})

        @app.route("/api/detector_channel", methods=["POST"])
        def api_detector_channel():
            """Enable or disable detector inference for one captured channel."""
            if not self.detector:
                return jsonify({"error": "Detector not running"}), 503
            payload = request.get_json(silent=True) or {}
            if "channel_index" not in payload or "enabled" not in payload:
                return jsonify({"error": "channel_index and enabled are required"}), 400
            try:
                status = self.detector.set_channel_enabled(
                    int(payload["channel_index"]),
                    bool(payload["enabled"]),
                )
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            return jsonify({"status": "detector_channel_updated", "detector": status})

        @app.route("/api/clear_alarm", methods=["POST"])
        def api_clear_alarm():
            """Manually clear the GPIO alert from UI."""
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
            if not self.recorder:
                return jsonify({"rms": 0.0, "peak": 0.0, "db": -120.0})
            r = self.recorder.recorder
            rms = r.get_rms_level()
            db = rms_to_dbfs(rms)
            return jsonify(
                {
                    "rms": round(rms, 4),
                    "peak": min(1.0, rms * 3.0),
                    "db": round(db, 1),
                    "vu_percent": round(dbfs_to_vu_percent(db), 1),
                }
            )

        @app.route("/api/mels")
        def api_mels():
            """Return compact recent mel frames for up to four captured channels."""
            if not self.recorder:
                return jsonify({"channels": [], "error": "Recorder not available"}), 503
            try:
                seconds = min(
                    120.0,
                    max(1.0, float(request.args.get("seconds", "8"))),
                )
                max_columns = min(
                    80,
                    max(2, int(request.args.get("columns", "24"))),
                )
                source = request.args.get("source", "ring_buffer")
                r = self.recorder.recorder
                det_cfg = self.detector.status if self.detector else {}
                n_mels = int(getattr(self.detector, "n_mels", det_cfg.get("n_mels", 128)))
                n_fft = int(getattr(self.detector, "n_fft", det_cfg.get("n_fft", 1024)))
                win_length = int(det_cfg.get("win_length", n_fft))
                win_length = int(getattr(self.detector, "win_length", win_length))
                hop_length = int(
                    getattr(self.detector, "hop_length", det_cfg.get("hop_length", 160))
                )
                model_sr = int(
                    getattr(
                        self.detector,
                        "model_sample_rate",
                        det_cfg.get("model_sample_rate", r.sample_rate),
                    )
                )

                cached = (
                    getattr(self.detector, "latest_mels", None)
                    if self.detector
                    else None
                )
                if source == "detector_cache" and cached and cached.get("channels"):
                    channels = []
                    for ch in cached["channels"][:4]:
                        mel = self._compact_mel(ch["mel"], max_columns)
                        channels.append(
                            {
                                "channel_index": int(ch["channel_index"]),
                                "frames": int(mel.shape[1]),
                                "n_mels": int(mel.shape[0]),
                                "min_db": round(float(np.min(mel)), 2),
                                "max_db": round(float(np.max(mel)), 2),
                                "mel": np.round(mel, 1).tolist(),
                            }
                        )
                    return jsonify(
                        {
                            "timestamp": cached.get("timestamp"),
                            "seconds": seconds,
                            "sample_rate": r.sample_rate,
                            "model_sample_rate": cached.get(
                                "sample_rate",
                                model_sr,
                            ),
                            "channels_available": len(cached["channels"]),
                            "source": "detector_cache",
                            "channels": channels,
                        }
                    )

                from audio_features import compute_mel_spectrogram, resample_audio

                try:
                    audio = r.ring_buffer.get_last_n_seconds(
                        seconds,
                        r.sample_rate,
                        channel=None,
                    )
                except TypeError:
                    audio = r.ring_buffer.get_last_n_seconds(seconds, r.sample_rate)
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim == 1:
                    audio = audio[:, np.newaxis]

                channels = []
                for channel_index in range(min(4, audio.shape[1])):
                    samples = audio[:, channel_index]
                    if samples.size == 0:
                        continue
                    if r.sample_rate != model_sr:
                        samples = resample_audio(samples, r.sample_rate, model_sr)
                    mel = compute_mel_spectrogram(
                        samples,
                        model_sr,
                        n_mels=n_mels,
                        n_fft=n_fft,
                        win_length=win_length,
                        hop_length=hop_length,
                    )
                    mel = self._compact_mel(mel, max_columns)
                    channels.append(
                        {
                            "channel_index": channel_index,
                            "frames": int(mel.shape[1]),
                            "n_mels": int(mel.shape[0]),
                            "min_db": round(float(np.min(mel)), 2),
                            "max_db": round(float(np.max(mel)), 2),
                            "mel": np.round(mel, 1).tolist(),
                        }
                    )
                return jsonify(
                    {
                        "timestamp": __import__("time").time(),
                        "seconds": seconds,
                        "sample_rate": r.sample_rate,
                        "model_sample_rate": model_sr,
                        "channels_available": int(audio.shape[1]),
                        "source": "ring_buffer_fallback",
                        "channels": channels,
                    }
                )
            except Exception as e:
                logger.error("Mel snapshot failed: %s", e)
                return jsonify({"channels": [], "error": str(e)}), 500

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

            payload = request.get_json(silent=True) or {}
            alert_level = str(payload.get("alert_level", "RED_ALERT")).upper()
            allowed_levels = {"RED_ALERT", "BLUE_ALERT", "UNKNOWN_ALERT"}
            if alert_level not in allowed_levels:
                return jsonify({"error": "invalid alert_level"}), 400

            color_by_level = {
                "RED_ALERT": ("RED", 0.95, 0.05),
                "BLUE_ALERT": ("BLUE", 0.05, 0.95),
                "UNKNOWN_ALERT": ("UNKNOWN", None, None),
            }
            drone_color, red_confidence, blue_confidence = color_by_level[alert_level]
            channel_index = int(payload.get("channel_index", 0) or 0)
            result = {
                "timestamp": time.time(),
                "alert_id": f"test_{int(time.time())}",
                "channel_index": channel_index,
                "channel_name": f"ch{channel_index}",
                "firing_channel_index": channel_index,
                "firing_channel_name": f"ch{channel_index}",
                "state": "YES",
                "alert_level": alert_level,
                "yes_confidence": 0.95,
                "threshold_yes": self.detector.threshold_yes
                if self.detector
                else 0.70,
                "test_alert": True,
                "drone_color": drone_color,
                "red_confidence": red_confidence,
                "blue_confidence": blue_confidence,
                "all_channel_results": [
                    {
                        "channel_index": i,
                        "channel_name": f"ch{i}",
                        "state": "YES" if i == channel_index else "NO",
                        "alert_level": alert_level if i == channel_index else "NO",
                        "yes_confidence": 0.95 if i == channel_index else 0.0,
                        "drone_color": drone_color if i == channel_index else "UNKNOWN",
                        "red_confidence": red_confidence if i == channel_index else None,
                        "blue_confidence": blue_confidence if i == channel_index else None,
                    }
                    for i in range(4)
                ],
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
            logger.warning("TEST %s TRIGGERED (simulated YES detection)", alert_level)
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
            return jsonify(system_stats())
