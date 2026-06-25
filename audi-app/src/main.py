#!/usr/bin/env python3
"""
AUDI Type A — Main Entrypoint

Orchestrates all subsystems:
  1. Loads configuration
  2. Starts audio recorder (continuous capture, ring buffer)
  3. Starts storage manager (FLAC compression, 32GB ring buffer eviction)
  4. Starts detection engine (TFLite, temporal hysteresis)
  5. Starts GPIO alert controller
  6. Starts web UI
  7. Graceful shutdown on SIGTERM/SIGINT
"""

import logging
import logging.handlers
import signal
import sys
import time
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

# Ensure src is importable
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_threshold_profiles(cfg: dict, config_file: Path) -> dict:
    """Load external threshold profiles referenced by the app config."""
    import yaml

    det_cfg = cfg.get("detection", {})
    profiles_file = det_cfg.get("threshold_profiles_file")
    if not profiles_file:
        return cfg

    profiles_path = Path(profiles_file)
    if not profiles_path.is_absolute():
        profiles_path = config_file.parent / profiles_path
    if not profiles_path.exists():
        logging.getLogger("audi").warning(
            "Threshold profiles file not found: %s", profiles_path
        )
        return cfg

    with open(profiles_path) as f:
        profiles_cfg = yaml.safe_load(f) or {}
    profiles = profiles_cfg.get("threshold_profiles", profiles_cfg)
    if not isinstance(profiles, dict):
        logging.getLogger("audi").warning(
            "Threshold profiles file has no usable profiles: %s",
            profiles_path,
        )
        return cfg

    det_cfg["threshold_profiles"] = profiles
    logging.getLogger("audi").info(
        "Threshold profiles loaded from %s", profiles_path
    )
    return cfg


def load_config(config_path: str = None) -> dict:
    """Load YAML config. Falls back to defaults if file missing."""
    import yaml

    paths = []
    if config_path:
        paths.append(Path(config_path))
    paths.extend(
        [
            HERE.parent / "config.yaml",
            Path("/etc/audi/config.yaml"),
            Path("config.yaml"),
        ]
    )

    for p in paths:
        if p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f) or {}
            logging.getLogger("audi").info("Config loaded from %s", p)
            return _load_threshold_profiles(cfg, p)

    # Defaults
    logging.getLogger("audi").warning(
        "No config file found — using defaults. Create config.yaml at %s",
        HERE.parent / "config.yaml",
    )
    return {
        "audio": {
            "device": "auto",
            "sample_rate": 16000,
            "channels": 4,
            "bit_depth": 32,
            "segment_duration": 300,
            "ring_buffer_seconds": 120,
            "device_retry_min": 2,
            "device_retry_max": 60,
        },
        "storage": {
            "max_size_gb": 32,
            "compress": True,
            "data_dir": "/data/recordings",
            "min_free_gb": 1,
            "cleanup_watermark_gb": 5,
            "alerts_dir": "/data/alerts",
            "max_alerts_gb": 2,
        },
        "detection": {
            "model_path": "/app/models/model_combined_mn10_mined_hardneg_blue_red.tflite",
            "model_type": "tflite",
            "model_sample_rate": 16000,
            "n_mels": 128,
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 160,
            "window_samples": 81920,
            "stride": 0.0625,
            "enabled_channels": None,
            "num_threads": 2,
            "active_threshold_profile": "mn10_p90",
            "threshold_profiles_file": "/app/threshold_profiles.yaml",
            "confidence_threshold_high": 0.6550,
            "red_color_threshold": 0.60,
            "blue_color_threshold": 0.60,
            "alert_on_red": True,
            "alert_on_blue": False,
            "alert_on_unknown": False,
            "save_color_trace": True,
            "inference_interval": 0.320,
            "hysteresis_window": 8,
            "hysteresis_ratio": 0.6,
            "hysteresis_margin": 0.05,
            "alarm_cooldown_s": 120,
            "labels": ["drone"],
            "alert_history_file": "/data/alerts/alert_history.json",
        },
        "gpio": {
            "enabled": True,
            "alert_pin": 2,
            "strobe_pin": 24,
            "reset_pin": 23,
            "record_led_pin": None,
            "record_button_pin": None,
            "pause_button_pin": None,
            "field_tag_green_pin": 22,
            "field_tag_yellow_pin": 27,
            "field_tag_red_pin": 17,
            "alert_duration_ms": 5000,
            "pulse_interval_ms": 500,
            "red_buzzer_on_ms": 120,
            "red_buzzer_off_ms": 80,
            "blue_buzzer_on_ms": 420,
            "blue_buzzer_off_ms": 280,
            "unknown_buzzer_on_ms": 250,
            "unknown_buzzer_off_ms": 250,
        },
        "web": {"host": "0.0.0.0", "port": 8080},
        "logging": {
            "level": "INFO",
            "file": "/var/log/audi.log",
            "max_size_mb": 10,
            "backup_count": 3,
        },
    }


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(cfg: dict):
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    # Root audi logger
    logger = logging.getLogger("audi")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s  %(name)-20s  %(levelname)-5s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stdout handler
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(level)
    stdout.setFormatter(formatter)
    logger.addHandler(stdout)

    # File handler (if path writable)
    log_file = log_cfg.get("file", "/var/log/audi.log")
    try:
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_cfg.get("max_size_mb", 10) * 1024 * 1024,
            backupCount=log_cfg.get("backup_count", 3),
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except (OSError, PermissionError):
        logger.warning("Cannot write log to %s — console only", log_file)

    return logger


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class AudioGuardApp:
    """Orchestrates all subsystems with graceful lifecycle."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("audi.main")
        self._shutdown_requested = False

        # Subsystems (initialized in start())
        self.recorder = None
        self.storage = None
        self.detector = None
        self.gpio = None
        self.webui = None

    def start(self):
        """Initialize and start all subsystems."""
        self.logger.info("=" * 60)
        self.logger.info("AUDI Type A starting...")
        self.logger.info("=" * 60)

        # 1. GPIO (first, so it's ready for alarms + buttons)
        from gpio_alarm import GPIOController

        self.gpio = GPIOController(self.config)
        gpio_ok = self.gpio.status()["has_gpio"]
        self.logger.info(
            "[1/5] GPIO: %s",
            "ready" if gpio_ok else "disabled",
        )

        # 2. Audio Recorder (continuous capture + ring buffer)
        from recorder import RecorderManager

        self.recorder = RecorderManager(self.config)
        self.recorder.recorder.start()
        self.logger.info(
            "[2/5] Recorder: %d Hz, %d ch, %ds segments, %ds ring buffer",
            self.recorder.recorder.sample_rate,
            self.recorder.recorder.channels,
            self.recorder.recorder.segment_duration,
            self.recorder.ring_buffer.max_samples
            // self.recorder.recorder.sample_rate,
        )

        # 3. Storage Manager (FLAC compression, 32GB ring buffer, alerts management)
        from storage import StorageManager

        self.storage = StorageManager(self.config)
        self.storage.start()
        self.logger.info(
            "[3/5] Storage: %.1f GB max (recordings), %.1f GB max (alerts), compress=%s",
            self.config["storage"]["max_size_gb"],
            self.config["storage"].get("max_alerts_gb", 2),
            self.config["storage"]["compress"],
        )

        # 4. Detection Engine (TFLite + alarm snapshots)
        from detector import DetectionEngine

        self.detector = DetectionEngine(
            self.config,
            ring_buffer=self.recorder.ring_buffer,
            on_alarm=self._on_alarm,
        )
        # Wire recorder reference for post-alarm audio capture
        self.detector.recorder = self.recorder
        self.detector.start()
        self.logger.info(
            "[4/5] Detector: YES≥%.2f interval=%.3fs",
            self.detector.threshold_yes,
            self.detector.inference_interval,
        )

        # 5. Wire GPIO button callbacks
        self.gpio.on_record_toggle = self._on_record_toggle
        self.gpio.on_pause_5m = self._on_pause_5m
        self.gpio.on_field_tag = self._on_field_tag
        # Turn on record LED immediately
        self.gpio.set_record_led(True)

        # 6. Web UI
        from webui_server import WebUI

        self.webui = WebUI(self.config)
        self.webui.recorder = self.recorder
        self.webui.storage = self.storage
        self.webui.detector = self.detector
        self.webui.gpio = self.gpio
        self.webui.start()
        self.logger.info(
            "[5/5] Web UI: http://%s:%s",
            self.config["web"]["host"],
            self.config["web"]["port"],
        )

        # 7. Start record LED heartbeat (syncs LED with recorder state)
        self._start_led_heartbeat()

        self.logger.info("-" * 60)
        self.logger.info("All systems running. Press Ctrl+C to stop.")
        self.logger.info("=" * 60)

    def _disable_radios(self):
        """Disable WiFi and Bluetooth via rfkill to reduce RF interference."""
        import subprocess

        for radio, label in [("wifi", "WiFi"), ("bluetooth", "Bluetooth")]:
            try:
                subprocess.run(
                    ["rfkill", "block", radio],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                self.logger.info("[0] %s: disabled via rfkill", label)
            except FileNotFoundError:
                self.logger.warning(
                    "[0] rfkill not found — cannot disable %s", label
                )
            except Exception as e:
                self.logger.warning("[0] Failed to disable %s: %s", label, e)

    def stop(self):
        """Graceful shutdown of all subsystems in reverse order."""
        self.logger.info("Shutting down...")

        if self.webui:
            self.webui.stop()
        if self.detector:
            self.detector.stop()
        if self.storage:
            self.storage.stop()
        if self.recorder:
            self.recorder.recorder.stop()
        if self.gpio:
            self.gpio.cleanup()

        self.logger.info("Shutdown complete.")

    def _on_alarm(self, detection: dict):
        """Callback when the detector finds something."""
        label = detection.get("alert_level") or detection.get("label", "unknown")
        confidence = detection.get("confidence", 0.0)
        if not confidence:
            confidence = detection.get("yes_confidence", 0.0)
        self.logger.warning("ALARM CALLBACK: %s (%.2f)", label, confidence)
        if self.gpio:
            self.gpio.trigger_alarm(label)

    def _on_record_toggle(self):
        """Toggle recording on/off (from GPIO record button or UI)."""
        if not self.recorder:
            return
        recording = self.recorder.recorder.toggle_start_stop()
        self.logger.info(
            "Record toggle: %s",
            "started" if recording else "stopped at end of segment",
        )
        # Update LED
        if self.gpio:
            self.gpio.set_record_led(recording)

    def _on_pause_5m(self):
        """Pause recording for 5 minutes (from GPIO pause button or UI)."""
        if not self.recorder:
            return
        r = self.recorder.recorder
        if r.is_paused:
            r.resume()
            self.logger.info("Pause cancelled — resuming recording")
        else:
            r.pause(300)
            self.logger.info("Pause 5m activated")
        # Update LED
        if self.gpio:
            self.gpio.set_record_led(not r.is_paused and not r.is_force_stopped)

    def _on_field_tag(self, tag: str):
        """Attach a field-button assessment to the most recent detection."""
        if not self.detector:
            self.logger.warning(
                "Field tag %s ignored: detector history is not available", tag
            )
            return
        try:
            updated = self.detector.alert_history.tag_latest(tag)
        except ValueError as e:
            self.logger.warning("Field tag %s rejected: %s", tag, e)
            return
        if updated is None:
            self.logger.warning("Field tag %s ignored: no detection history yet", tag)
            return
        self.logger.info(
            "Field tag %s applied to alert %s",
            tag,
            updated.get("alert_id", "<unknown>"),
        )

    def _start_led_heartbeat(self):
        """Background thread that keeps the record LED in sync with recorder state."""

        def _led_loop():
            while not getattr(self, "_shutdown_requested", False):
                try:
                    if self.recorder and self.gpio:
                        r = self.recorder.recorder
                        led_on = (
                            r.is_recording
                            and not r.is_paused
                            and not r.is_force_stopped
                        )
                        self.gpio.set_record_led(led_on)
                except Exception:
                    pass
                time.sleep(2)  # Check every 2 seconds

        threading.Thread(target=_led_loop, daemon=True).start()

    def run_forever(self):
        """Block until shutdown signal."""
        sig = threading.Event()

        def _signal_handler(signum, frame):
            if not self._shutdown_requested:
                self._shutdown_requested = True
                self.logger.info(
                    "Received signal %d — shutting down...", signum
                )
                sig.set()

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        sig.wait()
        self.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AUDI — Type A Audio Detection"
    )
    parser.add_argument("-c", "--config", help="Path to config.yaml")
    parser.add_argument(
        "--check", action="store_true", help="Check system and exit"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger("audi")

    # Check mode
    if args.check:
        logger.info("=== System Check ===")
        logger.info("Config: OK")
        logger.info("Audio: checking...")
        import subprocess

        try:
            r = subprocess.run(
                ["arecord", "--version"], capture_output=True, timeout=5
            )
            logger.info(
                "  arecord: %s", r.stdout.decode(errors="replace").strip()[:80]
            )
        except FileNotFoundError:
            logger.warning("  arecord: NOT FOUND (install alsa-utils)")
        try:
            r = subprocess.run(
                ["flac", "--version"], capture_output=True, timeout=5
            )
            logger.info(
                "  flac: %s", r.stdout.decode(errors="replace").strip()[:80]
            )
        except FileNotFoundError:
            logger.warning("  flac: NOT FOUND (install flac)")
        try:
            r = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, timeout=5
            )
            logger.info(
                "  ffmpeg: %s", r.stdout.decode(errors="replace").splitlines()[0][:80]
            )
        except FileNotFoundError:
            logger.warning("  ffmpeg: NOT FOUND (install ffmpeg for 9+ channels)")
        try:
            import_module("RPi.GPIO")
            logger.info("  RPi.GPIO: available")
        except (ImportError, RuntimeError) as exc:
            logger.warning("  RPi.GPIO: not usable on this host (%s)", exc)
        try:
            from ai_edge_litert.interpreter import Interpreter  # noqa: F401

            logger.info("  ai_edge_litert: available")
        except ImportError:
            logger.warning("  ai_edge_litert: not installed (using mock)")
        if find_spec("flask") is not None:
            logger.info("  flask: available")
        else:
            logger.warning("  flask: NOT FOUND")
        logger.info("Storage: %s", config["storage"]["data_dir"])
        Path(config["storage"]["data_dir"]).mkdir(parents=True, exist_ok=True)
        logger.info("  directory: OK")
        logger.info("=== Check complete ===")
        return

    # Start the app
    app = AudioGuardApp(config)
    try:
        app.start()
        app.run_forever()
    except KeyboardInterrupt:
        app.stop()
    except Exception as e:
        logger.critical("Fatal error: %s", e, exc_info=True)
        app.stop()
        sys.exit(1)


if __name__ == "__main__":
    import threading  # needed for run_forever

    main()
