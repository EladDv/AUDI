"""Smoke tests for audi-app (RPi deployment)."""

import importlib
import sys
import wave
from pathlib import Path

import numpy as np


def _import_from_audi_app(module_name: str):
    """Import a module from audi-app/src by adding it to sys.path."""
    app_src = Path(__file__).resolve().parents[1] / "audi-app" / "src"
    if str(app_src) not in sys.path:
        sys.path.insert(0, str(app_src))
    return importlib.import_module(module_name)


class TestAudiApp:
    def test_detector_imports(self):
        """detector.py imports without error."""
        mod = _import_from_audi_app("detector")
        assert hasattr(mod, "DetectionEngine")

    def test_recorder_imports(self):
        """recorder.py imports without error."""
        mod = _import_from_audi_app("recorder")
        assert hasattr(mod, "RecorderManager")
        assert hasattr(mod, "AudioRingBuffer")

    def test_storage_imports(self):
        """storage.py imports without error."""
        mod = _import_from_audi_app("storage")
        assert hasattr(mod, "StorageManager")

    def test_gpio_alarm_imports(self):
        """gpio_alarm.py imports without error."""
        mod = _import_from_audi_app("gpio_alarm")
        assert hasattr(mod, "GPIOController")

    def test_main_imports(self):
        """main.py imports without error."""
        mod = _import_from_audi_app("main")
        assert hasattr(mod, "AudioGuardApp")
        assert hasattr(mod, "load_config")

    def test_alarm_snapshot_writes_utc_metadata(self, tmp_path):
        """Alarm snapshots should not fail while writing UTC metadata."""
        detector = _import_from_audi_app("detector")

        class RingBuffer:
            def get_last_n_seconds(self, n_seconds, sample_rate):
                return np.zeros(int(n_seconds * sample_rate), dtype=np.float32)

        snapshotter = detector.AlarmSnapshotter(str(tmp_path), sample_rate=8000)
        snapshotter._stop_event.set()

        meta = snapshotter.save_snapshot(
            RingBuffer(),
            {
                "state": "YES",
                "yes_confidence": 0.9,
                "threshold_yes": 0.7,
            },
            recorder_ref=None,
        )

        assert meta is not None
        assert meta["timestamp_iso"].endswith("+00:00")
        with wave.open(meta["files"]["full_120s"]) as wav:
            assert wav.getframerate() == 8000

    def test_color_hysteresis_sticks_red_until_lower_exit_threshold(self):
        detector = _import_from_audi_app("detector")
        hyst = detector.ColorHysteresisState(
            enter_red_threshold=0.45,
            exit_red_threshold=0.35,
            window=1,
            ratio=1.0,
        )

        assert hyst.add(0.46) == "RED"
        assert hyst.add(0.40) == "RED"
        assert hyst.add(0.34) == "BLUE"

    def test_detection_threshold_profile_overrides_defaults(self, tmp_path):
        detector = _import_from_audi_app("detector")

        engine = detector.DetectionEngine(
            {
                "detection": {
                    "model_path": str(tmp_path / "missing.tflite"),
                    "alert_history_file": str(tmp_path / "alerts.jsonl"),
                    "active_threshold_profile": "sensitive",
                    "confidence_threshold_high": 0.9,
                    "threshold_profiles": {
                        "sensitive": {
                            "confidence_threshold_high": 0.72,
                            "blue_to_red_threshold": 0.42,
                            "red_to_blue_threshold": 0.30,
                        }
                    },
                },
                "audio": {"sample_rate": 16000},
                "storage": {"alerts_dir": str(tmp_path)},
            },
            ring_buffer=None,
        )

        assert engine.threshold_profile == "sensitive"
        assert engine.threshold_yes == 0.72
        assert engine.blue_to_red_threshold == 0.42
        assert engine.red_to_blue_threshold == 0.30

    def test_load_config_reads_external_threshold_profiles(self, tmp_path):
        main = _import_from_audi_app("main")
        profiles = tmp_path / "thresholds.yaml"
        profiles.write_text(
            """
threshold_profiles:
  field:
    confidence_threshold_high: 0.81
    blue_to_red_threshold: 0.41
    red_to_blue_threshold: 0.29
""",
            encoding="utf-8",
        )
        config = tmp_path / "config.yaml"
        config.write_text(
            """
detection:
  threshold_profiles_file: thresholds.yaml
  active_threshold_profile: field
""",
            encoding="utf-8",
        )

        cfg = main.load_config(str(config))

        assert cfg["detection"]["threshold_profiles"]["field"][
            "confidence_threshold_high"
        ] == 0.81
        assert cfg["detection"]["threshold_profiles"]["field"][
            "blue_to_red_threshold"
        ] == 0.41

    def test_alert_history_label_alert(self, tmp_path):
        detector = _import_from_audi_app("detector")
        history = detector.AlertHistory(str(tmp_path / "alerts.jsonl"))
        history.append({"alert_id": "a1", "state": "YES"})

        updated = history.label_alert("a1", "drone_red")

        assert updated is not None
        assert updated["operator_label"] == "drone_red"
        assert history.read_recent(1)[0]["operator_label"] == "drone_red"
