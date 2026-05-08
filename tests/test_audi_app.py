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
