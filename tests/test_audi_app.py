"""Smoke tests for audi-app (RPi deployment)."""

import importlib
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest


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

    def test_app_mel_preprocessing_matches_training_torchaudio(self):
        """Pi mel frontend must numerically match training preprocessing."""
        torch = pytest.importorskip("torch")
        torchaudio_transforms = pytest.importorskip("torchaudio.transforms")
        detector = _import_from_audi_app("detector")

        rng = np.random.default_rng(123)
        audio = rng.normal(0.0, 1.0, 81920).astype(np.float32)
        app_mel = detector.compute_mel_spectrogram(
            audio,
            16000,
            n_mels=128,
            n_fft=1024,
            hop_length=160,
        )[..., :512]

        wav = torch.from_numpy(audio).unsqueeze(0)
        train_mel = torchaudio_transforms.AmplitudeToDB()(
            torchaudio_transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=160,
                n_mels=128,
            )(wav)
        )[0].numpy()[..., :512]

        assert app_mel.shape == train_mel.shape == (128, 512)
        np.testing.assert_allclose(app_mel, train_mel, rtol=2e-5, atol=7e-4)

    def test_app_classifier_preprocess_matches_detector_checkpoint_contract(self):
        """Full app spec normalization should match detector _to_mel contract."""
        torch = pytest.importorskip("torch")
        torchaudio_transforms = pytest.importorskip("torchaudio.transforms")
        detector = _import_from_audi_app("detector")

        rng = np.random.default_rng(321)
        audio = rng.normal(0.0, 0.03, 81920).astype(np.float32)
        classifier = detector.TFLiteClassifier(
            model_path=str(Path("/missing/model.tflite")),
            n_mels=128,
            n_fft=1024,
            hop_length=160,
            model_sample_rate=16000,
            window_samples=81920,
            mel_mean=10.430418,
            mel_std=5.288271,
        )

        app_spec = classifier.preprocess(audio, 16000)

        wav = torch.from_numpy(audio).unsqueeze(0)
        wav_rms = torch.sqrt(torch.mean(wav.to(torch.float64) ** 2)).to(torch.float32)
        wav = wav / wav_rms
        peak = torch.max(torch.abs(wav))
        if peak > 0.98:
            wav = wav * (0.98 / peak)
        train_spec = torchaudio_transforms.AmplitudeToDB()(
            torchaudio_transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=160,
                n_mels=128,
            )(wav)
        )
        train_spec = (train_spec - 10.430418) / 5.288271
        train_spec = train_spec[..., :512].unsqueeze(1).expand(-1, 3, -1, -1)

        assert app_spec.shape == tuple(train_spec.shape) == (1, 3, 128, 512)
        np.testing.assert_allclose(
            app_spec,
            train_spec.numpy(),
            rtol=3e-5,
            atol=1e-3,
        )

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

    def test_gpio_falls_back_to_mock_unless_required(self, monkeypatch):
        """GPIO setup failures are fatal only when REQUIRE_GPIO is set."""
        mod = _import_from_audi_app("gpio_alarm")

        rpi_pkg = types.ModuleType("RPi")
        gpio_mod = types.ModuleType("RPi.GPIO")
        gpio_mod.BCM = "BCM"
        gpio_mod.OUT = "OUT"
        gpio_mod.IN = "IN"
        gpio_mod.LOW = 0
        gpio_mod.PUD_UP = "PUD_UP"
        gpio_mod.FALLING = "FALLING"
        gpio_mod.setmode = lambda mode: None
        gpio_mod.setwarnings = lambda enabled: None

        def fail_setup(*args, **kwargs):
            raise RuntimeError("gpio unavailable")

        gpio_mod.setup = fail_setup
        monkeypatch.setitem(sys.modules, "RPi", rpi_pkg)
        monkeypatch.setitem(sys.modules, "RPi.GPIO", gpio_mod)

        monkeypatch.delenv("REQUIRE_GPIO", raising=False)
        controller = mod.GPIOController({"gpio": {"enabled": True}})
        assert controller.status()["has_gpio"] is False

        monkeypatch.setenv("REQUIRE_GPIO", "true")
        with pytest.raises(RuntimeError):
            mod.GPIOController({"gpio": {"enabled": True}})

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

    def test_detection_threshold_profile_can_switch_at_runtime(self, tmp_path):
        detector = _import_from_audi_app("detector")

        engine = detector.DetectionEngine(
            {
                "detection": {
                    "model_path": str(tmp_path / "missing.tflite"),
                    "alert_history_file": str(tmp_path / "alerts.jsonl"),
                    "active_threshold_profile": "balanced",
                    "threshold_profiles": {
                        "balanced": {
                            "confidence_threshold_high": 0.80,
                            "blue_to_red_threshold": 0.45,
                            "red_to_blue_threshold": 0.35,
                        },
                        "sensitive": {
                            "confidence_threshold_high": 0.72,
                            "blue_to_red_threshold": 0.42,
                            "red_to_blue_threshold": 0.30,
                        },
                    },
                },
                "audio": {"sample_rate": 16000},
                "storage": {"alerts_dir": str(tmp_path)},
            },
            ring_buffer=None,
        )

        status = engine.set_threshold_profile("sensitive")

        assert status["threshold_profile"] == "sensitive"
        assert status["threshold_profiles"] == ["balanced", "sensitive"]
        assert engine.threshold_yes == 0.72
        assert engine.blue_to_red_threshold == 0.42
        assert engine.red_to_blue_threshold == 0.30

    def test_blue_and_unknown_alerts_are_disabled_by_default(self, tmp_path):
        detector = _import_from_audi_app("detector")

        engine = detector.DetectionEngine(
            {
                "detection": {
                    "model_path": str(tmp_path / "missing.tflite"),
                    "alert_history_file": str(tmp_path / "alerts.jsonl"),
                },
                "audio": {"sample_rate": 16000},
                "storage": {"alerts_dir": str(tmp_path)},
            },
            ring_buffer=None,
        )

        assert engine.alert_on_blue is False
        assert engine.alert_on_unknown is False
        assert (
            engine._resolve_alert_level(
                True,
                {
                    "drone_color": "BLUE",
                    "blue_confidence": 0.99,
                    "red_confidence": 0.01,
                },
            )
            == "DETECTED"
        )
        assert engine._resolve_alert_level(True, {"drone_color": "UNKNOWN"}) == "DETECTED"

    def test_alert_routing_can_enable_blue_and_unknown(self, tmp_path):
        detector = _import_from_audi_app("detector")

        engine = detector.DetectionEngine(
            {
                "detection": {
                    "model_path": str(tmp_path / "missing.tflite"),
                    "alert_history_file": str(tmp_path / "alerts.jsonl"),
                    "blue_alert_threshold": 0.50,
                },
                "audio": {"sample_rate": 16000},
                "storage": {"alerts_dir": str(tmp_path)},
            },
            ring_buffer=None,
        )

        status = engine.set_alert_routing(
            alert_on_blue=True,
            alert_on_unknown=True,
        )

        assert status["alert_on_blue"] is True
        assert status["alert_on_unknown"] is True
        assert (
            engine._resolve_alert_level(
                True,
                {
                    "drone_color": "BLUE",
                    "blue_confidence": 0.99,
                    "red_confidence": 0.01,
                },
            )
            == "BLUE_ALERT"
        )
        assert (
            engine._resolve_alert_level(True, {"drone_color": "UNKNOWN"})
            == "UNKNOWN_ALERT"
        )

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

    def test_webui_threshold_profile_endpoint(self):
        webui_server = _import_from_audi_app("webui_server")

        class Detector:
            def set_threshold_profile(self, profile):
                assert profile == "quiet"
                return {"threshold_profile": profile}

        webui = webui_server.WebUI({"web": {"port": 0}})
        webui.detector = Detector()

        response = webui._app.test_client().post(
            "/api/threshold_profile",
            json={"profile": "quiet"},
        )

        assert response.status_code == 200
        assert response.get_json()["detector"]["threshold_profile"] == "quiet"

    def test_webui_alert_routing_endpoint(self):
        webui_server = _import_from_audi_app("webui_server")

        class Detector:
            def set_alert_routing(self, *, alert_on_blue=None, alert_on_unknown=None):
                return {
                    "alert_on_blue": alert_on_blue,
                    "alert_on_unknown": alert_on_unknown,
                }

        webui = webui_server.WebUI({"web": {"port": 0}})
        webui.detector = Detector()

        response = webui._app.test_client().post(
            "/api/alert_routing",
            json={"alert_on_blue": True, "alert_on_unknown": False},
        )

        assert response.status_code == 200
        assert response.get_json()["detector"] == {
            "alert_on_blue": True,
            "alert_on_unknown": False,
        }

    def test_webui_test_alert_accepts_color_alert_level(self):
        webui_server = _import_from_audi_app("webui_server")

        class GPIO:
            def __init__(self):
                self.triggered = []

            def trigger_alarm(self, alert_level):
                self.triggered.append(alert_level)

        gpio = GPIO()
        webui = webui_server.WebUI({"web": {"port": 0}})
        webui.gpio = gpio

        response = webui._app.test_client().post(
            "/api/test_alert",
            json={"alert_level": "BLUE_ALERT"},
        )

        assert response.status_code == 200
        payload = response.get_json()["result"]
        assert payload["alert_level"] == "BLUE_ALERT"
        assert payload["drone_color"] == "BLUE"
        assert payload["blue_confidence"] == 0.95
        assert gpio.triggered == ["BLUE_ALERT"]

    def test_webui_test_alert_rejects_invalid_alert_level(self):
        webui_server = _import_from_audi_app("webui_server")
        webui = webui_server.WebUI({"web": {"port": 0}})

        response = webui._app.test_client().post(
            "/api/test_alert",
            json={"alert_level": "NO"},
        )

        assert response.status_code == 400

    def test_webui_audio_level_uses_dbfs_with_floor(self):
        webui_server = _import_from_audi_app("webui_server")

        class RawRecorder:
            def __init__(self, rms):
                self.rms = rms

            def get_rms_level(self):
                return self.rms

        class Recorder:
            def __init__(self, rms):
                self.recorder = RawRecorder(rms)

        webui = webui_server.WebUI({"web": {"port": 0}})
        webui.recorder = Recorder(10 ** (-70 / 20))

        response = webui._app.test_client().get("/api/audio_level")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["db"] == -70.0
        assert payload["vu_percent"] == pytest.approx(41.7)

        webui.recorder = Recorder(10 ** (-700 / 20))
        response = webui._app.test_client().get("/api/audio_level")

        assert response.status_code == 200
        assert response.get_json()["db"] == -120.0
        assert response.get_json()["vu_percent"] == 0.0

        webui.recorder = Recorder(1.0)
        response = webui._app.test_client().get("/api/audio_level")

        assert response.status_code == 200
        assert response.get_json()["db"] == 0.0
        assert response.get_json()["vu_percent"] == 100.0
