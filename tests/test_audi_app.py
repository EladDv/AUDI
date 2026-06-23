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
    def test_app_mel_preprocessing_matches_training_torchaudio(self):
        """Pi mel frontend must numerically match training preprocessing."""
        torch = pytest.importorskip("torch")
        torchaudio_transforms = pytest.importorskip("torchaudio.transforms")
        audio_features = _import_from_audi_app("audio_features")

        rng = np.random.default_rng(123)
        audio = rng.normal(0.0, 1.0, 81920).astype(np.float32)
        app_mel = audio_features.compute_mel_spectrogram(
            audio,
            16000,
            n_mels=128,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
        )[..., :512]

        wav = torch.from_numpy(audio).unsqueeze(0)
        train_mel = torchaudio_transforms.AmplitudeToDB()(
            torchaudio_transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                win_length=1024,
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
        audio_features = _import_from_audi_app("audio_features")

        rng = np.random.default_rng(321)
        audio = rng.normal(0.0, 0.03, 81920).astype(np.float32)
        classifier = audio_features.TFLiteClassifier(
            model_path=str(Path("/missing/model.tflite")),
            n_mels=128,
            n_fft=1024,
            win_length=1024,
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
                win_length=1024,
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
        assert classifier.last_mel_db is not None
        assert classifier.last_mel_db.shape == (128, 512)

    def test_app_classifier_default_window_matches_deployment_model(self):
        audio_features = _import_from_audi_app("audio_features")

        classifier = audio_features.TFLiteClassifier(
            model_path=str(Path("/missing/model.tflite")),
        )

        assert classifier.window_samples == 81920

    def test_app_classifier_incremental_preprocess_matches_full_windows(self):
        """Rolling preprocessing must match a full recompute on shifted windows."""
        audio_features = _import_from_audi_app("audio_features")

        rng = np.random.default_rng(4321)
        step_samples = 5120
        window_samples = 81920
        audio = rng.normal(
            0.0,
            0.03,
            window_samples + step_samples,
        ).astype(np.float32)

        rolling = audio_features.TFLiteClassifier(
            model_path=str(Path("/missing/model.tflite")),
            n_mels=128,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            model_sample_rate=16000,
            window_samples=window_samples,
            mel_mean=10.430418,
            mel_std=5.288271,
            incremental_step_samples=step_samples,
        )
        rolling.expected_frames = 512
        rolling.preprocess(audio[:window_samples], 16000)
        incremental_spec = rolling.preprocess(audio[step_samples:], 16000)

        full = audio_features.TFLiteClassifier(
            model_path=str(Path("/missing/model.tflite")),
            n_mels=128,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            model_sample_rate=16000,
            window_samples=window_samples,
            mel_mean=10.430418,
            mel_std=5.288271,
            incremental_preprocess=False,
        )
        full.expected_frames = 512
        full_spec = full.preprocess(audio[step_samples:], 16000)

        assert rolling._preprocess_cache_hits == 1
        assert rolling._preprocess_last_reused_frames == 473
        np.testing.assert_allclose(
            incremental_spec,
            full_spec,
            rtol=3e-5,
            atol=1e-3,
        )

    def test_app_mel_preprocessing_matches_training_custom_win_length(self):
        """Pi mel frontend must match torchaudio when win_length < n_fft."""
        torch = pytest.importorskip("torch")
        torchaudio_transforms = pytest.importorskip("torchaudio.transforms")
        audio_features = _import_from_audi_app("audio_features")

        rng = np.random.default_rng(7)
        audio = rng.normal(0.0, 0.25, 40960).astype(np.float32)
        app_mel = audio_features.compute_mel_spectrogram(
            audio,
            16000,
            n_mels=128,
            n_fft=1024,
            win_length=512,
            hop_length=160,
        )

        wav = torch.from_numpy(audio).unsqueeze(0)
        train_mel = torchaudio_transforms.AmplitudeToDB()(
            torchaudio_transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                win_length=512,
                hop_length=160,
                n_mels=128,
            )(wav)
        )[0].numpy()

        np.testing.assert_allclose(app_mel, train_mel, rtol=2e-5, atol=7e-4)

    def test_recorder_defaults_match_deployment_sample_rate(self, tmp_path):
        recorder = _import_from_audi_app("recorder")

        raw = recorder.ALSARecorder(str(tmp_path / "hot"))

        assert raw.sample_rate == 16000
        assert raw.ring_buffer.max_samples == 120 * 16000

    def test_recorder_manager_fallback_sample_rate_matches_deployment(self, tmp_path):
        recorder = _import_from_audi_app("recorder")

        manager = recorder.RecorderManager(
            {
                "audio": {},
                "storage": {"data_dir": str(tmp_path / "data")},
            }
        )

        assert manager.status["sample_rate"] == 16000
        assert manager.status["ring_buffer_capacity"] == 120

    def test_audio_ring_buffer_preserves_multichannel_frames(self):
        recorder = _import_from_audi_app("recorder")
        ring = recorder.AudioRingBuffer(max_samples=5, channels=4)

        frames = np.array(
            [
                [0.0, 0.1, 0.2, 0.3],
                [1.0, 1.1, 1.2, 1.3],
                [2.0, 2.1, 2.2, 2.3],
            ],
            dtype=np.float32,
        )
        ring.append(frames)

        all_channels = ring.get_recent(2, channel=None)

        assert all_channels.shape == (2, 4)
        np.testing.assert_allclose(all_channels[:, 2], [1.2, 2.2])
        np.testing.assert_allclose(ring.get_recent(2, channel=3), [1.3, 2.3])

    def test_recorder_status_reports_per_channel_health(self, tmp_path):
        recorder = _import_from_audi_app("recorder")
        raw = recorder.ALSARecorder(str(tmp_path / "hot"), channels=4)
        raw.ring_buffer.append(
            np.tile(
                np.array([[0.01, 0.02, 0.0, 0.5]], dtype=np.float32),
                (1600, 1),
            )
        )
        raw._last_audio_timestamp = __import__("time").time()

        health = raw.channel_health()

        assert [row["channel_index"] for row in health] == [0, 1, 2, 3]
        assert health[0]["healthy"] is True
        assert health[2]["healthy"] is False
        assert health[3]["peak"] == 0.5

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

    def test_gpio_defaults_include_field_tag_buttons(self, monkeypatch):
        mod = _import_from_audi_app("gpio_alarm")

        setups = []
        events = {}
        rpi_pkg = types.ModuleType("RPi")
        gpio_mod = types.ModuleType("RPi.GPIO")
        gpio_mod.BCM = "BCM"
        gpio_mod.OUT = "OUT"
        gpio_mod.IN = "IN"
        gpio_mod.LOW = 0
        gpio_mod.HIGH = 1
        gpio_mod.PUD_UP = "PUD_UP"
        gpio_mod.FALLING = "FALLING"
        gpio_mod.setmode = lambda mode: None
        gpio_mod.setwarnings = lambda enabled: None
        gpio_mod.output = lambda pin, value: None
        gpio_mod.cleanup = lambda: None

        def setup(pin, mode, **kwargs):
            setups.append((pin, mode, kwargs))

        def add_event_detect(pin, edge, callback, bouncetime):
            events[pin] = {"edge": edge, "callback": callback}

        gpio_mod.setup = setup
        gpio_mod.add_event_detect = add_event_detect
        monkeypatch.setitem(sys.modules, "RPi", rpi_pkg)
        monkeypatch.setitem(sys.modules, "RPi.GPIO", gpio_mod)

        controller = mod.GPIOController({"gpio": {"enabled": True}})
        tags = []
        controller.on_field_tag = tags.append

        assert controller.status()["alert_pin"] == 2
        assert controller.status()["record_led_pin"] is None
        assert controller.status()["record_button_pin"] is None
        assert controller.status()["field_tag_button_pins"] == {
            "green": 22,
            "yellow": 27,
            "red": 17,
        }
        assert (2, "OUT", {"initial": 0}) in setups
        assert (22, "IN", {"pull_up_down": "PUD_UP"}) in setups
        assert (27, "IN", {"pull_up_down": "PUD_UP"}) in setups
        assert (17, "IN", {"pull_up_down": "PUD_UP"}) in setups
        assert set(events) >= {17, 22, 27}
        assert events[22]["edge"] == "FALLING"
        assert events[27]["edge"] == "FALLING"
        assert events[17]["edge"] == "FALLING"

        events[22]["callback"](22)
        events[27]["callback"](27)
        events[17]["callback"](17)

        assert tags == ["green", "yellow", "red"]

    def test_alarm_snapshot_writes_utc_metadata(self, tmp_path):
        """Alarm snapshots should not fail while writing UTC metadata."""
        storage = _import_from_audi_app("storage")

        class RingBuffer:
            def get_last_n_seconds(self, n_seconds, sample_rate):
                return np.zeros(int(n_seconds * sample_rate), dtype=np.float32)

        assert storage.AlarmSnapshotter(str(tmp_path / "default")).sample_rate == 16000

        snapshotter = storage.AlarmSnapshotter(str(tmp_path), sample_rate=8000)
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

    def test_alarm_snapshot_saves_firing_channel_and_all_channels(self, tmp_path):
        storage = _import_from_audi_app("storage")
        recorder = _import_from_audi_app("recorder")

        ring = recorder.AudioRingBuffer(max_samples=8000, channels=4)
        ring.append(
            np.tile(
                np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                (8000, 1),
            )
        )
        snapshotter = storage.AlarmSnapshotter(str(tmp_path), sample_rate=8000)
        snapshotter._stop_event.set()

        meta = snapshotter.save_snapshot(
            ring,
            {
                "alert_id": "field-1",
                "state": "YES",
                "alert_level": "RED_ALERT",
                "yes_confidence": 0.9,
                "channel_index": 2,
                "channel_name": "ch2",
                "firing_channel_index": 2,
                "firing_channel_name": "ch2",
                "all_channel_results": [{"channel_index": i} for i in range(4)],
            },
            recorder_ref=None,
        )

        assert meta is not None
        assert meta["firing_channel_index"] == 2
        assert meta["channel_count"] == 4
        assert meta["all_channel_results"] == [{"channel_index": i} for i in range(4)]
        with wave.open(meta["files"]["full_120s"]) as wav:
            assert wav.getnchannels() == 1
        with wave.open(meta["files"]["full_120s_all_channels"]) as wav:
            assert wav.getnchannels() == 4

    def test_detector_runs_inference_on_all_audio_channels(self, tmp_path, monkeypatch):
        detector = _import_from_audi_app("detector")
        recorder = _import_from_audi_app("recorder")

        ring = recorder.AudioRingBuffer(max_samples=24, channels=4)
        ring.append(
            np.tile(
                np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32),
                (6, 1),
            )
        )
        engine = detector.DetectionEngine(
            {
                "detection": {
                    "model_path": str(tmp_path / "missing.tflite"),
                    "alert_history_file": str(tmp_path / "alerts.jsonl"),
                    "model_sample_rate": 4,
                    "window_samples": 4,
                    "confidence_threshold_high": 0.5,
                    "hysteresis_window": 1,
                    "hysteresis_ratio": 1.0,
                    "hysteresis_margin": 0.0,
                    "color_hysteresis_window": 1,
                    "color_hysteresis_ratio": 1.0,
                    "blue_red_min_detection_score": 0.0,
                    "red_alert_threshold": 0.5,
                    "alarm_cooldown_s": 0,
                },
                "audio": {"sample_rate": 4, "channels": 4},
                "storage": {"alerts_dir": str(tmp_path / "alerts")},
            },
            ring_buffer=ring,
        )
        snapshot_requests = []
        alarms = []
        engine._save_snapshot_and_alert = snapshot_requests.append
        engine.on_alarm = alarms.append

        def preprocess(audio, capture_sr):
            engine.classifier.last_mel_db = np.full(
                (4, 3),
                fill_value=float(audio[0]),
                dtype=np.float32,
            )
            return np.array([audio[0]], dtype=np.float32)

        def predict_logits(spec):
            channel_marker = int(spec[0])
            if channel_marker == 2:
                return np.array([2.0, 0.0, 4.0], dtype=np.float32)
            return np.array([-2.0, 4.0, 0.0], dtype=np.float32)

        monkeypatch.setattr(engine.classifier, "preprocess", preprocess)
        monkeypatch.setattr(engine.classifier, "predict_logits", predict_logits)

        result = engine.force_inference()

        assert result["firing_channel_index"] == 2
        assert result["alert_level"] == "RED_ALERT"
        assert len(result["channels"]) == 4
        assert len(result["all_channel_results"]) == 4
        assert engine.status["total_inferences"] == 4
        assert engine.status["channels"][2]["alert_level"] == "RED_ALERT"
        assert len(engine.latest_mels["channels"]) == 4
        assert engine.latest_mels["channels"][2]["mel"][0, 0] == 2.0
        assert alarms and alarms[0]["firing_channel_index"] == 2

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

    def test_detection_hysteresis_ratio_uses_ceiling(self):
        detector = _import_from_audi_app("detector")
        hyst = detector.HysteresisState(
            threshold=0.5,
            window=8,
            ratio=0.6,
            margin=0.0,
        )

        for _ in range(4):
            assert hyst.add(0.4) is False
        for _ in range(4):
            assert hyst.add(0.6) is False
        assert hyst.add(0.6) is True

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
        assert engine.capture_sample_rate == 16000
        assert engine.window_samples == 81920
        assert engine.labels == ["drone"]
        assert engine.threshold_yes == 0.655
        assert engine.blue_to_red_threshold == 0.37
        assert engine.red_to_blue_threshold == 0.56
        assert engine.alarm_cooldown_s == 120.0
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
        storage = _import_from_audi_app("storage")
        history = storage.AlertHistory(str(tmp_path / "alerts.jsonl"))
        history.append({"alert_id": "a1", "state": "YES"})

        updated = history.label_alert("a1", "drone_red")

        assert updated is not None
        assert updated["operator_label"] == "drone_red"
        assert history.read_recent(1)[0]["operator_label"] == "drone_red"

    def test_alert_history_field_tag_latest(self, tmp_path):
        storage = _import_from_audi_app("storage")
        history = storage.AlertHistory(str(tmp_path / "alerts.jsonl"))
        history.append({"alert_id": "a1", "state": "YES"})
        history.append({"alert_id": "a2", "state": "YES"})

        updated = history.tag_latest("red")
        entries = history.read_recent(2)

        assert updated is not None
        assert updated["alert_id"] == "a2"
        assert updated["field_tag_color"] == "red"
        assert updated["field_tag"] == "incorrect_detection"
        assert updated["operator_label"] == "incorrect_detection"
        assert "field_tag" not in entries[0]

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

    def test_webui_serves_checked_in_index_html(self):
        webui_server = _import_from_audi_app("webui_server")
        webui = webui_server.WebUI({"web": {"port": 0}})

        response = webui._app.test_client().get("/")

        assert response.status_code == 200
        assert b"AUDI" in response.data

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

    def test_webui_mels_prefers_detector_preprocess_cache(self):
        webui_server = _import_from_audi_app("webui_server")

        class RingBuffer:
            def get_last_n_seconds(self, *args, **kwargs):
                raise AssertionError("ring-buffer mel recompute should not run")

        class RawRecorder:
            sample_rate = 16000
            ring_buffer = RingBuffer()

        class Recorder:
            recorder = RawRecorder()

        class Detector:
            n_mels = 4
            n_fft = 16
            win_length = 16
            hop_length = 4
            model_sample_rate = 16000

            @property
            def status(self):
                return {
                    "n_mels": self.n_mels,
                    "n_fft": self.n_fft,
                    "win_length": self.win_length,
                    "hop_length": self.hop_length,
                    "model_sample_rate": self.model_sample_rate,
                }

            @property
            def latest_mels(self):
                return {
                    "timestamp": 123.0,
                    "sample_rate": 16000,
                    "channels": [
                        {
                            "channel_index": 0,
                            "mel": np.arange(16, dtype=np.float32).reshape(4, 4),
                        }
                    ],
                }

        webui = webui_server.WebUI({"web": {"port": 0}})
        webui.recorder = Recorder()
        webui.detector = Detector()

        response = webui._app.test_client().get("/api/mels?columns=2")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["source"] == "detector_cache"
        assert payload["channels_available"] == 1
        assert payload["channels"][0]["frames"] == 2
        assert payload["channels"][0]["mel"] == [
            [0.5, 2.5],
            [4.5, 6.5],
            [8.5, 10.5],
            [12.5, 14.5],
        ]
