"""Smoke tests for audi-app (RPi deployment)."""

import importlib
import subprocess
import sys
import time
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
        assert raw.bit_depth == 32
        assert raw.ring_buffer.max_samples == 120 * 16000

    def test_recorder_auto_discover_uses_alsa_plug_device(self, monkeypatch):
        recorder = _import_from_audi_app("recorder")

        def fake_run(*args, **kwargs):
            return types.SimpleNamespace(
                stdout=(
                    "card 2: US4x4HR [US-4x4HR], "
                    "device 0: USB Audio [USB Audio]\n"
                ),
                stderr="",
                returncode=0,
            )

        monkeypatch.setattr(recorder.subprocess, "run", fake_run)

        assert recorder.auto_discover_device() == "plughw:2,0"

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
        assert manager.recorder.bit_depth == 32

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
        assert controller.status()["pause_button_pin"] is None
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
        assert 18 not in events
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

    def test_alarm_snapshot_saves_mvdr_beam_outputs(self, tmp_path):
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

        def beamform_callback(audio_by_channel):
            assert audio_by_channel.shape == (8000, 4)
            return [
                {
                    "index": 0,
                    "name": "beam0_az000_el05",
                    "audio": np.full(8000, 0.05, dtype=np.float32),
                },
                {
                    "index": 1,
                    "name": "beam1_az180_el05",
                    "audio": np.full(8000, 0.75, dtype=np.float32),
                },
                {
                    "index": 2,
                    "name": "beam2_az270_el05",
                    "audio": np.full(8000, -0.25, dtype=np.float32),
                },
            ]

        meta = snapshotter.save_snapshot(
            ring,
            {
                "alert_id": "field-mvdr-1",
                "state": "YES",
                "alert_level": "RED_ALERT",
                "yes_confidence": 0.9,
                "channel_index": 1,
                "channel_name": "beam1_az180_el05",
                "input_mode": "mvdr_beam",
                "beam_azimuth_deg": 180.0,
                "beam_elevation_deg": 5.0,
                "beam_mic_indices": [0, 1, 2, 3],
            },
            recorder_ref=None,
            beamform_callback=beamform_callback,
        )

        assert meta is not None
        assert meta["input_mode"] == "mvdr_beam"
        assert meta["mvdr_beam_count"] == 3
        assert meta["channel_count"] == 4
        assert meta["beam_azimuth_deg"] == 180.0
        assert "full_120s_all_mvdr_beams" in meta["files"]

        with wave.open(meta["files"]["full_120s"], "rb") as wav:
            assert wav.getnchannels() == 1
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            selected_mean = float(np.mean(pcm.astype(np.float32) / 32768.0))
        assert selected_mean == pytest.approx(0.75, abs=1e-3)

        with wave.open(meta["files"]["full_120s_all_channels"], "rb") as wav:
            assert wav.getnchannels() == 4
        with wave.open(meta["files"]["full_120s_all_mvdr_beams"], "rb") as wav:
            assert wav.getnchannels() == 3

    def test_storage_compresses_16_channel_wav_as_wavpack(
        self, tmp_path, monkeypatch
    ):
        storage = _import_from_audi_app("storage")
        wav_path = tmp_path / "seg_16ch.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(16)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.zeros((160, 16), dtype=np.int16).tobytes())

        calls = []

        def fake_run(cmd, capture_output, timeout):
            calls.append(cmd)
            Path(cmd[-1]).write_bytes(b"wv")
            return subprocess.CompletedProcess(cmd, 0, stderr=b"")

        monkeypatch.setattr(subprocess, "run", fake_run)

        compressed = storage.FlacCompressor().compress(str(wav_path))

        assert compressed == str(wav_path.with_suffix(".wv"))
        assert not wav_path.exists()
        assert calls[0][-2:] == ["wavpack", str(wav_path.with_suffix(".wv"))]

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
                    "red_color_threshold": 0.5,
                    "blue_color_threshold": 0.5,
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
        assert len(engine.status["channel_score_history"]) == 4
        assert engine.status["channel_score_history"][2]["history"][-1][
            "channel_index"
        ] == 2
        assert len(engine.latest_mels["channels"]) == 4
        assert engine.latest_mels["channels"][2]["mel"][0, 0] == 2.0
        assert alarms and alarms[0]["firing_channel_index"] == 2

    def test_detector_disabled_channel_is_recorded_but_not_inferred(
        self, tmp_path, monkeypatch
    ):
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
                    "enabled_channels": [0, 1, 3],
                    "confidence_threshold_high": 0.5,
                    "hysteresis_window": 1,
                    "hysteresis_ratio": 1.0,
                    "hysteresis_margin": 0.0,
                    "red_color_threshold": 0.5,
                    "alarm_cooldown_s": 0,
                },
                "audio": {"sample_rate": 4, "channels": 4},
                "storage": {"alerts_dir": str(tmp_path / "alerts")},
            },
            ring_buffer=ring,
        )
        alarms = []
        seen = []
        engine.on_alarm = alarms.append

        def preprocess(audio, capture_sr):
            seen.append(int(audio[0]))
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

        assert seen == [0, 1, 3]
        assert result["alert_level"] == "NO"
        assert alarms == []
        assert engine.status["total_inferences"] == 3
        assert engine.status["enabled_channels"] == [0, 1, 3]
        assert engine.status["channels"][2]["detector_enabled"] is False
        assert result["all_channel_results"][2]["detector_enabled"] is False
        np.testing.assert_allclose(ring.get_recent(1, channel=None)[0], [0, 1, 2, 3])

    def test_detector_loop_overrun_uses_start_to_start_interval(self, monkeypatch):
        detector = _import_from_audi_app("detector")

        class FakeClock:
            def __init__(self):
                self.now = 100.0

            def time(self):
                return self.now

        class FakeStopEvent:
            def __init__(self):
                self.stopped = False
                self.waits = []

            def is_set(self):
                return self.stopped

            def wait(self, seconds):
                self.waits.append(seconds)
                clock.now += seconds
                if len(pass_starts) >= 2:
                    self.stopped = True

        clock = FakeClock()
        stop_event = FakeStopEvent()
        pass_starts = []

        def slow_inference_cycle():
            pass_starts.append(clock.now)
            clock.now += 1.0

        engine = types.SimpleNamespace(
            _stop_event=stop_event,
            _last_inference_time=0.0,
            inference_interval=0.32,
            _inference_cycle=slow_inference_cycle,
        )

        monkeypatch.setattr(detector.time, "time", clock.time)

        detector.DetectionEngine._run(engine)

        assert pass_starts == [100.0, 101.01]
        assert stop_event.waits == [0.01, 0.01]

    def test_detector_mvdr_beam_mode_runs_inference_on_beams(
        self, tmp_path, monkeypatch
    ):
        detector = _import_from_audi_app("detector")
        recorder = _import_from_audi_app("recorder")

        ring = recorder.AudioRingBuffer(max_samples=24, channels=16)
        ring.append(np.ones((6, 16), dtype=np.float32))
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
                    "red_color_threshold": 0.5,
                    "alarm_cooldown_s": 0,
                    "input_beamforming": {
                        "enabled": True,
                        "beam_count": 2,
                        "elevation_count": 1,
                    },
                },
                "audio": {"sample_rate": 4, "channels": 16},
                "storage": {"alerts_dir": str(tmp_path / "alerts")},
            },
            ring_buffer=ring,
        )

        class FakeBeamformer:
            def __init__(self):
                self.has_covariance = False
                self.covariance_updates = []

            def update_covariance(self, audio, sample_rate):
                self.covariance_updates.append(np.array(audio, copy=True))
                self.has_covariance = True
                return True

            def beamform(self, audio, sample_rate, noise_audio=None):
                assert audio.shape == (4, 16)
                assert sample_rate == 4
                return [
                    {
                        "index": 0,
                        "name": "beam0_az000_el05",
                        "audio": np.zeros(4, dtype=np.float32),
                        "azimuth_deg": 0.0,
                        "elevation_deg": 5.0,
                        "mic_indices": list(range(16)),
                    },
                    {
                        "index": 1,
                        "name": "beam1_az180_el05",
                        "audio": np.ones(4, dtype=np.float32),
                        "azimuth_deg": 180.0,
                        "elevation_deg": 5.0,
                        "mic_indices": list(range(16)),
                    },
                ]

        fake_beamformer = FakeBeamformer()
        engine._beamformer = fake_beamformer
        alarms = []
        engine.on_alarm = alarms.append

        def preprocess(audio, capture_sr):
            engine.classifier.last_mel_db = np.full((4, 3), audio[0], dtype=np.float32)
            return np.array([audio[0]], dtype=np.float32)

        def predict_logits(spec):
            return (
                np.array([2.0, 0.0, 4.0], dtype=np.float32)
                if int(spec[0]) == 1
                else np.array([-2.0, 4.0, 0.0], dtype=np.float32)
            )

        monkeypatch.setattr(engine.classifier, "preprocess", preprocess)
        monkeypatch.setattr(engine.classifier, "predict_logits", predict_logits)

        result = engine.force_inference()

        assert result["input_mode"] == "mvdr_beam"
        assert result["firing_channel_index"] == 1
        assert result["channel_name"] == "beam1_az180_el05"
        assert result["beam_azimuth_deg"] == 180.0
        assert result["alert_level"] == "RED_ALERT"
        assert engine.status["input_mode"] == "mvdr_beam"
        assert engine.status["detector_input_count"] == 2
        assert len(fake_beamformer.covariance_updates) == 1
        assert engine.status["input_beamforming"]["last_covariance_update_reason"] == "startup"
        assert len(result["all_channel_results"]) == 2
        assert alarms and alarms[0]["beam_elevation_deg"] == 5.0

    def test_detector_refreshes_mvdr_covariance_after_clean_beam_pass(
        self, tmp_path, monkeypatch
    ):
        detector = _import_from_audi_app("detector")
        recorder = _import_from_audi_app("recorder")

        ring = recorder.AudioRingBuffer(max_samples=24, channels=16)
        ring.append(np.ones((6, 16), dtype=np.float32))
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
                    "input_beamforming": {
                        "enabled": True,
                        "beam_count": 2,
                        "elevation_count": 1,
                        "covariance_update_interval_seconds": 1.0,
                        "covariance_no_drone_threshold": 0.25,
                    },
                },
                "audio": {"sample_rate": 4, "channels": 16},
                "storage": {"alerts_dir": str(tmp_path / "alerts")},
            },
            ring_buffer=ring,
        )

        class FakeBeamformer:
            has_covariance = True

            def __init__(self):
                self.covariance_updates = []

            def update_covariance(self, audio, sample_rate):
                self.covariance_updates.append(np.array(audio, copy=True))
                return True

            def beamform(self, audio, sample_rate, noise_audio=None):
                return [
                    {
                        "index": 0,
                        "name": "beam0_az000_el05",
                        "audio": np.zeros(4, dtype=np.float32),
                        "azimuth_deg": 0.0,
                        "elevation_deg": 5.0,
                        "mic_indices": list(range(16)),
                    },
                    {
                        "index": 1,
                        "name": "beam1_az180_el05",
                        "audio": np.zeros(4, dtype=np.float32),
                        "azimuth_deg": 180.0,
                        "elevation_deg": 5.0,
                        "mic_indices": list(range(16)),
                    },
                ]

        fake_beamformer = FakeBeamformer()
        engine._beamformer = fake_beamformer
        engine._last_mvdr_covariance_update_time = time.time() - 2.0

        def preprocess(audio, capture_sr):
            engine.classifier.last_mel_db = np.zeros((4, 3), dtype=np.float32)
            return np.array([0.0], dtype=np.float32)

        monkeypatch.setattr(engine.classifier, "preprocess", preprocess)
        monkeypatch.setattr(
            engine.classifier,
            "predict_logits",
            lambda spec: np.array([-4.0, 0.0, 0.0], dtype=np.float32),
        )

        result = engine.force_inference()

        assert result["alert_level"] == "NO"
        assert len(fake_beamformer.covariance_updates) == 1
        assert engine.status["input_beamforming"]["last_covariance_update_reason"] == (
            "periodic_no_drone"
        )

    def test_mvdr_deglitch_smooths_bad_channel_jump(self):
        mvdr = _import_from_audi_app("mvdr_beamformer")

        channels = np.zeros((2, 32), dtype=np.float32)
        channels[0, :] = 0.01
        channels[0, 15:] += 1.0
        channels[1, :] = 0.01

        repaired = mvdr.deglitch_multichannel(
            channels,
            threshold=0.001,
            loudness_ratio=8.0,
            diff_ratio=12.0,
            window_samples=3,
        )

        assert np.max(np.abs(np.diff(repaired[0]))) < 0.5
        np.testing.assert_allclose(repaired[1], channels[1])

    def test_mvdr_deglitch_sensitivity_scales_with_local_loudness(self):
        mvdr = _import_from_audi_app("mvdr_beamformer")

        channels = np.zeros((2, 64), dtype=np.float32)
        channels[0, :] = 0.01
        channels[0, 32:] += 0.2
        channels[1, :] = 1.0
        channels[1, 32:] += 0.2

        repaired = mvdr.deglitch_multichannel(
            channels,
            threshold=0.001,
            loudness_ratio=8.0,
            diff_ratio=12.0,
            window_samples=4,
        )

        assert np.max(np.abs(np.diff(repaired[0]))) < 0.1
        np.testing.assert_allclose(repaired[1], channels[1])

    def test_mvdr_beamformer_uses_training_azimuth_and_phase_convention(self):
        mvdr = _import_from_audi_app("mvdr_beamformer")

        sample_rate = 16000
        seconds = 2.0
        t = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
        source_azimuth = 0.0
        source_elevation = 10.0
        source_frequency = 2500.0
        direction = mvdr.direction_unit_vector(source_azimuth, source_elevation)

        assert direction[1] > 0.98
        assert abs(direction[0]) < 1e-6

        positions = mvdr.UMA16_MIC_POSITIONS_M
        signal_by_channel = np.stack(
            [
                np.sin(
                    2.0
                    * np.pi
                    * source_frequency
                    * (t + float(np.dot(position, direction)) / 343.0)
                )
                for position in positions
            ],
            axis=1,
        ).astype(np.float32)

        rng = np.random.default_rng(123)
        noise_covariance = rng.normal(
            0.0,
            0.05,
            size=signal_by_channel.shape,
        ).astype(np.float32)
        beamformer = mvdr.MVDRBeamformer(
            mvdr.MVDRBeamformerConfig(
                beam_count=2,
                elevation_count=1,
                min_elevation_deg=source_elevation,
                max_elevation_deg=source_elevation,
                diagonal_loading=1e-4,
                deglitch_enabled=False,
            )
        )
        assert beamformer.update_covariance(noise_covariance, sample_rate)

        outputs = beamformer.beamform(signal_by_channel, sample_rate)
        matched = next(item for item in outputs if item["azimuth_deg"] == 0.0)
        opposite = next(item for item in outputs if item["azimuth_deg"] == 180.0)
        matched_rms = float(np.sqrt(np.mean(matched["audio"] ** 2)))
        opposite_rms = float(np.sqrt(np.mean(opposite["audio"] ** 2)))

        assert matched_rms > opposite_rms * 1.8

    def test_mvdr_beamformer_default_grid_is_12_horizon_beams(self):
        mvdr = _import_from_audi_app("mvdr_beamformer")

        beamformer = mvdr.MVDRBeamformer()
        assert len(beamformer.beams) == 12
        assert [beam.elevation_deg for beam in beamformer.beams] == [5.0] * 12
        assert [beam.azimuth_deg for beam in beamformer.beams] == [
            0.0,
            30.0,
            60.0,
            90.0,
            120.0,
            150.0,
            180.0,
            210.0,
            240.0,
            270.0,
            300.0,
            330.0,
        ]

    def test_mvdr_beamformer_incremental_stft_matches_full_recompute(self):
        mvdr = _import_from_audi_app("mvdr_beamformer")

        sample_rate = 16000
        step_samples = 512
        window_samples = 4096
        rng = np.random.default_rng(321)
        audio = rng.normal(
            0.0,
            0.02,
            size=(window_samples + step_samples, 16),
        ).astype(np.float32)
        noise = rng.normal(
            0.0,
            0.02,
            size=(window_samples, 16),
        ).astype(np.float32)
        cfg = mvdr.MVDRBeamformerConfig(
            beam_count=4,
            elevation_count=1,
            n_fft=512,
            hop_length=128,
            diagonal_loading=1e-4,
            deglitch_enabled=False,
        )
        rolling = mvdr.MVDRBeamformer(
            mvdr.MVDRBeamformerConfig(
                **{
                    **cfg.__dict__,
                    "incremental_step_samples": step_samples,
                }
            )
        )
        full = mvdr.MVDRBeamformer(cfg)

        assert rolling.update_covariance(noise, sample_rate)
        assert full.update_covariance(noise, sample_rate)

        rolling.beamform(audio[:window_samples], sample_rate)
        incremental_outputs = rolling.beamform(audio[step_samples:], sample_rate)
        full_outputs = full.beamform(audio[step_samples:], sample_rate)

        assert rolling._stft_cache_hits == 1
        assert rolling._stft_last_reused_frames > 0
        for incremental, recomputed in zip(
            incremental_outputs,
            full_outputs,
            strict=True,
        ):
            np.testing.assert_allclose(
                incremental["audio"],
                recomputed["audio"],
                rtol=2e-4,
                atol=2e-5,
            )

    def test_detector_runs_doa_only_after_positive_model_detection(
        self, tmp_path, monkeypatch
    ):
        detector = _import_from_audi_app("detector")
        recorder = _import_from_audi_app("recorder")

        ring = recorder.AudioRingBuffer(max_samples=24, channels=2)
        ring.append(np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (6, 1)))
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
                    "alarm_cooldown_s": 0,
                },
                "audio": {"sample_rate": 4, "channels": 2},
                "storage": {"alerts_dir": str(tmp_path / "alerts")},
            },
            ring_buffer=ring,
        )

        class DOA:
            def __init__(self):
                self.calls = 0

            @property
            def status(self):
                return {"enabled": True}

            def force_estimate(self):
                self.calls += 1
                return {"ok": True, "azimuth_deg": 45.0}

        doa = DOA()
        engine.doa_estimator = doa

        def preprocess(audio, capture_sr):
            return np.array([audio[0]], dtype=np.float32)

        def predict_logits(spec):
            return np.array([2.0 if int(spec[0]) == 1 else -2.0], dtype=np.float32)

        monkeypatch.setattr(engine.classifier, "preprocess", preprocess)
        monkeypatch.setattr(engine.classifier, "predict_logits", predict_logits)

        result = engine.force_inference()

        assert doa.calls == 1
        assert result["firing_channel_index"] == 1
        assert result["doa"] == {"ok": True, "azimuth_deg": 45.0}

    def test_detector_skips_doa_when_model_does_not_detect_drone(
        self, tmp_path, monkeypatch
    ):
        detector = _import_from_audi_app("detector")
        recorder = _import_from_audi_app("recorder")

        ring = recorder.AudioRingBuffer(max_samples=24, channels=2)
        ring.append(np.zeros((6, 2), dtype=np.float32))
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
                },
                "audio": {"sample_rate": 4, "channels": 2},
                "storage": {"alerts_dir": str(tmp_path / "alerts")},
            },
            ring_buffer=ring,
        )

        class DOA:
            @property
            def status(self):
                return {"enabled": True}

            def force_estimate(self):
                raise AssertionError("DOA should not run for all-NO cycles")

        engine.doa_estimator = DOA()
        monkeypatch.setattr(
            engine.classifier,
            "preprocess",
            lambda audio, capture_sr: np.array([0.0], dtype=np.float32),
        )
        monkeypatch.setattr(
            engine.classifier,
            "predict_logits",
            lambda spec: np.array([-2.0], dtype=np.float32),
        )

        result = engine.force_inference()

        assert result["alert_level"] == "NO"
        assert "doa" not in result

    def test_doa_profiles_and_channel_disable_are_runtime_configurable(self):
        doa_mod = _import_from_audi_app("doa_estimator")

        class Ring:
            def get_recent(self, *args, **kwargs):
                return np.zeros((2048, 16), dtype=np.float32)

        estimator = doa_mod.DOAEstimator(
            {
                "audio": {"sample_rate": 16000, "channels": 16},
                "doa": {
                    "enabled": True,
                    "active_profile": "triangle_3",
                    "profiles": {
                        "triangle_3": {"mic_indices": [0, 7, 14]},
                        "corners_4": {"mic_indices": [1, 7, 8, 14]},
                    },
                },
            },
            Ring(),
        )

        assert estimator.status["active_profile"] == "triangle_3"
        assert estimator.status["mic_indices"] == [0, 7, 14]

        status = estimator.set_profile("corners_4")
        assert status["active_profile"] == "corners_4"
        assert status["mic_indices"] == [1, 7, 8, 14]

        status = estimator.set_channel_enabled(7, False)
        assert status["mic_indices"] == [1, 8, 14]
        assert status["disabled_channels"] == [7]

        status = estimator.set_channel_enabled(7, True)
        assert status["mic_indices"] == [1, 7, 8, 14]
        assert status["disabled_channels"] == []

    def test_doa_hps_peak_search_defaults_to_600_hz_ceiling(self):
        doa_mod = _import_from_audi_app("doa_estimator")

        config = doa_mod.parse_doa_config({"doa": {"enabled": True}})
        settings = config.profiles[config.active_profile]

        assert settings.peak_fmax_hz == 600.0
        assert settings.window_s == 1.28
        assert settings.context_padding_s == 0.32
        assert doa_mod.analysis_window_s(settings) == pytest.approx(1.92)

    def test_doa_estimate_reads_padded_analysis_window(self, monkeypatch):
        doa_mod = _import_from_audi_app("doa_estimator")

        class Ring:
            def __init__(self):
                self.requested_samples = None

            def get_recent(self, samples, channel=None):
                assert channel is None
                self.requested_samples = samples
                return np.zeros((samples, 16), dtype=np.float32)

        ring = Ring()
        estimator = doa_mod.DOAEstimator(
            {
                "audio": {"sample_rate": 16000, "channels": 16},
                "doa": {"enabled": True},
            },
            ring,
        )
        freqs = np.fft.rfftfreq(2048, d=1.0 / 16000)

        monkeypatch.setattr(
            estimator,
            "_dominant_hps_frequency",
            lambda samples, settings: (445.3125, 12.0),
        )
        monkeypatch.setattr(
            estimator,
            "_stft_channels",
            lambda samples, settings: (
                freqs,
                np.zeros((samples.shape[0], len(freqs), 2), dtype=np.complex128),
            ),
        )
        monkeypatch.setattr(
            estimator,
            "_music_frequencies",
            lambda freqs, dominant_f0, settings: [445.3125],
        )

        def fake_spectrum(*args, **kwargs):
            spectrum = np.zeros(360, dtype=np.float64)
            spectrum[190] = 1.0
            return spectrum

        monkeypatch.setattr(doa_mod, "pyroom_azimuth_spectrum", fake_spectrum)

        result = estimator.force_estimate()

        assert result["ok"] is True
        assert ring.requested_samples == 30720
        assert result["window_s"] == 1.28
        assert result["context_padding_s"] == 0.32
        assert result["analysis_window_s"] == pytest.approx(1.92)

    def test_doa_rejects_disabling_below_two_active_mics(self):
        doa_mod = _import_from_audi_app("doa_estimator")

        class Ring:
            def get_recent(self, *args, **kwargs):
                return np.zeros((2048, 16), dtype=np.float32)

        estimator = doa_mod.DOAEstimator(
            {
                "audio": {"sample_rate": 16000, "channels": 16},
                "doa": {
                    "enabled": True,
                    "mic_indices": [0, 7],
                },
            },
            Ring(),
        )

        with pytest.raises(ValueError, match="at least two"):
            estimator.set_channel_enabled(7, False)

    def test_doa_smoothing_lowers_confidence_on_large_jump(self):
        doa_mod = _import_from_audi_app("doa_estimator")

        class Ring:
            def get_recent(self, *args, **kwargs):
                return np.zeros((2048, 16), dtype=np.float32)

        estimator = doa_mod.DOAEstimator(
            {
                "audio": {"sample_rate": 16000, "channels": 16},
                "doa": {
                    "enabled": True,
                    "music": {
                        "smoothing_predictions": 3,
                        "confidence_jump_deg": 45,
                    },
                },
            },
            Ring(),
        )
        settings = estimator._active_settings()
        spectrum = np.array([0.1, 0.2, 1.0, 0.2, 0.1], dtype=np.float64)

        first = estimator._smooth_azimuth(10.0, spectrum, 12.0, settings)
        jumped = estimator._smooth_azimuth(130.0, spectrum, 12.0, settings)

        assert jumped["jump_deg"] == pytest.approx(120.0)
        assert jumped["confidence"] < first["confidence"]
        assert doa_mod.angular_distance_deg(jumped["azimuth_deg"], 130.0) > 1.0

    def test_doa_algorithm_names_map_to_pyroomacoustics(self):
        doa_mod = _import_from_audi_app("doa_estimator")
        pra = pytest.importorskip("pyroomacoustics")

        assert doa_mod._pyroom_doa_class("MUSIC") is pra.doa.MUSIC
        assert doa_mod._pyroom_doa_class("NormMUSIC") is pra.doa.NormMUSIC
        assert doa_mod._pyroom_doa_class("SRP-PHAT") is pra.doa.SRP
        assert doa_mod._pyroom_doa_class("srp") is pra.doa.SRP

        with pytest.raises(ValueError, match="unknown DOA algorithm"):
            doa_mod._pyroom_doa_class("bad-doa")

    def test_pyroom_music_matches_legacy_music_on_synthetic_snapshots(self):
        doa_mod = _import_from_audi_app("doa_estimator")
        pytest.importorskip("pyroomacoustics")

        settings = doa_mod.DOASettings(n_fft=2048, algorithm="MUSIC")
        mic_indices = (0, 7, 14)
        mic_positions = doa_mod.UMA16_MIC_POSITIONS_M[np.array(mic_indices)]
        azimuths = np.arange(-180.0, 180.0, 1.0)
        stft_freqs = np.fft.rfftfreq(settings.n_fft, d=1.0 / 16000)
        target_freqs = [296.875, 304.6875, 593.75, 601.5625, 890.625, 898.4375]
        source_azimuth = 40.0
        stft = np.zeros((len(mic_indices), len(stft_freqs), 16), dtype=np.complex128)
        rng = np.random.default_rng(4)
        azimuth_math_rad = np.deg2rad(90.0 - source_azimuth)
        direction = np.array(
            [np.cos(azimuth_math_rad), np.sin(azimuth_math_rad), 0.0],
            dtype=np.float64,
        )
        for freq_hz in target_freqs:
            freq_idx = int(np.searchsorted(stft_freqs, freq_hz))
            phase = (
                2.0
                * np.pi
                * freq_hz
                / doa_mod.SPEED_OF_SOUND_M_S
                * (mic_positions @ direction)
            )
            steering = np.exp(1j * phase)
            frame_signal = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, stft.shape[2]))
            stft[:, freq_idx, :] = steering[:, np.newaxis] * frame_signal[np.newaxis, :]
        stft += rng.normal(0.0, 0.005, stft.shape)
        stft += 1j * rng.normal(0.0, 0.005, stft.shape)

        class Ring:
            def get_recent(self, *args, **kwargs):
                return np.zeros((2048, 16), dtype=np.float32)

        estimator = doa_mod.DOAEstimator(
            {
                "audio": {"sample_rate": 16000, "channels": 16},
                "doa": {"enabled": True},
            },
            Ring(),
        )
        covariance = estimator._covariance(stft, stft_freqs, target_freqs, settings)
        legacy_spectrum = doa_mod.legacy_music_azimuth_spectrum(
            covariance,
            mic_positions,
            target_freqs,
            azimuths,
        )
        pyroom_spectrum = doa_mod.pyroom_azimuth_spectrum(
            stft,
            stft_freqs,
            mic_indices,
            target_freqs,
            azimuths,
            settings,
        )

        legacy_peak = float(azimuths[int(np.argmax(legacy_spectrum))])
        pyroom_peak = float(azimuths[int(np.argmax(pyroom_spectrum))])

        assert legacy_peak == pytest.approx(source_azimuth)
        assert doa_mod.angular_distance_deg(legacy_peak, pyroom_peak) <= 2.0

    def test_detector_defaults_to_all_capture_channels(self, tmp_path):
        detector = _import_from_audi_app("detector")

        engine = detector.DetectionEngine(
            {
                "detection": {
                    "model_path": str(tmp_path / "missing.tflite"),
                    "alert_history_file": str(tmp_path / "alerts.jsonl"),
                },
                "audio": {"sample_rate": 16000, "channels": 16},
                "storage": {"alerts_dir": str(tmp_path / "alerts")},
            },
            ring_buffer=None,
        )

        assert engine.status["enabled_channels"] == list(range(16))

    def test_detector_accepts_comma_separated_enabled_channels(self, tmp_path):
        detector = _import_from_audi_app("detector")

        engine = detector.DetectionEngine(
            {
                "detection": {
                    "model_path": str(tmp_path / "missing.tflite"),
                    "alert_history_file": str(tmp_path / "alerts.jsonl"),
                    "enabled_channels": "1,3,7,8",
                },
                "audio": {"sample_rate": 16000, "channels": 16},
                "storage": {"alerts_dir": str(tmp_path / "alerts")},
            },
            ring_buffer=None,
        )

        assert engine.status["enabled_channels"] == [1, 3, 7, 8]

    def test_color_typing_returns_red_blue_or_unknown(self, tmp_path):
        detector = _import_from_audi_app("detector")
        engine = detector.DetectionEngine(
            {
                "detection": {
                    "model_path": str(tmp_path / "missing.tflite"),
                    "alert_history_file": str(tmp_path / "alerts.jsonl"),
                    "red_color_threshold": 0.60,
                    "blue_color_threshold": 0.60,
                },
                "audio": {"sample_rate": 16000},
                "storage": {"alerts_dir": str(tmp_path)},
            },
            ring_buffer=None,
        )

        assert (
            engine._classify_blue_red(True, np.array([0.0, 1.0]))["drone_color"]
            == "RED"
        )
        assert (
            engine._classify_blue_red(True, np.array([1.0, 0.0]))["drone_color"]
            == "BLUE"
        )
        assert (
            engine._classify_blue_red(True, np.array([0.0, 0.0]))["drone_color"]
            == "UNKNOWN"
        )
        assert (
            engine._classify_blue_red(False, np.array([0.0, 1.0]))["drone_color"]
            == "UNKNOWN"
        )

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
                            "red_color_threshold": 0.62,
                            "blue_color_threshold": 0.64,
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
        assert engine.red_color_threshold == 0.62
        assert engine.blue_color_threshold == 0.64

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
                            "red_color_threshold": 0.65,
                            "blue_color_threshold": 0.66,
                        },
                        "sensitive": {
                            "confidence_threshold_high": 0.72,
                            "red_color_threshold": 0.62,
                            "blue_color_threshold": 0.64,
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
        assert engine.red_color_threshold == 0.62
        assert engine.blue_color_threshold == 0.64

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
        assert engine.red_color_threshold == 0.60
        assert engine.blue_color_threshold == 0.60
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
    red_color_threshold: 0.61
    blue_color_threshold: 0.62
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
            "red_color_threshold"
        ] == 0.61

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

    def test_webui_detector_channel_endpoint(self):
        webui_server = _import_from_audi_app("webui_server")

        class Detector:
            def set_channel_enabled(self, channel_index, enabled):
                assert channel_index == 2
                assert enabled is False
                return {"enabled_channels": [0, 1, 3]}

        webui = webui_server.WebUI({"web": {"port": 0}})
        webui.detector = Detector()

        response = webui._app.test_client().post(
            "/api/detector_channel",
            json={"channel_index": 2, "enabled": False},
        )

        assert response.status_code == 200
        assert response.get_json()["detector"] == {"enabled_channels": [0, 1, 3]}

    def test_webui_status_includes_latest_doa_without_triggering_estimate(self):
        webui_server = _import_from_audi_app("webui_server")

        class DOA:
            @property
            def status(self):
                return {
                    "enabled": True,
                    "azimuth_deg": 45.0,
                    "last_result": {"ok": True, "azimuth_deg": 45.0},
                }

        webui = webui_server.WebUI({"web": {"port": 0}})
        webui.doa = DOA()

        response = webui._app.test_client().get("/api/status")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["doa"]["azimuth_deg"] == 45.0

    def test_webui_doa_profile_and_channel_endpoints(self):
        webui_server = _import_from_audi_app("webui_server")

        class DOA:
            @property
            def status(self):
                return {"active_profile": "triangle_3"}

            def set_profile(self, profile):
                assert profile == "corners_4"
                return {"active_profile": profile}

            def set_channel_enabled(self, channel_index, enabled):
                assert channel_index == 7
                assert enabled is False
                return {"mic_indices": [1, 8, 14], "disabled_channels": [7]}

        webui = webui_server.WebUI({"web": {"port": 0}})
        webui.doa = DOA()
        client = webui._app.test_client()

        profile_response = client.post(
            "/api/doa_profile",
            json={"profile": "corners_4"},
        )
        channel_response = client.post(
            "/api/doa_channel",
            json={"channel_index": 7, "enabled": False},
        )

        assert profile_response.status_code == 200
        assert profile_response.get_json()["doa"] == {"active_profile": "corners_4"}
        assert channel_response.status_code == 200
        assert channel_response.get_json()["doa"]["disabled_channels"] == [7]

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

    def test_webui_mels_uses_ring_buffer_by_default(self):
        webui_server = _import_from_audi_app("webui_server")
        recorder = _import_from_audi_app("recorder")

        ring = recorder.AudioRingBuffer(max_samples=160, channels=4)
        ring.append(np.ones((160, 4), dtype=np.float32) * 0.01)
        class RawRecorder:
            sample_rate = 16000
            ring_buffer = ring

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
        assert payload["source"] == "ring_buffer_fallback"
        assert payload["channels_available"] == 4
        assert [ch["channel_index"] for ch in payload["channels"]] == [0, 1, 2, 3]

    def test_webui_mels_can_return_detector_preprocess_cache(self):
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

        response = webui._app.test_client().get(
            "/api/mels?source=detector_cache&columns=2"
        )

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
