"""
Pi Audio Guard — Detection Engine (TFLite int8)

Real TFLite inference with three-state detection (YES/BLUE/NO).

Pipeline:
  1. Capture audio from ring buffer (configurable SR, default 48kHz)
  2. Resample to 16kHz via scipy
  3. Compute Mel spectrogram (numpy/scipy, no torch needed)
  4. Normalize + convert to 3-channel grayscale
  5. Feed to TFLite int8 interpreter
  6. Sigmoid → confidence score
  7. Temporal smoothing → YES/BLUE/NO decision

Falls back to mock if model not found or ai_edge_litert unavailable.
"""

import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger("audio_guard.detector")

DEFAULT_LABELS = ["general_alert"]


# ===========================================================================
# Mel Spectrogram (lightweight numpy/scipy — no torch/torchaudio needed)
# ===========================================================================


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Build a mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    if f_max is None:
        f_max = sample_rate / 2.0

    n_freqs = n_fft // 2 + 1
    mel_min = _hz_to_mel(np.array(f_min))
    mel_max = _hz_to_mel(np.array(f_max))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        f_left = bins[m]
        f_center = bins[m + 1]
        f_right = bins[m + 2]
        for k in range(f_left, f_center):
            filters[m, k] = (k - f_left) / max(1, f_center - f_left)
        for k in range(f_center, f_right):
            filters[m, k] = (f_right - k) / max(1, f_right - f_center)
    return filters


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 160,
    f_min: float = 0.0,
    f_max: float | None = None,
    power: float = 2.0,
) -> np.ndarray:
    """Compute mel spectrogram matching torchaudio (librosa backend).

    Uses librosa with center=True (pad on both sides), periodic Hann
    window, and ``10 * log10(S)`` conversion — identical to
    ``torchaudio.transforms.MelSpectrogram + AmplitudeToDB`` defaults.
    """
    import librosa

    if f_max is None:
        f_max = sample_rate / 2.0

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=power,
        window="hann",
        center=True,
    )

    # Power → dB (torchaudio AmplitudeToDB default: 10*log10, ref=1.0)
    mel = np.maximum(mel, 1e-10)
    mel = 10.0 * np.log10(mel)

    return mel.astype(np.float32)


# ===========================================================================
# Resampling (lightweight scipy)
# ===========================================================================


def resample_audio(
    audio: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample audio using scipy. Returns float32 array."""
    if orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio.astype(np.float64), up, down).astype(np.float32)


# ===========================================================================
# TFLite Classifier
# ===========================================================================


class TFLiteClassifier:
    """Load and run a TFLite int8 model for drone detection."""

    def __init__(
        self,
        model_path: str,
        num_threads: int = 2,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 160,
        model_sample_rate: int = 16000,
        window_samples: int = 40960,  # 16000 * 2.56
        mel_mean: float | None = None,
        mel_std: float | None = None,
    ):
        self.model_path = model_path
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model_sample_rate = model_sample_rate
        self.window_samples = window_samples
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._loaded = False

        try:
            self._load_model(num_threads)
        except Exception as e:
            logger.warning("TFLite model not loaded: %s — using mock", e)

    def _load_model(self, num_threads: int):
        from ai_edge_litert.interpreter import Interpreter

        self._interpreter = Interpreter(
            model_path=self.model_path,
            num_threads=num_threads,
        )
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._loaded = True

        inp_shape = self._input_details[0]["shape"]
        logger.info(
            "TFLite model loaded: %s, input=%s, threads=%d",
            Path(self.model_path).name,
            inp_shape,
            num_threads,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def preprocess(self, audio: np.ndarray, capture_sr: int) -> np.ndarray:
        """Convert raw audio to model-ready spectrogram.

        Args:
            audio: Float32 audio array at capture_sr.
            capture_sr: Sample rate of the input audio.

        Returns:
            Spectrogram of shape (1, 3, n_mels, n_frames).
        """
        # Resample to model sample rate
        if capture_sr != self.model_sample_rate:
            audio = resample_audio(audio, capture_sr, self.model_sample_rate)

        # Pad/trim to window_samples
        if len(audio) < self.window_samples:
            audio = np.pad(audio, (0, self.window_samples - len(audio)))
        else:
            audio = audio[: self.window_samples]

        # RMS-normalize + peak-limit (matches training pipeline)
        audio_rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        if audio_rms > 1e-8:
            audio = audio / audio_rms
        peak = float(np.max(np.abs(audio)))
        if peak > 0.98:
            audio = audio * (0.98 / peak)

        # Compute mel spectrogram
        mel = compute_mel_spectrogram(
            audio,
            self.model_sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )[..., :-1]

        # Normalize
        if self.mel_mean is not None and self.mel_std is not None:
            mel = (mel - self.mel_mean) / max(self.mel_std, 1e-8)
        else:
            # RMS normalization
            rms = np.sqrt(np.mean(mel**2)) + 1e-8
            mel = mel / rms

        # Convert to 3-channel (grayscale → RGB)
        mel = np.stack([mel] * 3, axis=0)

        # Add batch dimension: (1, 3, n_mels, n_frames)
        mel = mel[np.newaxis, ...]

        return mel.astype(np.float32)

    def predict(self, spec: np.ndarray) -> float:
        """Run inference on a preprocessed spectrogram.

        Returns:
            Raw logit (before sigmoid).
        """
        if not self._loaded:
            return 0.0

        self._interpreter.set_tensor(self._input_details[0]["index"], spec)
        self._interpreter.invoke()
        logit = self._interpreter.get_tensor(self._output_details[0]["index"])
        return float(logit.flatten()[0])


# ===========================================================================
# Schmitt-Trigger Hysteresis
# ===========================================================================


class HysteresisState:
    """Schmitt-trigger state tracker with moving-average confirmation.

    Uses asymmetric thresholds: turns ON only when >= ratio of the last
    ``window`` scores exceed ``threshold + margin``, and turns OFF only when
    >= ratio fall below ``threshold - margin``.

    Mirrors ``audi.hysteresis.apply_hysteresis`` for standalone RPi use.
    """

    def __init__(
        self,
        threshold: float = 0.70,
        window: int = 5,
        ratio: float = 0.6,
        margin: float = 0.05,
    ):
        self.threshold = threshold
        self.window = window
        self.ratio = ratio
        self.margin = margin
        self.history: list[float] = []
        self.state = False  # OFF initially

    def add(self, score: float) -> bool:
        """Feed a new score, return current hysteresis state."""
        self.history.append(score)
        if len(self.history) > self.window:
            self.history.pop(0)

        recent = self.history
        k = max(1, int(len(recent) * self.ratio))
        lo = self.threshold - self.margin
        hi = self.threshold + self.margin

        if self.state:
            # ON: stay ON unless >= k recent scores drop below LO
            below = sum(1 for s in recent if s < lo)
            if below >= k:
                self.state = False
        else:
            # OFF: turn ON if >= k recent scores exceed HI
            above = sum(1 for s in recent if s > hi)
            if above >= k:
                self.state = True

        return self.state

    def clear(self):
        self.history.clear()
        self.state = False

    @property
    def confidence(self) -> float:
        """Mean of recent scores (for display)."""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)


# ===========================================================================
# Alarm Snapshotter (unchanged from original)
# ===========================================================================


class AlarmSnapshotter:
    """Saves pre/post alarm audio to alerts directory."""

    def __init__(self, alerts_dir: str, sample_rate: int = 48000):
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self._stop_event = threading.Event()

    def save_snapshot(
        self, ring_buffer, detection: dict, recorder_ref
    ) -> dict | None:
        ts = int(time.time())
        state = detection.get("state", "YES")
        yes_conf = detection.get("yes_confidence", 0.0)

        event_dir = self.alerts_dir / f"yes_{ts}"
        event_dir.mkdir(parents=True, exist_ok=True)

        try:
            pre_samples = ring_buffer.get_last_n_seconds(60, self.sample_rate)
            pre_path = event_dir / "pre_60s.wav"
            self._samples_to_wav(pre_samples, str(pre_path))
            pre_size = pre_path.stat().st_size

            post_path = event_dir / "post_60s.wav"
            self._stop_event.wait(60)
            post_samples = ring_buffer.get_last_n_seconds(60, self.sample_rate)
            self._samples_to_wav(post_samples, str(post_path))
            post_size = post_path.stat().st_size
            combined = np.concatenate([pre_samples, post_samples])
            combined_path = event_dir / "full_120s.wav"
            self._samples_to_wav(combined, str(combined_path))
            combined_size = combined_path.stat().st_size

            meta = {
                "timestamp": ts,
                "timestamp_iso": datetime.fromtimestamp(ts, tz=UTC).isoformat(),
                "state": state,
                "yes_confidence": yes_conf,
                "threshold_yes": detection.get("threshold_yes", 0.70),
                "files": {
                    "pre_60s": str(pre_path),
                    "post_60s": str(post_path),
                    "full_120s": str(combined_path),
                },
                "sizes_bytes": {
                    "pre_60s": pre_size,
                    "post_60s": post_size,
                    "full_120s": combined_size,
                },
                "sample_rate": self.sample_rate,
            }
            meta_path = event_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(
                "Alarm snapshot saved: %s (pre=%dKB, post=%dKB)",
                event_dir,
                pre_size // 1024,
                post_size // 1024,
            )
            return meta
        except Exception as e:
            logger.error("Failed to save alarm snapshot: %s", e)
            return None

    def _samples_to_wav(self, samples: np.ndarray, filepath: str):
        import wave

        pcm = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())


# ===========================================================================
# Alert History
# ===========================================================================


class AlertHistory:
    """Append-only alert history stored as JSON lines."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, entry: dict):
        with self._lock:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    def read_recent(self, limit: int = 50) -> list:
        entries = []
        try:
            with open(self.filepath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            return []
        return entries[-limit:]

    @property
    def count(self) -> int:
        try:
            with open(self.filepath) as f:
                return sum(1 for line in f if line.strip())
        except FileNotFoundError:
            return 0


# ===========================================================================
# Detection Engine
# ===========================================================================


class DetectionEngine:
    """Orchestrates periodic inference with TFLite model.

    Three output states:
      YES  — confidence >= high threshold → GPIO alarm + snapshot + history
      BLUE — low <= confidence < high      → log + history only
      NO   — confidence < low              → nothing
    """

    def __init__(
        self,
        config: dict,
        ring_buffer,
        on_alarm: Callable[[dict], None] | None = None,
    ):
        det_cfg = config.get("detection", {})

        # Model config
        self.model_path = det_cfg.get("model_path", "/app/models/model.tflite")
        self.model_type = det_cfg.get("model_type", "tflite")
        self.num_threads = det_cfg.get("num_threads", 2)
        self.model_sample_rate = det_cfg.get("model_sample_rate", 16000)
        self.n_mels = det_cfg.get("n_mels", 128)
        self.n_fft = det_cfg.get("n_fft", 1024)
        self.hop_length = det_cfg.get("hop_length", 160)
        self.window_samples = det_cfg.get(
            "window_samples", 40960
        )  # 16000 * 2.56
        self.stride = det_cfg.get("stride", 0.0625)

        # Thresholds
        self.threshold_yes = det_cfg.get("confidence_threshold_high", 0.70)
        self.labels = det_cfg.get("labels", DEFAULT_LABELS)
        self.inference_interval = det_cfg.get("inference_interval", 0.320)

        # Debug mode — logs per-window timing at DEBUG level
        self.debug = det_cfg.get("debug", False)

        self.ring_buffer = ring_buffer
        self.on_alarm = on_alarm

        # Audio config from the main audio section
        audio_cfg = config.get("audio", {})
        self.capture_sample_rate = audio_cfg.get("sample_rate", 48000)

        # Load TFLite classifier
        self.classifier = TFLiteClassifier(
            model_path=self.model_path,
            num_threads=self.num_threads,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            model_sample_rate=self.model_sample_rate,
            window_samples=self.window_samples,
            mel_mean=det_cfg.get("mel_mean"),
            mel_std=det_cfg.get("mel_std"),
        )

        # Hysteresis state tracker (Schmitt-trigger)
        self.hysteresis = HysteresisState(
            threshold=self.threshold_yes,
            window=det_cfg.get("hysteresis_window", 5),
            ratio=det_cfg.get("hysteresis_ratio", 0.6),
            margin=det_cfg.get("hysteresis_margin", 0.05),
        )

        self.snapshotter = AlarmSnapshotter(
            config.get("storage", {}).get("alerts_dir", "/data/alerts"),
            sample_rate=self.capture_sample_rate,
        )
        self.alert_history = AlertHistory(
            det_cfg.get(
                "alert_history_file", "/data/alerts/alert_history.json"
            ),
        )
        # State
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._recorder_ref = None
        self._last_inference: dict | None = None
        self._last_inference_time = 0.0
        self._total_inferences = 0
        self._yes_count = 0
        self._last_alarm_time = 0.0
        self.alarm_cooldown_s = det_cfg.get("alarm_cooldown_s", 0)
        self._last_timing: dict = {}  # {preprocess_ms, inference_ms, total_ms, n_windows}

        # Rolling timing stats (last 128 cycles)
        self._timing_window: list[float] = []
        self._timing_log_interval = 100  # log summary every N inferences

        # Rolling score history for debug UI (last 200 cycles)
        self._score_history: list[dict] = []
        self._score_history_max = 200

    @property
    def recorder(self):
        return self._recorder_ref

    @recorder.setter
    def recorder(self, val):
        self._recorder_ref = val
        if val:
            self.snapshotter.sample_rate = val.recorder.sample_rate

    @property
    def is_real_model(self) -> bool:
        return self.classifier.is_loaded

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(
            f"Detection started — YES≥{self.threshold_yes:.2f} "
            f"interval={self.inference_interval:.3f}s model={'real' if self.is_real_model else 'mock'}",
        )

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self):
        while not self._stop_event.is_set():
            now = time.time()
            elapsed = now - self._last_inference_time
            if elapsed >= self.inference_interval:
                try:
                    self._inference_cycle()
                    self._last_inference_time = time.time()
                except Exception as e:
                    logger.error("Inference cycle failed: %s", e)
            # Sleep remaining time, min 10ms to avoid busy-wait
            remaining = self.inference_interval - (
                time.time() - self._last_inference_time
            )
            self._stop_event.wait(max(0.01, remaining))

    def _inference_cycle(self):
        """Take latest window from ring buffer, run single inference, determine state."""
        t0 = time.perf_counter()

        # Get exactly one window of audio from the ring buffer
        window_seconds = self.window_samples / self.model_sample_rate
        audio = self.ring_buffer.get_last_n_seconds(
            window_seconds,
            self.capture_sample_rate,
        )
        if len(audio) < self.window_samples:
            logger.debug(
                "Not enough audio: %d < %d samples",
                len(audio),
                self.window_samples,
            )
            return

        # Single inference — no sliding windows
        t_pre = time.perf_counter()
        spec = self.classifier.preprocess(audio, self.capture_sample_rate)
        t_mid = time.perf_counter()
        logit = self.classifier.predict(spec)
        t_post = time.perf_counter()

        preprocess_ms = (t_mid - t_pre) * 1000
        inference_ms = (t_post - t_mid) * 1000
        total_ms = (time.perf_counter() - t0) * 1000

        self._last_timing = {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "total_ms": round(total_ms, 2),
            "n_windows": 1,
        }

        if self.debug:
            logger.debug(
                "Inference: pre=%.1fms inf=%.1fms total=%.1fms",
                preprocess_ms,
                inference_ms,
                total_ms,
            )

        # Hysteresis → binary YES/NO
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
        raw_score = float(prob)
        alarm = self.hysteresis.add(raw_score)
        state = "YES" if alarm else "NO"

        result = {
            "timestamp": time.time(),
            "state": state,
            "yes_confidence": round(self.hysteresis.confidence, 4),
            "raw_score": round(raw_score, 4),
            "threshold_yes": self.threshold_yes,
        }

        self._last_inference = result
        self._total_inferences += 1

        # Record in score history for debug UI
        self._score_history.append(
            {"ts": time.time(), "raw": raw_score, "state": state}
        )
        if len(self._score_history) > self._score_history_max:
            self._score_history.pop(0)

        # Rolling timing stats (log summary every N inferences)
        self._timing_window.append(total_ms)
        if len(self._timing_window) > 128:
            self._timing_window.pop(0)
        if self._total_inferences % self._timing_log_interval == 0:
            import statistics

            avg = statistics.mean(self._timing_window)
            p50 = statistics.median(self._timing_window)
            p99 = sorted(self._timing_window)[
                int(len(self._timing_window) * 0.99)
            ]
            logger.info(
                "Timing (last %d): avg=%.1fms p50=%.1fms p99=%.1fms",
                len(self._timing_window),
                avg,
                p50,
                p99,
            )

        if state == "YES":
            self._yes_count += 1
            now = time.time()
            in_cooldown = (
                self.alarm_cooldown_s > 0
                and (now - self._last_alarm_time) < self.alarm_cooldown_s
            )
            if in_cooldown:
                logger.info(
                    "YES suppressed (cooldown: %.1fs remaining): conf=%.2f (raw=%.2f)",
                    self.alarm_cooldown_s - (now - self._last_alarm_time),
                    self.hysteresis.confidence,
                    raw_score,
                )
            else:
                self._last_alarm_time = now
                logger.warning(
                    "YES ALARM: conf=%.2f (raw=%.2f)",
                    self.hysteresis.confidence,
                    raw_score,
                )
                self.alert_history.append(result)
                threading.Thread(
                    target=self._save_snapshot_and_alert,
                    args=(result,),
                    daemon=True,
                ).start()
                if self.on_alarm:
                    self.on_alarm(result)

        else:
            logger.debug(
                "NO: conf=%.2f (raw=%.2f)",
                self.hysteresis.confidence,
                raw_score,
            )

    def _save_snapshot_and_alert(self, detection: dict):
        self.snapshotter.save_snapshot(
            self.ring_buffer, detection, self._recorder_ref
        )

    def force_inference(self) -> dict | None:
        self._inference_cycle()
        return self._last_inference

    @property
    def status(self) -> dict:
        last = self._last_inference or {}
        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "labels": self.labels,
            "threshold_yes": self.threshold_yes,
            "inference_interval": self.inference_interval,
            "window_samples": self.window_samples,
            "stride": self.stride,
            "model_sample_rate": self.model_sample_rate,
            "running": self._thread is not None and self._thread.is_alive(),
            "total_inferences": self._total_inferences,
            "yes_count": self._yes_count,
            "alert_history_count": self.alert_history.count,
            "current_state": last.get("state", "NO"),
            "yes_confidence": last.get("yes_confidence", 0.0),
            "timing": self._last_timing,
            "last_inference": self._last_inference,
            "real_model": self.is_real_model,
            "debug": self.debug,
            "alarm_cooldown_s": self.alarm_cooldown_s,
            "in_cooldown": (
                self.alarm_cooldown_s > 0
                and (time.time() - self._last_alarm_time)
                < self.alarm_cooldown_s
            ),
            "timing_avg_ms": (
                round(sum(self._timing_window) / len(self._timing_window), 1)
                if self._timing_window
                else 0
            ),
            "timing_p50_ms": (
                round(
                    sorted(self._timing_window)[len(self._timing_window) // 2],
                    1,
                )
                if self._timing_window
                else 0
            ),
            "score_history": (self._score_history if self.debug else None),
        }

    @property
    def last_inference(self) -> dict | None:
        return self._last_inference

    @property
    def smoothed_predictions(self) -> dict:
        last = self._last_inference or {}
        return {
            "yes": last.get("yes_confidence", 0.0),
            "state": last.get("state", "NO"),
        }
