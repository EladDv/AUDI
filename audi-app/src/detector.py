"""
AUDI Type A — Detection Engine (TFLite FP32)

Real TFLite inference with drone detection and optional blue/red typing.

Pipeline:
  1. Capture audio from ring buffer (configurable SR, default 16kHz)
  2. Resample to 16kHz via scipy
  3. Compute Mel spectrogram (numpy/scipy, no torch needed)
  4. Normalize + convert to 3-channel grayscale
  5. Feed to TFLite FP32 interpreter
  6. Sigmoid -> confidence score
  7. Optional combined blue/red output from the same model
  8. Temporal smoothing -> YES/NO decision

Falls back to mock if model not found or ai_edge_litert unavailable.
"""

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from audio_features import TFLiteClassifier
from storage import AlarmSnapshotter, AlertHistory

logger = logging.getLogger("audi.detector")

DEFAULT_LABELS = ["drone"]
DEFAULT_CAPTURE_SAMPLE_RATE = 16000
DEFAULT_WINDOW_SAMPLES = 81920
DEFAULT_DRONE_THRESHOLD = 0.6550
DEFAULT_RED_ENTER_THRESHOLD = 0.37
DEFAULT_RED_EXIT_THRESHOLD = 0.56
DEFAULT_HYSTERESIS_WINDOW = 8
DEFAULT_HYSTERESIS_RATIO = 0.6
DEFAULT_HYSTERESIS_MARGIN = 0.05
DEFAULT_ALARM_COOLDOWN_S = 120.0


class HysteresisState:
    """Schmitt-trigger state tracker with moving-average confirmation."""

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
        self.state = False

    def add(self, score: float) -> bool:
        self.history.append(score)
        if len(self.history) > self.window:
            self.history.pop(0)

        recent = self.history
        k = max(1, math.ceil(len(recent) * self.ratio))
        lo = self.threshold - self.margin
        hi = self.threshold + self.margin

        if self.state:
            below = sum(1 for s in recent if s < lo)
            if below >= k:
                self.state = False
        else:
            above = sum(1 for s in recent if s > hi)
            if above >= k:
                self.state = True

        return self.state

    def clear(self):
        self.history.clear()
        self.state = False

    @property
    def confidence(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)


class ColorHysteresisState:
    """Stateful blue/red typing with sticky RED behavior."""

    def __init__(
        self,
        enter_red_threshold: float = 0.45,
        exit_red_threshold: float = 0.35,
        window: int = 5,
        ratio: float = 0.6,
    ):
        self.enter_red_threshold = enter_red_threshold
        self.exit_red_threshold = exit_red_threshold
        self.window = window
        self.ratio = ratio
        self.history: list[float] = []
        self.state = "UNKNOWN"

    def add(self, red_score: float) -> str:
        self.history.append(red_score)
        if len(self.history) > self.window:
            self.history.pop(0)

        recent = self.history
        k = max(1, math.ceil(len(recent) * self.ratio))
        above_enter = sum(1 for s in recent if s >= self.enter_red_threshold)
        below_exit = sum(1 for s in recent if s <= self.exit_red_threshold)

        if self.state == "RED":
            if below_exit >= k:
                self.state = "BLUE"
        else:
            if above_enter >= k:
                self.state = "RED"
            elif below_exit >= k:
                self.state = "BLUE"
        return self.state

    def clear(self):
        self.history.clear()
        self.state = "UNKNOWN"

    @property
    def confidence(self) -> float | None:
        if not self.history:
            return None
        return sum(self.history) / len(self.history)


@dataclass
class ChannelDetectionState:
    hysteresis: HysteresisState
    color_hysteresis: ColorHysteresisState
    last_alarm_time: float = 0.0
    last_inference: dict | None = None
    score_history: list[dict] = field(default_factory=list)
    color_trace: list[dict] = field(default_factory=list)


# ===========================================================================
# Detection Engine
# ===========================================================================


class DetectionEngine:
    """Orchestrates periodic inference with one combined TFLite model.

    Output shape may be a legacy single detector logit or the combined
    [detector_logit, blue_logit, red_logit]. RED is the positive color class.
    """

    def __init__(
        self,
        config: dict,
        ring_buffer,
        on_alarm: Callable[[dict], None] | None = None,
    ):
        self._base_detection_config = dict(config.get("detection", {}))
        self.threshold_profile = self._base_detection_config.get(
            "active_threshold_profile"
        )
        self.threshold_profiles = self._base_detection_config.get(
            "threshold_profiles", {}
        )
        det_cfg = dict(self._base_detection_config)
        if (
            self.threshold_profile
            and self.threshold_profile in self.threshold_profiles
        ):
            profile_cfg = self.threshold_profiles[self.threshold_profile] or {}
            det_cfg.update(profile_cfg)

        # Model config
        self.model_path = det_cfg.get(
            "model_path",
            "/app/models/model_combined_mn10_mined_hardneg_blue_red.tflite",
        )
        self.model_type = det_cfg.get("model_type", "tflite")
        self.num_threads = det_cfg.get("num_threads", 2)
        self.model_sample_rate = det_cfg.get("model_sample_rate", 16000)
        self.n_mels = det_cfg.get("n_mels", 128)
        self.n_fft = det_cfg.get("n_fft", 1024)
        self.win_length = det_cfg.get("win_length", self.n_fft)
        self.hop_length = det_cfg.get("hop_length", 160)
        self.window_samples = det_cfg.get(
            "window_samples", DEFAULT_WINDOW_SAMPLES
        )  # 16000 * 5.12
        self.stride = det_cfg.get("stride", 0.0625)

        # Thresholds
        self.threshold_yes = det_cfg.get(
            "confidence_threshold_high", DEFAULT_DRONE_THRESHOLD
        )
        self.threshold_blue = det_cfg.get(
            "confidence_threshold_low", self.threshold_yes
        )
        self.labels = det_cfg.get("labels", DEFAULT_LABELS)
        self.inference_interval = det_cfg.get("inference_interval", 0.320)

        # Combined model output: [detector_logit, blue_logit, red_logit].
        self.blue_red_threshold = det_cfg.get("blue_red_threshold", 0.5)
        self.blue_red_min_detection_score = det_cfg.get(
            "blue_red_min_detection_score", self.threshold_blue
        )
        self.red_alert_threshold = det_cfg.get(
            "red_alert_threshold", DEFAULT_RED_ENTER_THRESHOLD
        )
        self.blue_alert_threshold = det_cfg.get("blue_alert_threshold", 0.5)
        self.blue_to_red_threshold = det_cfg.get(
            "blue_to_red_threshold", DEFAULT_RED_ENTER_THRESHOLD
        )
        self.red_to_blue_threshold = det_cfg.get(
            "red_to_blue_threshold", DEFAULT_RED_EXIT_THRESHOLD
        )
        self.alert_on_red = det_cfg.get("alert_on_red", True)
        self.alert_on_blue = det_cfg.get("alert_on_blue", False)
        self.alert_on_unknown = det_cfg.get("alert_on_unknown", False)
        self.save_color_trace = det_cfg.get("save_color_trace", True)

        # Debug mode — logs per-window timing at DEBUG level
        self.debug = det_cfg.get("debug", False)

        self.ring_buffer = ring_buffer
        self.on_alarm = on_alarm

        # Audio config from the main audio section
        audio_cfg = config.get("audio", {})
        self.capture_sample_rate = audio_cfg.get(
            "sample_rate", DEFAULT_CAPTURE_SAMPLE_RATE
        )
        self.input_channels = max(1, int(audio_cfg.get("channels", 4)))

        # Load TFLite classifier
        self.classifier = TFLiteClassifier(
            model_path=self.model_path,
            num_threads=self.num_threads,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            model_sample_rate=self.model_sample_rate,
            window_samples=self.window_samples,
            mel_mean=det_cfg.get("mel_mean"),
            mel_std=det_cfg.get("mel_std"),
            incremental_step_samples=int(round(self.window_samples * self.stride)),
        )
        if self.classifier.expected_frames:
            expected_window_samples = (
                self.classifier.expected_frames * self.hop_length
            )
            if expected_window_samples != self.window_samples:
                logger.warning(
                    "Adjusting window_samples from %d to %d to match model "
                    "input frames=%d",
                    self.window_samples,
                    expected_window_samples,
                    self.classifier.expected_frames,
                )
                self.window_samples = expected_window_samples
                self.classifier.window_samples = expected_window_samples

        self._hysteresis_window = det_cfg.get(
            "hysteresis_window", DEFAULT_HYSTERESIS_WINDOW
        )
        self._hysteresis_ratio = det_cfg.get(
            "hysteresis_ratio", DEFAULT_HYSTERESIS_RATIO
        )
        self._hysteresis_margin = det_cfg.get(
            "hysteresis_margin", DEFAULT_HYSTERESIS_MARGIN
        )
        self._color_hysteresis_window = det_cfg.get(
            "color_hysteresis_window",
            self._hysteresis_window,
        )
        self._color_hysteresis_ratio = det_cfg.get(
            "color_hysteresis_ratio",
            self._hysteresis_ratio,
        )
        self._channel_states: list[ChannelDetectionState] = []
        self._sync_channel_states(self.input_channels)

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
        self._last_channel_results: list[dict] = []
        self._last_channel_mels: dict[int, np.ndarray] = {}
        self._last_mel_timestamp = 0.0
        self._last_inference_time = 0.0
        self._total_inferences = 0
        self._yes_count = 0
        self._red_count = 0
        self._blue_count = 0
        self._last_alarm_time = 0.0
        self.alarm_cooldown_s = det_cfg.get(
            "alarm_cooldown_s", DEFAULT_ALARM_COOLDOWN_S
        )
        self._last_timing: dict = {}

        # Rolling timing stats (last 128 cycles)
        self._timing_window: list[float] = []
        self._timing_log_interval = 100  # log summary every N inferences

        # Rolling score history for debug UI (last 200 cycles)
        self._score_history: list[dict] = []
        self._score_history_max = 200
        self._color_trace: list[dict] = []
        self._color_trace_max = 200

    def _new_channel_state(self) -> ChannelDetectionState:
        return ChannelDetectionState(
            hysteresis=HysteresisState(
                threshold=self.threshold_yes,
                window=self._hysteresis_window,
                ratio=self._hysteresis_ratio,
                margin=self._hysteresis_margin,
            ),
            color_hysteresis=ColorHysteresisState(
                enter_red_threshold=self.blue_to_red_threshold,
                exit_red_threshold=self.red_to_blue_threshold,
                window=self._color_hysteresis_window,
                ratio=self._color_hysteresis_ratio,
            ),
        )

    def _sync_channel_states(self, channel_count: int) -> None:
        channel_count = max(1, int(channel_count))
        while len(self._channel_states) < channel_count:
            self._channel_states.append(self._new_channel_state())
        if len(self._channel_states) > channel_count:
            del self._channel_states[channel_count:]
        self.input_channels = channel_count
        self.hysteresis = self._channel_states[0].hysteresis
        self.color_hysteresis = self._channel_states[0].color_hysteresis

    def _reset_channel_states(self, channel_count: int | None = None) -> None:
        if channel_count is None:
            channel_count = len(self._channel_states) or self.input_channels
        self._channel_states = []
        self._sync_channel_states(channel_count)

    def _profiled_detection_config(self, profile: str | None) -> dict:
        det_cfg = dict(self._base_detection_config)
        if profile:
            profile_cfg = self.threshold_profiles.get(profile)
            if profile_cfg is None:
                raise ValueError(f"Unknown threshold profile: {profile}")
            det_cfg.update(profile_cfg or {})
        return det_cfg

    def set_threshold_profile(self, profile: str) -> dict:
        """Apply a configured threshold profile without restarting the app."""
        det_cfg = self._profiled_detection_config(profile)
        self.threshold_profile = profile
        self.threshold_yes = det_cfg.get(
            "confidence_threshold_high", DEFAULT_DRONE_THRESHOLD
        )
        self.threshold_blue = det_cfg.get(
            "confidence_threshold_low", self.threshold_yes
        )
        self.inference_interval = det_cfg.get(
            "inference_interval", self.inference_interval
        )
        self.blue_red_threshold = det_cfg.get("blue_red_threshold", 0.5)
        self.blue_red_min_detection_score = det_cfg.get(
            "blue_red_min_detection_score", self.threshold_blue
        )
        self.red_alert_threshold = det_cfg.get(
            "red_alert_threshold", DEFAULT_RED_ENTER_THRESHOLD
        )
        self.blue_alert_threshold = det_cfg.get("blue_alert_threshold", 0.5)
        self.blue_to_red_threshold = det_cfg.get(
            "blue_to_red_threshold", DEFAULT_RED_ENTER_THRESHOLD
        )
        self.red_to_blue_threshold = det_cfg.get(
            "red_to_blue_threshold", DEFAULT_RED_EXIT_THRESHOLD
        )
        self.alert_on_red = det_cfg.get("alert_on_red", True)
        self.alert_on_blue = det_cfg.get("alert_on_blue", False)
        self.alert_on_unknown = det_cfg.get("alert_on_unknown", False)
        self.save_color_trace = det_cfg.get("save_color_trace", True)
        self.alarm_cooldown_s = det_cfg.get(
            "alarm_cooldown_s", self.alarm_cooldown_s
        )
        self._hysteresis_window = det_cfg.get(
            "hysteresis_window", DEFAULT_HYSTERESIS_WINDOW
        )
        self._hysteresis_ratio = det_cfg.get(
            "hysteresis_ratio", DEFAULT_HYSTERESIS_RATIO
        )
        self._hysteresis_margin = det_cfg.get(
            "hysteresis_margin", DEFAULT_HYSTERESIS_MARGIN
        )
        self._color_hysteresis_window = det_cfg.get(
            "color_hysteresis_window",
            self._hysteresis_window,
        )
        self._color_hysteresis_ratio = det_cfg.get(
            "color_hysteresis_ratio",
            self._hysteresis_ratio,
        )
        self._reset_channel_states()
        logger.info("Threshold profile switched to %s", profile)
        return self.status

    def set_alert_routing(
        self,
        *,
        alert_on_blue: bool | None = None,
        alert_on_unknown: bool | None = None,
    ) -> dict:
        """Update which non-red detection types are allowed to trigger GPIO."""
        if alert_on_blue is not None:
            self.alert_on_blue = bool(alert_on_blue)
        if alert_on_unknown is not None:
            self.alert_on_unknown = bool(alert_on_unknown)
        logger.info(
            "Alert routing updated: red=%s blue=%s unknown=%s",
            self.alert_on_red,
            self.alert_on_blue,
            self.alert_on_unknown,
        )
        return self.status

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

    @property
    def has_blue_red_model(self) -> bool:
        return self.classifier.is_loaded and self.classifier.output_size >= 3

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(
            f"Detection started - YES>={self.threshold_yes:.2f} "
            f"interval={self.inference_interval:.3f}s "
            f"model={'real' if self.is_real_model else 'mock'} "
            f"combined_blue_red={'on' if self.has_blue_red_model else 'off'}",
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

    def _classify_blue_red(
        self,
        detection_score: float,
        color_logits: np.ndarray | None,
        channel_state: ChannelDetectionState | None = None,
    ) -> dict:
        """Classify a detected drone window as BLUE or RED.

        RED is the positive class. The color is reported only when the drone
        detector score is high enough; otherwise the color head would be
        classifying background.
        """
        result = {
            "drone_color": "UNKNOWN",
            "red_confidence": None,
            "blue_confidence": None,
            "red_logit": None,
            "blue_logit": None,
            "blue_red_enabled": color_logits is not None
            and np.asarray(color_logits).size >= 2,
            "blue_red_threshold": self.blue_red_threshold,
            "blue_red_min_detection_score": self.blue_red_min_detection_score,
        }
        if detection_score < self.blue_red_min_detection_score:
            return result

        if color_logits is None:
            return result
        logits = np.asarray(color_logits, dtype=np.float32)

        if logits.size < 2:
            logger.warning(
                "Blue/red model returned %d logits; expected [blue, red]",
                logits.size,
            )
            return result

        logits = logits[:2].astype(np.float64)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / max(float(exp.sum()), 1e-12)
        blue_conf = float(probs[0])
        red_conf = float(probs[1])
        raw_color = "RED" if red_conf >= self.blue_red_threshold else "BLUE"
        color_hysteresis = (
            channel_state.color_hysteresis
            if channel_state is not None
            else self.color_hysteresis
        )
        color = color_hysteresis.add(red_conf)

        result.update(
            {
                "drone_color": color,
                "raw_drone_color": raw_color,
                "red_confidence": round(red_conf, 4),
                "blue_confidence": round(blue_conf, 4),
                "red_logit": round(float(logits[1]), 4),
                "blue_logit": round(float(logits[0]), 4),
            }
        )
        return result

    def _resolve_alert_level(self, alarm: bool, color_result: dict) -> str:
        """Map detector state and color confidence to an operator alert level."""
        if not alarm:
            return "NO"

        red_conf = color_result.get("red_confidence")
        blue_conf = color_result.get("blue_confidence")
        color = color_result.get("drone_color")

        if color == "RED":
            if (
                self.alert_on_red
                and red_conf is not None
                and red_conf >= self.red_alert_threshold
            ):
                return "RED_ALERT"
            return "DETECTED"
        if color == "BLUE":
            if (
                self.alert_on_blue
                and blue_conf is not None
                and blue_conf >= self.blue_alert_threshold
            ):
                return "BLUE_ALERT"
            return "DETECTED"
        if color == "UNKNOWN" and self.alert_on_unknown:
            return "UNKNOWN_ALERT"
        return "DETECTED"

    def _should_trigger_alarm(self, alert_level: str) -> bool:
        return alert_level in {"RED_ALERT", "BLUE_ALERT", "UNKNOWN_ALERT"}

    def _recent_audio_by_channel(self, window_seconds: float) -> tuple[np.ndarray, list[int]]:
        try:
            audio = self.ring_buffer.get_last_n_seconds(
                window_seconds,
                self.capture_sample_rate,
                channel=None,
            )
        except TypeError:
            audio = self.ring_buffer.get_last_n_seconds(
                window_seconds,
                self.capture_sample_rate,
            )

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            return audio[:, np.newaxis], [0]
        if audio.ndim != 2:
            raise ValueError(f"Expected 1D or 2D ring-buffer audio, got {audio.shape}")
        if audio.shape[1] == 0:
            return audio.reshape(audio.shape[0], 0), []
        return audio, list(range(audio.shape[1]))

    def _select_primary_result(self, results: list[dict]) -> dict | None:
        if not results:
            return None
        priority = {
            "RED_ALERT": 50,
            "BLUE_ALERT": 40,
            "UNKNOWN_ALERT": 30,
            "DETECTED": 20,
            "NO": 0,
        }
        return max(
            results,
            key=lambda r: (
                priority.get(r.get("alert_level", "NO"), 0),
                r.get("raw_score", 0.0),
            ),
        )

    def _inference_cycle(self):
        """Run independent inference on every captured channel."""
        t0 = time.perf_counter()

        window_seconds = self.window_samples / self.model_sample_rate
        audio_by_channel, channel_ids = self._recent_audio_by_channel(window_seconds)
        if len(audio_by_channel) < self.window_samples:
            logger.debug(
                "Not enough audio: %d < %d samples",
                len(audio_by_channel),
                self.window_samples,
            )
            return
        if not channel_ids:
            logger.debug("No audio channels available for inference")
            return
        self._sync_channel_states(len(channel_ids))

        t_pre = time.perf_counter()
        channel_specs = []
        channel_mels: dict[int, np.ndarray] = {}
        for channel_index in channel_ids:
            spec = self.classifier.preprocess(
                audio_by_channel[:, channel_index],
                self.capture_sample_rate,
            )
            channel_specs.append(spec)
            mel = getattr(self.classifier, "last_mel_db", None)
            if mel is not None:
                channel_mels[channel_index] = np.array(
                    mel,
                    dtype=np.float32,
                    copy=True,
                )
        t_mid = time.perf_counter()
        logits_by_channel = [
            self.classifier.predict_logits(spec) for spec in channel_specs
        ]
        t_post = time.perf_counter()

        preprocess_ms = (t_mid - t_pre) * 1000
        inference_ms = (t_post - t_mid) * 1000
        total_ms = (time.perf_counter() - t0) * 1000

        self._last_timing = {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "total_ms": round(total_ms, 2),
            "n_windows": len(channel_ids),
            "channels": len(channel_ids),
        }

        if self.debug:
            logger.debug(
                "Inference: channels=%d pre=%.1fms inf=%.1fms total=%.1fms",
                len(channel_ids),
                preprocess_ms,
                inference_ms,
                total_ms,
            )

        cycle_timestamp = time.time()
        if channel_mels:
            self._last_channel_mels = channel_mels
            self._last_mel_timestamp = cycle_timestamp
        results: list[dict] = []
        for batch_index, channel_index in enumerate(channel_ids):
            logits = np.asarray(
                logits_by_channel[batch_index], dtype=np.float32
            ).flatten()
            channel_state = self._channel_states[channel_index]

            det_logit = float(logits[0]) if logits.size else 0.0
            color_logits = logits[1:3] if logits.size >= 3 else None
            prob = 1.0 / (1.0 + np.exp(-np.clip(det_logit, -50, 50)))
            raw_score = float(prob)
            alarm = channel_state.hysteresis.add(raw_score)
            state = "YES" if alarm else "NO"
            color_result = self._classify_blue_red(
                raw_score,
                color_logits,
                channel_state=channel_state,
            )
            alert_level = self._resolve_alert_level(alarm, color_result)
            alert_id = (
                f"{int(cycle_timestamp)}_ch{channel_index}_"
                f"{self._total_inferences + batch_index + 1}"
            )

            result = {
                "timestamp": cycle_timestamp,
                "alert_id": alert_id,
                "channel_index": channel_index,
                "channel_name": f"ch{channel_index}",
                "state": state,
                "alert_level": alert_level,
                "yes_confidence": round(channel_state.hysteresis.confidence, 4),
                "raw_score": round(raw_score, 4),
                "threshold_yes": self.threshold_yes,
                "threshold_profile": self.threshold_profile,
                **color_result,
            }
            channel_state.last_inference = result
            results.append(result)

            score_entry = {
                "ts": cycle_timestamp,
                "channel_index": channel_index,
                "raw": raw_score,
                "state": state,
                "red": color_result.get("red_confidence"),
                "blue": color_result.get("blue_confidence"),
                "color": color_result.get("drone_color"),
            }
            channel_state.score_history.append(score_entry)
            if len(channel_state.score_history) > self._score_history_max:
                channel_state.score_history.pop(0)
            self._score_history.append(score_entry)
            if len(self._score_history) > self._score_history_max:
                self._score_history.pop(0)

            trace_entry = {
                "ts": result["timestamp"],
                "alert_id": alert_id,
                "channel_index": channel_index,
                "state": state,
                "alert_level": alert_level,
                "raw_score": result["raw_score"],
                "yes_confidence": result["yes_confidence"],
                "det_logit": round(det_logit, 4),
                "red_confidence": color_result.get("red_confidence"),
                "blue_confidence": color_result.get("blue_confidence"),
                "drone_color": color_result.get("drone_color"),
                "red_logit": color_result.get("red_logit"),
                "blue_logit": color_result.get("blue_logit"),
            }
            channel_state.color_trace.append(trace_entry)
            if len(channel_state.color_trace) > self._color_trace_max:
                channel_state.color_trace.pop(0)
            self._color_trace.append(trace_entry)
            if len(self._color_trace) > self._color_trace_max:
                self._color_trace.pop(0)

        # Rolling timing stats (log summary every N inferences)
        self._timing_window.append(total_ms)
        if len(self._timing_window) > 128:
            self._timing_window.pop(0)
        self._total_inferences += len(results)
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

        self._last_channel_results = results
        all_channel_results = [
            {
                "channel_index": r.get("channel_index"),
                "channel_name": r.get("channel_name"),
                "state": r.get("state"),
                "alert_level": r.get("alert_level"),
                "yes_confidence": r.get("yes_confidence"),
                "raw_score": r.get("raw_score"),
                "drone_color": r.get("drone_color"),
                "red_confidence": r.get("red_confidence"),
                "blue_confidence": r.get("blue_confidence"),
            }
            for r in results
        ]
        for result in results:
            result["firing_channel_index"] = result["channel_index"]
            result["firing_channel_name"] = result["channel_name"]
            result["all_channel_results"] = all_channel_results

        primary = self._select_primary_result(results)
        if primary is not None:
            primary_with_channels = dict(primary)
            primary_with_channels["channels"] = results
            primary_with_channels["channel_count"] = len(results)
            self._last_inference = primary_with_channels

        for result in results:
            channel_state = self._channel_states[result["channel_index"]]
            raw_score = float(result["raw_score"])
            if result["state"] != "YES":
                logger.debug(
                    "NO ch%d: conf=%.2f (raw=%.2f)",
                    result["channel_index"],
                    result["yes_confidence"],
                    raw_score,
                )
                continue

            self._yes_count += 1
            if result.get("drone_color") == "RED":
                self._red_count += 1
            elif result.get("drone_color") == "BLUE":
                self._blue_count += 1
            now = time.time()
            in_cooldown = (
                self.alarm_cooldown_s > 0
                and (now - channel_state.last_alarm_time) < self.alarm_cooldown_s
            )
            should_trigger_alarm = self._should_trigger_alarm(result["alert_level"])
            if in_cooldown:
                logger.info(
                    "%s ch%d suppressed (cooldown: %.1fs remaining): "
                    "conf=%.2f (raw=%.2f, color=%s, red=%s)",
                    result["alert_level"],
                    result["channel_index"],
                    self.alarm_cooldown_s - (now - channel_state.last_alarm_time),
                    result["yes_confidence"],
                    raw_score,
                    result.get("drone_color"),
                    result.get("red_confidence"),
                )
            elif not should_trigger_alarm:
                logger.info(
                    "%s ch%d observed without GPIO alert: "
                    "conf=%.2f (raw=%.2f, color=%s, red=%s)",
                    result["alert_level"],
                    result["channel_index"],
                    result["yes_confidence"],
                    raw_score,
                    result.get("drone_color"),
                    result.get("red_confidence"),
                )
            else:
                channel_state.last_alarm_time = now
                self._last_alarm_time = max(
                    state.last_alarm_time for state in self._channel_states
                )
                logger.warning(
                    "%s ch%d: conf=%.2f (raw=%.2f, color=%s, red=%s)",
                    result["alert_level"],
                    result["channel_index"],
                    result["yes_confidence"],
                    raw_score,
                    result.get("drone_color"),
                    result.get("red_confidence"),
                )
                if self.save_color_trace:
                    result["color_trace"] = channel_state.color_trace[-60:]
                self.alert_history.append(result)
                threading.Thread(
                    target=self._save_snapshot_and_alert,
                    args=(result,),
                    daemon=True,
                ).start()
                if self.on_alarm:
                    self.on_alarm(result)

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
        channel_status = [
            {
                "channel_index": idx,
                "state": (state.last_inference or {}).get("state", "NO"),
                "alert_level": (state.last_inference or {}).get("alert_level", "NO"),
                "yes_confidence": (state.last_inference or {}).get(
                    "yes_confidence", 0.0
                ),
                "raw_score": (state.last_inference or {}).get("raw_score", 0.0),
                "drone_color": (state.last_inference or {}).get(
                    "drone_color", "UNKNOWN"
                ),
                "red_confidence": (state.last_inference or {}).get(
                    "red_confidence"
                ),
                "blue_confidence": (state.last_inference or {}).get(
                    "blue_confidence"
                ),
                "in_cooldown": (
                    self.alarm_cooldown_s > 0
                    and (time.time() - state.last_alarm_time)
                    < self.alarm_cooldown_s
                ),
            }
            for idx, state in enumerate(self._channel_states)
        ]
        return {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "labels": self.labels,
            "input_channels": self.input_channels,
            "threshold_yes": self.threshold_yes,
            "threshold_blue": self.threshold_blue,
            "inference_interval": self.inference_interval,
            "window_samples": self.window_samples,
            "stride": self.stride,
            "model_sample_rate": self.model_sample_rate,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "threshold_profile": self.threshold_profile,
            "threshold_profiles": sorted(self.threshold_profiles.keys()),
            "blue_red_model_loaded": bool(
                self.has_blue_red_model
                or last.get("blue_red_enabled", False)
            ),
            "blue_red_threshold": self.blue_red_threshold,
            "blue_red_min_detection_score": self.blue_red_min_detection_score,
            "red_alert_threshold": self.red_alert_threshold,
            "blue_alert_threshold": self.blue_alert_threshold,
            "blue_to_red_threshold": self.blue_to_red_threshold,
            "red_to_blue_threshold": self.red_to_blue_threshold,
            "color_hysteresis_state": self.color_hysteresis.state,
            "color_hysteresis_confidence": self.color_hysteresis.confidence,
            "channels": channel_status,
            "alert_on_red": self.alert_on_red,
            "alert_on_blue": self.alert_on_blue,
            "alert_on_unknown": self.alert_on_unknown,
            "running": self._thread is not None and self._thread.is_alive(),
            "total_inferences": self._total_inferences,
            "yes_count": self._yes_count,
            "red_count": self._red_count,
            "blue_count": self._blue_count,
            "alert_history_count": self.alert_history.count,
            "current_state": last.get("state", "NO"),
            "alert_level": last.get("alert_level", "NO"),
            "yes_confidence": last.get("yes_confidence", 0.0),
            "drone_color": last.get("drone_color", "UNKNOWN"),
            "red_confidence": last.get("red_confidence"),
            "blue_confidence": last.get("blue_confidence"),
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
            "color_trace": self._color_trace[-80:],
        }

    @property
    def last_inference(self) -> dict | None:
        return self._last_inference

    @property
    def latest_mels(self) -> dict:
        return {
            "timestamp": self._last_mel_timestamp,
            "sample_rate": self.model_sample_rate,
            "n_mels": self.n_mels,
            "hop_length": self.hop_length,
            "channels": [
                {
                    "channel_index": channel_index,
                    "mel": np.array(mel, dtype=np.float32, copy=True),
                }
                for channel_index, mel in sorted(self._last_channel_mels.items())
            ],
        }

    @property
    def smoothed_predictions(self) -> dict:
        last = self._last_inference or {}
        return {
            "yes": last.get("yes_confidence", 0.0),
            "red": last.get("red_confidence"),
            "blue": last.get("blue_confidence"),
            "drone_color": last.get("drone_color", "UNKNOWN"),
            "alert_level": last.get("alert_level", "NO"),
            "state": last.get("state", "NO"),
        }
