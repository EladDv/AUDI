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
from mvdr_beamformer import MVDRBeamformer, MVDRBeamformerConfig
from storage import AlarmSnapshotter, AlertHistory

logger = logging.getLogger("audi.detector")

DEFAULT_LABELS = ["drone"]
DEFAULT_CAPTURE_SAMPLE_RATE = 16000
DEFAULT_WINDOW_SAMPLES = 81920
DEFAULT_DRONE_THRESHOLD = 0.6550
DEFAULT_RED_COLOR_THRESHOLD = 0.60
DEFAULT_BLUE_COLOR_THRESHOLD = 0.60
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


@dataclass
class ChannelDetectionState:
    hysteresis: HysteresisState
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
        self.labels = det_cfg.get("labels", DEFAULT_LABELS)
        self.inference_interval = det_cfg.get("inference_interval", 0.320)

        # Combined model output: [detector_logit, blue_logit, red_logit].
        self.red_color_threshold = det_cfg.get(
            "red_color_threshold", DEFAULT_RED_COLOR_THRESHOLD
        )
        self.blue_color_threshold = det_cfg.get(
            "blue_color_threshold", DEFAULT_BLUE_COLOR_THRESHOLD
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
        self.enabled_channel_indices = self._normalize_enabled_channels(
            det_cfg.get("enabled_channels"),
            self.input_channels,
        )
        beamforming_cfg = dict(det_cfg.get("input_beamforming", {}) or {})
        self.input_beamforming_enabled = bool(beamforming_cfg.get("enabled", False))
        self._beamformer: MVDRBeamformer | None = None
        self._beam_metadata: list[dict] = []
        self._mvdr_covariance_update_interval_s = float(
            beamforming_cfg.get("covariance_update_interval_seconds", 300.0)
        )
        self._mvdr_covariance_no_drone_threshold = float(
            beamforming_cfg.get("covariance_no_drone_threshold", 0.25)
        )
        self._last_mvdr_covariance_update_time = 0.0
        self._last_mvdr_covariance_update_reason: str | None = None
        self._last_mvdr_covariance_error: str | None = None
        if self.input_beamforming_enabled:
            mic_indices = beamforming_cfg.get("mic_indices")
            if isinstance(mic_indices, str):
                mic_indices = [
                    int(part.strip())
                    for part in mic_indices.split(",")
                    if part.strip()
                ]
            self._beamformer = MVDRBeamformer(
                MVDRBeamformerConfig(
                    beam_count=int(beamforming_cfg.get("beam_count", 36)),
                    elevation_count=int(beamforming_cfg.get("elevation_count", 3)),
                    min_elevation_deg=float(
                        beamforming_cfg.get("min_elevation_deg", 5.0)
                    ),
                    max_elevation_deg=float(
                        beamforming_cfg.get("max_elevation_deg", 70.0)
                    ),
                    n_fft=int(beamforming_cfg.get("n_fft", 512)),
                    hop_length=int(beamforming_cfg.get("hop_length", 160)),
                    diagonal_loading=float(
                        beamforming_cfg.get("diagonal_loading", 1e-2)
                    ),
                    mic_indices=tuple(mic_indices) if mic_indices else None,
                    deglitch_enabled=bool(
                        beamforming_cfg.get("deglitch_enabled", True)
                    ),
                    deglitch_threshold=float(
                        beamforming_cfg.get("deglitch_threshold", 0.001)
                    ),
                    deglitch_loudness_ratio=float(
                        beamforming_cfg.get("deglitch_loudness_ratio", 8.0)
                    ),
                    deglitch_diff_ratio=float(
                        beamforming_cfg.get("deglitch_diff_ratio", 12.0)
                    ),
                    deglitch_window_samples=int(
                        beamforming_cfg.get("deglitch_window_samples", 64)
                    ),
                )
            )
            self._beam_metadata = [
                {
                    "channel_index": beam.index,
                    "channel_name": beam.name,
                    "azimuth_deg": beam.azimuth_deg,
                    "elevation_deg": beam.elevation_deg,
                }
                for beam in self._beamformer.beams
            ]
            self.enabled_channel_indices = self._normalize_enabled_channels(
                beamforming_cfg.get("enabled_beams", det_cfg.get("enabled_channels")),
                len(self._beam_metadata),
            )

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
        self._channel_states: list[ChannelDetectionState] = []
        self._detector_input_count = (
            len(self._beam_metadata)
            if self.input_beamforming_enabled
            else self.input_channels
        )
        self._sync_channel_states(self._detector_input_count)

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
        self.doa_estimator = None
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
            )
        )

    def _sync_channel_states(self, channel_count: int) -> None:
        channel_count = max(1, int(channel_count))
        while len(self._channel_states) < channel_count:
            self._channel_states.append(self._new_channel_state())
        if len(self._channel_states) > channel_count:
            del self._channel_states[channel_count:]
        self._detector_input_count = channel_count
        self.hysteresis = self._channel_states[0].hysteresis

    def _reset_channel_states(self, channel_count: int | None = None) -> None:
        if channel_count is None:
            channel_count = len(self._channel_states) or self.input_channels
        self._channel_states = []
        self._sync_channel_states(channel_count)

    @staticmethod
    def _normalize_enabled_channels(value, channel_count: int) -> set[int]:
        if value is None:
            return set(range(max(1, int(channel_count))))
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "all", "*", "null", "none"}:
                return set(range(max(1, int(channel_count))))
            value = [part.strip() for part in normalized.split(",")]
        enabled: set[int] = set()
        for item in value:
            idx = int(item)
            if 0 <= idx < channel_count:
                enabled.add(idx)
        return enabled

    def set_channel_enabled(self, channel_index: int, enabled: bool) -> dict:
        channel_index = int(channel_index)
        if channel_index < 0 or channel_index >= self._detector_input_count:
            raise ValueError(
                "channel_index must be between 0 and "
                f"{self._detector_input_count - 1}"
            )
        self._sync_channel_states(self._detector_input_count)
        if enabled:
            self.enabled_channel_indices.add(channel_index)
        else:
            self.enabled_channel_indices.discard(channel_index)
            self._channel_states[channel_index].hysteresis.clear()
            self._channel_states[channel_index].last_inference = None
        return self.status

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
        self.inference_interval = det_cfg.get(
            "inference_interval", self.inference_interval
        )
        self.red_color_threshold = det_cfg.get(
            "red_color_threshold", DEFAULT_RED_COLOR_THRESHOLD
        )
        self.blue_color_threshold = det_cfg.get(
            "blue_color_threshold", DEFAULT_BLUE_COLOR_THRESHOLD
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

    @property
    def input_mode(self) -> str:
        return "mvdr_beam" if self.input_beamforming_enabled else "channel"

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
            pass_started_at = time.time()
            elapsed = pass_started_at - self._last_inference_time
            if elapsed >= self.inference_interval:
                self._last_inference_time = pass_started_at
                try:
                    self._inference_cycle()
                except Exception as e:
                    logger.error("Inference cycle failed: %s", e)
            # inference_interval is start-to-start spacing. If inference is
            # slower than the interval, only yield the loop briefly.
            remaining = self.inference_interval - (
                time.time() - self._last_inference_time
            )
            self._stop_event.wait(max(0.01, remaining))

    def _classify_blue_red(
        self,
        detected: bool,
        color_logits: np.ndarray | None,
    ) -> dict:
        """Classify a positive detection as RED, BLUE, or UNKNOWN.

        RED is the positive class. The color is reported only when the drone
        detector is already in a positive state; otherwise the color head would
        be classifying background.
        """
        result = {
            "drone_color": "UNKNOWN",
            "red_confidence": None,
            "blue_confidence": None,
            "red_logit": None,
            "blue_logit": None,
            "blue_red_enabled": color_logits is not None
            and np.asarray(color_logits).size >= 2,
            "red_color_threshold": self.red_color_threshold,
            "blue_color_threshold": self.blue_color_threshold,
        }
        if not detected:
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
        if red_conf >= self.red_color_threshold:
            color = "RED"
        elif blue_conf >= self.blue_color_threshold:
            color = "BLUE"
        else:
            color = "UNKNOWN"

        result.update(
            {
                "drone_color": color,
                "red_confidence": round(red_conf, 4),
                "blue_confidence": round(blue_conf, 4),
                "red_logit": round(float(logits[1]), 4),
                "blue_logit": round(float(logits[0]), 4),
            }
        )
        return result

    def _resolve_alert_level(self, alarm: bool, color_result: dict) -> str:
        """Map detector state and final color state to an operator alert level."""
        if not alarm:
            return "NO"

        color = color_result.get("drone_color")

        if color == "RED" and self.alert_on_red:
            return "RED_ALERT"
        if color == "BLUE" and self.alert_on_blue:
            return "BLUE_ALERT"
        if color == "UNKNOWN" and self.alert_on_unknown:
            return "UNKNOWN_ALERT"
        return "DETECTED"

    def _should_trigger_alarm(self, alert_level: str) -> bool:
        return alert_level in {"RED_ALERT", "BLUE_ALERT", "UNKNOWN_ALERT"}

    def _recent_audio_by_channel(
        self, window_seconds: float
    ) -> tuple[np.ndarray, list[int]]:
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

    def _detector_inputs_from_audio(
        self,
        audio_by_channel: np.ndarray,
        channel_ids: list[int],
    ) -> tuple[list[dict], dict[int, dict]]:
        if not self.input_beamforming_enabled:
            inputs = [
                {
                    "channel_index": idx,
                    "channel_name": f"ch{idx}",
                    "audio": audio_by_channel[:, idx],
                    "detector_enabled": idx in self.enabled_channel_indices,
                }
                for idx in channel_ids
            ]
            return inputs, {}

        if self._beamformer is None:
            return [], {}
        if not self._beamformer.has_covariance:
            self._refresh_mvdr_covariance(audio_by_channel, reason="startup")
        beams = self._beamformer.beamform(audio_by_channel, self.capture_sample_rate)
        inputs = []
        for beam in beams:
            idx = int(beam["index"])
            inputs.append(
                {
                    "channel_index": idx,
                    "channel_name": beam["name"],
                    "audio": beam["audio"],
                    "detector_enabled": idx in self.enabled_channel_indices,
                    "beam_azimuth_deg": beam["azimuth_deg"],
                    "beam_elevation_deg": beam["elevation_deg"],
                    "beam_mic_indices": beam["mic_indices"],
                }
            )
        beam_lookup = {
            int(item["channel_index"]): {
                "channel_index": int(item["channel_index"]),
                "channel_name": str(item["channel_name"]),
                "beam_azimuth_deg": float(item["beam_azimuth_deg"]),
                "beam_elevation_deg": float(item["beam_elevation_deg"]),
            }
            for item in inputs
        }
        return inputs, beam_lookup

    def _refresh_mvdr_covariance(self, audio_by_channel: np.ndarray, reason: str) -> bool:
        if not self.input_beamforming_enabled or self._beamformer is None:
            return False
        try:
            ok = self._beamformer.update_covariance(
                audio_by_channel,
                self.capture_sample_rate,
            )
        except Exception as exc:
            self._last_mvdr_covariance_error = str(exc)
            logger.warning("MVDR covariance update failed (%s): %s", reason, exc)
            return False
        if not ok:
            self._last_mvdr_covariance_error = "not_enough_audio_or_channels"
            return False
        self._last_mvdr_covariance_update_time = time.time()
        self._last_mvdr_covariance_update_reason = reason
        self._last_mvdr_covariance_error = None
        logger.info("MVDR covariance updated (%s)", reason)
        return True

    def _maybe_refresh_mvdr_covariance(
        self,
        audio_by_channel: np.ndarray,
        results: list[dict],
        cycle_timestamp: float,
    ) -> None:
        if (
            not self.input_beamforming_enabled
            or self._beamformer is None
            or self._mvdr_covariance_update_interval_s <= 0
        ):
            return
        if (
            cycle_timestamp - self._last_mvdr_covariance_update_time
            < self._mvdr_covariance_update_interval_s
        ):
            return
        enabled_results = [
            result
            for result in results
            if int(result.get("channel_index", -1)) in self.enabled_channel_indices
        ]
        if not enabled_results:
            return
        max_raw_score = max(
            float(result.get("raw_score", 1.0)) for result in enabled_results
        )
        any_positive_state = any(result.get("state") == "YES" for result in enabled_results)
        if any_positive_state or max_raw_score > self._mvdr_covariance_no_drone_threshold:
            return
        self._refresh_mvdr_covariance(audio_by_channel, reason="periodic_no_drone")

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
        detector_inputs, detector_metadata = self._detector_inputs_from_audio(
            audio_by_channel,
            channel_ids,
        )
        if not detector_inputs:
            logger.debug("No detector inputs available for inference")
            return
        self._sync_channel_states(len(detector_inputs))
        enabled_inputs = [
            item for item in detector_inputs if item["detector_enabled"]
        ]
        if not enabled_inputs:
            self._last_timing = {
                "preprocess_ms": 0.0,
                "inference_ms": 0.0,
                "total_ms": round((time.perf_counter() - t0) * 1000, 2),
                "n_windows": 0,
                "channels": 0,
                "input_mode": self.input_mode,
            }
            return

        t_pre = time.perf_counter()
        channel_specs = []
        channel_mels: dict[int, np.ndarray] = {}
        for item in enabled_inputs:
            channel_index = int(item["channel_index"])
            spec = self.classifier.preprocess(
                item["audio"],
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
            "n_windows": len(enabled_inputs),
            "channels": len(enabled_inputs),
            "input_mode": self.input_mode,
        }

        if self.debug:
            logger.debug(
                "Inference: channels=%d pre=%.1fms inf=%.1fms total=%.1fms",
                len(enabled_inputs),
                preprocess_ms,
                inference_ms,
                total_ms,
            )

        cycle_timestamp = time.time()
        if channel_mels:
            self._last_channel_mels = channel_mels
            self._last_mel_timestamp = cycle_timestamp
        results: list[dict] = []
        for batch_index, item in enumerate(enabled_inputs):
            channel_index = int(item["channel_index"])
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
                alarm,
                color_logits,
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
                "channel_name": str(item["channel_name"]),
                "state": state,
                "alert_level": alert_level,
                "yes_confidence": round(channel_state.hysteresis.confidence, 4),
                "raw_score": round(raw_score, 4),
                "threshold_yes": self.threshold_yes,
                "threshold_profile": self.threshold_profile,
                **color_result,
            }
            if self.input_beamforming_enabled:
                result.update(
                    {
                        "input_mode": "mvdr_beam",
                        "beam_azimuth_deg": item.get("beam_azimuth_deg"),
                        "beam_elevation_deg": item.get("beam_elevation_deg"),
                        "beam_mic_indices": item.get("beam_mic_indices"),
                    }
                )
            channel_state.last_inference = result
            results.append(result)

            score_entry = {
                "ts": cycle_timestamp,
                "channel_index": channel_index,
                "channel_name": result["channel_name"],
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
                "channel_name": result["channel_name"],
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

        self._maybe_refresh_mvdr_covariance(
            audio_by_channel,
            results,
            cycle_timestamp,
        )
        self._last_channel_results = results
        results_by_channel = {r.get("channel_index"): r for r in results}
        all_channel_results = []
        for idx, state in enumerate(self._channel_states):
            meta = detector_metadata.get(idx, {})
            channel_name = meta.get("channel_name", f"ch{idx}")
            r = results_by_channel.get(idx)
            if r is None:
                last_for_channel = state.last_inference or {}
                all_channel_results.append(
                    {
                        "channel_index": idx,
                        "channel_name": channel_name,
                        "detector_enabled": idx in self.enabled_channel_indices,
                        "state": "NO",
                        "alert_level": "NO",
                        "yes_confidence": 0.0,
                        "raw_score": 0.0,
                        "drone_color": last_for_channel.get(
                            "drone_color", "UNKNOWN"
                        ),
                        "red_confidence": None,
                        "blue_confidence": None,
                        **{
                            key: meta[key]
                            for key in ("beam_azimuth_deg", "beam_elevation_deg")
                            if key in meta
                        },
                    }
                )
                continue
            all_channel_results.append(
                {
                    "channel_index": r.get("channel_index"),
                    "channel_name": r.get("channel_name"),
                    "detector_enabled": True,
                    "state": r.get("state"),
                    "alert_level": r.get("alert_level"),
                    "yes_confidence": r.get("yes_confidence"),
                    "raw_score": r.get("raw_score"),
                    "drone_color": r.get("drone_color"),
                    "red_confidence": r.get("red_confidence"),
                    "blue_confidence": r.get("blue_confidence"),
                    **{
                        key: r[key]
                        for key in ("beam_azimuth_deg", "beam_elevation_deg")
                        if key in r
                    },
                }
            )
        doa_result = None
        for result in results:
            result["firing_channel_index"] = result["channel_index"]
            result["firing_channel_name"] = result["channel_name"]
            result["all_channel_results"] = all_channel_results
            if result["state"] == "YES":
                if doa_result is None:
                    doa_result = self._compute_doa_estimate()
                if doa_result is not None:
                    result["doa"] = doa_result

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

    def _compute_doa_estimate(self) -> dict | None:
        """Compute DOA only after the model reports a drone."""
        if not self.doa_estimator:
            return None
        try:
            status = self.doa_estimator.status
            if not status.get("enabled"):
                return None
            return self.doa_estimator.force_estimate()
        except Exception as exc:
            logger.warning("DOA estimate failed for detection: %s", exc)
            return {
                "ok": False,
                "timestamp": time.time(),
                "error": str(exc),
            }

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
                "detector_enabled": idx in self.enabled_channel_indices,
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
            "input_mode": self.input_mode,
            "detector_input_count": self._detector_input_count,
            "input_beamforming": {
                "enabled": self.input_beamforming_enabled,
                "beams": self._beam_metadata,
                "covariance_update_interval_seconds": (
                    self._mvdr_covariance_update_interval_s
                ),
                "covariance_no_drone_threshold": (
                    self._mvdr_covariance_no_drone_threshold
                ),
                "last_covariance_update_time": (
                    self._last_mvdr_covariance_update_time
                ),
                "last_covariance_update_reason": (
                    self._last_mvdr_covariance_update_reason
                ),
                "last_covariance_error": self._last_mvdr_covariance_error,
            },
            "enabled_channels": sorted(self.enabled_channel_indices),
            "threshold_yes": self.threshold_yes,
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
            "red_color_threshold": self.red_color_threshold,
            "blue_color_threshold": self.blue_color_threshold,
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
            "channel_score_history": [
                {
                    "channel_index": idx,
                    "history": state.score_history[-self._score_history_max :],
                }
                for idx, state in enumerate(self._channel_states)
            ],
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
