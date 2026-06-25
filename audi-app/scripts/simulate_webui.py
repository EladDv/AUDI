#!/usr/bin/env python3
"""Run the AUDI Flask UI against simulated recorder/detector/GPIO inputs."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = APP_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from webui_server import WebUI  # noqa: E402


class SimRingBuffer:
    def __init__(self, sample_rate: int = 16000, channels: int = 4):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_samples = sample_rate * 120

    @property
    def total_samples(self) -> int:
        return self.max_samples

    def get_last_n_seconds(
        self,
        n_seconds: float,
        sample_rate: int,
        channel: int | None = 0,
    ) -> np.ndarray:
        frames = max(1, int(float(n_seconds) * sample_rate))
        wall_time = time.time()
        t = np.arange(frames, dtype=np.float64) / float(sample_rate)
        base = []
        for idx in range(self.channels):
            components = np.zeros(frames, dtype=np.float32)
            freqs = np.geomspace(
                90.0 + idx * 35.0,
                7200.0 - idx * 450.0,
                num=28,
            )
            for harmonic_idx, freq in enumerate(freqs):
                wobble = 1.0 + 0.012 * np.sin(2.0 * np.pi * (0.07 + idx * 0.02) * t)
                phase = idx * 0.4 + harmonic_idx * 0.9
                band_gate = 0.55 + 0.45 * np.sin(
                    wall_time * 0.35
                    + idx * 0.8
                    + harmonic_idx * 0.37
                )
                weight = (0.45 + 0.55 * band_gate) / (1.0 + harmonic_idx * 0.035)
                components += (
                    weight
                    * np.sin(2.0 * np.pi * freq * wobble * t + phase)
                ).astype(np.float32)
            components /= max(1.0, float(np.max(np.abs(components))))
            slow_gate = 0.55 + 0.35 * np.sin(
                2.0 * np.pi * (0.11 + idx * 0.03) * t + idx
            )
            fast_gate = 0.75 + 0.20 * np.sin(
                2.0 * np.pi * (0.9 + idx * 0.17) * t
            )
            pulse = 0.22 + 0.20 * max(0.0, math.sin(wall_time * 0.7 + idx))
            if idx == 2 and int(wall_time) % 18 < 7:
                pulse += 0.35
            signal = (pulse * slow_gate * fast_gate * components).astype(np.float32)
            base.append(signal)
        audio = np.stack(base, axis=1)
        if channel is None:
            return audio
        return audio[:, int(channel)]


class SimRawRecorder:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 4
        self.segment_duration = 300
        self.ring_buffer = SimRingBuffer(self.sample_rate, self.channels)
        self._running = True
        self._paused_until = 0.0
        self._force_stopped = False
        self._bytes_captured_total = 42_000_000
        self._segment_count = 8

    @property
    def is_recording(self) -> bool:
        return self._running and not self._force_stopped

    @property
    def is_paused(self) -> bool:
        return self._paused_until > time.time()

    @property
    def is_force_stopped(self) -> bool:
        return self._force_stopped

    @property
    def audio_healthy(self) -> bool:
        return self.is_recording and not self.is_paused

    def start(self):
        self._running = True
        self._force_stopped = False

    def stop(self):
        self._force_stopped = True

    def pause(self, duration_seconds: int = 300):
        self._paused_until = time.time() + duration_seconds

    def resume(self):
        self._paused_until = 0.0

    def toggle_start_stop(self) -> bool:
        self._force_stopped = not self._force_stopped
        return not self._force_stopped

    def get_rms_level(self, window_seconds: float = 0.01) -> float:
        audio = self.ring_buffer.get_last_n_seconds(
            window_seconds,
            self.sample_rate,
            channel=None,
        )
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    def channel_health(self) -> list[dict]:
        audio = self.ring_buffer.get_last_n_seconds(1.0, self.sample_rate, channel=None)
        rows = []
        for idx in range(self.channels):
            ch = audio[:, idx]
            rms = float(np.sqrt(np.mean(ch.astype(np.float64) ** 2)))
            peak = float(np.max(np.abs(ch)))
            rows.append(
                {
                    "channel_index": idx,
                    "rms": round(rms, 6),
                    "peak": round(peak, 6),
                    "dbfs": round(20.0 * math.log10(max(rms, 1e-9)), 1),
                    "clipping_percent": 0.0,
                    "healthy": bool(rms > 1e-6 and self.audio_healthy),
                    "frames": int(ch.size),
                }
            )
        return rows


class SimRecorderManager:
    def __init__(self):
        self.recorder = SimRawRecorder()

    @property
    def status(self) -> dict:
        r = self.recorder
        return {
            "running": r.is_recording,
            "paused": r.is_paused,
            "stopped": r.is_force_stopped,
            "device": "simulated",
            "sample_rate": r.sample_rate,
            "channels": r.channels,
            "segment_duration": r.segment_duration,
            "ring_buffer_seconds": 120,
            "ring_buffer_capacity": 120,
            "bytes_captured": r._bytes_captured_total,
            "segments_recorded": r._segment_count,
            "audio_healthy": r.audio_healthy,
            "rms_level": r.get_rms_level(),
            "channel_health": r.channel_health(),
        }


class SimStorage:
    @property
    def status(self) -> dict:
        return {
            "data_dir": "simulated",
            "max_size_gb": 32.0,
            "used_gb": 3.8,
            "free_gb": 54.2,
            "wav_files": 2,
            "flac_files": 6,
            "compress_enabled": True,
            "over_budget": False,
            "low_disk": False,
            "alerts_dir": "simulated-alerts",
            "alerts_used_mb": 218.4,
            "alerts_max_mb": 2048.0,
        }


class SimAlertHistory:
    def __init__(self):
        now = time.time()
        self.entries = [
            {
                "timestamp": now - 74,
                "alert_id": "sim-red-ch2",
                "state": "YES",
                "alert_level": "RED_ALERT",
                "yes_confidence": 0.88,
                "raw_score": 0.94,
                "drone_color": "RED",
                "red_confidence": 0.91,
                "blue_confidence": 0.09,
                "channel_index": 2,
                "channel_name": "ch2",
                "firing_channel_index": 2,
                "firing_channel_name": "ch2",
            },
            {
                "timestamp": now - 31,
                "alert_id": "sim-blue-ch1",
                "state": "YES",
                "alert_level": "DETECTED",
                "yes_confidence": 0.71,
                "raw_score": 0.78,
                "drone_color": "BLUE",
                "red_confidence": 0.16,
                "blue_confidence": 0.84,
                "channel_index": 1,
                "channel_name": "ch1",
                "firing_channel_index": 1,
                "firing_channel_name": "ch1",
                "field_tag_color": "yellow",
                "field_tag": "correct_detection_incorrect_classification",
                "operator_label": "correct_detection_incorrect_classification",
            },
        ]

    def append(self, entry: dict):
        self.entries.append(dict(entry))

    def read_recent(self, limit: int = 50) -> list:
        return self.entries[-limit:]

    def label_alert(self, alert_id: str, label: str) -> dict | None:
        for entry in self.entries:
            if entry.get("alert_id") == alert_id:
                entry["operator_label"] = label
                return entry
        return None

    @property
    def count(self) -> int:
        return len(self.entries)


class SimDetector:
    def __init__(self):
        self.threshold_yes = 0.655
        self.threshold_profile = "field"
        self.alert_on_blue = False
        self.alert_on_unknown = False
        self.n_mels = 128
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = 160
        self.model_sample_rate = 16000
        self.alert_history = SimAlertHistory()

    def _channels(self) -> list[dict]:
        now = time.time()
        rows = []
        for idx in range(4):
            raw = 0.15 + 0.16 * (math.sin(now * 0.5 + idx) + 1.0)
            if idx == 2 and int(now) % 18 < 7:
                raw = 0.91
            state = "YES" if raw >= self.threshold_yes else "NO"
            alert_level = "RED_ALERT" if idx == 2 and state == "YES" else "NO"
            rows.append(
                {
                    "channel_index": idx,
                    "state": state,
                    "alert_level": alert_level,
                    "yes_confidence": round(raw, 4),
                    "raw_score": round(raw, 4),
                    "drone_color": "RED" if alert_level == "RED_ALERT" else "UNKNOWN",
                    "red_confidence": 0.89 if alert_level == "RED_ALERT" else None,
                    "blue_confidence": 0.11 if alert_level == "RED_ALERT" else None,
                    "in_cooldown": False,
                }
            )
        return rows

    @property
    def status(self) -> dict:
        channels = self._channels()
        primary = max(channels, key=lambda row: row["raw_score"])
        return {
            "model_path": "simulated-mn10.tflite",
            "model_type": "tflite",
            "labels": ["drone"],
            "input_channels": 4,
            "threshold_yes": self.threshold_yes,
            "threshold_blue": self.threshold_yes,
            "inference_interval": 0.32,
            "window_samples": 81920,
            "stride": 0.0625,
            "model_sample_rate": self.model_sample_rate,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "threshold_profile": self.threshold_profile,
            "threshold_profiles": ["field", "quiet", "sensitive"],
            "blue_red_model_loaded": True,
            "blue_red_threshold": 0.5,
            "blue_red_min_detection_score": 0.655,
            "red_alert_threshold": 0.37,
            "blue_alert_threshold": 0.5,
            "blue_to_red_threshold": 0.37,
            "red_to_blue_threshold": 0.56,
            "color_hysteresis_state": primary.get("drone_color", "UNKNOWN"),
            "color_hysteresis_confidence": primary.get("red_confidence"),
            "channels": channels,
            "alert_on_red": True,
            "alert_on_blue": self.alert_on_blue,
            "alert_on_unknown": self.alert_on_unknown,
            "running": True,
            "total_inferences": int(time.time()) * 4,
            "yes_count": 12,
            "red_count": 5,
            "blue_count": 2,
            "alert_history_count": self.alert_history.count,
            "current_state": primary["state"],
            "alert_level": primary["alert_level"],
            "yes_confidence": primary["yes_confidence"],
            "drone_color": primary["drone_color"],
            "red_confidence": primary["red_confidence"],
            "blue_confidence": primary["blue_confidence"],
            "timing": {"preprocess_ms": 44.0, "inference_ms": 62.0, "total_ms": 116.0},
            "last_inference": primary,
            "real_model": False,
            "debug": True,
            "alarm_cooldown_s": 120.0,
            "in_cooldown": False,
            "timing_avg_ms": 116.0,
            "timing_p50_ms": 114.0,
            "score_history": [
                {"ts": time.time() - i, "raw": 0.15 + 0.7 * abs(math.sin(i / 9.0))}
                for i in range(80, 0, -1)
            ],
            "color_trace": [
                {
                    "ts": time.time() - i * 2,
                    "channel_index": i % 4,
                    "alert_level": "RED_ALERT" if i % 7 == 0 else "NO",
                    "raw_score": 0.25 + 0.7 * abs(math.sin(i)),
                    "red_confidence": 0.9 if i % 7 == 0 else None,
                    "blue_confidence": 0.1 if i % 7 == 0 else None,
                    "drone_color": "RED" if i % 7 == 0 else "UNKNOWN",
                }
                for i in range(20, 0, -1)
            ],
        }

    @property
    def last_inference(self) -> dict:
        return self.status["last_inference"]

    @property
    def smoothed_predictions(self) -> dict:
        return {
            "yes": self.status["yes_confidence"],
            "red": self.status["red_confidence"],
            "blue": self.status["blue_confidence"],
            "drone_color": self.status["drone_color"],
            "alert_level": self.status["alert_level"],
            "state": self.status["current_state"],
        }

    def force_inference(self) -> dict:
        return self.last_inference

    def set_threshold_profile(self, profile: str) -> dict:
        self.threshold_profile = profile
        return self.status

    def set_alert_routing(
        self,
        *,
        alert_on_blue: bool | None = None,
        alert_on_unknown: bool | None = None,
    ) -> dict:
        if alert_on_blue is not None:
            self.alert_on_blue = bool(alert_on_blue)
        if alert_on_unknown is not None:
            self.alert_on_unknown = bool(alert_on_unknown)
        return self.status


class SimGPIO:
    def __init__(self):
        now = time.time()
        self._alarming = True
        self._events = [
            {"name": "green", "gpio": 22, "timestamp": now - 58},
            {"name": "yellow", "gpio": 27, "timestamp": now - 36},
            {"name": "red", "gpio": 17, "timestamp": now - 14},
        ]

    def status(self) -> dict:
        return {
            "enabled": True,
            "has_gpio": False,
            "alarming": self._alarming,
            "record_led": True,
            "alert_pin": 2,
            "strobe_pin": 24,
            "reset_pin": 23,
            "record_led_pin": None,
            "record_button_pin": None,
            "pause_button_pin": None,
            "field_tag_button_pins": {"green": 22, "yellow": 27, "red": 17},
            "button_events": self._events,
            "active_alert_level": "RED_ALERT",
        }

    def clear_alarm(self):
        self._alarming = False

    def trigger_alarm(self, alert_level: str = "RED_ALERT"):
        self._alarming = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    ui = WebUI({"web": {"host": args.host, "port": args.port}})
    ui.recorder = SimRecorderManager()
    ui.storage = SimStorage()
    ui.detector = SimDetector()
    ui.gpio = SimGPIO()
    print(f"Simulated AUDI Flask UI: http://{args.host}:{args.port}", flush=True)
    ui._app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
