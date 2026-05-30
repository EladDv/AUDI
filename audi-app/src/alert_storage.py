"""Alert snapshot and history storage for the Pi detector."""

import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger("audio_guard.detector")


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
                "alert_id": detection.get("alert_id"),
                "alert_level": detection.get("alert_level"),
                "state": state,
                "yes_confidence": yes_conf,
                "threshold_yes": detection.get("threshold_yes", 0.70),
                "drone_color": detection.get("drone_color"),
                "red_confidence": detection.get("red_confidence"),
                "blue_confidence": detection.get("blue_confidence"),
                "color_trace": detection.get("color_trace"),
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

    def label_alert(self, alert_id: str, label: str) -> dict | None:
        """Persist an operator label onto an existing alert history entry."""
        allowed = {"no_drone", "drone_red", "drone_blue", "unclear"}
        if label not in allowed:
            raise ValueError(f"Unsupported label: {label}")

        with self._lock:
            entries = self._read_all_unlocked()
            updated = None
            for entry in entries:
                if str(entry.get("alert_id", "")) == str(alert_id):
                    entry["operator_label"] = label
                    entry["operator_labeled_at"] = time.time()
                    updated = entry
                    break
            if updated is None:
                return None
            with open(self.filepath, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry, default=str) + "\n")
            return updated

    def _read_all_unlocked(self) -> list:
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
        return entries

    def read_recent(self, limit: int = 50) -> list:
        with self._lock:
            return self._read_all_unlocked()[-limit:]

    @property
    def count(self) -> int:
        try:
            with open(self.filepath) as f:
                return sum(1 for line in f if line.strip())
        except FileNotFoundError:
            return 0
