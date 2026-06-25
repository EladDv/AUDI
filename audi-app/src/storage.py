"""
AUDI Type A — Storage Manager

Manages a ring buffer on disk of up to 32GB of audio recordings.

Strategy:
  1. New WAV segments arrive from the recorder into the data directory.
  2. Background thread compresses old WAVs to FLAC or WavPack.
  3. When disk usage exceeds the cap, deletes oldest files first
     (compressed segments deleted before WAVs).
  4. Filesystem is checked on a 30-second heartbeat.
"""

import json
import logging
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger("audi.storage")
FLAC_MAX_CHANNELS = 8
FIELD_TAGS = {
    "green": "correct_detection_correct_classification",
    "yellow": "correct_detection_incorrect_classification",
    "red": "incorrect_detection",
}

# ---------------------------------------------------------------------------
# Lossless compression
# ---------------------------------------------------------------------------

class FlacCompressor:
    """Compress WAV files losslessly in-place, then remove the WAV.

    FLAC is used for channel counts it supports. Wider multichannel captures
    are encoded as WavPack through ffmpeg because FLAC is limited to 8 channels.
    """

    def __init__(self, compression_level: int = 5):
        self.level = compression_level  # 0 (fast) to 8 (best)

    def compress(self, wav_path: str) -> str | None:
        """Compress a WAV. Returns the compressed path, or None on failure."""
        import subprocess
        import wave

        wav = Path(wav_path)
        if wav.suffix.lower() != ".wav":
            return None

        try:
            with wave.open(str(wav), "rb") as wf:
                channels = wf.getnchannels()
        except wave.Error as e:
            logger.warning("Could not inspect WAV channels for %s: %s", wav.name, e)
            return None

        if channels > FLAC_MAX_CHANNELS:
            return self._compress_wavpack_with_ffmpeg(wav, channels)

        flac_path = wav.with_suffix(".flac")

        # Delete existing FLAC if somehow present
        if flac_path.exists():
            flac_path.unlink()

        cmd = [
            "flac",
            f"--compression-level-{self.level}",
            "--delete-input-file",   # removes the WAV on success
            "--silent",
            str(wav),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            if result.returncode == 0 and flac_path.exists():
                logger.info("Compressed: %s → %s (%.1f MB → %.1f MB)",
                            wav.name,
                            flac_path.name,
                            wav.stat().st_size / (1024 * 1024),
                            flac_path.stat().st_size / (1024 * 1024))
                return str(flac_path)
            stderr = result.stderr.decode(errors="replace")
            logger.warning("FLAC compression failed for %s: %s",
                           wav.name, stderr[:200])
            return self._compress_flac_with_ffmpeg(wav, flac_path)
        except FileNotFoundError:
            logger.warning("flac not found; trying ffmpeg fallback")
            return self._compress_flac_with_ffmpeg(wav, flac_path)
        except Exception as e:
            logger.error("FLAC exception for %s: %s", wav.name, e)
            return None

    def _compress_flac_with_ffmpeg(self, wav: Path, flac_path: Path) -> str | None:
        import subprocess

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav),
            "-c:a",
            "flac",
            str(flac_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
        except FileNotFoundError:
            logger.error("ffmpeg not found — install ffmpeg or flac")
            return None
        except Exception as e:
            logger.error("FFmpeg FLAC exception for %s: %s", wav.name, e)
            return None

        if result.returncode == 0 and flac_path.exists():
            try:
                wav.unlink()
            except OSError as e:
                logger.warning("Could not delete source WAV %s: %s", wav.name, e)
            logger.info(
                "Compressed with ffmpeg: %s → %s",
                wav.name,
                flac_path.name,
            )
            return str(flac_path)

        stderr = result.stderr.decode(errors="replace")
        logger.warning("FFmpeg FLAC compression failed for %s: %s",
                       wav.name, stderr[:200])
        return None

    def _compress_wavpack_with_ffmpeg(self, wav: Path, channels: int) -> str | None:
        import subprocess

        wv_path = wav.with_suffix(".wv")
        if wv_path.exists():
            wv_path.unlink()

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav),
            "-c:a",
            "wavpack",
            str(wv_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
        except FileNotFoundError:
            logger.error("ffmpeg not found — install ffmpeg for WavPack compression")
            return None
        except Exception as e:
            logger.error("FFmpeg WavPack exception for %s: %s", wav.name, e)
            return None

        if result.returncode == 0 and wv_path.exists():
            try:
                wav.unlink()
            except OSError as e:
                logger.warning("Could not delete source WAV %s: %s", wav.name, e)
            logger.info(
                "Compressed with WavPack: %s → %s (%d channels)",
                wav.name,
                wv_path.name,
                channels,
            )
            return str(wv_path)

        stderr = result.stderr.decode(errors="replace")
        logger.warning("FFmpeg WavPack compression failed for %s: %s",
                       wav.name, stderr[:200])
        return None


class AlarmSnapshotter:
    """Saves pre/post alarm audio snapshots to the alerts directory."""

    def __init__(self, alerts_dir: str, sample_rate: int = 16000):
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

        alert_id = str(detection.get("alert_id") or ts)
        safe_alert_id = "".join(
            c if c.isalnum() or c in {"-", "_"} else "_" for c in alert_id
        )
        event_dir = self.alerts_dir / f"yes_{safe_alert_id}"
        event_dir.mkdir(parents=True, exist_ok=True)

        try:
            channel = int(detection.get("channel_index", 0) or 0)
            channel_name = str(detection.get("channel_name") or f"ch{channel}")
            safe_channel_name = "".join(
                c if c.isalnum() or c in {"-", "_"} else "_" for c in channel_name
            )
            pre_samples = self._get_channel_samples(
                ring_buffer, 60, channel=channel
            )
            pre_all_samples = self._get_all_channel_samples(ring_buffer, 60)
            pre_path = event_dir / f"pre_60s_{safe_channel_name}.wav"
            self._samples_to_wav(pre_samples, str(pre_path))
            pre_size = pre_path.stat().st_size
            pre_all_path = event_dir / "pre_60s_all_channels.wav"
            self._samples_to_wav(pre_all_samples, str(pre_all_path))
            pre_all_size = pre_all_path.stat().st_size

            post_path = event_dir / f"post_60s_{safe_channel_name}.wav"
            self._stop_event.wait(60)
            post_samples = self._get_channel_samples(
                ring_buffer, 60, channel=channel
            )
            post_all_samples = self._get_all_channel_samples(ring_buffer, 60)
            self._samples_to_wav(post_samples, str(post_path))
            post_size = post_path.stat().st_size
            post_all_path = event_dir / "post_60s_all_channels.wav"
            self._samples_to_wav(post_all_samples, str(post_all_path))
            post_all_size = post_all_path.stat().st_size
            combined = np.concatenate([pre_samples, post_samples])
            combined_path = event_dir / f"full_120s_{safe_channel_name}.wav"
            self._samples_to_wav(combined, str(combined_path))
            combined_size = combined_path.stat().st_size
            combined_all = np.concatenate([pre_all_samples, post_all_samples], axis=0)
            combined_all_path = event_dir / "full_120s_all_channels.wav"
            self._samples_to_wav(combined_all, str(combined_all_path))
            combined_all_size = combined_all_path.stat().st_size

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
                "channel_index": channel,
                "channel_name": channel_name,
                "firing_channel_index": detection.get(
                    "firing_channel_index",
                    channel,
                ),
                "firing_channel_name": detection.get(
                    "firing_channel_name",
                    channel_name,
                ),
                "channel_count": self._channel_count(pre_all_samples),
                "all_channel_results": detection.get("all_channel_results"),
                "color_trace": detection.get("color_trace"),
                "files": {
                    "pre_60s": str(pre_path),
                    "post_60s": str(post_path),
                    "full_120s": str(combined_path),
                    "pre_60s_all_channels": str(pre_all_path),
                    "post_60s_all_channels": str(post_all_path),
                    "full_120s_all_channels": str(combined_all_path),
                },
                "sizes_bytes": {
                    "pre_60s": pre_size,
                    "post_60s": post_size,
                    "full_120s": combined_size,
                    "pre_60s_all_channels": pre_all_size,
                    "post_60s_all_channels": post_all_size,
                    "full_120s_all_channels": combined_all_size,
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

    def _get_all_channel_samples(self, ring_buffer, n_seconds: float) -> np.ndarray:
        try:
            samples = ring_buffer.get_last_n_seconds(
                n_seconds,
                self.sample_rate,
                channel=None,
            )
        except TypeError:
            samples = ring_buffer.get_last_n_seconds(n_seconds, self.sample_rate)
        return np.asarray(samples, dtype=np.float32)

    def _get_channel_samples(
        self, ring_buffer, n_seconds: float, *, channel: int
    ) -> np.ndarray:
        try:
            samples = ring_buffer.get_last_n_seconds(
                n_seconds,
                self.sample_rate,
                channel=channel,
            )
        except TypeError:
            samples = ring_buffer.get_last_n_seconds(n_seconds, self.sample_rate)
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 2:
            channel = min(max(0, int(channel)), samples.shape[1] - 1)
            samples = samples[:, channel]
        return samples

    @staticmethod
    def _channel_count(samples: np.ndarray) -> int:
        samples = np.asarray(samples)
        if samples.ndim == 2:
            return int(samples.shape[1])
        return 1

    def _samples_to_wav(self, samples: np.ndarray, filepath: str):
        import wave

        pcm = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
        if pcm.ndim == 1:
            channels = 1
        elif pcm.ndim == 2:
            channels = pcm.shape[1]
        else:
            raise ValueError(f"Expected 1D or 2D samples, got {pcm.shape}")
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(channels)
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
        allowed = {
            "no_drone",
            "drone_red",
            "drone_blue",
            "unclear",
            *FIELD_TAGS.values(),
        }
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

    def tag_latest(self, tag: str) -> dict | None:
        """Persist a field-button assessment onto the most recent alert."""
        normalized_tag = str(tag).strip().lower()
        if normalized_tag not in FIELD_TAGS:
            raise ValueError(f"Unsupported field tag: {tag}")

        with self._lock:
            entries = self._read_all_unlocked()
            if not entries:
                return None
            updated = entries[-1]
            updated["field_tag_color"] = normalized_tag
            updated["field_tag"] = FIELD_TAGS[normalized_tag]
            updated["field_tagged_at"] = time.time()
            updated["operator_label"] = FIELD_TAGS[normalized_tag]
            updated["operator_labeled_at"] = updated["field_tagged_at"]
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


# ---------------------------------------------------------------------------
# Storage Manager
# ---------------------------------------------------------------------------

class StorageManager:
    """Enforces a max-storage cap on the recordings directory.

    Runs a background thread that periodically:
      - Compresses WAVs older than N seconds to FLAC
      - Deletes oldest files when over budget

    The alerts directory (/data/alerts) is managed separately with its
    own capacity cap and is NOT subject to main-recording eviction.
    """

    def __init__(self, config: dict):
        storage_cfg = config.get("storage", {})
        self.data_dir = Path(storage_cfg.get("data_dir", "/data/recordings"))
        self.max_size_bytes = storage_cfg.get("max_size_gb", 32) * (1024 ** 3)
        self.min_free_bytes = storage_cfg.get("min_free_gb", 1) * (1024 ** 3)
        self.cleanup_watermark_bytes = storage_cfg.get("cleanup_watermark_gb", 5) * (1024 ** 3)
        self.compress_enabled = storage_cfg.get("compress", True)
        self.compress_delay = 60

        # Alerts directory — separate cap, not subject to main eviction
        self.alerts_dir = Path(storage_cfg.get("alerts_dir", "/data/alerts"))
        self.max_alerts_bytes = storage_cfg.get("max_alerts_gb", 2) * (1024 ** 3)

        self.compressor = FlacCompressor()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

    def _alerts_size_bytes(self) -> int:
        """Total size of all snapshots in alerts directory."""
        total = 0
        for f in self.alerts_dir.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    def _evict_oldest_alerts(self):
        """Delete oldest alert snapshots if over the alerts cap.

        Evicts oldest event directories first (by directory mtime).
        """
        over = self._alerts_size_bytes() > self.max_alerts_bytes
        if not over:
            return

        target = int(self.max_alerts_bytes * 0.8)  # evict down to 80%
        logger.info("Alerts over budget (%.1f / %.1f GB) — evicting oldest",
                     self._alerts_size_bytes() / (1024 ** 3),
                     self.max_alerts_bytes / (1024 ** 3))

        event_dirs = sorted(
            [d for d in self.alerts_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )
        for d in event_dirs:
            if self._alerts_size_bytes() <= target:
                break
            try:
                sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(str(d))
                logger.info("Evicted alert: %s (%.1f MB)", d.name, sz / (1024 * 1024))
            except OSError as e:
                logger.warning("Could not evict alert %s: %s", d.name, e)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Storage manager started — max %.1f GB, compress=%s",
                     self.max_size_bytes / (1024 ** 3), self.compress_enabled)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self):
        """Heartbeat loop: check disk, compress, evict."""
        while not self._stop_event.is_set():
            try:
                self._maintenance_cycle()
            except Exception as e:
                logger.error("Maintenance error: %s", e)
            self._stop_event.wait(30)  # Check every 30 seconds

    def _maintenance_cycle(self):
        """One cycle: compress old WAVs, then evict if over budget."""
        if self.compress_enabled:
            self._compress_old_files()

        # Evict old alerts if over their separate cap
        self._evict_oldest_alerts()

        # Check both: total audio size AND free space
        over_budget = self._is_over_budget()
        low_disk = self._is_low_on_disk()
        if over_budget or low_disk:
            target = self.max_size_bytes - self.cleanup_watermark_bytes
            self._evict_oldest(target=target)
            logger.info("Cleanup complete: %.2f GB used, %.2f GB free",
                         self._total_size_bytes() / (1024 ** 3),
                         self._free_bytes() / (1024 ** 3))
        else:
            # Log periodic status
            logger.debug("Storage OK: %.2f GB / %.1f GB, %.2f GB free",
                          self._total_size_bytes() / (1024 ** 3),
                          self.max_size_bytes / (1024 ** 3),
                          self._free_bytes() / (1024 ** 3))

    def _compress_old_files(self):
        """Compress WAVs older than compress_delay seconds."""
        now = time.time()
        for wav in sorted(self.data_dir.glob("*.wav")):
            if self._stop_event.is_set():
                return
            age = now - wav.stat().st_mtime
            if age > self.compress_delay:
                self.compressor.compress(str(wav))

    def _evict_oldest(self, target: int):
        """Delete oldest files until total size ≤ target.

        Deletes compressed files first (they're archived originals), then
        any remaining WAVs. Always deletes oldest by mtime.
        """
        if target <= 0:
            target = self.max_size_bytes // 2  # Safety floor

        # Gather files oldest-first
        all_files = []
        for ext in ("*.flac", "*.wv", "*.wav"):
            all_files.extend(self.data_dir.glob(ext))
        all_files.sort(key=lambda p: p.stat().st_mtime)

        for f in all_files:
            if self._stop_event.is_set():
                return
            if self._total_size_bytes() <= target:
                break
            try:
                sz = f.stat().st_size
                f.unlink()
                logger.info("Evicted: %s (%.1f MB)", f.name, sz / (1024 * 1024))
            except OSError as e:
                logger.warning("Could not delete %s: %s", f.name, e)

    def _total_size_bytes(self) -> int:
        """Sum of all tracked audio files in data_dir."""
        total = 0
        for ext in ("*.wav", "*.flac", "*.wv"):
            for f in self.data_dir.glob(ext):
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    def _free_bytes(self) -> int:
        """Free space on the filesystem containing data_dir."""
        try:
            _, _, free = shutil.disk_usage(str(self.data_dir))
            return free
        except Exception:
            return 0

    def _is_over_budget(self) -> bool:
        return self._total_size_bytes() > self.max_size_bytes

    def _is_low_on_disk(self) -> bool:
        return self._free_bytes() < self.min_free_bytes

    @property
    def status(self) -> dict:
        total = self._total_size_bytes()
        free = self._free_bytes()
        alerts_total = self._alerts_size_bytes()
        wav_count = len(list(self.data_dir.glob("*.wav")))
        flac_count = len(list(self.data_dir.glob("*.flac")))
        wv_count = len(list(self.data_dir.glob("*.wv")))
        return {
            "data_dir": str(self.data_dir),
            "max_size_gb": round(self.max_size_bytes / (1024 ** 3), 1),
            "used_gb": round(total / (1024 ** 3), 2),
            "free_gb": round(free / (1024 ** 3), 2),
            "wav_files": wav_count,
            "flac_files": flac_count,
            "wavpack_files": wv_count,
            "compress_enabled": self.compress_enabled,
            "over_budget": total > self.max_size_bytes,
            "low_disk": free < self.min_free_bytes,
            "alerts_dir": str(self.alerts_dir),
            "alerts_used_mb": round(alerts_total / (1024 * 1024), 1),
            "alerts_max_mb": round(self.max_alerts_bytes / (1024 * 1024), 1),
        }
