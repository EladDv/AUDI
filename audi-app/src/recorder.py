"""
AUDI Type A — Continuous Audio Recorder

Captures audio from the Pi's audio input (USB mic, I2S mic, or line-in)
in fixed-duration segments. Maintains an in-memory ring buffer of the
last 120 seconds of raw PCM samples (float32, [-1, 1]) for ML temporal
smoothing and alarm pre/post snapshots.

Segments are written as WAV files to the hot directory for the storage
manager to compress and archive.
"""

import logging
import re
import subprocess
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger("audi.recorder")


def auto_discover_device() -> str:
    """Find the first USB audio capture device via arecord -l.

    Returns the ALSA hardware device string (e.g. 'hw:2,0') or 'default'
    if no USB capture device is found.
    """
    try:
        r = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("arecord not available, using 'default' device")
        return "default"

    logger.info("arecord -l output:\n%s", r.stdout)
    logger.info("arecord stderr:\n%s", r.stderr)

    # Parse lines like: "card 2: CODEC [USB Audio CODEC], device 0: USB Audio [USB Audio]"
    for line in r.stdout.splitlines():
        if "USB" not in line:
            continue
        m = re.match(r"card\s+(\d+):.*device\s+(\d+):", line)
        if m:
            dev = f"hw:{m.group(1)},{m.group(2)}"
            logger.info("Auto-discovered USB audio device: %s", dev)
            return dev

    logger.warning("No USB audio device found via arecord -l, using 'default'")
    return "default"

# ---------------------------------------------------------------------------
# In-memory ring buffer
# ---------------------------------------------------------------------------


class AudioRingBuffer:
    """Circular buffer of recent audio samples (float32, [-1.0, 1.0]).

    Stores up to `max_samples` of mono audio. New data overwrites oldest.
    """

    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self._buffer = np.zeros(max_samples, dtype=np.float32)
        self._write_pos = 0
        self._count = 0
        self._lock = threading.Lock()

    def append(self, samples: np.ndarray):
        """Append float32 samples [-1, 1]. Thread-safe."""
        n = len(samples)
        if n <= 0:
            return
        with self._lock:
            if n >= self.max_samples:
                self._buffer[:] = samples[-self.max_samples :]
                self._write_pos = 0
                self._count = self.max_samples
                return

            end = self._write_pos + n
            if end <= self.max_samples:
                self._buffer[self._write_pos : end] = samples
            else:
                first_chunk = self.max_samples - self._write_pos
                self._buffer[self._write_pos :] = samples[:first_chunk]
                self._buffer[: n - first_chunk] = samples[first_chunk:]

            self._write_pos = end % self.max_samples
            self._count = min(self._count + n, self.max_samples)

    @property
    def total_samples(self) -> int:
        return self._count

    def get_recent(self, num_samples: int) -> np.ndarray:
        """Return the most recent `num_samples`. Ordered oldest→newest."""
        num_samples = int(num_samples)
        with self._lock:
            if num_samples > self._count:
                num_samples = self._count

            if self._count < self.max_samples:
                start = max(0, self._count - num_samples)
                return self._buffer[start : self._count].copy()

            idx = (self._write_pos - num_samples) % self.max_samples
            if idx + num_samples <= self.max_samples:
                return self._buffer[idx : idx + num_samples].copy()
            else:
                first = self._buffer[idx:].copy()
                second = self._buffer[: num_samples - len(first)]
                return np.concatenate([first, second])

    def get_last_n_seconds(
        self, n_seconds: float, sample_rate: int
    ) -> np.ndarray:
        needed = int(n_seconds * sample_rate)
        return self.get_recent(needed)

    def get_samples_at_timestamp(
        self,
        seconds_ago_start: float,
        duration_seconds: float,
        sample_rate: int,
    ) -> np.ndarray:
        """Get a specific window of audio from the past.

        Args:
            seconds_ago_start: How far back to start (e.g., 120 = 2 min ago)
            duration_seconds: How many seconds to retrieve
            sample_rate: Sample rate for calculation

        Returns:
            Float32 array of samples, oldest→newest. May be shorter if
            not enough data in buffer.
        """
        start_offset = int(seconds_ago_start * sample_rate)
        needed = int(duration_seconds * sample_rate)
        # We want samples starting from (write_pos - start_offset) going forward
        # But get_recent gives us the last N samples, so:
        # To get a window starting at seconds_ago_start ago with duration:
        # total = seconds_ago_start + duration seconds ago
        total_offset = start_offset + needed
        recent = self.get_recent(total_offset)
        if len(recent) < needed:
            return recent  # return whatever we have
        return recent[:needed]  # first `needed` samples of the recent window

    def clear(self):
        with self._lock:
            self._buffer.fill(0.0)
            self._write_pos = 0
            self._count = 0


# ---------------------------------------------------------------------------
# ALSA recorder
# ---------------------------------------------------------------------------


class ALSARecorder:
    """Captures audio via arecord, writes WAV files, feeds ring buffer."""

    def __init__(
        self,
        hot_dir: str,
        device: str = "default",
        sample_rate: int = 16000,
        channels: int = 1,
        bit_depth: int = 16,
        segment_duration: int = 300,
        ring_buffer_duration: int = 120,
        on_segment: Callable[[str], None] | None = None,
    ):
        self.hot_dir = Path(hot_dir)
        self.hot_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._device_is_auto = (device == "auto")  # re-discover on failure
        self.sample_rate = sample_rate
        self.channels = channels
        self.bit_depth = bit_depth
        self.segment_duration = segment_duration
        self.on_segment = on_segment

        # In-memory ring buffer — float32 mono, 120s for pre/post alarm snapshots
        ring_samples = ring_buffer_duration * sample_rate * channels
        self.ring_buffer = AudioRingBuffer(ring_samples)

        # Audio health tracking
        self._bytes_captured_total = 0
        self._segment_count = 0
        self._last_audio_timestamp = 0.0

        # Device hotplug recovery
        self._device_retry_min = 2
        self._device_retry_max = 60
        self._device_retry_delay = 0  # 0 = no failure yet

        # Pause mechanism
        self._paused_until = 0.0  # time.time() threshold; 0 = not paused
        self._force_stopped = False  # True when user pressed stop

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            logger.warning("Recorder already running")
            return
        self._force_stopped = False
        self._paused_until = 0.0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(
            "Recorder started — %ds segments, %ds ring buffer",
            self.segment_duration,
            self.ring_buffer.max_samples // self.sample_rate,
        )

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Recorder stopped")

    def pause(self, duration_seconds: int = 300):
        """Pause recording for `duration_seconds`. Resumes automatically."""
        self._paused_until = time.time() + duration_seconds
        logger.info(
            "Recorder paused for %ds (until %.1f)",
            duration_seconds,
            self._paused_until,
        )

    def resume(self):
        """Resume recording immediately."""
        self._paused_until = 0.0
        logger.info("Recorder resumed")

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return self._paused_until > time.time()

    @property
    def is_force_stopped(self) -> bool:
        return self._force_stopped

    def toggle_start_stop(self):
        """Toggle between recording and stopped."""
        if self._force_stopped:
            self._force_stopped = False
            if not self._thread or not self._thread.is_alive():
                self.start()
            logger.info("Recorder: started")
        else:
            self._force_stopped = True
            logger.info("Recorder: stop requested at end of current segment")
        return not self._force_stopped

    @property
    def audio_healthy(self) -> bool:
        """True if audio bytes have arrived in the last 2 seconds."""
        return (time.time() - self._last_audio_timestamp) < 5.0

    def _run(self):
        segment_index = 0
        chunk_bytes = 512  # feed ring buffer every ~10ms at 16kHz/16bit
        while not self._stop_event.is_set():
            # Check force-stop
            if self._force_stopped:
                self._stop_event.wait(1)
                continue

            # Check pause
            if self.is_paused:
                remaining = self._paused_until - time.time()
                logger.debug("Recorder paused — %ds remaining", int(remaining))
                self._stop_event.wait(min(remaining, 5.0))
                continue

            segment_index += 1
            filename = f"seg_{int(time.time())}_{segment_index:06d}.wav"
            filepath = self.hot_dir / filename

            try:
                raw_pcm = self._record_segment_streaming(chunk_bytes)
                self._write_wav(str(filepath), raw_pcm)

                # Health tracking
                self._bytes_captured_total += len(raw_pcm)
                self._segment_count += 1

                logger.info(
                    "Recorded: %s (%.1f MB, ring=%ds audio/%dcap)",
                    filepath,
                    len(raw_pcm) / (1024 * 1024),
                    self.ring_buffer.total_samples // self.sample_rate,
                    self.ring_buffer.max_samples // self.sample_rate,
                )
                # Reset retry on success
                self._device_retry_delay = 0
                if self.on_segment:
                    self.on_segment(str(filepath))
            except Exception as e:
                logger.error("Record failed for %s: %s", filename, e)
                self._handle_device_failure()

    def _record_segment_streaming(self, chunk_bytes: int = 4096) -> bytes:
        """Stream audio from arecord, feeding ring buffer in small chunks.

        Reads from arecord stdout in ``chunk_bytes`` increments, feeding each
        chunk to the ring buffer immediately so RMS/detection are live.
        Accumulates all chunks and returns the full raw PCM at the end.
        """
        import subprocess

        sample_fmt = {
            8: "U8",
            16: "S16_LE",
            24: "S24_LE",
            32: "S32_LE",
        }.get(self.bit_depth, "S16_LE")

        cmd = [
            "arecord",
            "-D",
            self.device,
            "-r",
            str(self.sample_rate),
            "-c",
            str(self.channels),
            "-f",
            sample_fmt,
            "-t",
            "raw",
            "-d",
            str(self.segment_duration),
        ]

        chunks: list[bytes] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error("arecord not found — install alsa-utils")
            raise

        try:
            while True:
                chunk = proc.stdout.read(chunk_bytes)
                if not chunk:
                    break
                chunks.append(chunk)
                self._feed_ring_buffer(chunk)

            proc.wait(timeout=30)
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode(errors="replace")
                logger.warning("arecord stderr: %s", stderr[:300])
                if (
                    "No such device" in stderr
                    or "Device or resource busy" in stderr
                ):
                    time.sleep(2)
                    raise RuntimeError(
                        f"Audio device unavailable: {stderr[:200]}"
                    )
                raise RuntimeError(f"arecord failed: {stderr[:200]}")

        except Exception:
            proc.kill()
            proc.wait()
            raise

        return b"".join(chunks)

    def _feed_ring_buffer(self, raw_pcm: bytes):
        """Convert raw PCM bytes to float32 [-1,1] and append to ring buffer."""
        dtype = {8: np.uint8, 16: np.int16, 24: np.int32, 32: np.int32}[
            self.bit_depth
        ]
        samples = np.frombuffer(raw_pcm, dtype=dtype).astype(np.float32)

        if self.bit_depth == 8:
            samples = (samples - 128.0) / 128.0
        elif self.bit_depth == 16:
            samples /= 32768.0
        elif self.bit_depth == 24:
            samples /= 8388608.0
        else:
            samples /= 2147483648.0

        np.clip(samples, -1.0, 1.0, out=samples)

        if self.channels > 1:
            samples = samples.reshape(-1, self.channels).mean(axis=1)

        self.ring_buffer.append(samples)
        self._last_audio_timestamp = time.time()

    def _handle_device_failure(self):
        """Exponential backoff retry for audio device errors.

        On first failure, retry after 2s. Doubles up to 60s max.
        Resets to 0 on success (see _run).

        When device was set to 'auto', re-discovers the USB device
        before each retry — the card may have re-enumerated.
        """
        if self._device_is_auto:
            new_dev = auto_discover_device()
            if new_dev != self.device:
                logger.info(
                    "Device re-enumerated: %s → %s", self.device, new_dev
                )
                self.device = new_dev

        if self._device_retry_delay == 0:
            self._device_retry_delay = self._device_retry_min
        else:
            self._device_retry_delay = min(
                self._device_retry_delay * 2,
                self._device_retry_max,
            )

        logger.warning(
            "Audio device failure — retrying in %ds "
            "(exponential backoff, max %ds)",
            self._device_retry_delay,
            self._device_retry_max,
        )
        self._stop_event.wait(self._device_retry_delay)

    def _write_wav(self, filepath: str, raw_data: bytes):
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.bit_depth // 8)
            wf.setframerate(self.sample_rate)
            wf.writeframes(raw_data)

    def capture_preview_seconds(self, n_seconds: float) -> bytes:
        """Get the last N seconds from the ring buffer as WAV bytes.

        Returns raw WAV bytes suitable for streaming to the web UI
        or saving to disk.
        """
        samples = self.ring_buffer.get_last_n_seconds(
            n_seconds, self.sample_rate
        )
        if len(samples) == 0:
            return b""

        # Convert float32 [-1,1] back to int16 PCM
        pcm = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)

        import io

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def get_rms_level(self, window_seconds: float = 0.01) -> float:
        """Return the current RMS audio level (0.0–1.0) for VU meter."""
        window_seconds = max(0.01, window_seconds)
        samples = self.ring_buffer.get_last_n_seconds(
            window_seconds, self.sample_rate
        )
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples**2)))


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class RecorderManager:
    """Manages start/stop/status of the audio recorder."""

    def __init__(self, config: dict):
        audio_cfg = config.get("audio", {})
        buf_sec = audio_cfg.get("ring_buffer_seconds", 120)
        device = audio_cfg.get("device", "default")
        if device == "auto":
            device = auto_discover_device()
        self.recorder = ALSARecorder(
            hot_dir=config["storage"]["data_dir"],
            device=device,
            sample_rate=audio_cfg.get("sample_rate", 16000),
            channels=audio_cfg.get("channels", 1),
            bit_depth=audio_cfg.get("bit_depth", 16),
            segment_duration=audio_cfg.get("segment_duration", 300),
            ring_buffer_duration=buf_sec,
        )
        # Wire hotplug retry settings from config
        self.recorder._device_retry_min = audio_cfg.get("device_retry_min", 2)
        self.recorder._device_retry_max = audio_cfg.get("device_retry_max", 60)

    @property
    def ring_buffer(self) -> AudioRingBuffer:
        return self.recorder.ring_buffer

    @property
    def status(self) -> dict:
        r = self.recorder
        return {
            "running": r.is_recording,
            "paused": r.is_paused,
            "stopped": r.is_force_stopped,
            "device": r.device,
            "sample_rate": r.sample_rate,
            "segment_duration": r.segment_duration,
            "ring_buffer_seconds": r.ring_buffer.total_samples // r.sample_rate,
            "ring_buffer_capacity": r.ring_buffer.max_samples // r.sample_rate,
            "bytes_captured": r._bytes_captured_total,
            "segments_recorded": r._segment_count,
            "audio_healthy": r.audio_healthy,
            "rms_level": r.get_rms_level(),
        }
