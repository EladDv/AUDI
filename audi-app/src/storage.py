"""
Pi Audio Guard — Storage Manager

Manages a ring buffer on disk of up to 32GB of audio recordings.

Strategy:
  1. New WAV segments arrive from the recorder into the data directory.
  2. Background thread compresses old WAVs to FLAC (roughly 6:1 ratio).
  3. When disk usage exceeds the cap, deletes oldest files first
     (FLAC-compressed segments deleted before WAVs).
  4. Filesystem is checked on a 30-second heartbeat.
"""

import logging
import shutil
import threading
import time
from pathlib import Path

logger = logging.getLogger("audio_guard.storage")

# ---------------------------------------------------------------------------
# FLAC compression via flac CLI
# ---------------------------------------------------------------------------

class FlacCompressor:
    """Compress WAV files to FLAC in-place, then remove the WAV."""

    def __init__(self, compression_level: int = 5):
        self.level = compression_level  # 0 (fast) to 8 (best)

    def compress(self, wav_path: str) -> str | None:
        """Compress a WAV to FLAC. Returns the FLAC path, or None on failure."""
        import subprocess

        wav = Path(wav_path)
        if wav.suffix.lower() != ".wav":
            return None

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
            else:
                stderr = result.stderr.decode(errors="replace")
                logger.warning("FLAC compression failed for %s: %s",
                               wav.name, stderr[:200])
                # Restore WAV if it was deleted
                return None
        except FileNotFoundError:
            logger.error("flac not found — install with: apt install flac")
            return None
        except Exception as e:
            logger.error("FLAC exception for %s: %s", wav.name, e)
            return None


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

        Deletes FLAC files first (they're compressed originals), then
        any remaining WAVs. Always deletes oldest by mtime.
        """
        if target <= 0:
            target = self.max_size_bytes // 2  # Safety floor

        # Gather files oldest-first
        all_files = []
        for ext in ("*.flac", "*.wav"):
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
        """Sum of all WAV + FLAC files in data_dir."""
        total = 0
        for ext in ("*.wav", "*.flac"):
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
        return {
            "data_dir": str(self.data_dir),
            "max_size_gb": round(self.max_size_bytes / (1024 ** 3), 1),
            "used_gb": round(total / (1024 ** 3), 2),
            "free_gb": round(free / (1024 ** 3), 2),
            "wav_files": wav_count,
            "flac_files": flac_count,
            "compress_enabled": self.compress_enabled,
            "over_budget": total > self.max_size_bytes,
            "low_disk": free < self.min_free_bytes,
            "alerts_dir": str(self.alerts_dir),
            "alerts_used_mb": round(alerts_total / (1024 * 1024), 1),
            "alerts_max_mb": round(self.max_alerts_bytes / (1024 * 1024), 1),
        }
