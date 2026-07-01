"""On-the-fly mixing dataset for drone detection training.

Creates training examples by mixing drone audio into background noise at
controlled SNR levels. Supports optional multi-noise layering and a
configurable augmentation pipeline.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from torch.utils.data import Dataset

from audi.augment import (
    _fit_length,
    _rms,
    distance_lowpass,
    doppler_shift,
    drone_fade,
    gain_jitter,
    highpass,
    lowpass,
    noise_inject,
    peak_limit,
    pitch_shift,
    random_eq,
    reverb,
    time_mask_waveform,
    time_stretch,
)
from audi.config import AugmentationConfig, MixConfig, SNRBin
from audi.training.hearability import scale_to_db


@dataclass
class BatchItem:
    """A single training/validation sample.

    Attributes:
        mix: Mixed waveform tensor of shape ``[T]``.
        label: Binary label tensor (0.0 or 1.0).
        bin_idx: SNR bin index (long tensor), -1 for negatives.
        drone: Drone-only waveform, zero for negatives. Present only when
            ``return_components=True``.
        noise: Background-only waveform. Present only when
            ``return_components=True``.
        snr_db: Actual SNR value in dB, -99.9 for negatives.
    """

    mix: torch.Tensor
    label: torch.Tensor
    bin_idx: torch.Tensor
    drone: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    snr_db: torch.Tensor | None = None

    def to_tuple(
        self, *, return_bin: bool = False, return_components: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """Convert to the tuple format expected by DataLoader."""
        if return_components:
            return (
                self.mix,
                self.label,
                self.bin_idx,
                self.drone
                if self.drone is not None
                else torch.zeros_like(self.mix),
                self.noise if self.noise is not None else self.mix,
                self.snr_db if self.snr_db is not None else torch.tensor(-99.9),
            )
        if return_bin:
            return (self.mix, self.label, self.bin_idx)
        return (self.mix, self.label)


def _resample_if_needed(
    audio: np.ndarray, source_sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    audio = _finite_audio(audio)
    if int(source_sample_rate) == int(target_sample_rate):
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    common = gcd(int(source_sample_rate), int(target_sample_rate))
    up = int(target_sample_rate) // common
    down = int(source_sample_rate) // common
    return _finite_audio(resample_poly(audio.astype(np.float64), up, down))


def _finite_audio(audio: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(audio, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)


def _rms_normalize(audio: np.ndarray) -> np.ndarray:
    audio = _finite_audio(audio)
    rms = _rms(audio)
    if not np.isfinite(rms) or rms <= 1e-8:
        return np.zeros_like(audio, dtype=np.float32)
    return _finite_audio(audio / rms)


def _path_str(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path))


def waveform_config_payload(cfg: MixConfig) -> dict[str, Any]:
    """Return the waveform-side HP payload for precompute compatibility.

    This intentionally excludes model, mel/frontend, optimizer, and training-loop
    HPs. Precomputed waveforms may be reused only when this payload matches.
    """
    aug_payload = asdict(cfg.aug) if cfg.aug is not None else None
    return {
        "noise_path": _path_str(cfg.noise_path),
        "drone_path": _path_str(cfg.drone_path),
        "hard_noise_path": _path_str(cfg.hard_noise_path),
        "hard_noise_prob": cfg.hard_noise_prob,
        "noise2_path": _path_str(cfg.noise2_path),
        "noise2_prob": cfg.noise2_prob,
        "noise2_multi_noise_prob": cfg.noise2_multi_noise_prob,
        "noise2_count": cfg.noise2_count,
        "noise2_max_attenuation_db": cfg.noise2_max_attenuation_db,
        "snr_bins": [asdict(b) for b in cfg.snr_bins],
        "target_length_samples": cfg.target_length_samples,
        "positive_probability": cfg.positive_probability,
        "highpass_hz": cfg.highpass_hz,
        "sample_rate": cfg.sample_rate,
        "aug": aug_payload,
    }


def waveform_config_hash(cfg: MixConfig) -> str:
    payload = waveform_config_payload(cfg)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_precomputed_manifest(path: str | Path, cfg: MixConfig, *, split: str) -> None:
    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing precomputed manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_hash = waveform_config_hash(cfg)
    got_hash = manifest.get("waveform_config_hash")
    if got_hash != expected_hash:
        expected_payload = waveform_config_payload(cfg)
        got_payload = manifest.get("waveform_config", {})
        raise SystemExit(
            "Precomputed dataset waveform HP mismatch for "
            f"{path} split={split}.\n"
            f"expected_hash={expected_hash}\n"
            f"manifest_hash={got_hash}\n"
            f"expected_waveform_config={json.dumps(expected_payload, sort_keys=True)}\n"
            f"manifest_waveform_config={json.dumps(got_payload, sort_keys=True)}"
        )
    got_split = manifest.get("split")
    if got_split != split:
        raise SystemExit(
            f"Precomputed dataset split mismatch for {path}: "
            f"expected {split!r}, manifest has {got_split!r}"
        )


def frontend_config_payload(mel_cfg: Any) -> dict[str, Any]:
    """Return frontend HP payload for precomputed feature compatibility."""
    payload = asdict(mel_cfg) if hasattr(mel_cfg, "__dataclass_fields__") else dict(mel_cfg)
    # JSON has no tuples; normalize recursively through json.
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def frontend_config_hash(mel_cfg: Any) -> str:
    payload = frontend_config_payload(mel_cfg)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_precomputed_feature_manifest(
    path: str | Path,
    cfg: MixConfig,
    mel_cfg: Any,
    *,
    split: str,
) -> None:
    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing precomputed feature manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_wave_hash = waveform_config_hash(cfg)
    expected_frontend_hash = frontend_config_hash(mel_cfg)
    got_wave_hash = manifest.get("waveform_config_hash")
    got_frontend_hash = manifest.get("frontend_config_hash")
    if got_wave_hash != expected_wave_hash or got_frontend_hash != expected_frontend_hash:
        raise SystemExit(
            "Precomputed feature dataset HP mismatch for "
            f"{path} split={split}.\n"
            f"expected_waveform_hash={expected_wave_hash}\n"
            f"manifest_waveform_hash={got_wave_hash}\n"
            f"expected_frontend_hash={expected_frontend_hash}\n"
            f"manifest_frontend_hash={got_frontend_hash}\n"
            f"expected_waveform_config={json.dumps(waveform_config_payload(cfg), sort_keys=True)}\n"
            "manifest_waveform_config="
            f"{json.dumps(manifest.get('waveform_config', {}), sort_keys=True)}\n"
            "expected_frontend_config="
            f"{json.dumps(frontend_config_payload(mel_cfg), sort_keys=True)}\n"
            "manifest_frontend_config="
            f"{json.dumps(manifest.get('frontend_config', {}), sort_keys=True)}"
        )
    got_split = manifest.get("split")
    if got_split != split:
        raise SystemExit(
            f"Precomputed feature split mismatch for {path}: "
            f"expected {split!r}, manifest has {got_split!r}"
        )


class EpochSliceDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Expose one disjoint fixed-size slice of a larger dataset per epoch."""

    def __init__(
        self,
        dataset: Dataset[tuple[torch.Tensor, ...]],
        *,
        samples_per_epoch: int,
    ) -> None:
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        start = self.epoch * self.samples_per_epoch
        remaining = len(self.dataset) - start
        return max(0, min(self.samples_per_epoch, remaining))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        n = len(self)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(idx)
        return self.dataset[self.epoch * self.samples_per_epoch + idx]


class PrecomputedFeatureDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Dataset backed by precomputed normalized frontend feature shards."""

    def __init__(self, path: str | Path, *, return_bin: bool = False) -> None:
        self.path = Path(path)
        self.return_bin = return_bin
        self.shards = sorted(self.path.glob("*.pt"))
        if not self.shards:
            raise SystemExit(f"No precomputed feature shards found in {self.path}")
        first = torch.load(self.shards[0], map_location="cpu", weights_only=False)
        first_n = int(first["spec"].shape[0])
        if len(self.shards) == 1:
            self._lengths = [first_n]
        else:
            last = torch.load(self.shards[-1], map_location="cpu", weights_only=False)
            last_n = int(last["spec"].shape[0])
            self._lengths = [first_n] * (len(self.shards) - 1) + [last_n]
        self.length = sum(self._lengths)
        self._cache_idx: int | None = None
        self._cache: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return self.length

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        base = 0
        for shard_idx, n in enumerate(self._lengths):
            if idx < base + n:
                return shard_idx, idx - base
            base += n
        raise IndexError(idx)

    def _load_shard(self, shard_idx: int) -> dict[str, torch.Tensor]:
        if self._cache_idx != shard_idx:
            self._cache = torch.load(
                self.shards[shard_idx], map_location="cpu", weights_only=False
            )
            self._cache_idx = shard_idx
        assert self._cache is not None
        return self._cache

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        shard_idx, local_idx = self._locate(idx)
        shard = self._load_shard(shard_idx)
        spec = shard["spec"][local_idx].float()
        label = shard["label"][local_idx].float()
        bin_idx = shard["bin_idx"][local_idx].long()
        if self.return_bin:
            return spec, label, bin_idx
        return spec, label


class PrecomputedDetectionDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Dataset backed by precomputed waveform shards.

    Shards are ``*.pt`` files containing tensor batches. This keeps the model
    frontend in the training loop while removing expensive HF audio decode,
    random mixing, and waveform augmentation from every dataloader item.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        return_bin: bool = False,
        return_components: bool = False,
    ) -> None:
        self.path = Path(path)
        self.return_bin = return_bin or return_components
        self.return_components = return_components
        self.shards = sorted(self.path.glob("*.pt"))
        if not self.shards:
            raise SystemExit(f"No precomputed waveform shards found in {self.path}")
        first = torch.load(self.shards[0], map_location="cpu", weights_only=False)
        first_n = int(first["mix"].shape[0])
        if len(self.shards) == 1:
            self._lengths = [first_n]
        else:
            last = torch.load(self.shards[-1], map_location="cpu", weights_only=False)
            last_n = int(last["mix"].shape[0])
            self._lengths = [first_n] * (len(self.shards) - 1) + [last_n]
        self.length = sum(self._lengths)
        self._cache_idx: int | None = None
        self._cache: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return self.length

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        base = 0
        for shard_idx, n in enumerate(self._lengths):
            if idx < base + n:
                return shard_idx, idx - base
            base += n
        raise IndexError(idx)

    def _load_shard(self, shard_idx: int) -> dict[str, torch.Tensor]:
        if self._cache_idx != shard_idx:
            self._cache = torch.load(
                self.shards[shard_idx], map_location="cpu", weights_only=False
            )
            self._cache_idx = shard_idx
        assert self._cache is not None
        return self._cache

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        shard_idx, local_idx = self._locate(idx)
        shard = self._load_shard(shard_idx)
        mix = shard["mix"][local_idx].float()
        label = shard["label"][local_idx].float()
        bin_idx = shard["bin_idx"][local_idx].long()
        drone = shard.get("drone")
        noise = shard.get("noise")
        snr_db = shard.get("snr_db")
        return BatchItem(
            mix=mix,
            label=label,
            bin_idx=bin_idx,
            drone=drone[local_idx].float() if drone is not None else None,
            noise=noise[local_idx].float() if noise is not None else None,
            snr_db=snr_db[local_idx].float() if snr_db is not None else None,
        ).to_tuple(
            return_bin=self.return_bin, return_components=self.return_components
        )


class HFDetectionDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Dataset backed by an already-mixed HF detector DatasetDict split."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str,
        target_length_samples: int,
        sample_rate: int,
        return_bin: bool = False,
        return_components: bool = False,
    ) -> None:
        self.ds = _load_split(Path(path), split)
        if len(self.ds) <= 0:
            raise SystemExit(f"Split {split!r} in {path} is empty")
        self.target_length_samples = int(target_length_samples)
        self.sample_rate = int(sample_rate)
        self.return_bin = return_bin or return_components
        self.return_components = return_components

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        row = self.ds[idx]
        audio = row["audio"]
        wav = np.asarray(audio["array"], dtype=np.float32).reshape(-1)
        source_sr = self._audio_sample_rate(audio)
        wav = _resample_if_needed(wav, source_sr, self.sample_rate)
        wav = _fit_length(wav, self.target_length_samples)
        wav = np.resize(wav, self.target_length_samples).astype(np.float32)
        wav_tensor = torch.as_tensor(_finite_audio(wav), dtype=torch.float32)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        bin_idx = torch.tensor(int(row.get("bin_idx", -1)), dtype=torch.long)
        snr_db = torch.tensor(float(row.get("snr_db", -99.9)), dtype=torch.float32)

        return BatchItem(
            mix=wav_tensor,
            label=label,
            bin_idx=bin_idx,
            drone=torch.zeros_like(wav_tensor),
            noise=wav_tensor,
            snr_db=snr_db,
        ).to_tuple(
            return_bin=self.return_bin,
            return_components=self.return_components,
        )

    @staticmethod
    def _audio_sample_rate(audio: Any) -> int:
        if isinstance(audio, dict) and audio.get("sampling_rate") is not None:
            return int(audio["sampling_rate"])
        return 16000


class MixedDataset(Dataset[tuple[torch.Tensor, ...]]):
    """On-the-fly dataset that mixes drone audio into background noise.

    Positive samples: drone + background at a randomly chosen SNR.
    Negative samples: background only.
    Optional multi-noise: layers 1-3 additional background sources.

    Attributes:
        noise_ds: Background noise HF dataset split.
        drone_ds: Drone audio HF dataset split.
        snr_bins: SNR bins for positive sample mixing.
        target_length_samples: Fixed segment length in samples.
        positive_probability: Fraction of samples containing a drone.
        highpass_hz: Highpass cutoff applied to all sources.
        sample_rate: Audio sample rate in Hz.
        length: Virtual dataset size (default: max(|noise_ds|, |drone_ds|)).
    """

    def __init__(
        self,
        *,
        noise_ds: Any,
        drone_ds: Any,
        snr_bins: list[SNRBin],
        target_length_samples: int,
        positive_probability: float,
        highpass_hz: float = 0.0,
        sample_rate: int = 16000,
        length: int | None = None,
        return_bin: bool = False,
        return_components: bool = False,
        aug: AugmentationConfig | None = None,
        hard_noise_ds: Any = None,
        hard_noise_prob: float = 0.0,
        # Multi-noise training
        noise2_ds: Any = None,
        noise2_prob: float = 0.25,
        noise2_multi_noise_prob: float = 0.5,
        noise2_count: int = 3,
        noise2_max_attenuation_db: float = -40.0,
    ) -> None:
        if len(noise_ds) <= 0 or len(drone_ds) <= 0:
            raise ValueError("datasets must be non-empty")
        if not snr_bins:
            raise ValueError("snr_bins must not be empty")
        self.noise_ds = noise_ds
        self.drone_ds = drone_ds
        self.hard_noise_ds = hard_noise_ds
        self.hard_noise_prob = float(hard_noise_prob)
        self.noise2_ds = noise2_ds
        self.noise2_count = max(1, min(noise2_count, 5))
        self.noise2_max_att = float(noise2_max_attenuation_db)
        self.noise2_prob = float(noise2_prob)
        self.noise2_multi_noise_prob = float(noise2_multi_noise_prob)
        self.snr_bins = list(snr_bins)
        self.target_length_samples = int(target_length_samples)
        self.positive_probability = float(positive_probability)
        self.highpass_hz = float(highpass_hz)
        self.sample_rate = int(sample_rate)
        self.length = (
            int(max(len(noise_ds), len(drone_ds))) if length is None else length
        )

        self.return_bin = return_bin or return_components
        self.return_components = return_components
        self.aug = aug if aug is not None else AugmentationConfig(enable=False)
        self._bin_name_to_idx: dict[str, int] = {
            b.name: i for i, b in enumerate(self.snr_bins)
        }

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        L = self.target_length_samples
        bg = self._load_background(L)
        bg = peak_limit(_rms_normalize(bg))
        label = random.random() >= self.positive_probability
        if not label:
            # ── Negative sample: background only ────────────────
            mix = self.augment_mix(bg)
            mix = peak_limit(_rms_normalize(mix))  # remove energy cue
            snr_db = torch.tensor(-99.9)
            bin_idx = torch.tensor(-1, dtype=torch.long)
            drone = torch.zeros(mix.shape)

        else:
            # ── Positive sample: drone + background ─────────────────
            bin_instance, target_snr = self._pick_snr()
            drone = self._load_drone(L, target_snr)
            scaled_drone = scale_to_db(drone, bg, target_snr)
            mix = bg + scaled_drone
            snr_db = torch.tensor(target_snr, dtype=torch.float32)
            bin_idx = torch.tensor(
                self._bin_name_to_idx[bin_instance.name], dtype=torch.long
            )
        mix = self.augment_mix(mix)

        # ── Post-processing ──────────────────────────────
        mix = peak_limit(_rms_normalize(mix))  # remove energy cue
        bg = _finite_audio(bg)
        drone = _finite_audio(drone)

        return BatchItem(
            mix=torch.as_tensor(mix, dtype=torch.float32),
            label=torch.tensor(float(label)),
            bin_idx=bin_idx,
            drone=torch.as_tensor(drone, dtype=torch.float32),
            noise=torch.as_tensor(bg, dtype=torch.float32),
            snr_db=snr_db,
        ).to_tuple(
            return_bin=self.return_bin, return_components=self.return_components
        )

    def _load_background(self, L: int) -> np.ndarray:
        """Load and preprocess a background noise segment.

        If ``noise2_ds`` is set, layers 0–3 additional noise sources on top.
        """
        base_ds = self.noise_ds
        if (
            self.hard_noise_ds is not None
            and random.random() < self.hard_noise_prob
        ):
            base_ds = self.hard_noise_ds
        bg = self._load_raw_segment(base_ds, L)

        # ── Multi-noise: layer extra backgrounds ────────────────
        if self.noise2_ds is not None and random.random() < self.noise2_prob:
            n_extra = self._pick_extra_count()
            for _ in range(n_extra):
                extra = self._load_raw_segment(self.noise2_ds, L)
                extra = _rms_normalize(extra)
                att_db = random.uniform(self.noise2_max_att, 0.0)
                scale = 10.0 ** (att_db / 20.0)
                bg = bg + extra * scale
            bg = peak_limit(_finite_audio(bg))

        return bg

    def augment_mix(self, mix: np.ndarray) -> np.ndarray:
        """Augment the mix."""
        aug = self.aug
        if aug.enable:
            if random.random() < aug.reverb_prob:
                try:
                    mix = reverb(mix, self.sample_rate, aug.reverb_decay)
                except Exception:
                    pass
            if random.random() < aug.eq_prob:
                try:
                    mix = random_eq(mix, self.sample_rate, aug.eq_gain_db)
                except Exception:
                    pass
            if random.random() < aug.noise_inject_prob:
                try:
                    mix = noise_inject(mix, aug.noise_inject_db)
                except Exception:
                    pass
            if random.random() < aug.time_mask_prob:
                try:
                    mix = time_mask_waveform(
                        mix, aug.time_mask_count, aug.time_mask_max_ratio
                    )
                except Exception:
                    pass
            if random.random() < aug.lowpass_prob:
                try:
                    mix = lowpass(
                        mix, self.sample_rate, aug.lowpass_cutoff_range
                    )
                except Exception:
                    pass
            if random.random() < aug.pitch_prob:
                try:
                    mix = pitch_shift(
                        mix, self.sample_rate, aug.pitch_semitones
                    )
                except Exception:
                    pass
            if random.random() < aug.gain_jitter_db:
                try:
                    mix = gain_jitter(mix, aug.gain_jitter_db)
                except Exception:
                    pass
        return _finite_audio(mix)

    def _load_drone(
        self, L: int, target_snr: float | None = None
    ) -> np.ndarray:
        """Load, preprocess, and optionally augment a drone segment.

        Args:
            L: Target length in samples.
            target_snr: If provided, applies atmospheric absorption lowpass
                        based on SNR (lower SNR = more filtering).
        """
        idx = random.randint(0, len(self.drone_ds) - 1)
        drone = self._load_audio_array(self.drone_ds, idx)
        drone = _fit_length(drone, L)
        if self.highpass_hz > 0.0:
            drone = highpass(
                drone, cutoff_hz=self.highpass_hz, sample_rate=self.sample_rate
            )
        drone = np.resize(drone, L).astype(np.float32)
        drone = _rms_normalize(drone)

        aug = self.aug
        if aug.enable:
            drone = gain_jitter(drone, aug.gain_jitter_db)
            if random.random() < aug.pitch_prob:
                try:
                    drone = pitch_shift(
                        drone, self.sample_rate, aug.pitch_semitones
                    )
                except Exception:
                    pass
            if random.random() < aug.stretch_prob:
                try:
                    drone = time_stretch(drone, aug.time_stretch_range, L)
                except Exception:
                    pass
            if random.random() < aug.shift_prob:
                drone = drone_fade(
                    drone, aug.shift_max_ratio, sample_rate=self.sample_rate
                )
            if random.random() < aug.atmospheric_prob:
                drone = distance_lowpass(
                    drone,
                    target_snr,
                    self.sample_rate,
                    snr_min=aug.atmospheric_snr_min,
                    snr_max=aug.atmospheric_snr_max,
                    cutoff_min=aug.atmospheric_cutoff_min,
                    cutoff_max=aug.atmospheric_cutoff_max,
                )
            if random.random() < aug.doppler_prob:
                drone = doppler_shift(
                    drone,
                    self.sample_rate,
                    max_speed_mps=aug.doppler_max_speed_mps,
                    target_length=L,
                )
        return _finite_audio(drone)

    def _load_raw_segment(self, ds: Any, L: int) -> np.ndarray:
        """
        Load a raw segment from a dataset,
        fit to length, highpass, RMS-normalize
        """
        idx = random.randint(0, len(ds) - 1)
        raw = self._load_audio_array(ds, idx)
        seg = _fit_length(raw, L)
        if self.highpass_hz > 0.0:
            seg = highpass(
                seg, cutoff_hz=self.highpass_hz, sample_rate=self.sample_rate
            )
        seg = np.resize(seg, L).astype(np.float32)
        return _rms_normalize(seg)

    def _load_audio_array(self, ds: Any, idx: int) -> np.ndarray:
        row = ds[idx]
        audio = row["audio"]
        raw = np.asarray(audio["array"], dtype=np.float32).reshape(-1)
        source_sr = self._audio_sample_rate(ds, audio)
        return _resample_if_needed(raw, source_sr, self.sample_rate)

    @staticmethod
    def _audio_sample_rate(ds: Any, audio: Any) -> int:
        if isinstance(audio, dict) and audio.get("sampling_rate") is not None:
            return int(audio["sampling_rate"])
        features = getattr(ds, "features", None)
        if features is not None:
            audio_feature = features.get("audio") if hasattr(features, "get") else None
            sample_rate = getattr(audio_feature, "sampling_rate", None)
            if sample_rate is not None:
                return int(sample_rate)
        return 16000

    def _pick_snr(self) -> tuple[SNRBin, float]:
        """Randomly select an SNR bin and value."""
        probs = [b.probability for b in self.snr_bins]
        bin_obj = random.choices(self.snr_bins, weights=probs, k=1)[0]
        lo, hi = (
            float(min(bin_obj.low_db, bin_obj.high_db)),
            float(max(bin_obj.low_db, bin_obj.high_db)),
        )
        val = float(random.uniform(lo, hi)) if hi > lo else lo
        return bin_obj, val

    def _pick_extra_count(self) -> int:
        """Weighted random according to self.noise2_multi_noise_prob."""
        num_extra = 0
        while (
            random.random() < self.noise2_multi_noise_prob
            and num_extra < self.noise2_count
        ):
            num_extra += 1
        return num_extra


def _load_split(data_path: Path, split: str) -> Dataset:
    """Load a dataset split from Arrow or parquet format."""
    # Try Arrow format first (load_from_disk)
    if (data_path / "dataset_dict.json").exists():
        dd = load_from_disk(str(data_path))
        if split in dd:
            return dd[split]
        raise SystemExit(f"Split {split!r} not found in {data_path}")
    # Try parquet format: {path}/{split}/data.parquet
    pq_file = data_path / split / "data.parquet"
    if pq_file.exists():
        return HFDataset.from_parquet(str(pq_file))
    raise SystemExit(f"No dataset found at {data_path} (neither Arrow nor parquet)")


def make_dataset(
    *,
    cfg: MixConfig,
    split: str,
    return_bin: bool = False,
    return_components: bool = False,
) -> MixedDataset:
    """Create a MixedDataset from a MixConfig and dataset split.

    Args:
        cfg: Mixing configuration.
        split: Dataset split name ("train", "validation", "test").
        return_bin: Whether samples include the SNR bin index.
        return_components: Whether samples include drone/noise/SNR components.

    Returns:
        A configured MixedDataset instance.

    Raises:
        SystemExit: If ``split`` is not found in either dataset.
    """
    noise_ds = _load_split(cfg.noise_path, split)
    drone_dd = load_from_disk(str(cfg.drone_path))
    if split not in drone_dd:
        raise SystemExit(f"Split {split!r} not found in drone dataset")
    drone_ds = drone_dd[split]
    noise2_ds = None
    if cfg.noise2_path:
        noise2_ds = _load_split(cfg.noise2_path, split)

    hard_noise_ds = None
    if cfg.hard_noise_path:
        hard_noise_ds = _load_split(cfg.hard_noise_path, split)

    return MixedDataset(
        noise_ds=noise_ds,
        drone_ds=drone_ds,
        snr_bins=cfg.snr_bins,
        target_length_samples=cfg.target_length_samples,
        positive_probability=cfg.positive_probability,
        highpass_hz=cfg.highpass_hz,
        sample_rate=cfg.sample_rate,
        length=cfg.dataset_length,
        return_bin=return_bin,
        return_components=return_components,
        aug=cfg.aug,
        hard_noise_ds=hard_noise_ds,
        hard_noise_prob=cfg.hard_noise_prob,
        noise2_ds=noise2_ds,
        noise2_prob=cfg.noise2_prob,
        noise2_multi_noise_prob=cfg.noise2_multi_noise_prob,
        noise2_count=cfg.noise2_count,
        noise2_max_attenuation_db=cfg.noise2_max_attenuation_db,
    )
