"""On-the-fly mixing dataset for drone detection training.

Creates training examples by mixing drone audio into background noise at
controlled SNR levels. Supports optional multi-noise layering and a
configurable augmentation pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from datasets import Dataset as HFDataset
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
        bg = bg / _rms(bg)
        bg = peak_limit(bg)
        label = random.random() >= self.positive_probability
        if not label:
            # ── Negative sample: background only ────────────────
            mix = self.augment_mix(bg)
            mix = peak_limit(mix / _rms(mix))  # remove energy cue
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
        mix = peak_limit(mix / _rms(mix))  # remove energy cue

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
        idx = random.randint(0, len(self.noise_ds) - 1)
        bg = np.asarray(self.noise_ds[idx]["audio"]["array"], dtype=np.float32)
        bg = _fit_length(bg, L)
        if self.highpass_hz > 0.0:
            bg = highpass(
                bg, cutoff_hz=self.highpass_hz, sample_rate=self.sample_rate
            )
        bg = np.resize(bg, L).astype(np.float32)
        r = _rms(bg)
        if r > 1e-8:
            bg = bg / r

        # ── Multi-noise: layer extra backgrounds ────────────────
        if self.noise2_ds is not None:
            n_extra = self._pick_extra_count()
            for _ in range(n_extra):
                extra = self._load_raw_segment(self.noise2_ds, L)
                r = _rms(extra)
                if r > 1e-8:
                    extra = extra / r
                att_db = random.uniform(self.noise2_max_att, 0.0)
                scale = 10.0 ** (att_db / 20.0)
                bg = bg + extra * scale
            bg = peak_limit(bg)

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
            if random.random() < aug.stretch_prob:
                try:
                    mix = time_stretch(mix, aug.time_stretch_rate)
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
        return mix

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
        drone = np.asarray(
            self.drone_ds[idx]["audio"]["array"], dtype=np.float32
        )
        drone = _fit_length(drone, L)
        if self.highpass_hz > 0.0:
            drone = highpass(
                drone, cutoff_hz=self.highpass_hz, sample_rate=self.sample_rate
            )
        drone = np.resize(drone, L).astype(np.float32)
        r = _rms(drone)
        if r > 1e-8:
            drone = drone / r

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
        return drone

    def _load_raw_segment(self, ds: Any, L: int) -> np.ndarray:
        """
        Load a raw segment from a dataset,
        fit to length, highpass, RMS-normalize
        """
        idx = random.randint(0, len(ds) - 1)
        raw = np.asarray(ds[idx]["audio"]["array"], dtype=np.float32)
        seg = _fit_length(raw, L)
        if self.highpass_hz > 0.0:
            seg = highpass(
                seg, cutoff_hz=self.highpass_hz, sample_rate=self.sample_rate
            )
        seg = np.resize(seg, L).astype(np.float32)
        r = _rms(seg)
        if r > 1e-8:
            seg = seg / r
        return seg

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
    sr = int(noise_ds.features["audio"].sampling_rate)

    noise2_ds = None
    if cfg.noise2_path:
        noise2_ds = _load_split(cfg.noise2_path, split)

    return MixedDataset(
        noise_ds=noise_ds,
        drone_ds=drone_ds,
        snr_bins=cfg.snr_bins,
        target_length_samples=cfg.target_length_samples,
        positive_probability=cfg.positive_probability,
        highpass_hz=cfg.highpass_hz,
        sample_rate=sr,
        length=cfg.dataset_length,
        return_bin=return_bin,
        return_components=return_components,
        aug=cfg.aug,
        noise2_ds=noise2_ds,
        noise2_count=cfg.noise2_count,
        noise2_max_attenuation_db=cfg.noise2_max_attenuation_db,
    )
