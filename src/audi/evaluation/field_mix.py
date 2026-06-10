"""Deterministic field-background mixing validation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from audi.augment import _rms, peak_limit
from audi.config import SNRBin
from audi.training.dataset import _resample_if_needed
from audi.training.hearability import scale_to_db

SOURCE_NAMES = ("field_mix", "field_background", "field_hard_negative")
COLOR_NAMES = ("none", "blue", "red")


@dataclass(frozen=True)
class FieldMixItem:
    """Manifest row for a fixed validation example."""

    source: str
    label: int
    background_idx: int | None
    drone_idx: int | None
    hard_idx: int | None
    color: str
    snr_bin: str
    snr_db: float
    seed: int


class FieldMixDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Fixed field-like detector validation set.

    Positives are blue/red drone clips mixed into field backgrounds at known SNRs.
    Negatives are held-out field backgrounds plus mined hard-negative field clips.
    The generated manifest is deterministic so every checkpoint is scored on the
    exact same examples.
    """

    def __init__(
        self,
        *,
        background_ds: Any,
        drone_ds: Any,
        snr_bins: list[SNRBin],
        target_length_samples: int,
        samples_per_color_bin: int = 32,
        background_negatives: int = 192,
        hard_negative_ds: Any | None = None,
        hard_negatives: int = 192,
        seed: int = 42,
        sample_rate: int = 16000,
    ) -> None:
        if len(background_ds) <= 0:
            raise ValueError("background_ds must be non-empty")
        if len(drone_ds) <= 0:
            raise ValueError("drone_ds must be non-empty")
        if not snr_bins:
            raise ValueError("snr_bins must not be empty")

        self.background_ds = background_ds
        self.drone_ds = drone_ds
        self.hard_negative_ds = hard_negative_ds
        self.snr_bins = list(snr_bins)
        self.target_length_samples = int(target_length_samples)
        self.sample_rate = int(sample_rate)
        self.source_to_idx = {name: i for i, name in enumerate(SOURCE_NAMES)}
        self.color_to_idx = {name: i for i, name in enumerate(COLOR_NAMES)}
        self.bin_to_idx = {b.name: i for i, b in enumerate(self.snr_bins)}

        self._blue_indices = self._indices_for_color("blue", 0)
        self._red_indices = self._indices_for_color("red", 1)
        if not self._blue_indices:
            raise ValueError("drone_ds has no blue samples")
        if not self._red_indices:
            raise ValueError("drone_ds has no red samples")

        self.items = self._build_items(
            samples_per_color_bin=samples_per_color_bin,
            background_negatives=background_negatives,
            hard_negatives=hard_negatives,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        item = self.items[idx]
        rng = np.random.default_rng(item.seed)

        if item.label:
            assert item.background_idx is not None
            assert item.drone_idx is not None
            bg = self._load_audio(self.background_ds, item.background_idx, rng)
            drone = self._load_audio(self.drone_ds, item.drone_idx, rng)
            bg = self._normalize(bg)
            drone = self._normalize(drone)
            mix = bg + scale_to_db(drone, bg, item.snr_db)
        elif item.source == "field_hard_negative":
            assert self.hard_negative_ds is not None
            assert item.hard_idx is not None
            mix = self._load_audio(self.hard_negative_ds, item.hard_idx, rng)
        else:
            assert item.background_idx is not None
            mix = self._load_audio(self.background_ds, item.background_idx, rng)

        mix = self._normalize(peak_limit(mix))
        return (
            torch.as_tensor(mix, dtype=torch.float32),
            torch.tensor(float(item.label), dtype=torch.float32),
            torch.tensor(self.source_to_idx[item.source], dtype=torch.long),
            torch.tensor(self.color_to_idx[item.color], dtype=torch.long),
            torch.tensor(float(item.snr_db), dtype=torch.float32),
            torch.tensor(self.bin_to_idx.get(item.snr_bin, -1), dtype=torch.long),
        )

    def metadata(self) -> list[dict[str, object]]:
        rows = []
        for item in self.items:
            rows.append(
                {
                    "source": item.source,
                    "label": item.label,
                    "color": item.color,
                    "snr_bin": item.snr_bin,
                    "snr_db": item.snr_db,
                    "background_idx": item.background_idx,
                    "drone_idx": item.drone_idx,
                    "hard_idx": item.hard_idx,
                }
            )
        return rows

    def _build_items(
        self,
        *,
        samples_per_color_bin: int,
        background_negatives: int,
        hard_negatives: int,
        seed: int,
    ) -> list[FieldMixItem]:
        rng = np.random.default_rng(seed)
        rows: list[FieldMixItem] = []
        colors = [("blue", self._blue_indices), ("red", self._red_indices)]
        for snr_bin in self.snr_bins:
            lo = min(float(snr_bin.low_db), float(snr_bin.high_db))
            hi = max(float(snr_bin.low_db), float(snr_bin.high_db))
            for color, indices in colors:
                for _ in range(samples_per_color_bin):
                    rows.append(
                        FieldMixItem(
                            source="field_mix",
                            label=1,
                            background_idx=int(rng.integers(len(self.background_ds))),
                            drone_idx=int(indices[int(rng.integers(len(indices)))]),
                            hard_idx=None,
                            color=color,
                            snr_bin=snr_bin.name,
                            snr_db=float(rng.uniform(lo, hi) if hi > lo else lo),
                            seed=int(rng.integers(0, 2**31 - 1)),
                        )
                    )

        for _ in range(background_negatives):
            rows.append(
                FieldMixItem(
                    source="field_background",
                    label=0,
                    background_idx=int(rng.integers(len(self.background_ds))),
                    drone_idx=None,
                    hard_idx=None,
                    color="none",
                    snr_bin="none",
                    snr_db=-99.9,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
            )

        if self.hard_negative_ds is not None and len(self.hard_negative_ds) > 0:
            for _ in range(hard_negatives):
                rows.append(
                    FieldMixItem(
                        source="field_hard_negative",
                        label=0,
                        background_idx=None,
                        drone_idx=None,
                        hard_idx=int(rng.integers(len(self.hard_negative_ds))),
                        color="none",
                        snr_bin="none",
                        snr_db=-99.9,
                        seed=int(rng.integers(0, 2**31 - 1)),
                    )
                )

        return rows

    def _indices_for_color(self, name: str, label_id: int) -> list[int]:
        indices = []
        for idx in range(len(self.drone_ds)):
            row = self.drone_ds[idx]
            if row.get("label_id") == label_id or row.get("label") == name:
                indices.append(idx)
        return indices

    def _load_audio(self, ds: Any, idx: int, rng: np.random.Generator) -> np.ndarray:
        row = ds[idx]
        audio_data = row["audio"]
        audio = np.asarray(audio_data["array"], dtype=np.float32).reshape(-1)
        source_sr = self._audio_sample_rate(ds, audio_data)
        audio = _resample_if_needed(audio, source_sr, self.sample_rate)
        return self._fit_length(audio, self.target_length_samples, rng)

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

    @staticmethod
    def _fit_length(
        audio: np.ndarray, target: int, rng: np.random.Generator
    ) -> np.ndarray:
        if audio.size == 0:
            raise ValueError("empty audio")
        if audio.size < target:
            reps = int(np.ceil(target / audio.size))
            audio = np.tile(audio, reps)
        if audio.size == target:
            return audio.astype(np.float32, copy=False)
        start = int(rng.integers(0, audio.size - target + 1))
        return audio[start : start + target].astype(np.float32, copy=False)

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        rms = _rms(audio)
        if rms <= 1e-8:
            return np.asarray(audio, dtype=np.float32)
        return (np.asarray(audio, dtype=np.float32) / rms).astype(np.float32)
