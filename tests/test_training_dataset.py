import random

import numpy as np

from audi.config import SNRBin
from audi.training.dataset import MixedDataset


def _rows(value: float, count: int = 2):
    return [
        {"audio": {"array": np.full(32, value, dtype=np.float32)}}
        for _ in range(count)
    ]


def _dataset(**kwargs) -> MixedDataset:
    hard_noise_ds = kwargs.pop("hard_noise_ds", _rows(2.0))
    noise2_ds = kwargs.pop("noise2_ds", _rows(3.0))
    return MixedDataset(
        noise_ds=_rows(1.0),
        hard_noise_ds=hard_noise_ds,
        drone_ds=_rows(0.5),
        noise2_ds=noise2_ds,
        snr_bins=[SNRBin("test", -10.0, -10.0, 1.0)],
        target_length_samples=32,
        positive_probability=0.0,
        highpass_hz=0.0,
        sample_rate=16000,
        **kwargs,
    )


def test_noise2_prob_zero_disables_extra_noise(monkeypatch):
    ds = _dataset(noise2_prob=0.0, noise2_multi_noise_prob=1.0, noise2_count=3)
    calls = []
    original = ds._load_raw_segment

    def record(ds_arg, length):
        calls.append(ds_arg)
        return original(ds_arg, length)

    monkeypatch.setattr(ds, "_load_raw_segment", record)

    random.seed(1)
    ds[0]

    assert len(calls) == 1
    assert calls[0] is ds.noise_ds


def test_hard_noise_prob_one_uses_hard_noise_as_base(monkeypatch):
    ds = _dataset(hard_noise_prob=1.0, noise2_ds=None)
    calls = []
    original = ds._load_raw_segment

    def record(ds_arg, length):
        calls.append(ds_arg)
        return original(ds_arg, length)

    monkeypatch.setattr(ds, "_load_raw_segment", record)

    random.seed(1)
    ds[0]

    assert len(calls) == 1
    assert calls[0] is ds.hard_noise_ds
