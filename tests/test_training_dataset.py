import random

import numpy as np

from audi.config import SNRBin
from audi.training.dataset import HFDetectionDataset, MixedDataset, app_window_normalize
from audi.training.hearability import scale_to_db


def _rows(value: float, count: int = 2, sample_rate: int | None = None):
    audio = {"array": np.full(32, value, dtype=np.float32)}
    if sample_rate is not None:
        audio["sampling_rate"] = sample_rate
    return [
        {"audio": dict(audio)}
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


def test_load_raw_segment_resamples_to_target_sample_rate():
    ds = MixedDataset(
        noise_ds=_rows(1.0, sample_rate=16000),
        hard_noise_ds=None,
        drone_ds=_rows(0.5, sample_rate=16000),
        noise2_ds=None,
        snr_bins=[SNRBin("test", -10.0, -10.0, 1.0)],
        target_length_samples=16,
        positive_probability=0.0,
        highpass_hz=0.0,
        sample_rate=8000,
    )

    segment = ds._load_raw_segment(ds.noise_ds, 16)

    assert segment.shape == (16,)


def test_app_window_normalize_matches_app_rms_peak_limit():
    audio = np.array([0.5, 0.0, -0.5, 0.0], dtype=np.float32)

    normalized = app_window_normalize(audio)

    np.testing.assert_allclose(normalized, [0.98, 0.0, -0.98, 0.0], rtol=1e-6)


def test_hf_detection_dataset_normalizes_waveform_like_app():
    ds = HFDetectionDataset.__new__(HFDetectionDataset)
    ds.ds = [
        {
            "audio": {
                "array": np.full(8, 0.5, dtype=np.float32),
                "sampling_rate": 16000,
            },
            "label": 1.0,
            "bin_idx": 0,
            "snr_db": -10.0,
        }
    ]
    ds.target_length_samples = 8
    ds.sample_rate = 16000
    ds.return_bin = True
    ds.return_components = False

    wav, label, bin_idx = ds[0]

    np.testing.assert_allclose(wav.numpy(), np.full(8, 0.98, dtype=np.float32))
    assert label.item() == 1.0
    assert bin_idx.item() == 0


def test_dataset_sanitizes_non_finite_augmented_waveforms():
    ds = _dataset(noise2_ds=None)
    ds.augment_mix = lambda mix: np.full_like(mix, np.nan)

    wav, _label = ds[0]

    assert np.isfinite(wav.numpy()).all()
    assert np.all(wav.numpy() == 0.0)


def test_scale_to_db_equal_power_no_scaling():
    drone = np.sin(np.linspace(0, 100 * np.pi, 16000, dtype=np.float32))
    bg = drone.copy()

    scaled = scale_to_db(drone, bg, 0.0)

    assert np.abs(scaled - drone).mean() < 0.1


def test_scale_to_db_positive_snr_increases_gain():
    drone = np.ones(16000, dtype=np.float32) * 0.1
    bg = np.ones(16000, dtype=np.float32) * 0.1

    scaled = scale_to_db(drone, bg, 10.0)

    assert scaled.max() > drone.max()
