import importlib.util
from pathlib import Path

import numpy as np
import pytest

from audi.config import SNRBin
from audi.training import pyroom_dataset as pyroom_mod
from audi.training.pyroom_dataset import (
    NoiseSection,
    PyRoomDataset,
    PyRoomSimulationConfig,
    beam_alignment,
    deglitch_multichannel,
    direction_unit_vector,
    planar_array_positions,
    pyroom_config_payload,
    simulate_drone_free_space,
    soft_target_from_alignment,
    split_wavpack_files,
)


def _load_app_uma16_positions() -> np.ndarray:
    path = Path("audi-app/src/doa_estimator.py")
    spec = importlib.util.spec_from_file_location("doa_estimator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UMA16_MIC_POSITIONS_M


def test_mvdr_uses_deployed_uma16_geometry_order():
    assert np.allclose(planar_array_positions(), _load_app_uma16_positions())


def test_mvdr_rejects_non_uma16_geometry():
    with pytest.raises(ValueError, match="UMA16"):
        PyRoomSimulationConfig(spacing_m=0.05)


def test_direction_unit_vector_matches_doa_compass_convention():
    assert np.allclose(direction_unit_vector(0.0, 0.0), [0.0, 1.0, 0.0])
    assert np.allclose(direction_unit_vector(90.0, 0.0), [1.0, 0.0, 0.0])
    assert np.allclose(direction_unit_vector(0.0, 90.0), [0.0, 0.0, 1.0])


def test_pyroom_manifest_records_uma16_positions():
    payload = pyroom_config_payload(PyRoomSimulationConfig(soft_target_by_beam_alignment=True))

    assert payload["array_geometry"] == "uma16_app_channel_order"
    assert np.allclose(payload["mic_positions_m"], _load_app_uma16_positions())
    assert payload["simulator"] == "pyroomacoustics.AnechoicRoom"
    assert payload["air_absorption"] is True
    assert payload["soft_target_by_beam_alignment"] is True


def test_soft_target_scales_by_beam_alignment_with_floor():
    assert beam_alignment(0.0, 45.0, 0.0, 45.0) == pytest.approx(1.0)
    assert soft_target_from_alignment(0.0, 45.0, 0.0, 45.0) == pytest.approx(1.0)
    assert soft_target_from_alignment(0.0, 0.0, 90.0, 0.0) == pytest.approx(0.5)
    assert soft_target_from_alignment(0.0, 0.0, 60.0, 0.0) == pytest.approx(0.5)
    assert soft_target_from_alignment(0.0, 0.0, 45.0, 0.0) == pytest.approx(
        np.sqrt(0.5)
    )


def test_split_wavpack_files_is_stable(tmp_path):
    for idx in range(10):
        (tmp_path / f"{idx:02d}.wv").touch()

    first = split_wavpack_files(tmp_path, split="validation", seed=7)
    second = split_wavpack_files(tmp_path, split="validation", seed=7)
    test = split_wavpack_files(tmp_path, split="test", seed=7)
    train = split_wavpack_files(tmp_path, split="train", seed=7)

    assert first == second
    assert set(first).isdisjoint(train)
    assert set(first).isdisjoint(test)
    assert set(test).isdisjoint(train)


def test_free_space_render_keeps_far_drone_in_clip():
    cfg = PyRoomSimulationConfig(min_distance_m=450.0, max_distance_m=450.0)
    target_length = 1024
    drone = np.zeros(target_length, dtype=np.float32)
    drone[0] = 1.0

    rendered = simulate_drone_free_space(
        drone,
        positions_m=planar_array_positions(),
        azimuth_deg=0.0,
        elevation_deg=45.0,
        distance_m=450.0,
        sample_rate=16000,
        cfg=cfg,
        target_length=target_length,
    )

    assert rendered.shape == (16, target_length)
    assert np.max(np.abs(rendered)) > 0.0


def test_mvdr_uses_separate_noise_estimation_and_output_sections(monkeypatch):
    ds = PyRoomDataset.__new__(PyRoomDataset)
    ds.target_length_samples = 8
    ds.positive_probability = 0.0
    ds.sample_rate = 16000
    ds.cfg = PyRoomSimulationConfig(stft_n_fft=8, hop_length=4, sensor_noise_db=None)
    ds.positions_m = planar_array_positions()
    ds.return_components = True

    mvdr_noise = np.ones((16, 8), dtype=np.float32)
    output_noise = np.full((16, 8), 2.0, dtype=np.float32)
    info = pyroom_mod.WavpackInfo(Path("noise.wv"), 16000, 16, 10.0)
    calls = []

    def load_noise_section():
        calls.append(len(calls))
        channels = mvdr_noise if len(calls) == 1 else output_noise
        return NoiseSection(info=info, channels=channels, start_seconds=float(len(calls)))

    captured = {}

    def estimate(noise_channels, **kwargs):
        captured["estimated_from"] = noise_channels.copy()
        return np.ones((5, 16), dtype=np.complex128)

    def apply(channels, weights, **kwargs):
        del weights, kwargs
        if np.allclose(channels, output_noise):
            captured["applied_to_output_noise"] = True
        return channels[0]

    monkeypatch.setattr(ds, "_load_noise_section", load_noise_section)
    monkeypatch.setattr(pyroom_mod, "estimate_mvdr_weights", estimate)
    monkeypatch.setattr(pyroom_mod, "apply_mvdr_weights", apply)

    item = ds[0]

    assert calls == [0, 1]
    assert np.allclose(captured["estimated_from"], mvdr_noise)
    assert captured["applied_to_output_noise"] is True
    assert item[0].shape == (8,)


def test_mvdr_cache_uses_output_file_without_estimation_section(monkeypatch):
    ds = PyRoomDataset.__new__(PyRoomDataset)
    ds.target_length_samples = 8
    ds.positive_probability = 0.0
    ds.sample_rate = 16000
    ds.cfg = PyRoomSimulationConfig(
        stft_n_fft=8,
        hop_length=4,
        mvdr_cache_dir="cache",
        sensor_noise_db=None,
    )
    ds.positions_m = planar_array_positions()
    ds.return_components = True

    output_noise = np.full((16, 8), 2.0, dtype=np.float32)
    info = pyroom_mod.WavpackInfo(Path("noise.wv"), 16000, 16, 10.0)
    calls = []
    captured = {}

    def load_noise_section():
        calls.append(True)
        return NoiseSection(info=info, channels=output_noise, start_seconds=0.0)

    class Cache:
        def weights_for_file(self, cache_info, *, look_direction):
            captured["cache_info"] = cache_info
            captured["look_direction_norm"] = np.linalg.norm(look_direction)
            return np.ones((5, 16), dtype=np.complex128)

    def fail_estimate(*args, **kwargs):
        raise AssertionError("cached MVDR path must not estimate per-sample weights")

    def apply(channels, weights, **kwargs):
        del weights, kwargs
        return channels[0]

    ds.mvdr_cache = Cache()
    monkeypatch.setattr(ds, "_load_noise_section", load_noise_section)
    monkeypatch.setattr(pyroom_mod, "estimate_mvdr_weights", fail_estimate)
    monkeypatch.setattr(pyroom_mod, "apply_mvdr_weights", apply)

    item = ds[0]

    assert len(calls) == 1
    assert captured["cache_info"] == info
    assert captured["look_direction_norm"] == pytest.approx(1.0)
    assert item[0].shape == (8,)


def test_mean_beamformer_uses_only_output_section(monkeypatch):
    ds = PyRoomDataset.__new__(PyRoomDataset)
    ds.target_length_samples = 8
    ds.positive_probability = 0.0
    ds.sample_rate = 16000
    ds.cfg = PyRoomSimulationConfig(
        stft_n_fft=8,
        hop_length=4,
        beamformer="mean",
        sensor_noise_db=None,
    )
    ds.positions_m = planar_array_positions()
    ds.return_components = True

    output_noise = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) + 1.0
    info = pyroom_mod.WavpackInfo(Path("noise.wv"), 16000, 16, 10.0)
    calls = []

    def load_noise_section():
        calls.append(True)
        return NoiseSection(info=info, channels=output_noise, start_seconds=0.0)

    def fail_estimate(*args, **kwargs):
        raise AssertionError("mean beamformer must not estimate MVDR weights")

    monkeypatch.setattr(ds, "_load_noise_section", load_noise_section)
    monkeypatch.setattr(pyroom_mod, "estimate_mvdr_weights", fail_estimate)

    item = ds[0]

    assert len(calls) == 1
    assert item[0].shape == (8,)


def test_spatial_bg_is_rendered_before_measured_snr(monkeypatch):
    ds = PyRoomDataset.__new__(PyRoomDataset)
    ds.target_length_samples = 8
    ds.positive_probability = 1.0
    ds.sample_rate = 16000
    ds.cfg = PyRoomSimulationConfig(
        beamformer="mean",
        sensor_noise_db=None,
        spatial_bg_probability=1.0,
        spatial_bg_multi_probability=0.0,
        spatial_bg_count=1,
        spatial_bg_max_attenuation_db=0.0,
        min_distance_m=20.0,
        max_distance_m=20.0,
    )
    ds.positions_m = planar_array_positions()
    ds.return_components = True
    ds.snr_bins = [SNRBin("measured", 0.0, 10.0, 1.0)]
    ds.spatial_bg_ds = [{"audio": {"array": np.ones(8, dtype=np.float32)}}]

    info = pyroom_mod.WavpackInfo(Path("noise.wv"), 16000, 16, 10.0)
    base_bg = np.ones((16, 8), dtype=np.float32)
    drone_marker = np.full(8, 9.0, dtype=np.float32)
    rendered = []

    def load_noise_section():
        return NoiseSection(info=info, channels=base_bg.copy(), start_seconds=0.0)

    def load_drone():
        return drone_marker

    def load_spatial_bg_clip():
        return np.full(8, 3.0, dtype=np.float32)

    def simulate(audio, **kwargs):
        del kwargs
        if np.allclose(audio, drone_marker):
            rendered.append("drone")
            return np.full((16, 8), 4.0, dtype=np.float32)
        rendered.append("bg")
        return np.ones((16, 8), dtype=np.float32)

    monkeypatch.setattr(ds, "_load_noise_section", load_noise_section)
    monkeypatch.setattr(ds, "_load_drone", load_drone)
    monkeypatch.setattr(ds, "_load_spatial_bg_clip", load_spatial_bg_clip)
    monkeypatch.setattr(pyroom_mod, "simulate_drone_free_space", simulate)

    item = ds[0]

    assert rendered == ["bg", "drone"]
    assert item[2].item() == 0
    assert item[5].item() == pytest.approx(20.0 * np.log10(4.0 / 2.0))
    assert np.allclose(item[4].numpy(), 2.0)
    assert np.allclose(item[3].numpy(), 4.0)


def test_measured_snr_below_lowest_bin_folds_to_lowest_bin():
    ds = PyRoomDataset.__new__(PyRoomDataset)
    ds.snr_bins = [
        SNRBin("easy", -5.0, 0.0, 1.0),
        SNRBin("hard", -15.0, -10.0, 1.0),
        SNRBin("floor", -20.0, -20.0, 1.0),
    ]

    assert ds._bin_idx_for_snr(-25.0) == 2
    assert ds._bin_idx_for_snr(-20.0) == 2
    assert ds._bin_idx_for_snr(3.0) == 0


def test_non_mvdr_rejects_beam_calibration_options():
    with pytest.raises(ValueError, match="beam calibration"):
        PyRoomSimulationConfig(beamformer="mean", soft_target_by_beam_alignment=True)


def test_deglitch_multichannel_interpolates_large_channel_jump():
    channels = np.zeros((16, 64), dtype=np.float32)
    channels[7, 30] = 0.0016
    channels[7, 31] = -0.0081
    channels[7, 32] = -0.0039

    repaired = deglitch_multichannel(channels, threshold=0.001, window_samples=4)

    assert np.max(np.abs(np.diff(repaired[7, 26:36]))) < 0.004
    assert np.allclose(repaired[6], channels[6])


def test_deglitch_respects_local_loudness():
    channels = np.ones((16, 64), dtype=np.float32) * 0.02
    channels[7, 30] = 0.0216
    channels[7, 31] = 0.0119
    channels[7, 32] = 0.0161

    repaired = deglitch_multichannel(channels, threshold=0.001, window_samples=4)

    assert np.allclose(repaired[7], channels[7])
