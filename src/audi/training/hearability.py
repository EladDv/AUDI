from __future__ import annotations

import numpy as np

_EPS = 1e-12
_SR = 16000
_N_FFT = 1024
_HOP = 512
_ACTIVE_THRESH = 0.01


def _build_filters() -> np.ndarray:
    f_min, f_max, n_bands = 50.0, 8000.0, 32

    def erb(f: float) -> float:
        return 21.4 * np.log10(1.0 + 0.00437 * f)

    def erb_inv(e: float) -> float:
        return (10.0 ** (e / 21.4) - 1.0) / 0.00437

    edges = np.array(
        [erb_inv(e) for e in np.linspace(erb(f_min), erb(f_max), n_bands + 2)],
        dtype=np.float32,
    )
    n_bins = _N_FFT // 2 + 1
    freqs = np.linspace(0.0, _SR / 2.0, n_bins, dtype=np.float32)
    filters = np.zeros((n_bands, n_bins), dtype=np.float32)
    for i in range(n_bands):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        filters[i] = np.clip(
            np.minimum(
                (freqs - lo) / (mid - lo + _EPS),
                (hi - freqs) / (hi - mid + _EPS),
            ),
            0.0,
            1.0,
        )
    return filters


_FILTERS = _build_filters()
_WINDOW = np.hanning(_N_FFT).astype(np.float32)


def scale_to_db(
    drone: np.ndarray, background: np.ndarray, snr_db: float
) -> np.ndarray:
    drone_band = _band_power(drone)
    noise_band = _band_power(background)
    band_snr = 10.0 * np.log10((drone_band + _EPS) / (noise_band + _EPS))
    w = _w(drone_band)
    active = w >= _ACTIVE_THRESH
    if not active.any():
        active = w > 0.0
    current_snr = float(np.sum(band_snr[active] * _w(w[active])))
    gain = np.float32(10.0 ** ((float(snr_db) - current_snr) / 20.0))
    return (np.asarray(drone, dtype=np.float32) * gain).astype(
        np.float32, copy=False
    )


def _band_power(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if arr.size < _N_FFT:
        arr = np.pad(arr, (0, _N_FFT - arr.size))
    frames = [
        np.abs(np.fft.rfft(arr[s : s + _N_FFT] * _WINDOW)) ** 2
        for s in range(0, arr.size - _N_FFT + 1, _HOP)
    ] or [np.abs(np.fft.rfft(arr[:_N_FFT] * _WINDOW)) ** 2]
    return (_FILTERS @ np.mean(frames, axis=0).astype(np.float32)) + _EPS


def _w(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    s = x.sum()
    return x / s if s > 0 else np.full_like(x, 1.0 / max(1, x.size))
