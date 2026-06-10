#!/usr/bin/env python3
"""
augment.py - data augmentation, applied to TRAIN data only.

Two layers:
  * audio-domain (run on the raw clip before features): background-noise mix at
    a target SNR, Doppler pitch sweep, white Gaussian noise, gain jitter,
    time-shift. These create realistic variation a real flight would show.
  * spectrogram-domain (run on the (C,F,T) tensor during training): SpecAugment
    time/freq masking, frequency shift, additive Gaussian noise.

The same background-mix function powers evaluate.py's detection-vs-SNR sweep
(mix a clean drone over noise at a KNOWN dB and see if the model still fires).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# audio-domain
# --------------------------------------------------------------------------- #
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def mix_at_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Mix `sig` over `noise` so the result has the requested SNR (dB)."""
    rng = rng or np.random.default_rng()
    if len(noise) < len(sig):
        reps = int(np.ceil(len(sig) / len(noise)))
        noise = np.tile(noise, reps)
    start = rng.integers(0, max(1, len(noise) - len(sig) + 1))
    noise = noise[start:start + len(sig)]
    target_noise_rms = rms(sig) / (10 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / rms(noise))
    out = sig + noise
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak * 0.99).astype(np.float32) if peak > 1 else out.astype(np.float32)


def add_white_noise(x: np.ndarray, snr_db: float,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n = rng.standard_normal(len(x))
    n *= (rms(x) / (10 ** (snr_db / 20.0))) / rms(n)
    return (x + n).astype(np.float32)


def doppler(x: np.ndarray, max_pct: float = 0.03,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Mild time-varying resample to mimic a moving drone (a few % pitch sweep)."""
    rng = rng or np.random.default_rng()
    pct = rng.uniform(-max_pct, max_pct)
    n = len(x)
    # linear pitch sweep from -pct..+pct across the clip
    t = np.linspace(0, 1, n)
    warp = np.cumsum(1.0 + pct * (t - 0.5) * 2.0)
    warp = warp / warp[-1] * (n - 1)
    return np.interp(np.arange(n), warp, x).astype(np.float32)


def gain_jitter(x, max_db=6.0, rng=None):
    rng = rng or np.random.default_rng()
    return (x * 10 ** (rng.uniform(-max_db, max_db) / 20.0)).astype(np.float32)


def time_shift(x, rng=None):
    rng = rng or np.random.default_rng()
    return np.roll(x, int(rng.integers(-len(x) // 8, len(x) // 8))).astype(np.float32)


def audio_augment(x, backgrounds=None, rng=None, p=0.7):
    """Random chain of audio-domain augmentations for one training clip."""
    rng = rng or np.random.default_rng()
    if rng.random() < p:
        x = time_shift(x, rng)
    if rng.random() < p:
        x = gain_jitter(x, rng=rng)
    if rng.random() < 0.5:
        x = doppler(x, rng=rng)
    if backgrounds is not None and len(backgrounds) and rng.random() < 0.6:
        bg = backgrounds[rng.integers(len(backgrounds))]
        x = mix_at_snr(x, bg, snr_db=float(rng.uniform(-5, 20)), rng=rng)
    elif rng.random() < 0.4:
        x = add_white_noise(x, snr_db=float(rng.uniform(5, 25)), rng=rng)
    return x


# --------------------------------------------------------------------------- #
# spectrogram-domain (operates on a (C,F,T) tensor)
# --------------------------------------------------------------------------- #
def spec_augment(t: np.ndarray, rng=None, n_freq=2, n_time=2,
                 max_f=0.15, max_t=0.15) -> np.ndarray:
    rng = rng or np.random.default_rng()
    C, F, T = t.shape
    t = t.copy()
    for _ in range(n_freq):
        w = int(rng.uniform(0, max_f) * F)
        if w:
            s = rng.integers(0, max(1, F - w)); t[:, s:s + w, :] = t.mean()
    for _ in range(n_time):
        w = int(rng.uniform(0, max_t) * T)
        if w:
            s = rng.integers(0, max(1, T - w)); t[:, :, s:s + w] = t.mean()
    return t


def freq_shift(t, rng=None, max_bins=8):
    rng = rng or np.random.default_rng()
    return np.roll(t, int(rng.integers(-max_bins, max_bins + 1)), axis=1)


def spec_noise(t, rng=None, sigma=0.05):
    rng = rng or np.random.default_rng()
    return t + rng.standard_normal(t.shape).astype(np.float32) * sigma * t.std()


def spec_augment_batch(t, rng=None, p=0.6):
    rng = rng or np.random.default_rng()
    if rng.random() < p:
        t = spec_augment(t, rng)
    if rng.random() < 0.4:
        t = freq_shift(t, rng)
    if rng.random() < 0.3:
        t = spec_noise(t, rng)
    return t
