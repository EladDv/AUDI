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
# field-condition augmentations (hostile-terrain robustness)
# Applied to BOTH drones and negatives so the model learns the distortion, not
# the source. Wired into make_features.py via --field-aug.
# --------------------------------------------------------------------------- #
def atmospheric_lowpass(x, sr, cutoff_hz=4000.0, rng=None):
    """Long-range air absorption: high frequencies (esp. >4 kHz) are swallowed
    by the atmosphere, so a far drone sounds muffled. Lower cutoff = farther.
    One-pole-style smooth rolloff above the corner (cheap, FFT-domain)."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    H = 1.0 / (1.0 + (f / max(cutoff_hz, 1.0)) ** 2)   # ~-6 dB/oct beyond corner
    return np.fft.irfft(X * H, n=n).astype(np.float32)


def clip_saturate(x, frac=0.5, rng=None):
    """Hard-clip near a fraction of the peak: mimics a loud source close to the
    mic driving the input into saturation (adds harmonic distortion)."""
    peak = float(np.max(np.abs(x))) + 1e-9
    thr = peak * float(frac)
    return np.clip(x, -thr, thr).astype(np.float32)


def dropout_chunks(x, sr, rng=None, max_chunks=4, max_ms=30.0):
    """Zero a few short windows: mimics I2S glitches / Pi bottleneck dropouts."""
    rng = rng or np.random.default_rng()
    out = x.copy()
    for _ in range(int(rng.integers(1, max_chunks + 1))):
        w = int(rng.uniform(1.0, max_ms) * sr / 1000.0)
        if w >= len(x):
            continue
        s = int(rng.integers(0, len(x) - w))
        out[s:s + w] = 0.0
    return out.astype(np.float32)


def doppler_flyby(x, sr, max_pct=0.12, rng=None):
    """Strong NON-linear flyby: pitch starts high (approaching), sweeps through
    overhead, then drops low (receding). Breaks the steady harmonic comb the way
    a fast FPV pass does - much more aggressive than the mild `doppler` above."""
    rng = rng or np.random.default_rng()
    n = len(x)
    t = np.linspace(0, 1, n)
    center = float(rng.uniform(0.3, 0.7))
    steep = float(rng.uniform(8.0, 16.0))
    s = 1.0 / (1.0 + np.exp(steep * (t - center)))      # 1 -> 0 transition
    rate = float(max_pct) * (2.0 * s - 1.0)             # +max_pct -> -max_pct
    warp = np.cumsum(1.0 + rate)
    warp = warp / warp[-1] * (n - 1)
    return np.interp(np.arange(n), warp, x).astype(np.float32)


def reverb(x, sr, rng=None, n_taps=3, decay=(0.2, 0.6)):
    """Cheap multipath/reverberation: a few delayed, decaying echoes (built-up
    terrain - sound bounces off walls and arrives smeared in time/phase)."""
    rng = rng or np.random.default_rng()
    out = x.astype(np.float32).copy()
    for _ in range(int(rng.integers(2, n_taps + 1))):
        d = int(rng.uniform(0.005, 0.05) * sr)          # 5-50 ms delay
        g = float(rng.uniform(*decay))
        if 0 < d < len(x):
            out[d:] += g * x[:len(x) - d]
    in_peak = float(np.max(np.abs(x))) + 1e-9
    out_peak = float(np.max(np.abs(out))) + 1e-9
    return (out / out_peak * in_peak).astype(np.float32)


def field_augment(x, sr, rng=None, backgrounds=None,
                  p_doppler=0.4, p_lowpass=0.5, p_reverb=0.3,
                  p_bg=0.6, p_clip=0.25, p_dropout=0.3):
    """One random chain of field-condition distortions for a TRAIN clip.
    Order mirrors the real signal path: flight dynamics -> propagation (air,
    walls) -> ambient mix -> capture-chain artifacts (saturation, dropout)."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, np.float32)
    if rng.random() < p_doppler:
        x = doppler_flyby(x, sr, max_pct=float(rng.uniform(0.05, 0.15)), rng=rng)
    if rng.random() < p_lowpass:
        x = atmospheric_lowpass(x, sr, cutoff_hz=float(rng.uniform(2000.0, 8000.0)), rng=rng)
    if rng.random() < p_reverb:
        x = reverb(x, sr, rng=rng)
    if backgrounds is not None and len(backgrounds) and rng.random() < p_bg:
        bg = backgrounds[rng.integers(len(backgrounds))]
        x = mix_at_snr(x, bg, snr_db=float(rng.uniform(-5.0, 20.0)), rng=rng)
    if rng.random() < p_clip:
        x = clip_saturate(x, frac=float(rng.uniform(0.3, 0.7)), rng=rng)
    if rng.random() < p_dropout:
        x = dropout_chunks(x, sr, rng=rng)
    return x.astype(np.float32)


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
