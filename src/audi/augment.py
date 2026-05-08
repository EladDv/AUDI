"""Audio augmentation transforms for training mixture generation.

All functions operate on numpy float32 arrays and return numpy float32 arrays.
They are designed to be composable and individually testable.
"""

from __future__ import annotations

import random

import numpy as np
from scipy.signal import butter, iirpeak, sosfilt


def gain_jitter(y: np.ndarray, db: float) -> np.ndarray:
    """Apply random gain in ``[-db, +db]``.

    Args:
        y: Input waveform (float32).
        db: Maximum gain deviation in dB.

    Returns:
        Gain-adjusted waveform.
    """
    factor = 10 ** (random.uniform(-db, db) / 20.0)
    return (y * factor).astype(np.float32)


def pitch_shift(wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Pitch-shift by random amount in ``[-semitones, +semitones]``.

    Pure numpy/scipy — no torch dependency, safe for DataLoader workers.

    Args:
        wav: Input waveform (float32).
        sr: Sample rate in Hz.
        semitones: Maximum pitch shift.

    Returns:
        Pitch-shifted waveform.
    """
    from scipy.signal import resample

    shift = random.uniform(-semitones, semitones)
    if abs(shift) < 0.01:
        return wav
    factor = 2.0 ** (shift / 12.0)  # semitones → rate ratio
    arr = np.asarray(wav, dtype=np.float64).reshape(-1)
    n_in = len(arr)
    n_out = max(1, int(n_in / factor))
    shifted = resample(arr, n_out).astype(np.float32)
    if len(shifted) < n_in:
        shifted = np.pad(shifted, (0, n_in - len(shifted)))
    elif len(shifted) > n_in:
        shifted = shifted[:n_in]
    return shifted.astype(np.float32)


def time_stretch(
    wav: np.ndarray,
    rate_range: tuple[float, float],
    target_length: int,
    sr: int = 16000,
) -> np.ndarray:
    """Time-stretch by random factor, then crop/pad to target_length.

    Pure numpy/scipy — no torch dependency, safe for DataLoader workers.

    Args:
        wav: Input waveform.
        rate_range: (min, max) speed factor.
        target_length: Output length in samples.
        sr: Sample rate (unused, kept for API compatibility).

    Returns:
        Time-stretched waveform of exactly ``target_length`` samples.
    """
    from scipy.signal import resample

    rate = random.uniform(*rate_range)
    if abs(rate - 1.0) < 0.01:
        return _fit_length(wav, target_length)
    arr = np.asarray(wav, dtype=np.float64).reshape(-1)
    n_out = max(1, int(len(arr) * rate))
    stretched = resample(arr, n_out).astype(np.float32)
    return _fit_length(stretched, target_length)


def reverb(
    wav: np.ndarray, sr: int, decay_range: tuple[float, float]
) -> np.ndarray:
    """Apply exponential-decay reverb.

    Args:
        wav: Input waveform.
        sr: Sample rate.
        decay_range: (min, max) RT60 in seconds.

    Returns:
        Reverberated waveform.
    """
    decay = random.uniform(*decay_range)
    ir_len = int(sr * decay)
    ir = np.exp(-np.linspace(0, 5, ir_len)) * random.uniform(0.1, 0.5)
    ir = ir.astype(np.float32)
    wav_arr = np.asarray(wav, dtype=np.float32)
    wet = np.convolve(wav_arr, ir, mode="full")[: len(wav_arr)]
    wet_max = np.abs(wet).max()
    if wet_max > 0:
        wet = wet * (np.abs(wav_arr).max() / wet_max)
    return (wav_arr + wet * 0.3).astype(np.float32)


def random_eq(wav: np.ndarray, sr: int, gain_db: float) -> np.ndarray:
    """Apply random 2-band parametric EQ.

    Args:
        wav: Input waveform.
        sr: Sample rate.
        gain_db: Maximum gain per band in dB.

    Returns:
        EQ'd waveform (peak-limited to 0.99).
    """
    t = np.asarray(wav, dtype=np.float32).copy()
    for _ in range(2):
        freq = random.uniform(200, min(sr // 2 - 100, 6000))
        q = random.uniform(0.5, 2.0)  # clamped Q to avoid resonance
        db = random.uniform(-gain_db, gain_db)
        try:
            b, a = iirpeak(freq, q, sr)
            sos = np.atleast_2d(np.concatenate([b, a]))
            # Run filter in float64 to avoid overflow, then clip and cast
            filtered = sosfilt(sos, t.astype(np.float64))
            filtered = np.clip(filtered, -10.0, 10.0).astype(np.float32)
            t = t + (10 ** (db / 20) - 1) * filtered
            # Per-band peak limit
            peak = np.abs(t).max()
            if peak > 1.0:
                t = t * (0.99 / peak)
        except Exception:
            continue
    return t.astype(np.float32)


def noise_inject(y: np.ndarray, db: float) -> np.ndarray:
    """Add Gaussian noise at ``db`` dBFS relative to signal RMS.

    Args:
        y: Input waveform.
        db: Target noise level in dB below signal RMS.

    Returns:
        Noise-injected waveform.
    """
    sig_rms = _rms(y)
    if sig_rms <= 1e-8:
        return np.asarray(y, dtype=np.float32)
    noise_rms = sig_rms * (10 ** (db / 20.0))
    noise = np.random.randn(*y.shape).astype(np.float32) * noise_rms
    return (y + noise).astype(np.float32)


def time_mask_waveform(
    y: np.ndarray, count: int, max_ratio: float
) -> np.ndarray:
    """Zero out random contiguous segments of the waveform.

    Args:
        y: Input waveform.
        count: Number of masks to apply.
        max_ratio: Maximum fraction of total length per mask.

    Returns:
        Masked waveform.
    """
    arr = np.asarray(y, dtype=np.float32).copy()
    L = len(arr)
    for _ in range(count):
        mask_len = int(random.uniform(0.01, max_ratio) * L)
        if mask_len > 0:
            start = random.randint(0, max(1, L - mask_len))
            arr[start : start + mask_len] = 0.0
    return arr


def lowpass(
    wav: np.ndarray, sr: int, cutoff_range: tuple[float, float]
) -> np.ndarray:
    """Apply random low-pass Butterworth filter.

    Args:
        wav: Input waveform.
        sr: Sample rate.
        cutoff_range: (min, max) cutoff frequency in Hz.

    Returns:
        Low-pass filtered waveform.
    """
    cutoff = random.uniform(*cutoff_range)
    nyq = sr / 2.0
    cutoff_norm = min(cutoff / nyq, 0.99)
    sos = butter(4, cutoff_norm, btype="low", output="sos")
    return np.asarray(sosfilt(sos, wav), dtype=np.float32)


def highpass(
    wav: np.ndarray, cutoff_hz: float, sample_rate: int, order: int = 4
) -> np.ndarray:
    """Apply high-pass Butterworth filter.

    Args:
        wav: Input waveform.
        cutoff_hz: Cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.
        order: Filter order.

    Returns:
        High-pass filtered waveform.
    """
    sos = butter(order, cutoff_hz, btype="high", fs=sample_rate, output="sos")
    return np.asarray(sosfilt(sos, wav), dtype=np.float32)


# ── Utility helpers ────────────────────────────────────────────────


def _rms(x: np.ndarray, eps: float = 1e-8) -> float:
    """Root-mean-square of a numpy array."""
    y = np.asarray(x, dtype=np.float32).reshape(-1)
    sq = y * y
    if not np.isfinite(sq).all():
        return eps  # NaN/Inf guard
    return float(np.sqrt(np.mean(sq) + eps))


def _fit_length(y: np.ndarray, length: int) -> np.ndarray:
    """Crop or tile an array to exactly ``length`` samples."""
    arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty audio")
    if arr.size < length:
        repeats = int(np.ceil(length / arr.size))
        arr = np.tile(arr, repeats)
    if arr.size == length:
        return arr
    start = int(random.randint(0, arr.size - length))
    return arr[start : start + length].astype(np.float32)


def peak_limit(mix: np.ndarray, peak_target: float = 0.98) -> np.ndarray:
    """Hard-clip waveform peaks to ``peak_target``."""
    x = np.asarray(mix, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > peak_target and peak > 0.0:
        x = x * np.float32(peak_target / peak)
    return x


def drone_fade(
    wav: np.ndarray,
    max_ratio: float = 0.5,
    *,
    sample_rate: int = 16000,
    min_fade_seconds: float = 0.02,
) -> np.ndarray:
    """Apply random silence gap + exponential fade-in/out to drone audio.

    Randomly inserts a silence-gap with an exponential fade at the start
    (fade-in), at the end (fade-out), or both.  The total gap duration is
    a random fraction of the window (up to max_ratio).  Within that gap
    the fade portion is shorter than the full gap, leaving a short silence
    before the fade-in starts or after the fade-out finishes.

    Fade-in:  [silence …] [exp ramp 0→1] [full audio …]
    Fade-out: [… full audio] [exp ramp 1→0] [silence …]

    Args:
        wav: Input waveform (float32).
        max_ratio: Maximum total gap as fraction of total length
                   (e.g. 0.5 = up to 50% of window).
        sample_rate: Audio sample rate in Hz.
        min_fade_seconds: Minimum fade duration in seconds to avoid clicks.

    Returns:
        Faded waveform of the same length.
    """
    arr = np.asarray(wav, dtype=np.float32).copy()
    length = len(arr)
    if length < 8:
        return arr

    min_fade_samples = max(1, int(sample_rate * min_fade_seconds))
    max_gap = max(1, int(length * max_ratio))
    # Clamp min_fade to the max gap so short arrays still get fades
    min_fade_samples = min(min_fade_samples, max_gap // 2)

    # ── Fade-in (gap at start) ────────────────────────────────────
    gap_in_len = random.randint(0, max_gap)
    if gap_in_len >= min_fade_samples:
        # Fade occupies 25–100 % of the gap, weighted toward the high end
        lo = max(min_fade_samples, gap_in_len // 4)
        hi = gap_in_len
        # Skew toward hi using sqrt: uniform→power-law favoring large values
        u = random.random()
        fade_in_len = lo + int((hi - lo) * (u**0.5))
        silence_in = gap_in_len - fade_in_len  # silence before fade

        # Silence at very start
        arr[:silence_in] = 0.0

        # Exponential ramp 0 → 1 over fade_in_len
        t = np.arange(fade_in_len, dtype=np.float32) / fade_in_len
        env = np.exp(-4.0 * (1.0 - t))
        arr[silence_in:gap_in_len] *= env

    # ── Fade-out (gap at end) ─────────────────────────────────────
    gap_out_len = random.randint(0, max_gap)
    if gap_out_len >= min_fade_samples:
        lo = max(min_fade_samples, gap_out_len // 4)
        hi = gap_out_len
        u = random.random()
        fade_out_len = lo + int((hi - lo) * (u**0.5))
        silence_out = gap_out_len - fade_out_len  # silence after fade

        # Exponential ramp 1 → 0
        t = np.arange(fade_out_len, dtype=np.float32) / fade_out_len
        env = np.exp(-4.0 * t)
        arr[length - gap_out_len : length - silence_out] *= env

        # Silence at very end
        arr[length - silence_out :] = 0.0

    return arr


def distance_lowpass(
    wav: np.ndarray,
    snr_db: float,
    sample_rate: int = 16000,
    *,
    snr_min: float = -30.0,
    snr_max: float = 0.0,
    cutoff_min: float = 400.0,
    cutoff_max: float = 7500.0,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth lowpass simulating atmospheric absorption.

    Cutoff frequency is interpolated from SNR: close drones (high SNR)
    get little filtering; far drones (low SNR) get aggressive roll-off.

    Args:
        wav: Input drone waveform (float32).
        snr_db: Actual SNR value in dB.
        sample_rate: Audio sample rate in Hz.
        snr_min: SNR that maps to cutoff_min (most distant, e.g. -30 dB).
        snr_max: SNR that maps to cutoff_max (closest, e.g. 0 dB).
        cutoff_min: Lowest cutoff frequency in Hz.
        cutoff_max: Highest cutoff frequency in Hz.
        order: Butterworth filter order.

    Returns:
        Low-pass filtered waveform.
    """
    from scipy.signal import butter, sosfilt

    snr = max(snr_min, min(snr_max, snr_db))
    t = (snr - snr_min) / max(1e-6, snr_max - snr_min)
    cutoff = cutoff_min + t * (cutoff_max - cutoff_min)
    nyq = sample_rate / 2.0
    cutoff_norm = min(cutoff / nyq, 0.99)
    sos = butter(order, cutoff_norm, btype="low", output="sos")
    return np.asarray(sosfilt(sos, wav), dtype=np.float32)


def doppler_shift(
    wav: np.ndarray,
    sample_rate: int = 16000,
    *,
    max_speed_mps: float = 30.0,
    speed_of_sound_mps: float = 343.0,
    target_length: int | None = None,
) -> np.ndarray:
    """Apply Doppler shift simulating a drone moving toward/away from the mic.

    Resamples the waveform to simulate the frequency compression (approaching)
    or expansion (receding) caused by relative motion. Unlike independent
    pitch_shift + time_stretch, this preserves the physical relationship
    between pitch change and time dilation.

    The Doppler factor is ``f' = f * c / (c - v)`` where ``c`` is the speed
    of sound and ``v`` is the relative velocity (positive = approaching,
    negative = receding).

    Args:
        wav: Input drone waveform (float32).
        sample_rate: Audio sample rate in Hz.
        max_speed_mps: Maximum absolute velocity in m/s. A random velocity
                       in ``[-max_speed_mps, +max_speed_mps]`` is chosen.
                       Default 30 m/s ≈ 108 km/h (fast racing drone).
        speed_of_sound_mps: Speed of sound (default 343 m/s at 20°C).
        target_length: If provided, crop/pad output to this length.
                       If None, returns the Doppler-shifted length.

    Returns:
        Doppler-shifted waveform.
    """
    v = random.uniform(-max_speed_mps, max_speed_mps)
    if abs(v) < 0.5:
        return _fit_length(wav, target_length) if target_length else wav

    factor = speed_of_sound_mps / (speed_of_sound_mps - v)
    # factor > 1: approaching (compressed, higher pitch)
    # factor < 1: receding (expanded, lower pitch)

    # Resample using linear interpolation
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    n_in = len(arr)
    n_out = max(1, int(n_in / factor))

    new_indices = np.arange(n_out)
    # Map each output sample to a position in the input
    src_idx = new_indices * factor
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    frac = src_idx - lo.astype(float)

    shifted = arr[lo] * (1 - frac) + arr[hi] * frac
    shifted = np.clip(shifted, -10.0, 10.0).astype(np.float32)

    if target_length is not None:
        return _fit_length(shifted, target_length)
    return shifted
