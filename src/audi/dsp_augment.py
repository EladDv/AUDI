"""Physics-based drone audio augmentation transforms.

Composable numpy functions matching the existing audi.augment pattern.
Apply rotor-physics modifications to real drone recordings before mixing
with background noise in the training pipeline.

10-inch FPV: 3-4 blade props, 8000-12000 RPM → f0 125-350 Hz.
"""

from __future__ import annotations

import random

import numpy as np
from scipy.signal import resample

# ═══════════════════════════════════════════════════════════════
# Rotor-physics audio modifications
# ═══════════════════════════════════════════════════════════════

def rotor_modulation(
    wav: np.ndarray,
    sr: int = 16000,
    mod_hz: float = 25.0,
    mod_depth: float = 0.15,
    n_rotors: int = 4,
    f0_spread_hz: float = 5.0,
) -> np.ndarray:
    """Apply multi-rotor amplitude modulation.

    Simulates the beating/interference pattern of N unsynchronized
    rotors at slightly different RPM. Each rotor contributes an
    independent modulation envelope.

    Args:
        wav: Input waveform.
        sr: Sample rate.
        mod_hz: Base modulation frequency (RPM difference).
        mod_depth: Modulation depth (0-1).
        n_rotors: Number of rotors (4 for quad).
        f0_spread_hz: Max RPM deviation between rotors.

    Returns:
        Modulated waveform.
    """
    wav = np.asarray(wav, dtype=np.float64)
    n = len(wav)
    t = np.arange(n) / sr

    # Build combined modulation envelope
    envelope = np.ones(n, dtype=np.float64)
    for _ in range(n_rotors):
        f = mod_hz + random.uniform(-f0_spread_hz, f0_spread_hz)
        phase = random.uniform(0, 2 * np.pi)
        envelope += mod_depth * np.sin(2 * np.pi * f * t + phase)

    envelope /= n_rotors  # normalise

    return (wav * envelope).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Propagation effects (distance, atmosphere)
# ═══════════════════════════════════════════════════════════════

def harmonic_attenuation(
    wav: np.ndarray,
    sr: int = 16000,
    f0_hz: float = 200.0,
    distance_m: float = 200.0,
    humidity_pct: float = 50.0,
) -> np.ndarray:
    """Apply frequency-dependent atmospheric attenuation.

    Higher harmonics attenuate faster with distance due to air
    absorption. Models ISO 9613-1 atmospheric absorption.

    Args:
        wav: Input waveform.
        sr: Sample rate.
        f0_hz: Fundamental frequency.
        distance_m: Simulated distance in meters.
        humidity_pct: Relative humidity.

    Returns:
        Attenuated waveform.
    """
    import scipy.fft

    wav = np.asarray(wav, dtype=np.float64)
    n = len(wav)
    n_fft = 2 ** int(np.ceil(np.log2(n)))

    # FFT
    spec = scipy.fft.rfft(wav, n=n_fft)
    freqs = scipy.fft.rfftfreq(n_fft, 1.0 / sr)

    # ISO 9613-1 atmospheric absorption (simplified)
    # α ≈ 0.01 dB/kHz/m at 20°C, 50% RH
    alpha = 0.01  # dB / kHz / m  (approximate)
    att_db = -alpha * (freqs / 1000.0) * distance_m * (100.0 / humidity_pct)
    att_lin = 10.0 ** (att_db / 20.0)

    spec *= att_lin
    result = scipy.fft.irfft(spec, n=n_fft)[:n]
    return result.astype(np.float32)


def doppler_drift(
    wav: np.ndarray,
    sr: int = 16000,
    drift_rate_hz_s: float = 2.0,
) -> np.ndarray:
    """Apply linear frequency drift (Doppler from approaching/receding drone).

    Args:
        wav: Input waveform.
        sr: Sample rate.
        drift_rate_hz_s: Frequency drift rate in Hz/second.

    Returns:
        Drifted waveform.
    """
    wav = np.asarray(wav, dtype=np.float64)
    n = len(wav)
    dur = n / sr
    t = np.arange(n) / sr

    # Instantaneous frequency ratio
    ratio = 1.0 + drift_rate_hz_s * t / 200.0  # relative to 200 Hz f0

    # Phase accumulation
    phase = 2.0 * np.pi * sr * np.cumsum(1.0 / (sr * ratio)) / sr
    phase = phase - phase[0]

    # Resample by phase interpolation
    # Use linear interpolation to stretch/compress
    orig_t = np.linspace(0, dur, n)
    new_t = np.linspace(0, dur * ratio.mean(), n)
    result = np.interp(new_t, orig_t, wav)

    return result[:n].astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# DSP-based drone augmentation (compose multiple effects)
# ═══════════════════════════════════════════════════════════════

def drone_dsp_augment(
    wav: np.ndarray,
    sr: int = 16000,
    *,
    f0_hz: float | None = None,
    f0_range: tuple[float, float] = (125, 350),
    is_drone: bool = True,           # ← controls drone-specific transforms
    rotor_mod_prob: float = 0.7,
    harmonic_att_prob: float = 0.5,
    doppler_prob: float = 0.2,
    pitch_jitter_semitones: float = 1.0,
    gain_db_range: tuple[float, float] = (-30, 0),
) -> np.ndarray:
    """Physics-based augmentation for drone OR background audio.

    Recording/propagation effects (pitch jitter, gain, doppler, harmonic
    attenuation) are applied to both drones and backgrounds.
    Drone-specific transforms (harmonic stacking, rotor modulation) are
    ONLY applied to drone samples.

    Args:
        wav: Waveform [samples] (float32).
        sr: Sample rate.
        f0_hz: Override f0 (None = random in f0_range).
        f0_range: (min, max) f0 range.
        is_drone: If True, apply drone-specific physics transforms.
        (remaining args control transform probabilities)

    Returns:
        Augmented waveform.
    """
    wav = np.asarray(wav, dtype=np.float32).copy()

    if f0_hz is None:
        f0_hz = random.uniform(*f0_range)

    # ── Drone-specific (only for positive samples) ─────────
    if is_drone:
        if random.random() < rotor_mod_prob:
            wav = rotor_modulation(wav, sr, mod_hz=random.uniform(15, 35),
                                    mod_depth=random.uniform(0.05, 0.25),
                                    n_rotors=random.choice([2, 4]),
                                    f0_spread_hz=random.uniform(2, 8))

    # ── Propagation / recording (both drone and BG) ───────
    if random.random() < harmonic_att_prob:
        wav = harmonic_attenuation(wav, sr, f0_hz,
                                    distance_m=random.uniform(50, 500),
                                    humidity_pct=random.uniform(30, 80))

    if random.random() < doppler_prob:
        wav = doppler_drift(wav, sr,
                            drift_rate_hz_s=random.uniform(-5, 5))

    if pitch_jitter_semitones > 0:
        shift = random.uniform(-pitch_jitter_semitones, pitch_jitter_semitones)
        if abs(shift) > 0.01:
            factor = 2.0 ** (shift / 12.0)
            orig_len = len(wav)
            n_out = max(1, int(orig_len / factor))
            wav = resample(wav.astype(np.float64), n_out).astype(np.float32)
            if len(wav) < orig_len:
                wav = np.pad(wav, (0, orig_len - len(wav)))
            else:
                wav = wav[:orig_len]

    gain_db = random.uniform(*gain_db_range)
    wav = wav * (10.0 ** (gain_db / 20.0))

    peak = np.max(np.abs(wav))
    if peak > 0.99:
        wav = wav * (0.99 / peak)

    return wav.astype(np.float32)
