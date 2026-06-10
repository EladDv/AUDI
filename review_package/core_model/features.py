#!/usr/bin/env python3
"""
features.py - the full feature menu from CHANNEL_OPTIONS.md.

Every feature takes a mono clip (1-D float32) and returns a (C, F, T) float32
array, where F == CFG.F and T == CFG.T, so any model can consume any feature.

FEATURES registry maps name -> spec:
    fn         : callable(clip, cfg) -> (C, F, T)
    channels   : C
    needs_4ch  : True if it requires the raw 4-mic array (spatial features)

Add a new representation = add one function + one registry line.
"""
from __future__ import annotations

import numpy as np

from config import CFG


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _zoom_FT(x: np.ndarray, F: int = None, T: int = None) -> np.ndarray:
    """Resample a 2-D array to (F, T) with light bilinear interpolation."""
    from scipy.ndimage import zoom
    F = F or CFG.F
    T = T or CFG.T
    if x.shape == (F, T):
        return x.astype(np.float32)
    zy, zx = F / x.shape[0], T / x.shape[1]
    return zoom(x, (zy, zx), order=1).astype(np.float32)


def _stft_mag(y, cfg):
    import librosa
    S = np.abs(librosa.stft(
        y,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
    ))
    return S  # (n_fft/2+1, frames)


# --------------------------------------------------------------------------- #
# 1-channel representations
# --------------------------------------------------------------------------- #
def feat_spectrogram(y, cfg):
    import librosa
    S = _stft_mag(y, cfg)
    db = librosa.amplitude_to_db(S, ref=np.max)
    return _zoom_FT(db)[None]


def feat_mel(y, cfg):
    import librosa
    m = librosa.feature.melspectrogram(
        y=y, sr=cfg.sr, n_fft=cfg.n_fft, win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.F, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0)
    return _zoom_FT(librosa.power_to_db(m, ref=np.max))[None]


def feat_pcen(y, cfg):
    import librosa
    S = librosa.feature.melspectrogram(
        y=y, sr=cfg.sr, n_fft=cfg.n_fft, win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.F, fmin=cfg.fmin, fmax=cfg.fmax, power=1.0)
    p = librosa.pcen(S * (2 ** 31), sr=cfg.sr, hop_length=cfg.hop_length)
    return _zoom_FT(p)[None]


_SCALO_FB: dict = {}


def _scalo_filterbank(sr, n_fft, fmin, fmax, n_bins):
    """Precomputed (cached) log-frequency (constant-Q-style) filterbank matrix.
    Row i = a triangular weight centred on a log-spaced frequency, L1-normalised.
    Built ONCE per (sr, n_fft, ...) then reused -> the scalogram becomes a matmul."""
    key = (sr, n_fft, round(fmin, 3), round(fmax, 3), n_bins)
    W = _SCALO_FB.get(key)
    if W is None:
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        cen = np.geomspace(fmin, fmax, n_bins)
        bw = np.gradient(cen)                          # local bandwidth (Hz)
        W = np.maximum(0.0, 1.0 - np.abs(cen[:, None] - freqs[None, :]) / bw[:, None])
        W = (W / (W.sum(1, keepdims=True) + 1e-9)).astype(np.float32)
        _SCALO_FB[key] = W
    return W


def feat_scalogram(y, cfg):
    """FAST log-frequency scalogram: one STFT magnitude * a precomputed constant-Q
    filterbank (matmul). ~50x faster than librosa.cqt, same log-freq picture.
    (Old exact CQT kept as 'scalogram_cqt' for reference / fallback.)"""
    import librosa
    fmin = max(cfg.fmin, 32.0)
    fmax = min(cfg.fmax, 0.95 * cfg.sr / 2.0)
    W = _scalo_filterbank(cfg.sr, cfg.n_fft, fmin, fmax, n_bins=96)
    S = _stft_mag(y, cfg)                              # (n_fft/2+1, frames)
    sca = W @ S                                        # (96, frames) - one matmul
    return _zoom_FT(librosa.amplitude_to_db(sca, ref=np.max))[None]


def feat_scalogram_cqt(y, cfg):
    """Exact constant-Q transform (slower; kept for reference/fallback)."""
    import librosa
    fmin = max(cfg.fmin, 32.0)
    fmax = min(cfg.fmax, 0.95 * cfg.sr / 2.0)
    bpo = 36
    n_bins = max(int(np.floor(bpo * np.log2(fmax / fmin))), 24)
    n_oct = int(np.ceil(n_bins / bpo))
    step = 1 << n_oct
    hop = max(step, (cfg.hop_length // step) * step or step)
    c = np.abs(librosa.cqt(
        y, sr=cfg.sr, hop_length=hop, fmin=fmin,
        n_bins=n_bins, bins_per_octave=bpo))
    return _zoom_FT(librosa.amplitude_to_db(c, ref=np.max))[None]


def feat_harmonic_stack(y, cfg):
    """HPSS harmonic component on a mel grid (sustained tones, transients gone)."""
    import librosa
    S = _stft_mag(y, cfg)
    H, _ = librosa.decompose.hpss(S)
    mel = librosa.feature.melspectrogram(
        S=H ** 2, sr=cfg.sr, n_mels=cfg.F, fmin=cfg.fmin, fmax=cfg.fmax)
    return _zoom_FT(librosa.power_to_db(mel, ref=np.max))[None]


def feat_cepstrum(y, cfg):
    """Real cepstrum per frame: how regularly spaced the harmonics are."""
    S = _stft_mag(y, cfg)
    logS = np.log(S + 1e-6)
    cep = np.fft.irfft(logS, axis=0).real          # (n_fft/2+1 -> quefrency, frames)
    cep = cep[: S.shape[0]]
    return _zoom_FT(cep)[None]


def feat_modulation(y, cfg):
    """Modulation spectrogram: rhythm / how energy pulses (rotor RPM)."""
    import librosa
    m = librosa.feature.melspectrogram(
        y=y, sr=cfg.sr, n_fft=cfg.n_fft, win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.F, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0)
    # FFT along time -> modulation frequency per mel band
    mod = np.abs(np.fft.rfft(m - m.mean(axis=1, keepdims=True), axis=1))
    return _zoom_FT(20 * np.log10(mod + 1e-6))[None]


def feat_cyclostationary(y, cfg):
    """Cheap cyclostationary proxy: autocorrelation of each band's envelope.
    Rotating machines (drones) show periodic structure; random noise does not."""
    import librosa
    m = librosa.feature.melspectrogram(
        y=y, sr=cfg.sr, n_fft=cfg.n_fft, win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.F, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0)
    env = m - m.mean(axis=1, keepdims=True)
    ac = np.fft.irfft(np.abs(np.fft.rfft(env, axis=1)) ** 2, axis=1).real
    return _zoom_FT(ac[:, : m.shape[1]])[None]


def feat_harmonic_sum(y, cfg):
    """Harmonic-sum spectrogram: sum buried harmonics to reveal a faint drone."""
    S = _stft_mag(y, cfg)
    acc = np.zeros_like(S)
    for h in (1, 2, 3, 4, 5):
        ds = S[::h]
        acc[: ds.shape[0]] += ds
    return _zoom_FT(20 * np.log10(acc + 1e-6))[None]


def feat_cfar(y, cfg):
    """CFAR contrast map: each cell vs its local background (adaptive threshold)."""
    from scipy.ndimage import uniform_filter
    import librosa
    S = librosa.amplitude_to_db(_stft_mag(y, cfg), ref=np.max)
    bg = uniform_filter(S, size=(9, 9), mode="nearest")
    return _zoom_FT(S - bg)[None]


# --------------------------------------------------------------------------- #
# multi-channel representations
# --------------------------------------------------------------------------- #
def feat_mel2(y, cfg):
    """[Mel+PCEN, scalogram] - the main 2-channel candidate."""
    return np.concatenate([feat_pcen(y, cfg), feat_scalogram(y, cfg)], axis=0)


def feat_mel3(y, cfg):
    """[PCEN-mel, scalogram, modulation] - the chosen 3-channel feature.

    Picked from our experiments:
      ch0 PCEN-mel  -> detection energy (the mel family won drone yes/no)
      ch1 scalogram -> best, most noise-consistent TYPE separator (EVO low / FPV high)
      ch2 modulation-> rhythm/RPM, the noise-robust type cue (best at 10 dB)
    """
    return np.concatenate([feat_pcen(y, cfg), feat_scalogram(y, cfg),
                           feat_modulation(y, cfg)], axis=0)


# --------------------------------------------------------------------------- #
# Explicit, human-readable EVO/FPV type cues (Option A).
# These are the SAME physics you read by eye in the 3 mel3 layers: where the
# harmonic comb sits (the >4 kHz cue) and how fast the rotor pulses (rhythm/SCD).
# Stage 2 fuses these alongside the CNN embedding so the TYPE call reads ch1/ch2/
# SCD DIRECTLY, instead of only through the pooled 128-d summary that blurs them.
# --------------------------------------------------------------------------- #
TYPE_STAT_NAMES = ["frac>4kHz", "frac>2kHz", "frac1-4kHz", "centroid_kHz",
                   "rolloff_kHz", "hi/lo_ratio", "scd_peak", "scd_ratio",
                   "scd_loc_Hz"]


def type_physics_stats(clip, cfg=CFG):
    """Return the named EVO/FPV physics cues for one clip -> (len(TYPE_STAT_NAMES),).

    High >4 kHz energy / high centroid / strong fast rhythm  => FPV-like.
    Low-band dominant / slow rhythm                          => EVO-like.
    Cheap (one FFT + the SCD profile), so it runs live per window.
    """
    from scd_probe import scd_alpha_profile
    y = clip.astype(np.float64)
    w = np.hanning(len(y))
    P = np.abs(np.fft.rfft(y * w)) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / cfg.sr)
    tot = P.sum() + 1e-12
    frac4 = P[f >= 4000].sum() / tot
    frac2 = P[f >= 2000].sum() / tot
    frac14 = P[(f >= 1000) & (f < 4000)].sum() / tot
    centroid = (f * P).sum() / tot
    cs = np.cumsum(P)
    rolloff = f[np.searchsorted(cs, 0.85 * cs[-1])] if cs[-1] > 0 else 0.0
    hi_lo = P[f >= 4000].sum() / (P[f < 2000].sum() + 1e-12)
    scd = scd_alpha_profile(clip, cfg.sr)
    speak = float(scd.max())
    sratio = speak / (float(scd.mean()) + 1e-9)
    sloc = 10.0 + (2000.0 - 10.0) * (int(np.argmax(scd)) / max(1, len(scd) - 1))
    return np.array([frac4, frac2, frac14, centroid / 1000.0, rolloff / 1000.0,
                     hi_lo, speak, sratio, sloc / 1000.0], np.float32)


def feat_gcc_spatial(y4, cfg):
    """GCC-PHAT lag image across mic pairs. Requires the raw 4-channel clip
    (shape (4, samples)); returns a (1, F, T) lag-time image."""
    import librosa
    if y4.ndim != 2 or y4.shape[0] < 2:
        raise ValueError("gcc_spatial needs a (>=2, samples) array clip")
    n_ch = y4.shape[0]
    pairs = [(a, b) for a in range(n_ch) for b in range(a + 1, n_ch)]
    win = cfg.n_fft
    hop = cfg.hop_length
    frames = 1 + (y4.shape[1] - win) // hop
    img = np.zeros((win, max(frames, 1)), np.float32)
    for f in range(max(frames, 1)):
        s = f * hop
        acc = np.zeros(win)
        for a, b in pairs:
            A = np.fft.rfft(y4[a, s:s + win] * np.hanning(win), win)
            B = np.fft.rfft(y4[b, s:s + win] * np.hanning(win), win)
            R = A * np.conj(B)
            R /= np.abs(R) + 1e-9
            acc += np.fft.irfft(R, win)
        img[:, f] = np.fft.fftshift(acc)
    return _zoom_FT(img)[None]


# --------------------------------------------------------------------------- #
FEATURES = {
    "spectrogram":     dict(fn=feat_spectrogram,     channels=1, needs_4ch=False),
    "mel":             dict(fn=feat_mel,             channels=1, needs_4ch=False),
    "pcen":            dict(fn=feat_pcen,            channels=1, needs_4ch=False),
    "scalogram":       dict(fn=feat_scalogram,       channels=1, needs_4ch=False),
    "scalogram_cqt":   dict(fn=feat_scalogram_cqt,   channels=1, needs_4ch=False),
    "harmonic_stack":  dict(fn=feat_harmonic_stack,  channels=1, needs_4ch=False),
    "cepstrum":        dict(fn=feat_cepstrum,        channels=1, needs_4ch=False),
    "modulation":      dict(fn=feat_modulation,      channels=1, needs_4ch=False),
    "cyclostationary": dict(fn=feat_cyclostationary, channels=1, needs_4ch=False),
    "harmonic_sum":    dict(fn=feat_harmonic_sum,    channels=1, needs_4ch=False),
    "cfar":            dict(fn=feat_cfar,            channels=1, needs_4ch=False),
    "mel2":            dict(fn=feat_mel2,            channels=2, needs_4ch=False),
    "mel3":            dict(fn=feat_mel3,            channels=3, needs_4ch=False),
    "gcc_spatial":     dict(fn=feat_gcc_spatial,     channels=1, needs_4ch=True),
}


def list_features() -> list[str]:
    return list(FEATURES)
