#!/usr/bin/env python3
"""
feature_gallery.py - READ-ONLY: draw the 4 classification features on REAL clips
so the HTML guide can show concrete examples (EVO vs FPV) and cite exactly which
file + which time-offset each picture came from.

Outputs (into array_guide_assets/ at the repo root):
    feature_gallery.png        4 features (rows) x 4 example clips (cols)
    spectrogram_falldown.png    mel vs scd, clean vs +noise (why mel collapses)
    feature_gallery_manifest.txt  the file + time-offset behind every column

No model, no data changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import CFG                                  # noqa: E402
from scd_probe import scd_alpha_profile                 # noqa: E402

REPO = HERE.parent.parent
ASSETS = REPO / "array_guide_assets"
RAW = HERE.parent / "data" / "raw"

# (label, distance-tag, path) - real files confirmed to exist
EXAMPLES = [
    ("EVO", "20 m", RAW / "live_session_20260528/drone/EVO/evo_20-20m_20260528T131513.wav"),
    ("EVO", "50 m", RAW / "live_session_20260528/drone/EVO/evo_50-50m_20260528T131821.wav"),
    ("FPV", "20 m", RAW / "live_session_20260528/drone/fpv/fpv_20-20m_20260528T142410.wav"),
    ("FPV", "25 m", RAW / "live_session_20260601/drone/fpv/fpv_25-25m_20260601T074851.wav"),
]
EVO_C = "#2563eb"
FPV_C = "#dc2626"


def take_clip(path):
    """Load mono and return (clip, t0_seconds) from the middle of the file."""
    import librosa
    y, _ = librosa.load(str(path), sr=CFG.sr, mono=True)
    n = int(CFG.clip_s * CFG.sr)
    if len(y) <= n:
        return np.pad(y, (0, n - len(y))).astype(np.float32), 0.0
    s = (len(y) - n) // 2
    return y[s:s + n].astype(np.float32), s / CFG.sr


# --------------------------------------------------------------------------- #
# the 4 classification views, computed with correct axes for plotting
# --------------------------------------------------------------------------- #
def harmonic_comb(y):
    """HPSS-harmonic mel spectrum, averaged over time -> a comb of rotor peaks."""
    import librosa
    S = np.abs(librosa.stft(
        y,
        n_fft=CFG.n_fft,
        win_length=CFG.win_length,
        hop_length=CFG.hop_length,
    ))
    H, _ = librosa.decompose.hpss(S)
    mel = librosa.feature.melspectrogram(S=H ** 2, sr=CFG.sr, n_mels=128,
                                         fmin=CFG.fmin, fmax=CFG.fmax)
    hz = librosa.mel_frequencies(n_mels=128, fmin=CFG.fmin, fmax=CFG.fmax)
    v = librosa.power_to_db(mel, ref=np.max).mean(axis=1)
    return hz, v


def scalogram_img(y):
    """CQT (log-frequency) dB image + freq axis."""
    import librosa
    fmin = max(CFG.fmin, 32.0)
    fmax = min(CFG.fmax, 0.95 * CFG.sr / 2.0)
    bpo = 36
    n_bins = max(int(np.floor(bpo * np.log2(fmax / fmin))), 24)
    c = np.abs(librosa.cqt(y, sr=CFG.sr, hop_length=256, fmin=fmin,
                           n_bins=n_bins, bins_per_octave=bpo))
    db = librosa.amplitude_to_db(c, ref=np.max)
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bpo)
    return db, freqs


def modulation_rhythm(y, fmax_hz=220.0):
    """High-rate envelope -> modulation (rhythm) spectrum up to ~220 Hz."""
    import librosa
    hop = 32
    S = librosa.feature.melspectrogram(y=y, sr=CFG.sr, n_fft=CFG.n_fft,
                                       win_length=CFG.win_length,
                                       hop_length=hop, n_mels=48,
                                       fmin=CFG.fmin, fmax=CFG.fmax, power=2.0)
    fr = CFG.sr / hop
    env = S - S.mean(axis=1, keepdims=True)
    M = np.abs(np.fft.rfft(env, axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(S.shape[1], 1.0 / fr)
    sel = f <= fmax_hz
    return f[sel], M[sel]


def scd_rhythm(y, fmax_hz=220.0):
    """SCD cyclic profile, sliced to the rhythm band for a clean line."""
    prof = scd_alpha_profile(y, CFG.sr)                 # 256 bins over 10..2000 Hz
    grid = np.linspace(10.0, 2000.0, len(prof))
    sel = grid <= fmax_hz
    return grid[sel], prof[sel]


# --------------------------------------------------------------------------- #
# averaged EVO-vs-FPV overlay (the version that makes the difference obvious)
# --------------------------------------------------------------------------- #
def _gather_live_files():
    """All EVO / FPV clips from the user's own array sessions (same mic)."""
    evo, fpv = [], []
    for w in RAW.rglob("*.wav"):
        p = str(w).replace("\\", "/").lower()
        if "live_session" not in p:
            continue
        if "fpv" in p:
            fpv.append(w)
        elif "evo" in p:
            evo.append(w)
    return evo, fpv


def _shape_spectrum(y):
    """Per-clip energy-normalised log-mel shape (so it's about WHERE energy is,
    not how loud) + freq axis."""
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=CFG.sr, n_fft=CFG.n_fft,
                                       win_length=CFG.win_length,
                                       hop_length=CFG.hop_length, n_mels=96,
                                       fmin=CFG.fmin, fmax=CFG.fmax, power=2.0)
    v = m.mean(axis=1)
    v = v / (v.sum() + 1e-12)
    hz = librosa.mel_frequencies(n_mels=96, fmin=CFG.fmin, fmax=CFG.fmax)
    return hz, v


def make_overlay(plt, librosa, max_per_class=200):
    evo_f, fpv_f = _gather_live_files()
    specs = {"EVO": [], "FPV": []}
    ratios = {"EVO": [], "FPV": []}
    hz_ref = None
    for label, files in (("EVO", evo_f), ("FPV", fpv_f)):
        for w in files:
            try:
                y, _ = librosa.load(str(w), sr=CFG.sr, mono=True)
            except Exception:
                continue
            n = int(CFG.clip_s * CFG.sr)
            if len(y) < n:
                continue
            starts = np.linspace(0, len(y) - n, 10).astype(int)
            for s in starts:
                clip = y[s:s + n].astype(np.float32)
                hz, v = _shape_spectrum(clip)              # energy-normalised (linear)
                hz_ref = hz
                specs[label].append(v)
                ratios[label].append(float(v[hz > 1500].sum()))   # share of energy >1.5 kHz
            if len(specs[label]) >= max_per_class:
                break

    counts = {k: len(v) for k, v in specs.items()}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # panel A: average spectrum shape in dB (so small high-band gaps are visible)
    ax[0].axvspan(1500, 6000, color="#dc2626", alpha=0.06)
    for label, col in (("EVO", EVO_C), ("FPV", FPV_C)):
        if not specs[label]:
            continue
        mu = np.stack(specs[label]).mean(0)
        mu_db = 10 * np.log10(mu / mu.max() + 1e-9)
        ax[0].plot(hz_ref, mu_db, color=col, lw=2, label=f"{label} (avg {counts[label]} clips)")
    ax[0].axvline(1500, ls="--", color="gray", lw=1)
    ax[0].set_xlim(150, 6000); ax[0].set_ylim(-50, 2)
    ax[0].set_title("AVERAGE energy shape (dB): how fast does it fall with frequency?")
    ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("relative energy (dB)")
    ax[0].text(3500, -8, "high band (>1.5 kHz):\nFPV should sit higher", color=FPV_C, fontsize=9, ha="center")
    ax[0].legend()

    # panel B: per-clip high-band energy ratio -> does it actually separate?
    rng = np.random.default_rng(0)
    for i, (label, col) in enumerate((("EVO", EVO_C), ("FPV", FPV_C))):
        r = np.array(ratios[label])
        x = i + 1 + rng.uniform(-0.12, 0.12, len(r))
        ax[1].scatter(x, r, s=14, alpha=0.45, color=col)
        ax[1].plot([i + 0.7, i + 1.3], [r.mean(), r.mean()], color=col, lw=3)
        ax[1].text(i + 1, -0.02, f"mean {r.mean():.2f}", color=col, ha="center", fontsize=10)
    ax[1].set_xticks([1, 2]); ax[1].set_xticklabels(["EVO", "FPV"])
    ax[1].set_xlim(0.5, 2.5); ax[1].set_ylim(-0.03, max(0.25, ax[1].get_ylim()[1]))
    ax[1].set_title("Share of energy above 1.5 kHz, per clip")
    ax[1].set_ylabel("high-band energy share")
    fig.suptitle("EVO vs FPV averaged over real array clips — honest view: at distance the gap is small\n"
                 "(the classifier still separates them using fine harmonic + rhythm structure, not gross energy)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(ASSETS / "evo_vs_fpv_overlay.png", dpi=120); plt.close(fig)
    print(f"saved -> evo_vs_fpv_overlay.png  (EVO {counts['EVO']} / FPV {counts['FPV']} clips)")


# --------------------------------------------------------------------------- #
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ASSETS.mkdir(exist_ok=True)

    clips, manifest = [], []
    for label, dist, path in EXAMPLES:
        if not path.exists():
            print(f"  ! missing {path}")
            continue
        y, t0 = take_clip(path)
        clips.append((label, dist, path.name, t0, y))
        manifest.append(f"{label:4s} {dist:6s}  t={t0:5.1f}-{t0+CFG.clip_s:4.1f}s  {path.name}")

    if not clips:
        raise SystemExit("no example files found")

    ncol = len(clips)
    fig, axes = plt.subplots(4, ncol, figsize=(3.4 * ncol, 11))
    if ncol == 1:
        axes = axes[:, None]

    for j, (label, dist, name, t0, y) in enumerate(clips):
        col = EVO_C if label == "EVO" else FPV_C
        # row 0: harmonic comb
        hz, v = harmonic_comb(y)
        ax = axes[0, j]; ax.plot(hz, v, color=col, lw=1.1); ax.set_xlim(0, 6000)
        ax.set_title(f"{label}  {dist}\n{name}\nt = {t0:.1f}s", fontsize=8)
        if j == 0: ax.set_ylabel("harmonic_stack\n(comb, dB)", fontsize=9)
        # row 1: scalogram
        db, freqs = scalogram_img(y)
        ax = axes[1, j]
        ax.imshow(db, aspect="auto", origin="lower", cmap="magma",
                  extent=[0, CFG.clip_s, 0, db.shape[0]])
        nt = [0, db.shape[0] // 2, db.shape[0] - 1]
        ax.set_yticks(nt); ax.set_yticklabels([f"{freqs[k]:.0f}" for k in nt], fontsize=7)
        if j == 0: ax.set_ylabel("scalogram\n(CQT, Hz)", fontsize=9)
        # row 2: modulation rhythm (peak-normalised for fair shape comparison)
        f, m = modulation_rhythm(y)
        m = m / (m.max() + 1e-9)
        ax = axes[2, j]; ax.plot(f, m, color=col, lw=1.1)
        ax.set_xlim(0, 220); ax.set_ylim(0, 1.05)
        if j == 0: ax.set_ylabel("modulation\n(rhythm, norm.)", fontsize=9)
        # row 3: scd cyclic (peak-normalised)
        g, p = scd_rhythm(y)
        p = p / (p.max() + 1e-9)
        ax = axes[3, j]; ax.plot(g, p, color=col, lw=1.1)
        ax.set_xlim(0, 220); ax.set_ylim(0, 1.05)
        ax.set_xlabel("cyclic freq (Hz)", fontsize=8)
        if j == 0: ax.set_ylabel("scd\n(cyclic)", fontsize=9)

    fig.suptitle("The 4 classification features on real clips  (blue = EVO low hum,  red = FPV high whine)",
                 fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(ASSETS / "feature_gallery.png", dpi=110); plt.close(fig)

    # ---- where the spectrogram falls down -------------------------------- #
    import librosa
    fpv = next(c for c in clips if c[0] == "FPV")
    y = fpv[4]
    rng = np.random.default_rng(0)
    noisy = (y + 3.0 * np.std(y) * rng.standard_normal(len(y))).astype(np.float32)  # ~0 dB

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for col, (lab, sig) in enumerate([("clean", y), ("+ heavy noise (~0 dB)", noisy)]):
        mel = librosa.power_to_db(librosa.feature.melspectrogram(
            y=sig, sr=CFG.sr, n_fft=CFG.n_fft, win_length=CFG.win_length,
            hop_length=CFG.hop_length,
            n_mels=128, fmin=CFG.fmin, fmax=CFG.fmax), ref=np.max)
        ax[0, col].imshow(mel, aspect="auto", origin="lower", cmap="magma")
        ax[0, col].set_title(f"MEL spectrogram - {lab}", fontsize=10)
        ax[0, col].set_ylabel("mel bin"); ax[0, col].set_xlabel("time")
        g, p = scd_rhythm(sig)
        ax[1, col].plot(g, p, color=FPV_C, lw=1.2)
        ax[1, col].set_title(f"SCD rhythm - {lab}", fontsize=10)
        ax[1, col].set_xlabel("cyclic freq (Hz)"); ax[1, col].set_xlim(0, 220)
    fig.suptitle("Why the spectrogram falls down: mel smears in noise (top-right blurs),\n"
                 "but the SCD rhythm peaks survive (bottom-right keeps its spikes)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(ASSETS / "spectrogram_falldown.png", dpi=110); plt.close(fig)

    # ---- averaged overlay: the CLEAR EVO-vs-FPV difference ---------------- #
    make_overlay(plt, librosa)

    (ASSETS / "feature_gallery_manifest.txt").write_text("\n".join(manifest), encoding="utf-8")
    print("saved -> feature_gallery.png, spectrogram_falldown.png")
    print("\nMANIFEST (file + time-offset behind each example):")
    print("\n".join("  " + m for m in manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
