#!/usr/bin/env python3
"""discrim_band.py - READ-ONLY: which FREQUENCIES separate the drone types?

For each pair it averages each clip's log-mel over time (one value per mel band),
then measures per-band separability between the two classes (|AUC-0.5|).
Plots separability vs frequency, marks the 15cm beamforming ceiling (1143 Hz),
and reports how much of the discriminative power sits below vs above it.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import CFG                                   # noqa: E402
from feature_probe import load_pair                      # noqa: E402

OUT = HERE / "feature_probe_out"
F_BEAM_15CM = 1143.0


def per_clip_logmel(audio):
    import librosa
    m = librosa.feature.melspectrogram(
        y=audio, sr=CFG.sr, n_fft=CFG.n_fft,
        win_length=getattr(CFG, "win_length", CFG.n_fft),
        hop_length=CFG.hop_length,
        n_mels=CFG.F, fmin=CFG.fmin, fmax=CFG.fmax, power=2.0)
    return librosa.power_to_db(m).mean(axis=1)          # (n_mels,)


def auc_per_band(X, y):
    from sklearn.metrics import roc_auc_score
    out = np.zeros(X.shape[1])
    for b in range(X.shape[1]):
        try:
            out[b] = abs(roc_auc_score(y, X[:, b]) - 0.5)
        except Exception:
            out[b] = 0.0
    return out


def main():
    import librosa
    freqs = librosa.mel_frequencies(n_mels=CFG.F, fmin=CFG.fmin, fmax=CFG.fmax)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, pair in zip(axes, ["red_blue", "array"]):
        rows = load_pair(pair)
        cls = np.array([r["cls"] for r in rows])
        classes = sorted(set(cls))
        y = (cls == classes[1]).astype(int)
        X = np.stack([per_clip_logmel(r["audio"]) for r in rows])
        sep = auc_per_band(X, y)
        ax.plot(freqs, sep, color="tab:blue")
        ax.fill_between(freqs, 0, sep, where=freqs <= F_BEAM_15CM,
                        color="tab:green", alpha=0.25,
                        label="<1143 Hz (15cm beamform-clean)")
        ax.fill_between(freqs, 0, sep, where=freqs > F_BEAM_15CM,
                        color="tab:red", alpha=0.20,
                        label=">1143 Hz (15cm aliased)")
        ax.axvline(F_BEAM_15CM, ls="--", color="k", lw=1)
        below = sep[freqs <= F_BEAM_15CM].sum()
        above = sep[freqs > F_BEAM_15CM].sum()
        frac_below = 100 * below / (below + above + 1e-9)
        peak_hz = freqs[int(np.argmax(sep))]
        ax.set_title(f"{pair}  ({classes[0]} vs {classes[1]}): "
                     f"{frac_below:.0f}% of separation is <1143 Hz, "
                     f"peak at {peak_hz:.0f} Hz")
        ax.set_ylabel("per-band separability |AUC-0.5|")
        ax.grid(alpha=0.3); ax.legend(loc="upper right")
        print(f"[{pair}] {classes}: peak={peak_hz:.0f} Hz, "
              f"{frac_below:.0f}% of discriminative power below 1143 Hz")
    axes[-1].set_xlabel("frequency (Hz)")
    axes[-1].set_xscale("log")
    fig.suptitle("Where does the drone-TYPE difference live in frequency?",
                 fontsize=13)
    fig.tight_layout()
    p = OUT / "discrim_band.png"
    fig.savefig(p, dpi=120); print("saved ->", p)


if __name__ == "__main__":
    main()
