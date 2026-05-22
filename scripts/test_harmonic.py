#!/usr/bin/env python3
"""Quick test: run harmonic folding detector on a field alert WAV."""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audi.harmonic_detector import HarmonicFoldingDetector, HarmonicConfig


def main():
    # Pick first alert WAV
    wav = ROOT / "data/field_recordings_20260514/alerts/yes_1778715244/full_120s.wav"
    if not wav.exists():
        print(f"Not found: {wav}")
        return 1

    audio, sr = sf.read(str(wav))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Run detector (native sample rate, no resample needed)
    cfg = HarmonicConfig(sample_rate=sr, n_fft=1024, hop_length=256,
                         f0_min=125, f0_max=350,
                         n_harmonics=12, noise_beta=0.01, stack_alpha=0.15)
    det = HarmonicFoldingDetector(cfg)

    t0 = time.time()
    raw, stacked, times = det.process(audio)
    elapsed = time.time() - t0
    print(f"Processed in {elapsed:.2f}s  ({len(times)} frames, hop={cfg.hop_length/cfg.sample_rate:.3f}s)")

    print(f"Raw score:     mean={raw.mean():.4f}  max={raw.max():.4f}")
    print(f"Stacked score: mean={stacked.mean():.4f}  max={stacked.max():.4f}")
    print(f"Threshold {cfg.threshold}: {int((stacked > cfg.threshold).sum())} / {len(stacked)} frames ({100*(stacked>cfg.threshold).sum()/len(stacked):.1f}%)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), facecolor="#0e1117",
                                    gridspec_kw={"height_ratios": [1, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#888")
        for s in ax.spines.values():
            s.set_color("#333")

    ax1.plot(times, raw, color="#4ECDC4", linewidth=0.8, label="Raw harmonic score")
    ax1.set_ylabel("Raw score", color="#ccc")
    ax1.set_title(f"Harmonic folding detector — wd_003 max=0.9568 alert", color="#ccc")
    ax1.legend(loc="upper right", facecolor="#222", edgecolor="#333", labelcolor="#ccc")

    ax2.plot(times, stacked, color="#FF6B6B", linewidth=1.0, label="Stacked score")
    ax2.axhline(cfg.threshold, color="#fff", linestyle="--", linewidth=0.6, alpha=0.4,
                label=f"Threshold={cfg.threshold}")
    ax2.set_xlabel("Time (s)", color="#ccc")
    ax2.set_ylabel("Stacked score", color="#ccc")
    ax2.legend(loc="upper right", facecolor="#222", edgecolor="#333", labelcolor="#ccc")
    ax2.grid(True, alpha=0.15, color="#444")

    fig.tight_layout(pad=1.2)
    out = ROOT / "data/harmonic_test.png"
    fig.savefig(str(out), dpi=140, facecolor="#0e1117")
    plt.close(fig)
    print(f"\nSaved plot: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
