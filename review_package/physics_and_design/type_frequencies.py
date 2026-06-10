#!/usr/bin/env python3
"""type_frequencies.py - READ-ONLY: at what frequencies does each drone operate?

For each drone type/size it averages the magnitude spectrum across all its files,
then reports the dominant tonal peaks (fundamental + strong harmonics) and the
band that holds most of the energy. Also overlays the mean spectra on one chart.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000
N_FFT = 4096
WIN_LENGTH = 4096
ROOT = Path(r"c:\Users\Nir\Desktop\drone_nir")
RAW = ROOT / "data_other" / "raw"
ARR = ROOT / "06_model_v1" / "data" / "raw"
OUT = ROOT / "06_model_v1" / "training" / "feature_probe_out" / "type_frequencies.png"

GROUPS = {
    'FPV 5"':  [RAW / "dataset_v2/dataset_v2/5inchP"],
    'FPV 7"':  [RAW / "dataset_v2/dataset_v2/7inchN"],
    'FPV 10"': [RAW / "dataset_v2/dataset_v2/10inchP"],
    'FPV 13"': [RAW / "dataset_v2/dataset_v2/13inchP"],
    'red_13 (13" probe)': [RAW / "data_clean_unit_06052026/target_drone"],
    'blue_7 (7" probe)':  [RAW / "data_clean_unit_06052026/other_drones"],
    'EVO (array)': [ARR / "live_session_20260528", ARR / "live_session_20260601"],
    'FPV (array)': [ARR / "live_session_20260528", ARR / "live_session_20260601"],
}


def collect(name, dirs):
    wavs = []
    for d in dirs:
        if not d.exists():
            continue
        for w in d.rglob("*.wav"):
            p = str(w).lower()
            if name == "blue_7 (7\" probe)" and not w.name.lower().startswith("blue_7"):
                continue
            if name == "red_13 (13\" probe)" and not w.name.lower().startswith("red_13"):
                continue
            if name == "EVO (array)" and "evo" not in p:
                continue
            if name == "FPV (array)" and "fpv" not in p:
                continue
            wavs.append(w)
    return wavs


def mean_spectrum(wavs):
    import librosa
    acc = None; n = 0
    for w in wavs:
        y, _ = librosa.load(str(w), sr=SR, mono=True)
        S = np.abs(librosa.stft(
            y,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=N_FFT // 2,
        ))
        m = S.mean(axis=1)
        acc = m if acc is None else acc + m
        n += 1
    if n == 0:
        return None
    spec = acc / n
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    return freqs, spec


def peaks(freqs, spec):
    from scipy.signal import find_peaks
    sdb = 20 * np.log10(spec + 1e-9)
    sdb = sdb - sdb.max()
    band = (freqs >= 80) & (freqs <= 7500)
    pk, props = find_peaks(sdb[band], height=-25, distance=5)
    fb = freqs[band][pk]; hb = sdb[band][pk]
    order = np.argsort(hb)[::-1][:5]
    top = sorted(fb[order])
    # energy band: where cumulative power covers 10-90%
    p = spec[band] ** 2
    c = np.cumsum(p) / p.sum()
    lo = freqs[band][np.searchsorted(c, 0.10)]
    hi = freqs[band][np.searchsorted(c, 0.90)]
    centroid = (freqs[band] * p).sum() / p.sum()
    return top, (lo, hi), centroid


def main():
    fig, ax = plt.subplots(figsize=(11, 6))
    print(f"{'type':22s} {'#':>3}  {'energy band (Hz)':>18}  {'centroid':>9}  top peaks (Hz)")
    for name, dirs in GROUPS.items():
        wavs = collect(name, dirs)
        res = mean_spectrum(wavs)
        if res is None:
            print(f"{name:22s}   0   (no files)")
            continue
        freqs, spec = res
        top, (lo, hi), cen = peaks(freqs, spec)
        sdb = 20 * np.log10(spec + 1e-9); sdb -= sdb.max()
        ax.plot(freqs, sdb, label=f"{name} (n={len(wavs)})", alpha=0.8)
        print(f"{name:22s} {len(wavs):>3}  {lo:7.0f} - {hi:<7.0f}  {cen:7.0f} Hz  "
              + ", ".join(f"{f:.0f}" for f in top))
    ax.axvline(1143, ls="--", color="k", lw=1, label="15cm beamform ceiling (1143 Hz)")
    ax.set_xscale("log"); ax.set_xlim(80, 8000); ax.set_ylim(-45, 2)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("mean spectrum (dB, normalised)")
    ax.set_title("Operating frequencies per drone type/size")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT, dpi=120)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
