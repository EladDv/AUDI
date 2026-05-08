"""cmd_analyze_snr data-building subcommand."""
from __future__ import annotations

from pathlib import Path


def run() -> None:
    import json
    
    import matplotlib
    
    matplotlib.use("Agg")
    import argparse
    
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np
    
    _EPS = 1e-12
    _SR = 16000
    _N_FFT = 1024
    _HOP = 512
    _ACTIVE_THRESH = 0.01
    _N_BANDS = 32
    _F_MIN, _F_MAX = 50.0, 8000.0
    
    def _build_filters():
        def erb(f):
            return 21.4 * np.log10(1.0 + 0.00437 * f)
    
        def erb_inv(e):
            return (10.0 ** (e / 21.4) - 1.0) / 0.00437
    
        edges = np.array(
            [
                erb_inv(e)
                for e in np.linspace(erb(_F_MIN), erb(_F_MAX), _N_BANDS + 2)
            ],
            dtype=np.float32,
        )
        n_bins = _N_FFT // 2 + 1
        freqs = np.linspace(0.0, _SR / 2.0, n_bins, dtype=np.float32)
        flt = np.zeros((_N_BANDS, n_bins), dtype=np.float32)
        for i in range(_N_BANDS):
            lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
            flt[i] = np.clip(
                np.minimum(
                    (freqs - lo) / (mid - lo + _EPS),
                    (hi - freqs) / (hi - mid + _EPS),
                ),
                0.0,
                1.0,
            )
        return flt, edges, freqs
    
    _FILTERS, _EDGES, _FREQS = _build_filters()
    _WINDOW = np.hanning(_N_FFT).astype(np.float32)
    _ERB_CENTERS = ((_EDGES[:-2] + _EDGES[1:-1]) / 2).astype(np.float32)
    
    def _band_power(y):
        arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if arr.size < _N_FFT:
            arr = np.pad(arr, (0, _N_FFT - arr.size))
        frames = [
            np.abs(np.fft.rfft(arr[s : s + _N_FFT] * _WINDOW)) ** 2
            for s in range(0, len(arr) - _N_FFT, _HOP)
        ]
        if not frames:
            return np.zeros(_N_BANDS, dtype=np.float32)
        return np.mean(
            np.dot(np.array(frames, dtype=np.float32), _FILTERS.T), axis=0
        )
    
    ap = argparse.ArgumentParser(
        description="Analyze SNR with per-band metrics."
    )
    ap.add_argument(
        "--drone-path",
        type=Path,
        default=Path("data/drone"),
        help="Drone audio files",
    )
    ap.add_argument(
        "--noise-path",
        type=Path,
        default=Path("data/noise"),
        help="Noise audio files",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/snr_analysis")
    )
    ap.add_argument(
        "--drone-gain-db",
        type=float,
        default=None,
        nargs="*",
        help="Drone gain in dB (e.g. 0 -10 -20). Default: auto from SNR targets",
    )
    args = ap.parse_args()
    
    drone_files = sorted(args.drone_path.glob("*.wav"))
    noise_files = sorted(args.noise_path.glob("*.wav"))
    print(f"Drone files: {len(drone_files)}, Noise files: {len(noise_files)}")
    print(f"ERB bands: {_N_BANDS} ({_F_MIN:.0f}–{_F_MAX:.0f} Hz)")
    
    drone_powers, noise_powers = [], []
    for fp in drone_files:
        y, _ = librosa.load(str(fp), sr=_SR, mono=True)
        bp = _band_power(y)
        drone_powers.append(bp)
        print(f"  Drone: {fp.name}: max band power = {bp.max():.2e}")
    for fp in noise_files:
        y, _ = librosa.load(str(fp), sr=_SR, mono=True)
        bp = _band_power(y)
        noise_powers.append(bp)
    drone_mean = (
        np.mean(drone_powers, axis=0) if drone_powers else np.ones(_N_BANDS)
    )
    noise_mean = (
        np.mean(noise_powers, axis=0)
        if noise_powers
        else np.ones(_N_BANDS) * 1e-12
    )
    snr_linear = drone_mean / np.maximum(noise_mean, _EPS)
    snr_db = 10.0 * np.log10(np.maximum(snr_linear, _EPS))
    print(f"\n{'=' * 50}")
    print("  ERB-Band SNR (Drone / Noise)")
    print(f"{'=' * 50}")
    print(
        f"  {'Band':>5}  {'Center':>7}  {'Drone':>10}  {'Noise':>10}  {'SNR(dB)':>8}"
    )
    for i in range(_N_BANDS):
        print(
            f"  {i:>5}  {_ERB_CENTERS[i]:>7.0f}  {drone_mean[i]:>10.2e}"
            f"  {noise_mean[i]:>10.2e}  {snr_db[i]:>8.2f}"
        )
    print(
        f"  {'Average':>5}  {'':>7}  {'':>10}  {'':>10}  {snr_db.mean():>8.2f}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "snr_analysis.json", "w") as f:
        json.dump(
            {
                "band_centers": _ERB_CENTERS.tolist(),
                "band_edges": _EDGES.tolist(),
                "drone_mean_power": drone_mean.tolist(),
                "noise_mean_power": noise_mean.tolist(),
                "snr_db": snr_db.tolist(),
                "overall_snr_db": float(snr_db.mean()),
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {args.output_dir / 'snr_analysis.json'}")
    
    plt.figure(figsize=(12, 4))
    plt.bar(
        range(_N_BANDS),
        snr_db,
        width=0.8,
        color=["#27ae60" if v > 0 else "#e74c3c" for v in snr_db],
    )
    plt.axhline(y=0, color="gray", ls="-", lw=0.5)
    plt.xticks(
        range(0, _N_BANDS, 4),
        [f"{_ERB_CENTERS[i]:.0f}" for i in range(0, _N_BANDS, 4)],
    )
    plt.xlabel("ERB-Band Center Frequency (Hz)")
    plt.ylabel("SNR (dB)")
    plt.title("Per-ERB-Band SNR — Drone / Noise")
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(args.output_dir / "snr_per_band.png", dpi=150)
    print(f"Plot: {args.output_dir / 'snr_per_band.png'}")
    
    
    # ====================================================================
    # mel-stats — Compute global mel-spectrogram mean and std
    # ====================================================================
    
    
