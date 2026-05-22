#!/usr/bin/env python3
"""Compare PCEN vs standard mel-dB preprocessing on real audio.

Shows the full DroneDetector preprocessing pipeline for both paths,
plus DSP feature extraction output. Useful for tuning PCEN params.

Usage:
    uv run python scripts/compare_pcen_mel.py [wav_path]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
import librosa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audi.config import MelConfig
from audi.training.detector import PCEN
from audi.dsp_features import DSPFeatureExtractor, DSPFeatureConfig


SR = 16000
N_FFT = 1024
HOP_LEN = 256  # match DSP hop for clean comparison
N_MELS = 128

# PCEN params to test
PCEN_CONFIGS = {
    "default": {"s": 0.025, "alpha": 0.98, "delta": 2.0, "r": 0.5},
    "fast_adapt": {"s": 0.05, "alpha": 0.80, "delta": 2.0, "r": 0.5},
    "slow_adapt": {"s": 0.01, "alpha": 0.95, "delta": 2.0, "r": 0.5},
    "high_floor": {"s": 0.025, "alpha": 0.98, "delta": 5.0, "r": 0.33},
    "aggressive": {"s": 0.025, "alpha": 0.70, "delta": 3.0, "r": 0.33},
}

# Independent librosa PCEN configs (different param space)
LIBROSA_PCEN_CONFIGS = {
    "librosa_default": {
        "alpha": 0.98,
        "delta": 2.0,
        "r": 0.5,
        "max_size": 1,
    },
    "librosa_size3": {
        "alpha": 0.98,
        "delta": 2.0,
        "r": 0.5,
        "max_size": 3,
    },
    "librosa_slower": {
        "alpha": 0.98,
        "delta": 2.0,
        "r": 0.5,
        "max_size": 1,
    },
    "librosa_faster": {
        "alpha": 0.80,
        "delta": 2.0,
        "r": 0.5,
        "max_size": 1,
    },
    "librosa_high_delta": {
        "alpha": 0.98,
        "delta": 10.0,
        "r": 0.5,
        "max_size": 1,
    },
    "librosa_strong_agc": {
        "alpha": 0.50,
        "delta": 2.0,
        "r": 0.33,
        "max_size": 1,
    },
    "librosa_linear": {
        "alpha": 1.0,
        "delta": 1.0,
        "r": 1.0,
        "max_size": 1,
    },
}


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import scipy.signal

        audio = scipy.signal.resample(audio, int(len(audio) * SR / sr))
    return audio.astype(np.float32)


def mel_db_pipeline(audio: np.ndarray) -> np.ndarray:
    """Standard pipeline: mel → AmplitudeToDB → scalar normalization."""
    mel_cfg = MelConfig(n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LEN)
    mel_transform = T.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        n_mels=N_MELS,
    )
    wav = torch.from_numpy(audio).unsqueeze(0)
    mel = mel_transform(wav)  # [1, n_mels, T]
    mel_db = T.AmplitudeToDB()(mel)
    if mel_cfg.mean_db is not None and mel_cfg.std_db is not None:
        mel_db = (mel_db - mel_cfg.mean_db) / mel_cfg.std_db
    return mel_db.squeeze(0).numpy()


def pcen_pipeline(audio: np.ndarray, pcen_cfg: dict) -> np.ndarray:
    """PCEN pipeline: mel → PCEN."""
    mel_transform = T.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        n_mels=N_MELS,
    )
    pcen = PCEN(**pcen_cfg)
    wav = torch.from_numpy(audio).unsqueeze(0)
    mel = mel_transform(wav)
    out = pcen(mel)
    return out.squeeze(0).numpy()


def librosa_pcen_pipeline(audio: np.ndarray, pcen_cfg: dict) -> np.ndarray:
    """librosa reference PCEN: mel power → pcen."""
    # Mel power spectrogram (numpy)

    S = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        n_mels=N_MELS,
        power=1,
    )
    # librosa.pcen with matching params
    return librosa.pcen(
        S * (2**31),
        sr=SR,
        hop_length=HOP_LEN,
        gain=pcen_cfg["alpha"],
        bias=pcen_cfg["delta"],
        power=pcen_cfg["r"],
        max_size=pcen_cfg["max_size"],
    )


def dsp_features(audio: np.ndarray) -> dict[str, np.ndarray]:
    cfg = DSPFeatureConfig(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        f0_min=125,
        f0_max=350,
        n_harmonics=12,
        noise_beta=0.0001,
        stack_alpha=0.50,
        enable_v3=True,
        enable_v4=False,
        enable_v5=True,
    )
    ext = DSPFeatureExtractor(cfg)
    feats = ext.extract(audio)
    return {name: feats[:, i] for i, name in enumerate(ext.feature_names)}


def main():
    # Pick audio
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    else:
        wav_path = str(ROOT / "data/attack_runs/run_number_1-03.wav")

    if not Path(wav_path).exists():
        print(f"Not found: {wav_path}")
        return 1

    audio = load_audio(wav_path)
    print(f"Audio: {len(audio) / SR:.1f}s @ {SR}Hz  ({Path(wav_path).name})")

    # ── Compute all pipelines ──
    mel_db = mel_db_pipeline(audio)
    pcen_specs = {
        name: pcen_pipeline(audio, cfg) for name, cfg in PCEN_CONFIGS.items()
    }
    librosa_specs = {
        name: librosa_pcen_pipeline(audio, cfg)
        for name, cfg in LIBROSA_PCEN_CONFIGS.items()
    }
    dsp = dsp_features(audio)

    # ── Plot ──
    n_pcen = len(PCEN_CONFIGS)
    n_librosa = len(librosa_specs)
    n_rows = (
        2 + n_librosa + n_pcen + len(dsp)
    )  # mel-dB + PCENs + librosa PCENs + DSP
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(18, 3 * n_rows),
        facecolor="#0e1117",
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # Helper
    def plot_spec(ax, spec, title, vmin=None, vmax=None):
        ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
        ax.set_title(title, color="#ccc", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mel bin", color="#888", fontsize=8)
        ax.tick_params(colors="#666", labelsize=7)

    def plot_hist(ax, values, title, color="#4ECDC4"):
        ax.hist(
            values.flatten(), bins=80, color=color, alpha=0.8, edgecolor=None
        )
        ax.set_title(title, color="#ccc", fontsize=9)
        ax.tick_params(colors="#666", labelsize=7)
        ax.axvline(0, color="#fff", linewidth=0.5, alpha=0.3)

    row = 0

    # ── Row 1: Standard mel-dB ──
    plot_spec(axes[row, 0], mel_db, "Standard: mel → dB → scalar norm")
    plot_hist(axes[row, 1], mel_db, "Value distribution")
    row += 1

    # ── Rows 2..: PCEN variants ──
    for name, spec in pcen_specs.items():
        cfg = PCEN_CONFIGS[name]
        label = f"PCEN (ours)  name={name}  s={cfg['s']} α={cfg['alpha']} δ={cfg['delta']} r={cfg['r']}"
        plot_spec(axes[row, 0], spec, label, vmin=None, vmax=None)
        plot_hist(axes[row, 1], spec, label)
        row += 1

    # ── Rows: librosa PCEN (reference) ──
    for name, spec in librosa_specs.items():
        cfg = LIBROSA_PCEN_CONFIGS[name]
        label = f"librosa.pcen  name={name}  α={cfg['alpha']} δ={cfg['delta']} r={cfg['r']} max_size={cfg['max_size']}"
        plot_spec(axes[row, 0], spec, label, vmin=None, vmax=None)
        plot_hist(axes[row, 1], spec, label)
        row += 1

    # ── Row after PCEN: DSP features ──
    for feat_name, values in dsp.items():
        if row >= len(axes):
            break
        axes[row, 0].plot(values, color="#4ECDC4", linewidth=0.8)
        axes[row, 0].set_title(
            f"DSP: {feat_name}", color="#ccc", fontsize=10, fontweight="bold"
        )
        axes[row, 0].set_ylabel("Value", color="#888", fontsize=8)
        axes[row, 0].tick_params(colors="#666", labelsize=7)
        axes[row, 0].set_facecolor("#0e1117")

        axes[row, 1].hist(values, bins=60, color="#FF6B6B", alpha=0.8)
        axes[row, 1].set_title(f"{feat_name} dist", color="#ccc", fontsize=9)
        axes[row, 1].tick_params(colors="#666", labelsize=7)
        row += 1

    # Hide unused axes
    for r in range(row, len(axes)):
        for c in range(2):
            axes[r, c].set_visible(False)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#0e1117")
            for spine in ax.spines.values():
                spine.set_color("#333")

    fig.suptitle(
        f"PCEN (ours + librosa) vs Mel-dB  |  {Path(wav_path).name}",
        color="#ccc",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(pad=1.5)

    out = ROOT / "data/pcen_comparison.png"
    fig.savefig(str(out), dpi=150, facecolor="#0e1117", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
