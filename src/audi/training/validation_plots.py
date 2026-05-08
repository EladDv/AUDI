"""Matplotlib rendering for validation figures.

Depends on `audi.training.validation` for computation.
Logs figures to TensorBoard via the Lightning logger.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

matplotlib.use("Agg")

from audi.training.validation import (
    compute_calibration,
    compute_det_curve,
    compute_pr_curve,
    compute_precision,
    compute_roc_values,
    find_threshold_at_precision,
)

_BIN_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]
_N_SPEC_SAMPLES = 16


# ── Per-row render helper ─────────────────────────────────────────────


def _make_bin_row(
    fig: plt.Figure,
    grid: plt.GridSpec,
    row: int,
    bin_label: str,
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    auc: float,
    precision: np.ndarray | None = None,
    *,
    color: str,
) -> None:
    """Populate one row: left=Recall+Precision vs threshold, right=ROC."""
    _PRECISION_TARGETS = [0.99, 0.95, 0.90, 0.80]

    # ── left: Recall and Precision vs threshold ────────────────────
    ax1 = fig.add_subplot(grid[row, 0])
    ax1.plot(
        thresholds,
        tpr,
        "--",
        color="#d62728",
        alpha=0.8,
        label="Recall",
        lw=1.0,
    )
    if precision is not None:
        ax1.plot(
            thresholds,
            precision,
            ":",
            color="#2ca02c",
            alpha=0.8,
            label="Precision",
            lw=1.0,
        )
        for pt in _PRECISION_TARGETS:
            if pt < precision.min() or pt > precision.max():
                continue
            th_pt, tp_at_pt, actual_p = find_threshold_at_precision(
                precision, tpr, thresholds, pt
            )
            if th_pt < thresholds.min() or th_pt > thresholds.max():
                continue
            ax1.axvline(
                th_pt, color="#2ca02c", linestyle=":", alpha=0.6, linewidth=0.8
            )
            ax1.annotate(
                f"TPR@{actual_p:.2f}P={tp_at_pt:.3f}",
                xy=(th_pt, tp_at_pt),
                fontsize=7,
                color="#2ca02c",
                rotation=90,
                textcoords="offset points",
                xytext=(3, 0),
                va="bottom",
            )
    ax1.set_xlabel("", fontsize=7)
    ax1.set_ylabel(bin_label, fontsize=7, rotation=0, ha="right", labelpad=18)
    ax1.legend(fontsize=6, loc="lower left", framealpha=0.6)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=6)

    # ── right: ROC ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(grid[row, 1])
    ax2.plot(fpr, tpr, "-", color=color, lw=1.5, label=f"AUC={auc:.3f}")
    ax2.plot([0, 1], [0, 1], ":", color="gray", alpha=0.8, lw=0.8)
    ax2.set_xlabel("FPR" if row == grid.nrows - 1 else "", fontsize=7)
    ax2.set_ylabel("TPR", fontsize=7)
    ax2.legend(fontsize=6, loc="lower right", framealpha=0.6)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=6)


# ── Validation grid ────────────────────────────────────────────────────


def render_validation_grid(
    full_logits: np.ndarray,
    full_labels: np.ndarray,
    per_bin_logits: dict[str, tuple[np.ndarray, np.ndarray]],
    epoch: int,
    logger: object,
    *,
    model_name: str = "model",
    bin_colors: Sequence[str] | None = None,
    bin_order: list[str] | None = None,
) -> None:
    """Build a grid figure and log it to TensorBoard.

    Grid layout (one row per SNR bin + one row for "All"):
      Left column  — Recall + Precision vs threshold.
      Right column — ROC curve.
    """
    if bin_colors is None:
        bin_colors = _BIN_COLORS

    full_fpr, full_tpr, full_th, full_auc = compute_roc_values(
        full_logits, full_labels
    )
    full_precision = compute_precision(full_logits, full_labels, full_th)

    if bin_order:
        bin_names = [b for b in bin_order if b in per_bin_logits]
    else:
        bin_names = sorted(per_bin_logits.keys())
    n_bins = len(bin_names)
    n_rows = n_bins + 1

    fig = plt.figure(figsize=(11, 1.5 * n_rows + 0.3), dpi=120)
    grid = fig.add_gridspec(
        n_rows, 2, wspace=0.08, hspace=0.15, width_ratios=[1, 0.33]
    )

    _make_bin_row(
        fig,
        grid,
        0,
        "All",
        full_fpr,
        full_tpr,
        full_th,
        full_auc,
        precision=full_precision,
        color="#2c3e50",
    )

    for i, bn in enumerate(bin_names):
        bl, ll = per_bin_logits[bn]
        bfpr, btpr, bth, bauc = compute_roc_values(bl, ll)
        bprec = compute_precision(bl, ll, bth)
        color = bin_colors[i % len(bin_colors)]
        _make_bin_row(
            fig,
            grid,
            i + 1,
            bn,
            bfpr,
            btpr,
            bth,
            bauc,
            precision=bprec,
            color=color,
        )

    fig.suptitle(
        f"Epoch {epoch}  —  {model_name}",
        fontsize=10,
        fontweight="bold",
        y=0.995,
    )
    if (
        logger is not None
        and hasattr(logger, "experiment")
        and callable(getattr(logger.experiment, "add_figure", None))
    ):
        logger.experiment.add_figure(
            f"validation_roc/{model_name}", fig, global_step=epoch
        )
    fig.subplots_adjust(top=0.985, bottom=-0.03, left=-0.03, right=1.03)
    plt.close(fig)


# ── DET / PR / Calibration ─────────────────────────────────────────────


def render_det_pr_calibration(
    full_logits: np.ndarray,
    full_labels: np.ndarray,
    per_bin: dict[str, tuple[np.ndarray, np.ndarray]],
    epoch: int,
    logger: object,
    *,
    model_name: str = "model",
    bin_colors: Sequence[str] | None = None,
    bin_order: list[str] | None = None,
) -> None:
    """Build a 3-panel figure (DET, Precision-Recall, Calibration) and log to TB."""
    from scipy.special import ndtri

    if bin_colors is None:
        bin_colors = _BIN_COLORS
    if bin_order:
        bin_names = [b for b in bin_order if b in per_bin]
    else:
        bin_names = sorted(per_bin.keys())

    full_fpr, full_tpr, _th, full_auc = compute_roc_values(
        full_logits, full_labels
    )
    full_det_x, full_det_y = compute_det_curve(full_fpr, full_tpr)
    full_pr, full_rec, _th, full_ap = compute_pr_curve(full_logits, full_labels)
    exp_p, obs_p, counts, ece = compute_calibration(full_logits, full_labels)

    fig = plt.figure(figsize=(18, 5.2), dpi=120)

    # ── DET ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(
        full_det_x,
        full_det_y,
        "-",
        color="#2c3e50",
        lw=1.5,
        label=f"All AUC={full_auc:.3f}",
    )
    for i, bn in enumerate(bin_names):
        bl, bll = per_bin[bn]
        bfpr, btpr, _bth, bauc = compute_roc_values(bl, bll)
        bx, by = compute_det_curve(bfpr, btpr)
        color = bin_colors[i % len(bin_colors)]
        ax1.plot(
            bx,
            by,
            "-",
            color=color,
            lw=1.0,
            alpha=0.8,
            label=f"{bn} AUC={bauc:.3f}",
        )
    for pct in [0.01, 0.05, 0.10, 0.20]:
        x = ndtri(pct)
        ax1.axvline(x, color="gray", linestyle=":", alpha=0.3, lw=0.8)
        ax1.axhline(x, color="gray", linestyle=":", alpha=0.3, lw=0.8)
    ax1.set_xlabel("FPR (probit)", fontsize=7)
    ax1.set_ylabel("FNR / Miss Rate (probit)", fontsize=7)
    ax1.set_title("DET Curve", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=5, loc="lower left", framealpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=6)

    # ── Precision-Recall ──────────────────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(
        full_rec,
        full_pr,
        "-",
        color="#2c3e50",
        lw=1.5,
        label=f"All AP={full_ap:.3f}",
    )
    baseline = full_labels.mean() if len(full_labels) else 0.5
    ax2.axhline(baseline, color="gray", linestyle=":", alpha=0.5, lw=0.8)
    for i, bn in enumerate(bin_names):
        bl, bll = per_bin[bn]
        bpr, brec, _bth, bap = compute_pr_curve(bl, bll)
        color = bin_colors[i % len(bin_colors)]
        ax2.plot(
            brec,
            bpr,
            "-",
            color=color,
            lw=1.0,
            alpha=0.8,
            label=f"{bn} AP={bap:.3f}",
        )
    ax2.set_xlabel("Recall", fontsize=7)
    ax2.set_ylabel("Precision", fontsize=7)
    ax2.set_title("Precision-Recall", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=5, loc="lower left", framealpha=0.5)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=6)

    # ── Calibration ───────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    if len(exp_p) > 0:
        ax3.bar(
            np.arange(len(exp_p)),
            counts / max(counts.max(), 1),
            width=0.8,
            color="lightgray",
            alpha=0.6,
            label="sample density",
        )
        ax3.plot(
            np.arange(len(exp_p)),
            exp_p,
            "s-",
            color="#2c3e50",
            ms=6,
            lw=1.5,
            label="expected",
        )
        ax3.plot(
            np.arange(len(exp_p)),
            obs_p,
            "o-",
            color="#d62728",
            ms=5,
            lw=1.5,
            label="observed",
        )
    ax3.set_xlabel("Probability bin", fontsize=7)
    ax3.set_ylabel("Probability", fontsize=7)
    ax3.set_title(f"Calibration (ECE={ece:.4f})", fontsize=9, fontweight="bold")
    ax3.legend(fontsize=6, loc="upper left", framealpha=0.5)
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.tick_params(labelsize=6)

    fig.suptitle(
        f"Epoch {epoch}  —  {model_name}",
        fontsize=10,
        fontweight="bold",
        y=0.995,
    )
    if (
        logger is not None
        and hasattr(logger, "experiment")
        and callable(getattr(logger.experiment, "add_figure", None))
    ):
        logger.experiment.add_figure(
            f"validation_det_pr_cal/{model_name}", fig, global_step=epoch
        )
    fig.subplots_adjust(
        top=0.90, bottom=0.12, left=0.06, right=0.98, wspace=0.25
    )
    plt.close(fig)


# ── Spectrogram helpers ─────────────────────────────────────────────────


def _log_mel_spec(
    waveform: torch.Tensor,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 160,
) -> np.ndarray:
    """Compute log-mel spectrogram from a ``[T]`` or ``[1, T]`` waveform."""
    from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel_t = MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    to_db = AmplitudeToDB()
    with torch.no_grad():
        spec = to_db(mel_t(waveform))
    return spec.squeeze(0).cpu().numpy()


def render_spectrogram_samples(
    mix_waveforms: torch.Tensor,
    drone_waveforms: torch.Tensor,
    noise_waveforms: torch.Tensor,
    snr_values: np.ndarray,
    logits: np.ndarray,
    epoch: int,
    logger: object,
    *,
    model_name: str = "model",
    sample_rate: int = 16000,
    max_samples: int = _N_SPEC_SAMPLES,
    output_dir: str | None = None,
) -> None:
    """Render a grid of spectrograms and log audio clips to TensorBoard.

    Grid layout: Noise | Drone (SNR) | Mix (pred=...) | cbar
    """
    n = min(len(mix_waveforms), max_samples)
    probs = 1.0 / (1.0 + np.exp(-logits[:n]))

    fig = plt.figure(figsize=(14, 1.1 * n + 0.3), dpi=120)
    grid = fig.add_gridspec(
        n, 4, hspace=0.08, wspace=0.08, width_ratios=[1, 1, 1, 0.03]
    )

    for i in range(n):
        noise_spec = _log_mel_spec(noise_waveforms[i])
        drone_spec = _log_mel_spec(drone_waveforms[i])
        mix_spec = _log_mel_spec(mix_waveforms[i])

        colormap = "magma"
        extent = [0, mix_spec.shape[1], 0, mix_spec.shape[0]]
        vmin = min(noise_spec.min(), drone_spec.min(), mix_spec.min())
        vmax = max(noise_spec.max(), drone_spec.max(), mix_spec.max())
        im_kw = dict(
            aspect="auto",
            origin="lower",
            cmap=colormap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )

        panels = [
            (0, "Noise", noise_spec),
            (1, f"Drone {snr_values[i]:+.0f}dB", drone_spec),
            (2, f"Mix p={probs[i]:.2f}", mix_spec),
        ]
        for col, title, data in panels:
            ax = fig.add_subplot(grid[i, col])
            im = ax.imshow(data, **im_kw)
            ax.text(
                0.5,
                0.99,
                title,
                transform=ax.transAxes,
                ha="center",
                fontsize=7,
                va="top",
                bbox=dict(
                    facecolor="white", alpha=0.75, pad=0, edgecolor="none"
                ),
            )
            ax.tick_params(labelsize=5, pad=1)
            if i < n - 1:
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        cax = fig.add_subplot(grid[i, 3])
        fig.colorbar(im, cax=cax)
        cax.tick_params(labelsize=4, pad=1)

    fig.suptitle(
        f"Epoch {epoch}  —  {model_name}",
        fontsize=10,
        fontweight="bold",
        y=0.995,
    )
    if (
        logger is not None
        and hasattr(logger, "experiment")
        and callable(getattr(logger.experiment, "add_figure", None))
    ):
        logger.experiment.add_figure(
            f"validation_samples/{model_name}", fig, global_step=epoch
        )
    fig.subplots_adjust(top=0.985, bottom=-0.03, left=-0.03, right=1.03)

    if output_dir:
        import soundfile as sf

        epoch_dir = Path(output_dir) / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            epoch_dir / "spectrograms.png", dpi=120, bbox_inches="tight"
        )
        for j in range(n):
            snr_str = f"{snr_values[j]:+.1f}".replace(".", "p").replace(
                "-", "neg"
            )
            pred_str = f"{probs[j]:.3f}".replace(".", "p")
            stem = f"sample_{j}__{snr_str}_{pred_str}"
            sr_float = float(sample_rate)

            def _to_wav(wf: torch.Tensor) -> np.ndarray:
                arr = wf.numpy()
                peak = max(np.abs(arr).max(), 1e-12)
                return arr * (0.99 / peak) if peak > 0.99 else arr

            sf.write(
                epoch_dir / f"{stem}_noise.wav",
                _to_wav(noise_waveforms[j]),
                int(sr_float),
            )
            sf.write(
                epoch_dir / f"{stem}_drone.wav",
                _to_wav(drone_waveforms[j]),
                int(sr_float),
            )
            sf.write(
                epoch_dir / f"{stem}_mix.wav",
                _to_wav(mix_waveforms[j]),
                int(sr_float),
            )

    plt.close(fig)

    _log_audio_clips(
        mix_waveforms,
        drone_waveforms,
        noise_waveforms,
        snr_values,
        probs,
        epoch,
        logger,
        model_name=model_name,
        sample_rate=sample_rate,
    )


def _log_audio_clips(
    mix_waveforms: torch.Tensor,
    drone_waveforms: torch.Tensor,
    noise_waveforms: torch.Tensor,
    snr_values: np.ndarray,
    probs: np.ndarray,
    epoch: int,
    logger: object,
    *,
    model_name: str = "model",
    sample_rate: int = 16000,
    max_clips: int = 8,
) -> None:
    """Log audio clips for the first *max_clips* samples."""
    if (
        logger is None
        or not hasattr(logger, "experiment")
        or not callable(getattr(logger.experiment, "add_audio", None))
    ):
        return

    n = min(len(mix_waveforms), max_clips)
    for i in range(n):
        tag_base = f"validation_clips_sample_{i}"

        def _safe(wf: torch.Tensor) -> torch.Tensor:
            peak = wf.abs().max().clamp(min=1e-12)
            return wf * (0.99 / peak) if peak > 0.99 else wf

        logger.experiment.add_audio(
            f"{tag_base}/noise",
            _safe(noise_waveforms[i]).unsqueeze(0),
            global_step=epoch,
            sample_rate=sample_rate,
        )
        logger.experiment.add_audio(
            f"{tag_base}/drone",
            _safe(drone_waveforms[i]).unsqueeze(0),
            global_step=epoch,
            sample_rate=sample_rate,
        )
        logger.experiment.add_audio(
            f"{tag_base}/mix",
            _safe(mix_waveforms[i]).unsqueeze(0),
            global_step=epoch,
            sample_rate=sample_rate,
        )
