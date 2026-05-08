"""Drone Detection Explorer — sweep model switcher + eval-calibrated FPR/TPR.

Scans phase3_full_* directory for models. Loads eval_data/*.pt + curves_*.npz
for proper FPR/TPR calibration. Interactive threshold slider.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 160
CHUNK_S = 2.56
CHUNK_N = int(CHUNK_S * SR)
OVERLAP = 0.75
HOP_N = int(CHUNK_N * (1 - OVERLAP))

TEST_FILES = {
    "200m attack run": "data/200m_attackrun.wav",
    "200m attack #2": "data/200m_attack_2.wav",
    "500m attack": "data/500m_attack.wav",
    "150m constant": "data/150m_constant.wav",
}
BG_FILE = "data/background.wav"

# ── Discover sweep directories ──────────────────────────────────────
SWEEP_GLOB = "checkpoints_v2/phase3_full_*"
sweep_dirs = sorted([d for d in ROOT.glob(SWEEP_GLOB) if d.is_dir()])


def discover_models():
    """Scan ALL phase3_full_* dirs for models with checkpoints."""
    models = {}
    for sweep_dir in sweep_dirs:
        if not sweep_dir.is_dir():
            continue
        for d in sorted(sweep_dir.iterdir()):
            if not d.is_dir():
                continue
            ckpts = sorted(
                d.glob("lightning_logs/version_*/checkpoints/*.ckpt")
            )
            eval_dir = d / "eval_data"
            if ckpts:
                # Use display name with sweep timestamp prefix
                display = f"{sweep_dir.name}/{d.name}"
                models[display] = {
                    "dir": d,
                    "checkpoints": ckpts,
                    "eval_dir": eval_dir,
                }
    return models


# ── Model loading (cached per checkpoint) ───────────────────────────
@st.cache_resource
def load_model(ckpt_path: str):
    sys.path.insert(0, str(ROOT / "src"))
    from audi.config import MelConfig, ModelConfig
    from audi.training.detector import DroneDetector

    sd = torch.load(ckpt_path, map_location="cpu")
    hp = sd["hyper_parameters"]
    model = DroneDetector(
        model=ModelConfig(
            arch=hp.get("model_arch", "cnn14"),
            pretrained=hp.get("pretrained_backbone", True),
            compile=hp.get("use_compile", True),
        ),
        mel=MelConfig(
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP,
            mean_db=hp.get("mel_mean"),
            std_db=hp.get("mel_std"),
        ),
        bin_names=hp.get("bin_names", []),
        loss_type=hp.get("loss_type", "bce"),
        spec_augment_prob=0.0,
        bn_momentum=hp.get("bn_momentum", 0.1),
    )
    model.load_state_dict(sd["state_dict"], strict=False)
    model.eval()
    return model, hp.get("model_arch", "?"), hp.get("loss_type", "?")


@torch.no_grad()
def _predict(chunk, model):
    r = float(np.sqrt(np.mean(chunk**2) + 1e-8))
    if r < 1e-8:
        return 0.0
    x = torch.from_numpy((chunk / r).astype(np.float32)).unsqueeze(0)
    return torch.sigmoid(model(x)).item()


@st.cache_data
def sliding_probs(filepath, _ckpt):
    model, _, _ = load_model(_ckpt)
    audio, _ = sf.read(str(ROOT / filepath))
    ts, ps = [], []
    for s in range(0, max(1, len(audio) - CHUNK_N), HOP_N):
        c = audio[s : s + CHUNK_N]
        if len(c) < CHUNK_N:
            c = np.pad(c, (0, CHUNK_N - len(c)))
        ts.append((s + CHUNK_N / 2) / SR)
        ps.append(_predict(c, model))
    return np.array(ts), np.array(ps)


def running_avg(arr, w=4):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


def segment_runs(audio):
    bs, bn = 0.05, int(0.05 * SR)
    nb = len(audio) // bn
    zm = np.abs(audio[: nb * bn]) < 1e-5
    bl = np.array([zm[i * bn : (i + 1) * bn].mean() > 0.7 for i in range(nb)])
    runs = []
    ir = False
    rs = 0.0
    for i, s in enumerate(bl):
        t = i * bs
        if not s and not ir:
            ir = True
            rs = t
        elif s and ir:
            ir = False
        if s and not ir and t - rs >= 0.5:
            runs.append((rs, t, t - rs))
    if ir:
        te = nb * bs
        if te - rs >= 0.5:
            runs.append((rs, te, te - rs))
    return runs


@st.cache_data
def load_eval_data(eval_dir_str):
    ed = Path(eval_dir_str)
    if not ed.exists():
        return None, None
    # Load predictions
    pt_f = ed / "predictions_best.pt"
    preds = None
    if pt_f.exists():
        preds = torch.load(str(pt_f), map_location="cpu", weights_only=False)
    # Load curves
    npz_f = ed / "curves_best.npz"
    curves = None
    if npz_f.exists():
        curves = dict(np.load(npz_f, allow_pickle=True))
    return preds, curves


# ── UI ──────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Drone Detection Explorer")
st.title("🛸 Drone Detection — Sweep Explorer")

# ── Sidebar ─────────────────────────────────────────────────────────
models = discover_models()
st.caption(f"Sweep dirs: {len(sweep_dirs)} found ({len(models)} models total)")
if not models:
    st.error("No sweep models found!")
    st.stop()

# Model picker
st.sidebar.header("Model")
model_names = sorted(models.keys())
selected_name = st.sidebar.selectbox("Sweep model", model_names)
mi = models[selected_name]
ckpts_sorted = sorted(mi["checkpoints"], key=lambda p: p.name)

# Checkpoint picker within model
ckpt_options = {}
for c in ckpts_sorted:
    label = f"{c.parent.parent.name}/{c.name}"
    ckpt_options[label] = str(c)
selected_ckpt_label = st.sidebar.selectbox(
    "Checkpoint", list(ckpt_options.keys()), index=len(ckpt_options) - 1
)
selected_ckpt = ckpt_options[selected_ckpt_label]

model, arch, loss = load_model(selected_ckpt)
st.sidebar.caption(f"{arch} | {loss}")

# ── Eval calibration ────────────────────────────────────────────────
preds, curves = load_eval_data(str(mi["eval_dir"]))
st.sidebar.divider()
st.sidebar.subheader("Eval Calibration")

if preds is not None and curves is not None:
    st.sidebar.success(f"✓ {mi['eval_dir'].name}")
    logits = (
        preds["logits"].numpy()
        if hasattr(preds["logits"], "numpy")
        else np.asarray(preds["logits"])
    )
    labels = (
        preds["labels"].numpy()
        if hasattr(preds["labels"], "numpy")
        else np.asarray(preds["labels"])
    )
    bin_idx = preds.get("bin_idx")
    bin_names_arr = preds.get("bin_names")
else:
    st.sidebar.warning("No eval data")

# Threshold slider
st.sidebar.divider()
st.sidebar.subheader("Threshold")
threshold = st.sidebar.slider("Detection threshold", 0.0, 1.0, 0.68, 0.005)

# Show FPR/TPR at current threshold (from eval)
if preds is not None:
    probs_eval = 1.0 / (1.0 + np.exp(-logits))
    fpr_eval = (probs_eval[labels < 0.5] >= threshold).mean()
    tpr_eval = (probs_eval[labels > 0.5] >= threshold).mean()
    st.sidebar.metric("Eval FPR", f"{fpr_eval:.2%}")
    st.sidebar.metric("Eval TPR", f"{tpr_eval:.2%}")
    st.sidebar.metric("Threshold", f"{threshold:.3f}")

# ── Run inference on test files ─────────────────────────────────────
st.divider()
st.subheader(f"Test Runs — {selected_name} :: {Path(selected_ckpt).name}")

with st.spinner(f"Running inference with {arch}..."):
    bg_t, bg_p = sliding_probs(BG_FILE, selected_ckpt)
    all_results = {}
    for name, path in TEST_FILES.items():
        t, p = sliding_probs(path, selected_ckpt)
        all_results[name] = dict(times=t, probs=p, path=path)

# Metrics
cols = st.columns(6)
cols[0].metric("BG chunks", len(bg_p))
cols[1].metric("BG ≥thresh", f"{(bg_p >= threshold).mean():.1%}")
all_drone = np.concatenate([r["probs"] for r in all_results.values()])
cols[2].metric("Drone chunks", len(all_drone))
cols[3].metric("Drone ≥thresh", f"{(all_drone >= threshold).mean():.1%}")
cols[4].metric("Drone mean", f"{all_drone.mean():.3f}")
cols[5].metric("Drone max", f"{all_drone.max():.3f}")

# ── Global overview plot ────────────────────────────────────────────
st.subheader("All Test Files — Detection Overview")
fig, axes = plt.subplots(
    len(all_results), 1, figsize=(12, 2.2 * len(all_results)), sharex=True
)
colors_c = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
for ax, (name, data), c in zip(axes, all_results.items(), colors_c, strict=False):
    t, p = data["times"], data["probs"]
    ra = running_avg(p)
    ax.plot(t, p, lw=0.2, color=c, alpha=0.3)
    ax.plot(t, ra, lw=1.5, color=c)
    ax.axhline(y=threshold, color="red", ls="--", lw=1)
    det = p >= threshold
    for i in range(len(t) - 1):
        if det[i]:
            ax.axvspan(t[i], t[i + 1], alpha=0.1, color="green")
    ax.set_ylabel(name, fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"Threshold = {threshold:.3f}", fontsize=12, fontweight="bold")
fig.tight_layout()
st.pyplot(fig)
plt.close()

# ── Per-file detail (collapsed) ────────────────────────────────────
for name, data in all_results.items():
    t, p = data["times"], data["probs"]
    ra = running_avg(p)
    dm = p >= threshold
    audio, _ = sf.read(str(ROOT / data["path"]))
    runs = segment_runs(audio)
    with st.expander(
        f"{name} — det rate: {dm.mean():.1%} | max: {p.max():.3f} | mean: {p.mean():.3f}",
        expanded=dm.mean() > 0,
    ):
        c1, c2 = st.columns([3, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 2.2))
            ax.plot(t, p, lw=0.3, color="steelblue", alpha=0.4)
            ax.plot(t, ra, lw=1.2, color="darkblue")
            ax.axhline(y=threshold, color="red", ls="--", lw=1)
            for i in range(len(t) - 1):
                if dm[i]:
                    ax.axvspan(t[i], t[i + 1], alpha=0.08, color="green")
            for s, e, _d in runs:
                ax.axvspan(s, e, alpha=0.04, color="orange")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Prob")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.2)
            st.pyplot(fig)
            plt.close()
        with c2:
            st.metric("Det rate", f"{dm.mean():.1%}")
            st.metric("Max", f"{p.max():.3f}")
            st.metric("Mean", f"{p.mean():.3f}")
            if dm.mean() > 0:
                st.success(f"🚨 {dm.mean():.0%}")

# ── Calibration plots (FPR/TPR vs threshold, per-bin ROC) ───────────
st.subheader("Eval Calibration Curves")

if curves is not None:
    # Per-bin FPR/TPR vs threshold
    bin_keys = sorted([k.split("/")[0] for k in curves.keys() if "/fpr" in k])
    if bin_keys:
        st.caption("Per-bin FPR / TPR vs Threshold")
        n_cols = min(3, len(bin_keys))
        rows = (len(bin_keys) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(rows, n_cols, figsize=(5 * n_cols, 3.5 * rows))
        if rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, bk in enumerate(bin_keys):
            ax = axes[i]
            if f"{bk}/fpr" in curves:
                fpr = curves[f"{bk}/fpr"]
                tpr = curves[f"{bk}/tpr"]
                ths = curves[f"{bk}/thresholds"]
                ax.plot(ths, fpr, lw=1.5, color="red", label="FPR")
                ax.plot(ths, tpr, lw=1.5, color="green", label="TPR")
                ax.axvline(x=threshold, color="gray", ls="--", lw=1)
                ax.set_title(bk)
                ax.set_xlabel("Threshold")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.2)
                ax.set_ylim(-0.02, 1.02)
        for i in range(len(bin_keys), len(axes)):
            axes[i].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ROC curves per bin
    st.caption("Per-bin ROC Curves")
    fig, ax = plt.subplots(figsize=(6, 5))
    for bk in bin_keys:
        if f"{bk}/fpr" in curves:
            ax.plot(
                curves[f"{bk}/fpr"],
                curves[f"{bk}/tpr"],
                lw=1.5,
                label=f"{bk} (AUC={curves[f'{bk}/auc']:.3f})",
            )
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC per SNR Bin")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close()

elif preds is not None:
    # Fallback: compute from raw predictions
    probs = 1.0 / (1.0 + np.exp(-logits))
    th_range = np.linspace(0, 1, 200)
    fpr_c = [(probs[labels < 0.5] >= t).mean() for t in th_range]
    tpr_c = [(probs[labels > 0.5] >= t).mean() for t in th_range]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(th_range, fpr_c, lw=1.5, color="red", label="FPR")
    ax1.plot(th_range, tpr_c, lw=1.5, color="green", label="TPR")
    ax1.axvline(x=threshold, color="gray", ls="--")
    ax1.set_xlabel("Threshold")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    ax2.plot(fpr_c, tpr_c, lw=2, color="darkblue")
    ax2.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.3)
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.set_title("ROC")
    ax2.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close()

else:
    # Bare fallback: BG-based
    fpr_vals = [(bg_p >= t).mean() for t in np.linspace(0, 1, 200)]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.linspace(0, 1, 200), fpr_vals, lw=2, color="darkred")
    ax.axvline(x=threshold, color="red", ls="--")
    ax.axhline(y=0.10, color="gray", ls=":", label="10% FPR")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("FPR (background)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close()

# ── Per-bin detection breakdown ─────────────────────────────────────
if preds is not None and bin_idx is not None and bin_names_arr is not None:
    st.subheader("Per-Bin Detection Rates (eval)")
    bi = bin_idx.numpy() if hasattr(bin_idx, "numpy") else np.asarray(bin_idx)
    bn = (
        bin_names_arr
        if isinstance(bin_names_arr, np.ndarray)
        else np.asarray(bin_names_arr)
    )
    tn = sorted(set(str(x) for x in bn if str(x) != ""))
    rows = []
    for bname in tn:
        mask = bn == bname
        if mask.sum() == 0:
            continue
        bp = probs_eval[mask]
        bl = labels[mask]
        rows.append(
            [
                bname,
                len(bp),
                f"{(bp >= threshold).mean():.1%}",
                f"{bp[bl > 0.5].mean():.3f}" if (bl > 0.5).any() else "-",
            ]
        )
    st.table(
        {
            "Bin": [r[0] for r in rows],
            "Samples": [r[1] for r in rows],
            "≥Thresh": [r[2] for r in rows],
            "Mean prob": [r[3] for r in rows],
        }
    )
