"""Ensemble viewer — multi-model fusion + smoothing for attack runs."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

from eval_app.audio_utils import (
    _ATTACK_DIR,
    _CLIP_S,
    _SR,
    compute_mel_image,
    load_audio,
    predict_windows,
)
from audi.hysteresis import apply_hysteresis
from eval_app.model_utils import discover_checkpoints, load_model

# ── Smoothing functions ──────────────────────────────────────────────────


def smooth_ema(scores: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    out = np.zeros_like(scores)
    out[0] = scores[0]
    for i in range(1, len(scores)):
        out[i] = alpha * scores[i] + (1 - alpha) * out[i - 1]
    return out


def smooth_moving_avg(scores: np.ndarray, window: int = 5) -> np.ndarray:
    return np.convolve(scores, np.ones(window) / window, mode="same")


def smooth_median(scores: np.ndarray, window: int = 5) -> np.ndarray:
    from scipy.signal import medfilt

    return medfilt(scores, kernel_size=window)


def smooth_hysteresis(scores: np.ndarray, high: float = 0.55, low: float = 0.45) -> np.ndarray:
    out = np.zeros_like(scores)
    state = False
    for i, s in enumerate(scores):
        if s > high:
            state = True
        elif s < low:
            state = False
        out[i] = 1.0 if state else s
    return out


# ── Ensemble viewer page ─────────────────────────────────────────────────


def ensemble_viewer_page() -> None:
    """Multi-model fusion + smoothing visualization."""
    st.title("🎯 Ensemble Viewer")

    st.sidebar.header("Audio File")
    audio_files = sorted(f.name for f in _ATTACK_DIR.glob("*.wav"))
    selected_file = st.sidebar.selectbox("Attack run", audio_files, key="ens_file")

    st.sidebar.header("Models")
    ckpts = discover_checkpoints()
    model_options = {f"{c['run']} / {c['exp_dir']}": c for c in ckpts}
    selected_labels = st.sidebar.multiselect(
        "Select models",
        list(model_options.keys()),
        default=list(model_options.keys())[:2],
        max_selections=5,
        key="ens_models",
    )

    st.sidebar.header("Fusion")
    fusion_method = st.sidebar.selectbox(
        "Method", ["max", "avg", "median", "min", "vote"], index=0, key="ens_fusion",
    )

    st.sidebar.header("Smoothing")
    smooth_method = st.sidebar.selectbox(
        "Method",
        ["none", "ema", "moving_avg", "median", "hysteresis"],
        index=0,
        key="ens_smooth",
    )
    smooth_params: dict[str, float] = {}
    if smooth_method == "ema":
        smooth_params["alpha"] = st.sidebar.slider("EMA alpha", 0.05, 0.95, 0.3, 0.05, key="ens_ema")  # noqa: E501
    elif smooth_method == "moving_avg":
        smooth_params["window"] = st.sidebar.slider("Window", 3, 15, 5, 2, key="ens_avg")
    elif smooth_method == "median":
        smooth_params["window"] = st.sidebar.slider("Median window", 3, 15, 5, 2, key="ens_med")
    elif smooth_method == "hysteresis":
        smooth_params["high"] = st.sidebar.slider("Hysteresis high", 0.3, 0.8, 0.55, 0.05, key="ens_hh")  # noqa: E501
        smooth_params["low"] = st.sidebar.slider("Hysteresis low", 0.2, 0.7, 0.45, 0.05, key="ens_hl")  # noqa: E501

    st.sidebar.header("Inference")
    stride = st.sidebar.slider("Stride", 0.05, 0.95, 0.125, 0.025, key="ens_stride")
    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.59, 0.01, key="ens_th")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not selected_labels or len(selected_labels) < 1:
        st.info("Select at least one model.")
        return

    audio_path = str(_ATTACK_DIR / selected_file)
    audio, sr = load_audio(audio_path)
    duration = len(audio) / sr

    models = {}
    for label in selected_labels:
        info = model_options[label]
        ckpt_path = info["path"]
        with st.spinner(f"Loading: {info['exp_dir']} ..."):
            models[info["exp_dir"]] = load_model(ckpt_path, device)

    n_models = len(models)
    m_names = list(models.keys())
    clip_samples = int(_SR * _CLIP_S)
    step = int(clip_samples * stride)

    # Build windows
    windows = []
    window_centers = []
    for i in range(0, len(audio) - clip_samples + 1, step):
        windows.append(audio[i : i + clip_samples])
        window_centers.append((i + clip_samples / 2) / _SR)
    window_centers = np.array(window_centers)
    n_windows = len(windows)

    # Run all models
    with st.spinner(f"Running {n_models} models on {n_windows} windows ..."):
        all_scores = np.zeros((n_windows, n_models))
        for j, (_name, model) in enumerate(models.items()):
            logits = predict_windows(model, audio, device, stride)
            all_scores[:, j] = torch.sigmoid(torch.as_tensor(logits)).numpy()

    # Fusion
    if fusion_method == "max":
        fused = all_scores.max(axis=1)
    elif fusion_method == "avg":
        fused = all_scores.mean(axis=1)
    elif fusion_method == "median":
        fused = np.median(all_scores, axis=1)
    elif fusion_method == "min":
        fused = all_scores.min(axis=1)
    elif fusion_method == "vote":
        fused = (all_scores > 0.5).mean(axis=1)
    else:
        fused = all_scores.max(axis=1)

    # Smoothing
    display_scores = fused.copy()
    if smooth_method == "ema":
        display_scores = smooth_ema(fused, smooth_params.get("alpha", 0.3))
    elif smooth_method == "moving_avg":
        display_scores = smooth_moving_avg(fused, smooth_params.get("window", 5))
    elif smooth_method == "median":
        display_scores = smooth_median(fused, smooth_params.get("window", 5))
    elif smooth_method == "hysteresis":
        display_scores = smooth_hysteresis(
            fused, smooth_params.get("high", 0.55), smooth_params.get("low", 0.45),
        )

    detections = apply_hysteresis(display_scores, threshold)
    n_det = int(detections.sum())

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("File", selected_file)
    col2.metric("Duration", f"{duration:.1f}s")
    col3.metric("Windows", str(n_windows))
    col4.metric("Models", str(n_models))
    col5.metric(
        "Detections", f"{n_det} / {n_windows}",
        delta=f"{100 * n_det / max(1, n_windows):.1f}%" if n_windows else "",
    )

    # Combined spectrogram + scores
    st.subheader("Spectrogram & Detection Scores")
    full_mel = compute_mel_image(audio, list(models.values())[0])
    mel_time_axis = np.linspace(0, duration, full_mel.shape[1])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.35, 0.65], vertical_spacing=0.02,
    )

    fig.add_trace(
        go.Scatter(x=window_centers, y=fused, mode="lines", name="fused",
                   line=dict(color="#3498db", width=1)),
        row=1, col=1,
    )
    if smooth_method != "none":
        fig.add_trace(
            go.Scatter(x=window_centers, y=display_scores, mode="lines", name="smoothed",
                       line=dict(color="#e74c3c", width=2.5)),
            row=1, col=1,
        )
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"σ={threshold:.2f}", row=1, col=1)
    if detections.any():
        fig.add_trace(
            go.Scatter(x=window_centers, y=detections.astype(float),
                       fill="tozeroy", mode="none",
                       fillcolor="rgba(255,0,0,0.10)", showlegend=False),
            row=1, col=1,
        )

    fig.add_trace(
        go.Heatmap(z=full_mel, x=mel_time_axis, colorscale="viridis",
                   showscale=False, name="mel"),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="Score", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Mel bin", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(
        height=500, margin=dict(l=20, r=20, t=10, b=20),
        hovermode="x unified", showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-model breakdown
    st.subheader("Per-Model Last Scan")
    cols = st.columns(len(m_names))
    for j, name in enumerate(m_names):
        det = int((all_scores[:, j] > threshold).sum())
        with cols[j]:
            st.metric(name, f"{det}/{n_windows}", delta=f"{all_scores[:, j].max():.3f} max")

    st.subheader("Audio")
    st.audio(audio_path)
