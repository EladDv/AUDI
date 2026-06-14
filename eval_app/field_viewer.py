"""Field recording viewer — inspect model predictions on field alert recordings."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

from audi.hysteresis import apply_hysteresis
from eval_app.audio_utils import (
    _CLIP_S,
    _CLIP_SAMPLES,
    _SR,
    compute_mel_image,
    load_audio,
    predict_windows,
    window_time_axis,
)
from eval_app.model_utils import (
    discover_checkpoints,
    load_precision_thresholds,
    load_model,
)

_FIELD_DIR = Path(__file__).resolve().parents[1] / "data" / "field_recordings_20260514"
PRECISION_THRESHOLDS = load_precision_thresholds()


def field_viewer_page() -> None:
    st.title("🎙️ Field Recording Viewer")

    # ── Model selection ────────────────────────────────────────────
    ckpts = discover_checkpoints()
    run_names = sorted(set(c["run"] for c in ckpts))
    selected_run = st.sidebar.selectbox("Sweep / run", run_names, key="field_run")
    run_ckpts = [c for c in ckpts if c["run"] == selected_run]
    selected_ckpt = st.sidebar.selectbox(
        "Checkpoint", run_ckpts, format_func=lambda c: c["label"], key="field_ckpt",
    )

    # ── Threshold ───────────────────────────────────────────────────
    stride_ms = st.sidebar.slider("Window stride (ms)", 160, 1280, 320, 160, key="field_stride_ms")
    stride = stride_ms / 1000 / _CLIP_S
    threshold = st.sidebar.slider("Detection threshold", 0.0, 1.0, 0.5, 0.01, key="field_thresh")
    gain_db = st.sidebar.slider("Audio gain (dB)", 10, 40, 28, 1, key="field_gain")

    # ── Load labels ─────────────────────────────────────────────────
    labels_csv = _FIELD_DIR / "labels.csv"
    alert_labels: dict[str, str] = {}
    if labels_csv.exists():
        with open(labels_csv) as f:
            for r in csv.DictReader(f):
                alert_labels[r["alert_dir"]] = r["label"]

    # ── File selection ──────────────────────────────────────────────
    alert_dir = _FIELD_DIR / "alerts"
    audio_choices = []
    if alert_dir.exists():
        for d in sorted(alert_dir.iterdir()):
            if not d.is_dir():
                continue
            fp = d / "full_120s.wav"
            if fp.exists() and fp.stat().st_size > 1000:
                label = alert_labels.get(d.name, "?")
                audio_choices.append((d.name, str(fp), label))

    if not audio_choices:
        st.warning("No field recordings found.")
        return

    selected_audio = st.sidebar.selectbox(
        "Recording",
        audio_choices,
        format_func=lambda x: f"{x[2]:<7} {x[0]}",
        key="field_audio",
    )
    audio_name, audio_path, audio_label = selected_audio

    # ── Load model ──────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner(f"Loading {selected_ckpt['label']} ..."):
        model = load_model(selected_ckpt["path"], device)

    try:
        audio, sr = load_audio(audio_path)
    except Exception:
        st.error(f"Failed to load {audio_name} — file may be corrupted.")
        return
    # Keep raw audio for inference, create gained copy for playback
    audio_gained = audio * (10 ** (gain_db / 20.0))
    duration = len(audio) / sr

    # ── Run inference ───────────────────────────────────────────────
    with st.spinner(f"Running inference ({duration:.0f}s, stride={stride}) ..."):
        logits = predict_windows(model, audio, device, stride)
        scores = torch.sigmoid(torch.as_tensor(logits)).numpy()
        times = window_time_axis(len(audio), _CLIP_SAMPLES, stride)

    detections = apply_hysteresis(scores, threshold)
    n_detections = int(detections.sum())
    n_windows = len(logits)

    # ── Precision thresholds ────────────────────────────────────────
    model_ref = f"{selected_ckpt.get('run','')}/{selected_ckpt.get('exp_dir','')}"
    P_THRESH = PRECISION_THRESHOLDS.get(model_ref, {})

    # ── Metrics ─────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("File", audio_name)
    col2.metric("Label", audio_label)
    col3.metric("Duration", f"{duration:.1f}s")
    col4.metric("Windows", str(n_windows))
    col5.metric("Detections", f"{n_detections}/{n_windows}",
                delta=f"{100*n_detections/max(1,n_windows):.1f}%")

    if P_THRESH:
        st.subheader("Detections at Precision Levels")
        cols = st.columns(len(P_THRESH))
        for i, (level, info) in enumerate(sorted(P_THRESH.items(), reverse=True)):
            th = info["sigma"]
            n_det = int((scores > th).sum())
            pct = 100 * n_det / max(1, n_windows)
            with cols[i]:
                st.metric(f"σ@{level} ({th:.3f})", f"{n_det}/{n_windows}",
                          delta=f"{pct:.1f}%")

    # ── Spectrogram + score timeline ────────────────────────────────
    st.subheader(f"Spectrogram & Scores — {audio_label}")
    full_mel = compute_mel_image(audio, model)
    mel_time_axis = np.linspace(0, duration, full_mel.shape[1])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.35, 0.65], vertical_spacing=0.02)

    fig.add_trace(go.Scatter(
        x=times, y=scores, mode="lines", name="score",
        line=dict(color="#3498db", width=1.5),
    ), row=1, col=1)

    fig.add_hline(y=threshold, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"σ={threshold:.2f}", row=1, col=1)

    above_th = apply_hysteresis(scores, threshold)
    if above_th.any():
        fig.add_trace(go.Scatter(
            x=times, y=above_th.astype(float) * 1.0,
            fill="tozeroy", mode="none", fillcolor="rgba(255,0,0,0.10)",
            showlegend=False,
        ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=full_mel, x=mel_time_axis, colorscale="viridis",
        showscale=False, name="mel",
    ), row=2, col=1)

    fig.update_yaxes(title_text="Score", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Mel bin", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=10, b=20),
                      hovermode="x unified", showlegend=False)
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"field_pr_{audio_name}_{selected_ckpt['exp_dir']}",
    )

    # ── Full audio playback ──────────────────────────────────────────
    st.subheader("Full Recording")
    st.audio(audio_gained, sample_rate=sr)

    # ── Top detection windows ───────────────────────────────────────
    if n_detections > 0:
        st.subheader(f"Top Detection Windows (σ={threshold})")
        det_indices = np.where(detections)[0]
        det_indices = det_indices[np.argsort(-scores[det_indices])]
        n_show = min(10, len(det_indices))
        for rank, idx in enumerate(det_indices[:n_show]):
            t_start = idx * int(_CLIP_SAMPLES * stride) / _SR
            score_val = scores[idx]
            with st.expander(f"#{rank+1}  t={t_start:.1f}s  score={score_val:.4f}"):
                step = int(_CLIP_SAMPLES * stride)
                wav_window = audio_gained[idx * step: idx * step + _CLIP_SAMPLES]
                if len(wav_window) < _CLIP_SAMPLES:
                    wav_window = np.pad(wav_window, (0, _CLIP_SAMPLES - len(wav_window)))
                col_a, col_s = st.columns([1, 2])
                with col_a:
                    st.audio(wav_window, sample_rate=sr)
                with col_s:
                    mel_img = compute_mel_image(
                        audio[idx * step: idx * step + _CLIP_SAMPLES],
                        model,
                    )
                    fig_w = go.Figure(data=go.Heatmap(z=mel_img, colorscale="viridis"))
                    fig_w.update_layout(height=200, margin=dict(l=10, r=10, t=5, b=5),
                                        xaxis_title="Frame", yaxis_title="Mel")
                    st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.info(f"No windows exceed threshold {threshold}.")
