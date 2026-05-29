#!/usr/bin/env python3
"""Interactive model evaluation dashboard for audi drone detection.

Run: uv run streamlit run eval_app/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

from audi.hearability_estimator import HearabilityEstimator
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
from eval_app.ensemble import ensemble_viewer_page
from eval_app.leaderboard import leaderboard_page
from eval_app.field_viewer import field_viewer_page
from eval_app.model_utils import (
    _CHECKPOINTS_DIR,
    discover_checkpoints,
    find_hearability_calib,
    find_predictions_file,
    get_model_arch_from_ckpt,
    load_model,
)
from eval_app.precision import compute_precision_recall_curve, load_precision_thresholds

# ── Page setup ───────────────────────────────────────────────────────────

st.set_page_config(page_title="audi — Attack Run Evaluator", layout="wide")
st.title("🚁 audi — Attack Run Evaluator")

# ── Helpers: bin classification (used in multiple places) ────────────────

BIN_ORDER = ["easy", "medium", "hard", "very_hard", "extreme", "far_field"]
BIN_COLORS_HEX = {
    "easy": "#2ecc71", "medium": "#27ae60", "hard": "#f39c12",
    "very_hard": "#e67e22", "extreme": "#e74c3c", "far_field": "#c0392b",
}
_ATTACK_DIR = _CHECKPOINTS_DIR.parent / "data" / "attack_runs"


def classify_bins_simple(scores, easy_th, medium_th, hard_th, vhard_th):
    """Fallback threshold-based bin classification when no calibrator."""
    result = []
    for s in scores:
        for bn, th in [("easy", easy_th), ("medium", medium_th),
                        ("hard", hard_th), ("very_hard", vhard_th)]:
            if s >= th:
                result.append(bn)
                break
        else:
            result.append("extreme")
    return np.array(result)

# ── Page selector ────────────────────────────────────────────────

page = st.sidebar.radio(
    "Page",
    ["Model Viewer", "Attack Run Leaderboard", "Field Viewer", "Ensemble Viewer"],
    index=0,
)

if page == "Attack Run Leaderboard":
    leaderboard_page()
    st.stop()
elif page == "Field Viewer":
    field_viewer_page()
    st.stop()
elif page == "Ensemble Viewer":
    ensemble_viewer_page()
    st.stop()

# ── Sidebar ─────────────────────────────────────────────────────────────

st.sidebar.header("Model")
ckpts = discover_checkpoints()
run_names = sorted(set(c["run"] for c in ckpts))
selected_run = st.sidebar.selectbox("Sweep / run", run_names)

run_ckpts = [c for c in ckpts if c["run"] == selected_run]
selected_ckpt = st.sidebar.selectbox(
    "Checkpoint", run_ckpts, format_func=lambda c: c["label"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

PRECISION_THRESHOLDS = load_precision_thresholds()

st.sidebar.header("Inference")
stride_ms = st.sidebar.slider(
    "Window stride (ms)", 160, 1280, 320, 160,
    help="Hop between windows in milliseconds. 160ms = 10% overlap on 5.12s window.",
)
stride = stride_ms / 1000 / _CLIP_S  # convert ms → fraction of window
threshold = st.sidebar.slider(
    "Detection threshold", 0.0, 1.0, 0.5, 0.01,
    help="Sigmoid score threshold. Windows with score > threshold are flagged.",
)

# ── SNR bin estimation ──────────────────────────────────────────────────

st.sidebar.header("SNR Bin Estimation")
calib_path = find_hearability_calib(selected_ckpt["path"])
estimator = None
if calib_path:
    estimator = HearabilityEstimator.load(Path(calib_path))
    st.sidebar.success(f"Bayesian calibrator: {Path(calib_path).parent.parent.name}")
else:
    exp = selected_ckpt.get("exp_dir", "?")
    st.sidebar.warning(f"No calibration for {exp}\nSelect a Phase 3 checkpoint")

if st.sidebar.button("🔄 Refresh models"):
    st.cache_resource.clear()
    st.rerun()

if estimator:
    sorted_bins = sorted(estimator.bins, key=lambda b: b.mean)
    boundaries = []
    for i in range(len(sorted_bins) - 1):
        a, b = sorted_bins[i], sorted_bins[i + 1]
        v_a, v_b = a.std**2, b.std**2
        m_a, m_b = a.mean, b.mean
        A = 1/v_a - 1/v_b if v_a > 0 and v_b > 0 else 0
        B = -2*m_a/v_a + 2*m_b/v_b if v_a > 0 and v_b > 0 else 0
        C_val = (  # noqa: E501
            m_a**2 / v_a - m_b**2 / v_b
            - 2 * np.log(
                max(a.prior, 1e-12) / max(b.prior, 1e-12)
                * max(b.std, 1e-8) / max(a.std, 1e-8)
            )
        ) if v_a > 0 and v_b > 0 else 0
        if abs(A) < 1e-12:
            boundary = -C_val / B if abs(B) > 1e-12 else (m_a + m_b) / 2
        else:
            disc = B**2 - 4*A*C_val
            boundary = (-B + np.sqrt(disc)) / (2*A) if disc >= 0 else (m_a+m_b)/2
        boundaries.append((f"{a.name}↔{b.name}", boundary))
    st.sidebar.caption("Decision boundaries (logit):")
    for label, boundary in boundaries:
        st.sidebar.caption(f"  {label}: {boundary:.2f}")
else:
    st.sidebar.caption("Score ranges → estimated SNR category")
    bin_easy_th = st.sidebar.slider("easy above", 0.0, 1.0, 0.80, 0.01, key="bin_easy")
    bin_medium_th = st.sidebar.slider("medium above", 0.0, 1.0, 0.65, 0.01, key="bin_medium")
    bin_hard_th = st.sidebar.slider("hard above", 0.0, 1.0, 0.50, 0.01, key="bin_hard")
    bin_vhard_th = st.sidebar.slider("very_hard above", 0.0, 1.0, 0.35, 0.01, key="bin_vhard")

st.sidebar.header("Audio File")
audio_files = sorted(f.name for f in _ATTACK_DIR.glob("*.wav"))
selected_file = st.sidebar.selectbox("Attack run", audio_files)

# ── Load model ──────────────────────────────────────────────────────────

with st.spinner(f"Loading model: {selected_ckpt['label']} ..."):
    model = load_model(selected_ckpt["path"], device)

audio_path = str(_ATTACK_DIR / selected_file)
audio, sr = load_audio(audio_path)
duration = len(audio) / sr

# ── Run inference ───────────────────────────────────────────────────────

with st.spinner(f"Running sliding-window inference ({duration:.0f}s audio, stride={stride}) ..."):
    logits = predict_windows(model, audio, device, stride)
    scores = torch.sigmoid(torch.as_tensor(logits)).numpy()
    times = window_time_axis(len(audio), _CLIP_SAMPLES, stride)

detections = apply_hysteresis(scores, threshold)
n_detections = int(detections.sum())
n_windows = len(logits)

# ── Precision thresholds for this model ─────────────────────────────────

model_arch = get_model_arch_from_ckpt(selected_ckpt["path"])
model_ref = f"{selected_ckpt.get('run','')}/{selected_ckpt.get('exp_dir','')}"
P_THRESH = PRECISION_THRESHOLDS.get(model_ref, {})

# ── Metrics row ─────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("File", selected_file)
col2.metric("Duration", f"{duration:.1f}s")
col3.metric("Windows", str(n_windows))
col4.metric("Detections", f"{n_detections} / {n_windows}",
            delta=f"{100*n_detections/max(1,n_windows):.1f}%" if n_windows else "")

# ── Precision-level detection counts ────────────────────────────────────

if P_THRESH:
    st.subheader("Detections at Precision Levels")
    cols = st.columns(len(P_THRESH))
    for i, (level, info) in enumerate(sorted(P_THRESH.items(), reverse=True)):
        th = info["sigma"]  # sigma is already sigmoid probability
        n_det = int((scores > th).sum())
        pct = 100 * n_det / max(1, n_windows)
        with cols[i]:
            st.metric(
                f"σ@{level.upper()} ({th:.3f})",
                f"{n_det}/{n_windows}",
                delta=f"{pct:.1f}% of windows",
            )

# ── SNR bin distribution ───────────────────────────────────────────────

bin_order = list(estimator._names) if estimator else BIN_ORDER

if estimator:
    bin_probs = np.array([list(estimator.predict(float(logit)).values()) for logit in logits])
    bin_labels_arr = np.array([bin_order[np.argmax(p)] for p in bin_probs])
else:
    bin_labels_arr = classify_bins_simple(
        scores, bin_easy_th, bin_medium_th, bin_hard_th, bin_vhard_th,
    )

bin_counts = {bn: int((bin_labels_arr == bn).sum()) for bn in bin_order}

st.subheader("Estimated SNR Bin Distribution")
cols = st.columns(len(bin_order))
for i, bn in enumerate(bin_order):
    cnt = bin_counts[bn]
    pct = 100 * cnt / max(1, n_windows)
    color = BIN_COLORS_HEX.get(bn, "#888")
    with cols[i]:
        st.markdown(
            f"<div style='background:{color}22;padding:8px;border-radius:6px;border-left:4px solid {color}'>"  # noqa: E501
            f"<small style='color:#888'>{bn}</small><br>"
            f"<b style='font-size:20px'>{cnt}</b> <small>({pct:.1f}%)</small></div>",
            unsafe_allow_html=True,
        )

# ── Combined spectrogram + score timeline ───────────────────────────────

st.subheader("Spectrogram & Detection Scores")

full_mel = compute_mel_image(audio, model)
mel_time_axis = np.linspace(0, duration, full_mel.shape[1])

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.35, 0.65],
    vertical_spacing=0.02,
)

fig.add_trace(go.Scatter(
    x=times, y=scores, mode="lines", name="score",
    line=dict(color="#3498db", width=1.5),
    hovertemplate="%{x:.1f}s — score: %{y:.4f}<extra></extra>",
), row=1, col=1)

# SNR bin bands
if estimator:
    sorted_bins = sorted(estimator.bins, key=lambda b: b.mean)
    from math import log as mlog
    decision_bounds = []
    for i in range(len(sorted_bins) - 1):
        a, b = sorted_bins[i], sorted_bins[i + 1]
        v_a, v_b = a.std**2, b.std**2
        m_a, m_b = a.mean, b.mean
        if v_a < 1e-8 or v_b < 1e-8:
            decision_bounds.append((a.mean + b.mean) / 2)
            continue
        A = 1/v_a - 1/v_b
        B = -2*m_a/v_a + 2*m_b/v_b
        C = (  # noqa: E501
            m_a**2 / v_a - m_b**2 / v_b
            - 2 * mlog(
                max(a.prior, 1e-12) / max(b.prior, 1e-12)
                * max(b.std, 1e-8) / max(a.std, 1e-8)
            )
        )
        if abs(A) < 1e-12:
            boundary = -C / B if abs(B) > 1e-12 else (m_a + m_b) / 2
        else:
            disc = B**2 - 4*A*C
            boundary = (-B + np.sqrt(disc)) / (2*A) if disc >= 0 else (m_a+m_b)/2
        decision_bounds.append(boundary)
    all_bounds = [float('-inf')] + decision_bounds + [float('inf')]
    bg_map = {"far_field": "rgba(192,57,43,0.08)", "extreme": "rgba(231,76,60,0.08)",
              "very_hard": "rgba(230,126,34,0.08)", "hard": "rgba(243,156,18,0.08)",
              "medium": "rgba(39,174,96,0.08)", "easy": "rgba(46,204,113,0.08)"}
    for i, bn in enumerate([b.name for b in sorted_bins]):
        lo = all_bounds[i]
        hi = all_bounds[i+1] if i+1 < len(all_bounds) else float('inf')
        lo_s = 1/(1+np.exp(-lo)) if lo > float('-inf') else 0.0
        hi_s = 1/(1+np.exp(-hi)) if hi < float('inf') else 1.0
        if hi_s > lo_s:
            fig.add_hrect(y0=lo_s, y1=hi_s, fillcolor=bg_map.get(bn, "rgba(128,128,128,0.05)"),
                          line_width=0, annotation_text=bn, annotation_position="left",
                          annotation_font_size=8, annotation_font_color="#888", row=1, col=1)
else:
    prev = 0.0
    for bn, th_val, color in [("extreme", 0.0, "231,76,60"), ("very_hard", bin_vhard_th, "230,126,34"),  # noqa: E501
                               ("hard", bin_hard_th, "243,156,18"), ("medium", bin_medium_th, "39,174,96"),  # noqa: E501
                               ("easy", bin_easy_th, "46,204,113")]:
        if th_val > prev:
            fig.add_hrect(y0=prev, y1=th_val, fillcolor=f"rgba({color},0.08)", line_width=0,
                          annotation_text=bn, annotation_position="left",
                          annotation_font_size=8, annotation_font_color="#888", row=1, col=1)
        prev = th_val

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
    z=full_mel, x=mel_time_axis,
    colorscale="viridis", showscale=False, name="mel",
    hovertemplate="t=%{x:.1f}s mel[%{y}]<extra></extra>",
), row=2, col=1)

fig.update_yaxes(title_text="Score", range=[0, 1.05], row=1, col=1)
fig.update_yaxes(title_text="Mel bin", row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_layout(
    height=500, margin=dict(l=20, r=20, t=10, b=20),
    hovermode="x unified", showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# ── Calibration curves ──────────────────────────────────────────────────

pred_file = find_predictions_file(selected_ckpt["path"])
if pred_file:
    st.subheader(f"Precision & Recall vs Threshold — {selected_ckpt['exp_dir']}")
    curve = compute_precision_recall_curve(pred_file)

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=curve["sig_thresholds"], y=curve["precisions"], mode="lines",
        name="precision", line=dict(color="#2ecc71", width=2),
    ))
    fig_cal.add_trace(go.Scatter(
        x=curve["sig_thresholds"], y=curve["recalls"], mode="lines",
        name="recall", line=dict(color="#3498db", width=2),
    ))
    # Overlay all P-level thresholds as markers on the precision curve
    if P_THRESH:
        level_colors = {
            "P50": "#888", "P60": "#888", "P70": "#888", "P75": "#888",
            "P80": "#888", "P85": "#888",
            "P90": "#f1c40f", "P95": "#e67e22", "P99": "#e74c3c",
        }
        for level in sorted(P_THRESH.keys()):
            sigma = P_THRESH[level]["sigma"]
            color = level_colors.get(level, "#888")
            size = 10 if level in ("P90", "P95", "P99") else 5
            # Find nearest precision value on the curve
            idx = int(np.argmin(np.abs(curve["sig_thresholds"] - sigma)))
            fig_cal.add_trace(go.Scatter(
                x=[sigma], y=[curve["precisions"][idx]],
                mode="markers+text", marker=dict(size=size, color=color),
                text=[level], textposition="top center",
                name=level, showlegend=False,
            ))

    user_prec = curve["precisions"][np.argmin(np.abs(curve["sig_thresholds"] - threshold))]
    user_rec = curve["recalls"][np.argmin(np.abs(curve["sig_thresholds"] - threshold))]
    fig_cal.add_vline(
        x=threshold, line_dash="dash", line_color="red", line_width=2,
        annotation_text=f"σ={threshold:.2f}  P={user_prec:.3f}  R={user_rec:.3f}",
    )
    fig_cal.update_layout(
        xaxis_title="Threshold (sigmoid)", yaxis_title="Value",
        height=300, margin=dict(l=20, r=20, t=10, b=20),
        hovermode="x unified", yaxis=dict(range=[0, 1.05]),
    )
    st.plotly_chart(fig_cal, use_container_width=True, key=f"pr_curve_{selected_ckpt['exp_dir']}")
    st.caption("Precision and recall on validation set. Adjust threshold to trade off.")
else:
    st.info("No eval_data predictions found — run postprocess first to see calibration curves.")

st.subheader("Audio")
st.audio(audio_path)

# ── Top detection windows ───────────────────────────────────────────────

if n_detections > 0:
    st.subheader(f"Top Detection Windows (threshold={threshold})")
    det_indices = np.where(detections)[0]
    det_indices = det_indices[np.argsort(-scores[det_indices])]
    n_show = min(10, len(det_indices))
    for rank, idx in enumerate(det_indices[:n_show]):
        t_start = idx * int(_CLIP_SAMPLES * stride) / _SR
        score_val = scores[idx]
        with st.expander(f"#{rank+1}  t={t_start:.1f}s  score={score_val:.4f}"):
            step = int(_CLIP_SAMPLES * stride)
            wav_window = audio[idx * step: idx * step + _CLIP_SAMPLES]
            if len(wav_window) < _CLIP_SAMPLES:
                wav_window = np.pad(wav_window, (0, _CLIP_SAMPLES - len(wav_window)))
            col_a, col_s = st.columns([1, 2])
            with col_a:
                st.audio(wav_window, sample_rate=sr)
            with col_s:
                mel_img = compute_mel_image(wav_window, model)
                fig_w = go.Figure(data=go.Heatmap(z=mel_img, colorscale="viridis"))
                fig_w.update_layout(
                    height=200, margin=dict(l=10, r=10, t=5, b=5),
                    xaxis_title="Frame", yaxis_title="Mel",
                )
                st.plotly_chart(fig_w, use_container_width=True)
else:
    st.info(f"No windows exceed threshold {threshold}. Try lowering the threshold.")

# ── All windows table ───────────────────────────────────────────────────

st.subheader("All Windows (sorted by score)")
with st.expander("Show table"):
    data = []
    step_s = _CLIP_SAMPLES * stride / _SR
    for i in range(n_windows):
        data.append({
            "window": i,
            "time": f"{i * step_s:.1f}s – {i * step_s + _CLIP_S:.1f}s",
            "score": round(float(scores[i]), 4),
            "est. SNR": bin_labels_arr[i],
            "detection": "🚨" if detections[i] else "",
        })
    st.dataframe(
        data, use_container_width=True,
        column_config={
            "score": st.column_config.ProgressColumn(
                "score", min_value=0, max_value=1, format="%.4f",
            ),
        },
    )
