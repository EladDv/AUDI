#!/usr/bin/env python3
"""551 Field Recordings Explorer — tag-anchored spectrogram + detection viewer.
Precomputes inference once, applies hysteresis on sigma change only."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audi.checkpoint import load_model_from_checkpoint
from audi.hysteresis import apply_hysteresis

ROOT = Path(__file__).resolve().parents[1]
REC_DIR = ROOT / "data/551/Device_1_MultiMicRecorder_8_5-11_5"
TAG_DIR = ROOT / "data/551/TAGS_PFK_Device_1_MultiMicRecorder_11.05"
CKPT = (
    ROOT
    / "checkpoints/bce_wd_sweep_20260518_122516/01_wd_003/checkpoints/epoch=12-step=3250.ckpt"
)

SR = 16000
CLIP_S = 5.12
CHUNK_N = int(CLIP_S * SR)
HOP_MS = 320  # deployment-consistent hop
HOP_N = int(HOP_MS / 1000 * SR)  # 5120 samples
BATCH_SIZE = 32
CONTEXT_S = 30.0
MIN_TAG_DUR = 10.0

TAG_COLORS = {
    "רחפן": "#e74c3c",
    "רחפן חלש": "#e67e22",
    "זיק": "#3498db",
    "כלי רכב": "#95a5a6",
    "ציפור": "#2ecc71",
    "דיבור": "#9b59b6",
    "נביחות": "#f39c12",
    "הודעה": "#1abc9c",
    "כלי טייס": "#e74c3c",
    "BG Detection": "#ff4444",
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_time(t: str) -> float:
    if ":" not in t:
        return float(t)
    mm, ss = t.split(":")
    return int(mm) * 60 + float(ss)


def classify_tag(name: str) -> str:
    n = name.strip()
    if "רחפן" in n:
        return "רחפן חלש" if "חלש" in n else "רחפן"
    if "זיק" in n:
        return "זיק"
    if "כלי רכב" in n or "רכב" in n:
        return "כלי רכב"
    if "ציפור" in n:
        return "ציפור"
    if "דיבור" in n:
        return "דיבור"
    if "נביחות" in n:
        return "נביחות"
    if "הודעה" in n:
        return "הודעה"
    if "כלי טייס" in n:
        return "כלי טייס"
    return "other"


def load_all_tags() -> dict[str, list[dict]]:
    all_tags: dict[str, list[dict]] = {}
    for csv_path in sorted(TAG_DIR.glob("*.csv")):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        wav_stem = None
        for row in rows:
            name = row.get("Name", "").strip()
            dur = parse_time(row.get("Duration", "0:00.000"))
            if dur == 0 and name.startswith("rec_"):
                wav_stem = name.replace(".wav", "")
                continue
            if dur > 0 and wav_stem:
                s = parse_time(row["Start"])
                all_tags.setdefault(wav_stem, []).append(
                    {
                        "name": name,
                        "start_s": s,
                        "end_s": s + dur,
                        "type": classify_tag(name),
                    }
                )
        if not wav_stem:
            stem = csv_path.stem
            wav_stem = (
                stem[len("Device1_") :] if stem.startswith("Device1_") else stem
            )
            if "rec_" in wav_stem:
                wav_stem = (
                    wav_stem.split("_16000Hz")[0]
                    if "_16000Hz" in wav_stem
                    else wav_stem
                )
    return all_tags


def get_color(tag_type: str) -> str:
    for key, color in TAG_COLORS.items():
        if key in tag_type:
            return color
    return "#888888"


@st.cache_resource
def load_model():
    return load_model_from_checkpoint(CKPT, device=device)


@st.cache_data
def precompute_all() -> dict:
    """Precompute scores, times, and audio for all 11 files. Cached across runs."""
    model = load_model()
    model.eval()
    all_tags = load_all_tags()
    results = {}

    for stem in sorted(all_tags):
        wav_path = REC_DIR / f"{stem}.wav"
        audio, sr = sf.read(str(wav_path))
        rms = np.sqrt(np.mean(audio**2) + 1e-8)
        audio_norm = audio / rms if rms > 0 else audio

        scores = []
        n = max(1, (len(audio_norm) - CHUNK_N) // HOP_N + 1)
        for b in range(0, n, BATCH_SIZE):
            be = min(b + BATCH_SIZE, n)
            chunks = [
                audio_norm[i * HOP_N : i * HOP_N + CHUNK_N]
                for i in range(b, be)
            ]
            chunks = [
                np.pad(c, (0, CHUNK_N - len(c))) if len(c) < CHUNK_N else c
                for c in chunks
            ]
            t = torch.from_numpy(np.stack(chunks)).float().to(device)
            p = torch.sigmoid(model(t)).detach().squeeze(-1).cpu().numpy()
            scores.extend(float(p) if p.ndim == 0 else p.tolist())

        scores = np.array(scores)
        times = np.arange(len(scores)) * (HOP_N / SR) + (CLIP_S / 2)

        tags = all_tags[stem]
        results[stem] = {
            "audio": audio,
            "scores": scores,
            "times": times,
            "tags": tags,
            "duration": len(audio) / SR,
        }

    return results


def find_bg_events(
    dets: np.ndarray, times: np.ndarray, drone_tags: list[dict]
) -> list[dict]:
    events = []
    in_event = False
    event_start = 0.0
    for i in range(len(dets)):
        t = times[i]
        if dets[i] and not in_event:
            in_event = True
            event_start = t
        elif not dets[i] and in_event:
            in_event = False
            dur = t - event_start
            overlaps = any(
                event_start < tag["end_s"] and t > tag["start_s"]
                for tag in drone_tags
            )
            if not overlaps and dur >= 1.0:
                events.append(
                    {
                        "name": f"BG false alarm ({dur:.1f}s)",
                        "start_s": event_start,
                        "end_s": t,
                        "type": "BG Detection",
                    }
                )
    if in_event:
        dur = times[-1] - event_start
        overlaps = any(
            event_start < t["end_s"] and times[-1] > t["start_s"]
            for t in drone_tags
        )
        if not overlaps and dur >= 1.0:
            events.append(
                {
                    "name": f"BG false alarm ({dur:.1f}s)",
                    "start_s": event_start,
                    "end_s": times[-1],
                    "type": "BG Detection",
                }
            )
    return events


def render_spec_png(
    audio: np.ndarray,
    model,
    view_start: float,
    view_end: float,
    tags: list[dict],
    bg_events: list[dict],
    selected_item: dict,
) -> bytes:
    margin_s = 2.0
    a_start = max(0, int((view_start - margin_s) * SR))
    a_end = min(len(audio), int((view_end + margin_s) * SR))
    chunk = audio[a_start:a_end]
    if len(chunk) < SR:
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(14, 1))
        ax.text(0.5, 0.5, "Zoom too narrow", ha="center", va="center")
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    rms = np.sqrt(np.mean(chunk**2) + 1e-8)
    chunk_norm = chunk / rms if rms > 0 else chunk
    with torch.no_grad():
        mel = model._to_mel(
            torch.from_numpy(chunk_norm).float().unsqueeze(0).to(device)
        )
        mel_db = model._to_db(mel).squeeze(0)[0].detach().cpu().numpy()
    hop = 160
    mel_t = np.arange(mel_db.shape[1]) * hop / SR + a_start / SR

    max_cols = 2000
    if mel_db.shape[1] > max_cols:
        ds = mel_db.shape[1] // max_cols
        mel_db = mel_db[:, ::ds]
        mel_t = mel_t[::ds]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=-120,
        vmax=0,
        extent=[mel_t[0], mel_t[-1], 0, mel_db.shape[0]],
    )

    # Non-drone then drone (on top)
    non_d = [t for t in tags if "רחפן" not in t["type"]]
    drone = [t for t in tags if "רחפן" in t["type"]]
    for tag in non_d:
        c = get_color(tag["type"])
        sel = (
            tag["start_s"] == selected_item["start_s"]
            and tag["end_s"] == selected_item["end_s"]
        )
        ax.axvspan(
            tag["start_s"],
            tag["end_s"],
            alpha=0.3 if sel else 0.08,
            color=c,
            linewidth=2 if sel else 0,
            edgecolor=c if sel else None,
        )
    for tag in drone:
        c = get_color(tag["type"])
        sel = (
            tag["start_s"] == selected_item["start_s"]
            and tag["end_s"] == selected_item["end_s"]
        )
        ax.axvspan(
            tag["start_s"],
            tag["end_s"],
            alpha=0.35 if sel else 0.18,
            color=c,
            linewidth=2 if sel else 0,
            edgecolor=c if sel else None,
        )
    for ev in bg_events:
        sel = (
            ev["start_s"] == selected_item["start_s"]
            and ev["end_s"] == selected_item["end_s"]
        )
        ax.axvspan(
            ev["start_s"],
            ev["end_s"],
            alpha=0.3 if sel else 0.15,
            color="#ff4444",
            linestyle="--",
            linewidth=1,
        )

    ax.set_xlim(view_start, view_end)
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Time (s)")
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def main():
    st.set_page_config(page_title="551 Field Explorer", layout="wide")
    st.title("🎙️ 551 Field Recordings — Tag Explorer")

    all_tags = load_all_tags()
    wav_stems = sorted(all_tags.keys())

    # ── Load precomputed data ──
    with st.spinner("Precomputing inference on all 11 files (one-time)..."):
        data = precompute_all()

    st.sidebar.header("Recording")
    selected_stem = st.sidebar.selectbox("File", wav_stems)
    file_data = data[selected_stem]
    audio = file_data["audio"]
    scores = file_data["scores"]
    times = file_data["times"]
    duration = file_data["duration"]

    tags = [
        t for t in file_data["tags"] if t["end_s"] - t["start_s"] >= MIN_TAG_DUR
    ]
    tag_types = sorted(set(t["type"] for t in tags))
    selected_types = st.sidebar.multiselect(
        "Tag types", tag_types, default=tag_types
    )
    show_bg = st.sidebar.checkbox("Show BG detections", value=True)

    st.sidebar.header("Detection")
    sigma = st.sidebar.slider("Sigma (threshold)", 0.0, 1.0, 0.5, 0.01)

    st.sidebar.metric("Total tags", str(sum(len(v) for v in all_tags.values())))
    st.sidebar.metric("File tags", str(len(tags)))

    # ── Hysteresis (fast, recomputed on sigma change) ──
    dets = apply_hysteresis(scores, sigma)
    n_det = int(dets.sum())

    if show_bg:
        all_drone_tags = [t for t in file_data["tags"] if "רחפן" in t["type"]]
        bg_events = find_bg_events(dets, times, all_drone_tags)
    else:
        bg_events = []

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{duration:.0f}s")
    col2.metric("Windows", str(len(scores)))
    col3.metric(
        "Detections",
        f"{n_det}/{len(scores)}",
        delta=f"{100 * n_det / max(1, len(scores)):.1f}%",
    )
    col4.metric("BG events", str(len(bg_events)))

    # ── Selector ──
    filtered_tags = [t for t in tags if t["type"] in selected_types]
    all_items = list(filtered_tags) + (bg_events if show_bg else [])

    st.subheader("Tagged Segments & BG Detections")
    item_labels = [
        f"[{t['type']}] {t['start_s']:.0f}s–{t['end_s']:.0f}s  {t['name'][:60]}"
        for t in all_items
    ]
    if not item_labels:
        st.info("No items to show.")
        return

    selected_idx = st.selectbox(
        "Jump to segment",
        range(len(item_labels)),
        format_func=lambda i: item_labels[i],
    )
    selected_item = all_items[selected_idx]

    seg_center = (selected_item["start_s"] + selected_item["end_s"]) / 2
    view_start = max(0, seg_center - CONTEXT_S)
    view_end = min(duration, seg_center + CONTEXT_S)

    # ── Decimate for plotting ──
    ds = max(1, len(scores) // 500)
    s_disp = scores[::ds]
    t_disp = times[::ds]
    d_disp = dets[::ds]

    # ── Score plot ──
    fig_score = go.Figure()
    fig_score.add_trace(
        go.Scatter(
            x=t_disp,
            y=s_disp,
            mode="lines",
            line=dict(color="#3498db", width=1.2),
        )
    )
    if d_disp.any():
        fig_score.add_trace(
            go.Scatter(
                x=t_disp,
                y=d_disp.astype(float) * 1.02,
                fill="tozeroy",
                mode="none",
                fillcolor="rgba(255,0,0,0.10)",
                showlegend=False,
            )
        )
    fig_score.add_hline(
        y=sigma,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"σ={sigma:.2f}",
    )
    non_drone = [t for t in filtered_tags if "רחפן" not in t["type"]]
    drone_plot = [t for t in filtered_tags if "רחפן" in t["type"]]
    for tag in non_drone:
        fig_score.add_vrect(
            x0=tag["start_s"],
            x1=tag["end_s"],
            fillcolor=get_color(tag["type"]),
            opacity=0.10,
            line_width=0,
        )
    for tag in drone_plot:
        fig_score.add_vrect(
            x0=tag["start_s"],
            x1=tag["end_s"],
            fillcolor=get_color(tag["type"]),
            opacity=0.20,
            line_width=0,
        )
    for ev in bg_events:
        fig_score.add_vrect(
            x0=ev["start_s"],
            x1=ev["end_s"],
            fillcolor="#ff4444",
            opacity=0.2,
            line_dash="dash",
            line_width=1,
        )
    fig_score.update_xaxes(range=[view_start, view_end], title_text="Time (s)")
    fig_score.update_yaxes(range=[0, 1.05], title_text="Score")
    fig_score.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=10, b=20),
        showlegend=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # ── Hysteresis plot ──
    fig_hyst = go.Figure()
    fig_hyst.add_trace(
        go.Scatter(
            x=t_disp,
            y=d_disp.astype(float),
            mode="lines",
            line=dict(color="#e74c3c", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.15)",
        )
    )
    fig_hyst.update_xaxes(range=[view_start, view_end])
    fig_hyst.update_yaxes(
        range=[-0.05, 1.1], tickvals=[0, 1], ticktext=["OFF", "ON"]
    )
    fig_hyst.update_layout(
        height=120, margin=dict(l=20, r=20, t=5, b=20), showlegend=False
    )
    st.plotly_chart(fig_hyst, use_container_width=True)

    # ── Spectrogram ──
    st.subheader("Spectrogram")
    model = load_model()
    png_bytes = render_spec_png(
        audio,
        model,
        view_start,
        view_end,
        filtered_tags,
        bg_events,
        selected_item,
    )
    st.image(png_bytes, use_container_width=True)

    # ── Audio ──
    st.subheader("Audio — Selected Segment (±5s)")
    a_s = max(0, int((selected_item["start_s"] - 5) * SR))
    a_e = min(len(audio), int((selected_item["end_s"] + 5) * SR))
    st.audio(audio[a_s:a_e], sample_rate=SR)

    # ── Legend ──
    st.subheader("Legend")
    cols = st.columns(5)
    items = dict(TAG_COLORS)
    if not show_bg:
        items.pop("BG Detection", None)
    for i, (tt, c) in enumerate(sorted(items.items())):
        with cols[i % 5]:
            dash = "┅┅" if tt == "BG Detection" else ""
            st.markdown(
                f"<span style='color:{c};font-size:18px'>{dash}■</span> {tt}",
                unsafe_allow_html=True,
            )

    # ── Table ──
    with st.expander(f"All {len(all_items)} segments"):
        rows = []
        for t in all_items:
            mask = (times >= t["start_s"]) & (times <= t["end_s"])
            ss = scores[mask]
            dd = dets[mask]
            rows.append(
                {
                    "type": t["type"],
                    "name": t["name"][:80],
                    "start": f"{t['start_s']:.1f}s",
                    "end": f"{t['end_s']:.1f}s",
                    "dur": f"{t['end_s'] - t['start_s']:.1f}s",
                    "score_max": f"{ss.max():.3f}" if len(ss) else "-",
                    "score_mean": f"{ss.mean():.3f}" if len(ss) else "-",
                    "n_det": int(dd.sum()) if len(ss) else 0,
                }
            )
        st.dataframe(rows, use_container_width=True)

    st.caption(
        f"**{selected_item['type']}**: {selected_item['name'][:100]}  "
        f"({selected_item['start_s']:.1f}s–{selected_item['end_s']:.1f}s, "
        f"{selected_item['end_s'] - selected_item['start_s']:.1f}s)"
    )


if __name__ == "__main__":
    main()
