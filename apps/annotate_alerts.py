#!/usr/bin/env python3
"""Annotate field alerts: mark drone time segments.
Playback with time-synced score + spectrogram, dB meter, gain, segment regions.

Usage:
    uv run python -m streamlit run apps/annotate_alerts.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/field_recordings_20260514"
SCORES_JSON = DATA / "wd003_scores.json"
FRAMES_NPZ = DATA / "wd003_frames.npz"
LABELS_CSV = DATA / "labels.csv"
SEGMENTS_JSON = DATA / "segments.json"

SR = 16000

st.set_page_config(page_title="Alert annotator", layout="wide")

st.markdown("""
<style>
  html, body, [class*="css"]  { font-size: 14px !important; }
  .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1500px; }
  h1 { font-size: 1.5rem !important; }
  h2 { font-size: 1.1rem !important; }
  .drone-btn button, .nodrone-btn button, .skip-btn button {
    height: 52px !important; width: 100% !important; font-size: 1.1rem !important;
    font-weight: 700 !important; border-radius: 12px !important; }
  .drone-btn button { background: #2F9E44 !important; border-color: #2F9E44 !important; }
  .nodrone-btn button { background: #E03131 !important; border-color: #E03131 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    scores = json.loads(SCORES_JSON.read_text())
    frames = np.load(FRAMES_NPZ)
    return scores, dict(frames)


@st.cache_data
def load_labels() -> dict[str, str]:
    labels = {}
    if LABELS_CSV.exists():
        for line in LABELS_CSV.read_text().strip().split("\n")[1:]:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels[parts[0]] = parts[1]
    return labels


def save_labels(labels: dict[str, str]):
    LABELS_CSV.write_text("\n".join(
        ["alert_dir,label"] + [f"{k},{v}" for k, v in sorted(labels.items())]
    ) + "\n")


def load_segments() -> dict[str, list[tuple[float, float]]]:
    """{alert_dir: [(start_s, end_s), ...]}"""
    if SEGMENTS_JSON.exists():
        raw = json.loads(SEGMENTS_JSON.read_text())
        return {k: [tuple(s) for s in v] for k, v in raw.items()}
    return {}


def save_segments(segments: dict[str, list[tuple[float, float]]]):
    SEGMENTS_JSON.write_text(json.dumps(
        {k: [list(s) for s in v] for k, v in segments.items()}, indent=2
    ))


@st.cache_data
def load_audio(wav_path_str: str) -> tuple[np.ndarray, float, float, float]:
    audio, sr = sf.read(wav_path_str)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio**2) + 1e-8))
    peak = float(np.max(np.abs(audio)))
    dur = len(audio) / sr
    rms_db = 20 * np.log10(rms) if rms > 0 else -100
    peak_db = 20 * np.log10(peak) if peak > 0 else -100
    return audio, rms_db, peak_db, dur


@st.cache_data
def load_spec(wav_path_str: str, hop_s: float) -> tuple[np.ndarray, float]:
    audio, sr = sf.read(wav_path_str)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio**2) + 1e-8)
    if rms > 0:
        audio = audio / rms
    S = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=1024, hop_length=160, n_mels=128, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80.0)
    spec_fs = 160 / SR
    target_fs = hop_s / 2
    stride = max(1, int(target_fs / spec_fs))
    return S_db[:, ::stride], stride * spec_fs


def compute_stats(alerts, labels, frame_scores, threshold):
    labeled = {k: v for k, v in labels.items() if v in ("drone", "nodrone")}
    if not labeled:
        return {"precision": 0, "recall": 0, "tp": 0, "fp": 0, "fn": 0,
                "n_labeled": 0, "n_drone": 0, "n_nodrone": 0}
    tp = fp = fn = 0
    for ad, gt in labeled.items():
        is_drone = gt == "drone"
        max_s = frame_scores.get(ad, np.array([0])).max()
        if is_drone and max_s >= threshold:
            tp += 1
        elif is_drone:
            fn += 1
        elif max_s >= threshold:
            fp += 1
    n_drone = sum(1 for v in labeled.values() if v == "drone")
    n_nodrone = sum(1 for v in labeled.values() if v == "nodrone")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn,
            "n_labeled": len(labeled), "n_drone": n_drone, "n_nodrone": n_nodrone}


def make_audio_bytes(audio: np.ndarray, gain_db: float) -> bytes:
    lin = 10.0 ** (gain_db / 20.0)
    y = np.clip(audio * lin, -1.0, 1.0).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def plot_both(alert_dir: str, frame_scores: np.ndarray, hop_s: float,
              thresholds: list[float], spec_db: np.ndarray, spec_fs: float,
              cursor_t: float, duration: float,
              segments: list[tuple[float, float]]):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 5.5), dpi=100,
                                          gridspec_kw={"height_ratios": [1, 1.4]},
                                          facecolor="#0e1117")
    score_t = np.arange(len(frame_scores)) * hop_s

    # ── Segment shading (behind score line) ──
    for s0, s1 in segments:
        ax_top.axvspan(s0, s1, color="#2F9E44", alpha=0.12)
        ax_bot.axvspan(s0, s1, color="#2F9E44", alpha=0.12)

    # ── Score plot ──
    ax_top.plot(score_t, frame_scores, color="#4ECDC4", linewidth=1.0)
    for th in thresholds:
        ax_top.axhline(th, color="#FF6B6B", linestyle="--", linewidth=0.6, alpha=0.7)
    ax_top.axvline(cursor_t, color="#FF0000", linewidth=1.5, alpha=0.9)
    ax_top.set_ylim(0, 1.02)
    ax_top.set_xlim(0, duration)
    ax_top.set_ylabel("Score", color="#ccc")
    title = f"{alert_dir}  max={frame_scores.max():.4f}  mean={frame_scores.mean():.4f}"
    if segments:
        total_drone = sum(e - s for s, e in segments)
        title += f"  |  drone={total_drone:.0f}s in {len(segments)} segment(s)"
    ax_top.set_title(title, color="#ccc", fontsize=10)
    ax_top.tick_params(colors="#888", labelsize=9)
    ax_top.set_facecolor("#0e1117")
    for spine in ax_top.spines.values():
        spine.set_color("#333")

    if 0 <= cursor_t <= duration:
        idx = min(int(cursor_t / hop_s), len(frame_scores) - 1)
        ax_top.annotate(f"{frame_scores[idx]:.3f}", (cursor_t, frame_scores[idx]),
                        textcoords="offset points", xytext=(6, 6), color="#FF0000", fontsize=9)

    # ── Spectrogram ──
    n_mels, n_frames = spec_db.shape
    extent = [0, n_frames * spec_fs, 0, n_mels]
    im = ax_bot.imshow(spec_db, aspect="auto", origin="lower", extent=extent,
                        cmap="magma", vmin=-80, vmax=0)
    ax_bot.axvline(cursor_t, color="#FF0000", linewidth=1.5, alpha=0.9)
    ax_bot.set_xlabel("Time (s)", color="#ccc")
    ax_bot.set_ylabel("Mel bin", color="#ccc")
    ax_bot.tick_params(colors="#888", labelsize=9)
    ax_bot.set_facecolor("#0e1117")
    for spine in ax_bot.spines.values():
        spine.set_color("#333")
    cbar = fig.colorbar(im, ax=ax_bot, fraction=0.02, pad=0.02)
    cbar.set_label("dB", color="#ccc")
    cbar.ax.tick_params(colors="#888", labelsize=8)

    fig.tight_layout(pad=1.2)
    return fig


def _export_dataset(segments: dict, frame_scores: dict, hop_s: float):
    """Export all marked segments as a CSV with per-segment features."""
    import csv
    rows = []
    for alert_dir, segs in segments.items():
        fs = frame_scores.get(alert_dir)
        if fs is None:
            continue
        score_t = np.arange(len(fs)) * hop_s
        for s_start, s_end in segs:
            mask = (score_t >= s_start) & (score_t <= s_end)
            seg_scores = fs[mask] if mask.any() else np.array([0])
            rows.append({
                "alert_dir": alert_dir,
                "start_s": f"{s_start:.1f}",
                "end_s": f"{s_end:.1f}",
                "duration_s": f"{s_end - s_start:.1f}",
                "max_score": f"{seg_scores.max():.4f}",
                "mean_score": f"{seg_scores.mean():.4f}",
                "p90_score": f"{np.percentile(seg_scores, 90):.4f}",
                "n_frames": len(seg_scores),
            })
    path = DATA / "segments_dataset.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["alert_dir"])
        w.writeheader()
        w.writerows(rows)


def main():
    st.title("Field Alert Annotator")
    st.caption("Scrub → mark drone start/end → segments shaded on both plots")

    scores_meta, frame_scores = load_data()
    alerts = scores_meta["alerts"]
    thresholds = [0.5, 0.75, 0.9]
    hop_s = scores_meta.get("hop_s", 0.32)

    if "labels" not in st.session_state:
        st.session_state.labels = load_labels()
    labels = st.session_state.labels

    if "segments" not in st.session_state:
        st.session_state.segments = load_segments()
    segments = st.session_state.segments

    # Order — sort by directory name
    ordered = sorted(alerts, key=lambda a: a["alert_dir"])

    if "alert_idx" not in st.session_state:
        st.session_state.alert_idx = 0
    idx = st.session_state.alert_idx
    idx = max(0, min(idx, len(ordered) - 1)) if ordered else 0

    # ── Sidebar ──
    with st.sidebar:
        st.header("Progress")
        n_labeled = sum(1 for v in labels.values() if v in ("drone", "nodrone"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Done", f"{n_labeled}/{len(alerts)}")
        c2.metric("Drone", sum(1 for v in labels.values() if v == "drone"))
        c3.metric("Noise", sum(1 for v in labels.values() if v == "nodrone"))

        if n_labeled > 0:
            st.divider()
            st.subheader("Precision")
            for th in thresholds:
                s = compute_stats(alerts, labels, frame_scores, th)
                st.markdown(f"**σ={th:.2f}**  {s['precision']:.2f}  "
                            f"(TP={s['tp']} FP={s['fp']} FN={s['fn']})")

        st.divider()
        if st.button("← Prev", use_container_width=True):
            st.session_state.alert_idx = max(0, idx - 1); st.rerun()
        if st.button("Next →", use_container_width=True):
            st.session_state.alert_idx = min(len(ordered) - 1, idx + 1) if ordered else 0; st.rerun()
        st.caption(f"{idx + 1} / {len(ordered)}")
        fst = next((i for i, a in enumerate(ordered) if a["alert_dir"] not in labels), None)
        if fst is not None:
            if st.button("→ Next unlabeled", use_container_width=True, type="primary"):
                st.session_state.alert_idx = fst; st.rerun()

        st.divider()
        if st.button("Export all"):
            save_labels(labels)
            save_segments(segments)
            n_seg = sum(len(v) for v in segments.values())
            st.success(f"Saved {n_labeled} labels, {n_seg} segments")

        if st.button("📦 Export dataset CSV"):
            _export_dataset(segments, frame_scores, hop_s)
            n_seg = sum(len(v) for v in segments.values())
            st.success(f"Exported {n_seg} segments to {DATA / 'segments_dataset.csv'}")

    if not ordered:
        st.info("No alerts."); return

    alert = ordered[idx]
    alert_dir = alert["alert_dir"]
    wav_path = DATA / "alerts" / alert_dir / "full_120s.wav"
    wav_str = str(wav_path)
    current_label = labels.get(alert_dir, "")
    file_segments = segments.get(alert_dir, [])

    # ── Label buttons ──
    bc1, bc2, bc3 = st.columns([1, 1, 1])
    with bc1:
        if st.button("🛸 DRONE", key="btn_drone", use_container_width=True):
            labels[alert_dir] = "drone"; st.session_state.labels = labels; save_labels(labels)
            nu = next((i for i in range(idx + 1, len(ordered))
                       if ordered[i]["alert_dir"] not in labels), None)
            st.session_state.alert_idx = nu if nu is not None else min(idx + 1, len(ordered) - 1)
            st.rerun()
    with bc2:
        if st.button("❌ NO DRONE", key="btn_nodrone", use_container_width=True):
            labels[alert_dir] = "nodrone"; st.session_state.labels = labels; save_labels(labels)
            nu = next((i for i in range(idx + 1, len(ordered))
                       if ordered[i]["alert_dir"] not in labels), None)
            st.session_state.alert_idx = nu if nu is not None else min(idx + 1, len(ordered) - 1)
            st.rerun()
    with bc3:
        if st.button("⏭ Skip", key="btn_skip", use_container_width=True):
            st.session_state.alert_idx = min(idx + 1, len(ordered) - 1); st.rerun()

    if current_label:
        clr = "#2F9E44" if current_label == "drone" else "#E03131"
        st.markdown(f"**Label:** <span style='color:{clr};font-size:1rem'>{current_label.upper()}</span>",
                    unsafe_allow_html=True)
        if st.button("Clear label"):
            labels.pop(alert_dir, None); st.session_state.labels = labels; save_labels(labels); st.rerun()

    # ── Segment controls ──
    # Use pending values from quick-set buttons
    if "pending_start" not in st.session_state:
        st.session_state.pending_start = 0.0
    if "pending_end" not in st.session_state:
        st.session_state.pending_end = 120.0

    seg_col1, seg_col2, seg_col3, seg_col4 = st.columns([1.2, 1.2, 0.8, 2])
    with seg_col1:
        seg_start = st.number_input("Start (s)", 0.0, 120.0,
                                     st.session_state.pending_start, 0.1, key="seg_start")
    with seg_col2:
        seg_end = st.number_input("End (s)", 0.0, 120.0,
                                   st.session_state.pending_end, 0.1, key="seg_end")
    with seg_col3:
        st.caption("")
        st.caption("")
        if st.button("➕ Add", key="btn_add_seg", use_container_width=True):
            s, e = min(seg_start, seg_end), max(seg_start, seg_end)
            if e - s >= 0.5:
                segments.setdefault(alert_dir, []).append((s, e))
                segments[alert_dir].sort()
                st.session_state.segments = segments
                save_segments(segments)
                st.session_state.pending_start = seg_start
                st.session_state.pending_end = seg_end
                st.rerun()
    with seg_col4:
        if file_segments:
            st.caption(f"{len(file_segments)} segment(s):")
            for i, (s, e) in enumerate(file_segments):
                cdel, clbl = st.columns([0.3, 2])
                with cdel:
                    if st.button("✕", key=f"del_seg_{i}"):
                        del segments[alert_dir][i]
                        if not segments[alert_dir]:
                            del segments[alert_dir]
                        st.session_state.segments = segments
                        save_segments(segments)
                        st.rerun()
                with clbl:
                    st.caption(f"  {s:.1f}s – {e:.1f}s  ({e-s:.1f}s)")

    # ── Quick set from cursor ──
    st.caption("Tip: scrub time slider below, then click 'Set start' / 'Set end'")

    # ── Metrics row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Max", f"{alert['max_score']:.4f}")
    c2.metric("Mean", f"{alert['mean_score']:.4f}")
    c3.metric("P90", f"{alert['p90_score']:.4f}")

    if wav_path.exists():
        audio, rms_db, peak_db, duration = load_audio(wav_str)
        c4.metric("RMS", f"{rms_db:.1f} dB")
        c5.metric("Peak", f"{peak_db:.1f} dB")
    else:
        audio = None
        duration = 120
        c4.metric("RMS", "—")
        c5.metric("Peak", "—")

    # ── Gain + audio player ──
    gc1, gc2 = st.columns([1, 5])
    with gc1:
        gain_db = st.slider("Gain (dB)", 10, 40, 30, 1, key="gain")
    with gc2:
        if wav_path.exists():
            audio_bytes = make_audio_bytes(audio, gain_db)
            st.audio(audio_bytes, format="audio/wav")

    # ── Time slider ──
    cursor_t = st.slider("Time position (s)", 0.0, duration, 0.0, 0.1, key="cursor")

    # Quick-set buttons below slider
    qc1, qc2 = st.columns(2)
    with qc1:
        if st.button("📍 Set start", key="btn_set_start", use_container_width=True):
            st.session_state.pending_start = cursor_t
            st.rerun()
    with qc2:
        if st.button("📍 Set end", key="btn_set_end", use_container_width=True):
            st.session_state.pending_end = cursor_t
            st.rerun()

    # ── Charts ──
    if wav_path.exists():
        fs = frame_scores.get(alert_dir)
        if fs is not None:
            spec_db, spec_fs = load_spec(wav_str, hop_s)
            fig = plot_both(alert_dir, fs, hop_s, thresholds, spec_db, spec_fs,
                            cursor_t, duration, file_segments)
            st.pyplot(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
