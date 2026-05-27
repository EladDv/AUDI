"""Streamlit app for AUDI attack eval results exploration."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torchaudio

st.set_page_config(page_title="AUDI Attack Eval", layout="wide")
st.title("AUDI Attack Eval Results")

PROJECT = Path(__file__).resolve().parent
CSV_PATH = PROJECT / "checkpoints" / "attack_run_precision_eval.csv"
ATTACK_DIR = PROJECT / "data" / "attack_runs"
FIELD_DIR = PROJECT / "data" / "field_recordings_20260514"
SR = 32000
STRIDE = 0.125
LEVELS = ["P50", "P60", "P70", "P75", "P80", "P85", "P90", "P95", "P99"]


@st.cache_data(ttl=1)
def load_data():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            r["cov_pct"] = float(r["cov_pct"])
            r["sigma"] = float(r["sigma"])
            r["first_pct"] = float(r.get("first_pct", 0) or 0)
            r["bg"] = int(r.get("bg", 0) or 0)
            r["bg_alerts"] = r.get("bg_alerts", "-") or "-"
            rows.append(r)

    pivoted = defaultdict(dict)
    for r in rows:
        key = (r["sweep"], r["model"])
        p = r["precision"]
        pivoted[key][p] = r
        pivoted[key]["sweep"] = r["sweep"]
        pivoted[key]["model"] = r["model"]

    data = []
    for (sweep, model), precs in pivoted.items():
        entry = {"model": model, "sweep": sweep}
        for lvl in LEVELS:
            p = precs.get(lvl, {})
            entry[f"{lvl}_cov"] = p.get("cov_pct", None)
            entry[f"{lvl}_bg"] = p.get("bg", None)
            entry[f"{lvl}_alerts"] = p.get("bg_alerts", None)
            entry[f"{lvl}_sigma"] = p.get("sigma", None)
        data.append(entry)

    df = pd.DataFrame(data)
    df = df.sort_values("P90_cov", ascending=False).reset_index(drop=True)
    return df


def cov_color(val):
    if pd.isna(val): return "color: #8b949e"
    if val >= 50: return "color: #3fb950"
    if val >= 30: return "color: #d2991d"
    return "color: #f85149"


def bg_color(val):
    if pd.isna(val) or val == "-": return "color: #8b949e"
    v = int(val) if val != "-" else -1
    if v == 0: return "color: #3fb950"
    if v <= 20: return "color: #d2991d"
    return "color: #f85149"


# ── Load sweep checkpoints ──────────────────────────────────────────────
@st.cache_data
def get_available_models():
    models = []
    for sweep_dir in sorted((PROJECT / "checkpoints").iterdir()):
        if not sweep_dir.is_dir():
            continue
        for run_dir in sorted(sweep_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            ckpt_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
            if not ckpt_dir.exists():
                ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            ckpts = sorted(ckpt_dir.glob("*.ckpt"))
            if ckpts:
                models.append(f"{sweep_dir.name}/{run_dir.name}")
    return sorted(models)


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: EVAL TABLE
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["Eval Table", "Model Explorer", "Scatter Compare"])

with tab1:
    df = load_data()
    sweeps = sorted(df["sweep"].unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        sweep_filter = st.selectbox("Sweep", ["All"] + sweeps)
    with col2:
        search = st.text_input("Search model", "")
    with col3:
        show_bg = st.checkbox("Show bg columns", True)
        show_alerts = st.checkbox("Show alert columns", False)

    filtered = df.copy()
    if sweep_filter != "All":
        filtered = filtered[filtered["sweep"] == sweep_filter]
    if search:
        mask = filtered["model"].str.contains(search, case=False) | filtered["sweep"].str.contains(search, case=False)
        filtered = filtered[mask]

    st.caption(f"{len(filtered)} models shown")

    # Merge field eval data if available
    FIELD_EVAL_PATH = PROJECT / "checkpoints" / "field_eval.csv"
    field_cols = []
    if FIELD_EVAL_PATH.exists():
        field_df = pd.read_csv(FIELD_EVAL_PATH)
        field_df = field_df[["sweep", "model", "bg_fp", "bg_total", "alert_tp", "alert_fn", "alert_total", "rec_alerts", "rec_segments"]]
        filtered = filtered.merge(field_df, on=["sweep", "model"], how="left")
        field_cols = ["bg_fp", "bg_total", "alert_tp", "alert_fn", "alert_total", "rec_alerts", "rec_segments"]

    # Build display dataframe with sweep info
    cov_cols = [f"{lvl}_cov" for lvl in LEVELS]
    display_cols = ["model", "sweep", *cov_cols, *field_cols]
    display = filtered[display_cols].copy()
    # Format cov columns as percentages
    for lvl in LEVELS:
        col = f"{lvl}_cov"
        display[col] = display[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "--")
    # Format field columns
    if "bg_fp" in display.columns:
        display["bg_fp"] = display["bg_fp"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "?")
        display["alert_tp"] = display["alert_tp"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "?")
        display["alert_fn"] = display["alert_fn"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "?")
        display["rec_alerts"] = display["rec_alerts"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "?")

    st.dataframe(
        display.set_index("model"),
        use_container_width=True, height=700,
    )

    # Show sigma for P90 on hover (tooltip substitute)
    with st.expander("P90 sigma values"):
        sigma_df = filtered[["model", "sweep"] + [f"{lvl}_sigma" for lvl in LEVELS]].copy()
        for lvl in LEVELS:
            col = f"{lvl}_sigma"
            sigma_df[col] = sigma_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "--")
        st.dataframe(sigma_df.set_index("model"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL EXPLORER
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    available = get_available_models()
    model_sel = st.selectbox("Model", available, key="explorer_model")

    col_a, col_b = st.columns(2)
    with col_a:
        sigma = st.number_input("Sigma (hysteresis threshold)", value=0.70, min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
    with col_b:
        run_btn = st.button("Run Analysis", type="primary")

    if run_btn:
        sweep_name, model_name = model_sel.split("/")
        sweep_dir = PROJECT / "checkpoints" / sweep_name
        run_dir = sweep_dir / model_name
        ckpt_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
        if not ckpt_dir.exists():
            ckpt_dir = run_dir / "checkpoints"
        ckpt_path = sorted(ckpt_dir.glob("*.ckpt"))[-1]

        with st.spinner(f"Loading {ckpt_path.name}..."):
            from audi.checkpoint import strip_compile_prefix, get_clip_seconds
            from audi.config import MelConfig, ModelConfig, OptimizerConfig
            from audi.training.detector import DroneDetector
            from audi.hysteresis import apply_hysteresis

            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            hp = ckpt["hyper_parameters"]
            model_hp = hp.get("model", {})
            if isinstance(model_hp, dict):
                model_cfg = ModelConfig(
                    arch=model_hp.get("arch", hp.get("model_arch", "cnn14")),
                    pretrained=model_hp.get("pretrained", hp.get("pretrained_backbone", True)),
                    compile=False,
                )
            else:
                model_cfg = ModelConfig(arch=model_hp.arch, pretrained=model_hp.pretrained, compile=False)
            mel_hp = hp.get("mel", {})
            if isinstance(mel_hp, dict):
                mel_cfg = MelConfig(
                    n_mels=mel_hp.get("n_mels", 128), n_fft=mel_hp.get("n_fft", 1024),
                    hop_length=mel_hp.get("hop_length", 160),
                )
            else:
                mel_cfg = mel_hp
            model = DroneDetector(
                model=model_cfg, mel=mel_cfg, optimizer=OptimizerConfig(),
                bin_names=hp.get("bin_names", []),
                loss_type=hp.get("loss_type", "bce"),
                label_smoothing=hp.get("label_smoothing", 0.0),
                dropout=hp.get("dropout", 0.0),
            )
            model.load_state_dict(strip_compile_prefix(ckpt["state_dict"]), strict=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            clip_s = get_clip_seconds(hp)
            del ckpt
            torch.cuda.empty_cache()

        st.success(f"Loaded: {model_cfg.arch} | clip={clip_s}s | mel={mel_cfg.n_mels} n_mels, hop={mel_cfg.hop_length}")

        # Helper functions
        def split_into_windows(audio, clip_s):
            win = int(SR * clip_s)
            step = int(win * STRIDE)
            if len(audio) < win:
                return []
            return [audio[i:i+win] for i in range(0, len(audio) - win + 1, step)]

        def split_by_zero_gaps(audio, min_dur=3.0, min_gap_s=0.5):
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            exact_zero = audio == 0.0
            zero_runs = []
            in_zero, start = False, 0
            for i in range(len(exact_zero) + 1):
                z = bool(exact_zero[i]) if i < len(exact_zero) else False
                if z and not in_zero:
                    start, in_zero = i, True
                elif not z and in_zero:
                    if (i - start) / SR >= min_gap_s:
                        zero_runs.append((start, i))
                    in_zero = False
            if not zero_runs:
                return [audio] if len(audio) / SR >= min_dur else []
            segments, prev = [], 0
            for zs, ze in zero_runs:
                if (zs - prev) / SR >= min_dur:
                    segments.append(audio[prev:zs].copy())
                prev = ze
            if (len(audio) - prev) / SR >= min_dur:
                segments.append(audio[prev:].copy())
            return segments

        @torch.no_grad()
        def predict(windows, batch_size=32):
            scores = []
            for i in range(0, len(windows), batch_size):
                batch = torch.as_tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
                logits = model(batch).cpu().numpy()
                scores.append(1.0 / (1.0 + np.exp(-logits)))
            return np.concatenate(scores).flatten() if scores else np.array([])

        def count_alerts(dets):
            if len(dets) == 0: return 0
            padded = np.pad(dets.astype(np.int8), (1, 0), constant_values=0)
            return int(np.sum((padded[1:] == 1) & (padded[:-1] == 0)))

        # ── Load audio ──
        audio_waveforms = {}
        for fp in sorted(ATTACK_DIR.glob("*.wav")):
            audio, sr_ = torchaudio.load(str(fp))
            audio_waveforms[fp.name] = audio.mean(dim=0).numpy().astype(np.float32)

        bg_names = sorted([n for n in audio_waveforms if n.startswith("background")])
        atk_names = sorted([n for n in audio_waveforms if not n.startswith("background")])

        # ══ BACKGROUND ANALYSIS ══
        st.subheader("Background Audio")
        all_bg_windows = []
        for name in bg_names:
            all_bg_windows.extend(split_into_windows(audio_waveforms[name], clip_s))

        bg_scores = predict(np.stack(all_bg_windows)) if all_bg_windows else np.array([])
        bg_dets = apply_hysteresis(bg_scores, sigma)
        bg_alerts = count_alerts(bg_dets)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Windows", len(bg_scores))
        col2.metric("Detections", f"{bg_dets.sum()} ({100*bg_dets.sum()/max(1,len(bg_dets)):.1f}%)")
        col3.metric("Alerts", bg_alerts)
        col4.metric("Max Score", f"{bg_scores.max():.3f}")

        # Score histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=bg_scores, nbinsx=50, marker_color="#58a6ff",
                                    name="Background scores"))
        fig.add_vline(x=sigma, line_dash="dash", line_color="#f85149",
                      annotation_text=f"σ={sigma:.4f}")
        fig.update_layout(
            title="Background Score Distribution",
            xaxis_title="Score", yaxis_title="Count",
            template="plotly_dark", height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-file breakdown
        per_file = []
        offset = 0
        for name in bg_names:
            n_win = len(split_into_windows(audio_waveforms[name], clip_s))
            file_scores = bg_scores[offset:offset+n_win]
            file_dets = apply_hysteresis(file_scores, sigma)
            file_alerts = count_alerts(file_dets)
            per_file.append({
                "file": name, "windows": n_win,
                "detections": int(file_dets.sum()),
                "alerts": file_alerts,
                "max_score": float(file_scores.max()),
                "mean_score": float(file_scores.mean()),
            })
            offset += n_win

        st.dataframe(pd.DataFrame(per_file).set_index("file"), use_container_width=True)

        # ══ ATTACK ANALYSIS ══
        st.subheader("Attack Audio")
        atk_results = []
        all_covs = []
        for name in atk_names:
            audio = audio_waveforms[name]
            segs = split_by_zero_gaps(audio)
            for si, seg in enumerate(segs):
                wins = split_into_windows(seg, clip_s)
                if not wins:
                    continue
                scores = predict(np.stack(wins))
                dets = apply_hysteresis(scores, sigma)
                cov = 100.0 * dets.sum() / len(dets)
                det_idx = np.where(dets)[0]
                first = 100.0 * det_idx[0] / len(dets) if len(det_idx) > 0 else 100.0
                all_covs.append(cov)
                atk_results.append({
                    "file": name, "segment": si,
                    "windows": len(wins), "cov%": round(cov, 1),
                    "first%": round(first, 1),
                    "max_score": float(scores.max()),
                })

        atk_df = pd.DataFrame(atk_results)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Segments", len(atk_df))
        col2.metric("Mean Cov", f"{atk_df['cov%'].mean():.1f}%")
        col3.metric("Median Cov", f"{atk_df['cov%'].median():.1f}%")
        col4.metric("100% coverage", int((atk_df['cov%'] == 100).sum()))

        # Coverage scatter
        fig2 = px.scatter(
            atk_df, x="first%", y="cov%", hover_data=["file", "segment", "windows"],
            title="Attack Segments: Coverage vs First Detection",
            template="plotly_dark", height=400,
        )
        fig2.update_traces(marker=dict(size=8, color="#3fb950", opacity=0.7))
        fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

        atk_df_display = atk_df.pivot_table(
            index="file", values=["cov%", "first%", "max_score"],
            aggfunc={"cov%": "mean", "first%": "median", "max_score": "max"}
        ).round(1)
        st.dataframe(atk_df_display, use_container_width=True)

        # ══ FIELD RECORDINGS ══
        if st.checkbox("Include field recordings"):
            st.subheader("Field Recordings")
            bg_dir = FIELD_DIR / "backgrounds"
            if bg_dir.exists():
                bg_files = sorted(bg_dir.glob("*.wav"))
                field_results = []
                for fp in bg_files:
                    try:
                        audio, sr_ = torchaudio.load(str(fp))
                    except Exception:
                        continue
                    audio = audio.mean(dim=0).numpy().astype(np.float32)
                    wins = split_into_windows(audio, clip_s)
                    if not wins:
                        continue
                    scores = predict(np.stack(wins))
                    dets = apply_hysteresis(scores, sigma)
                    alerts = count_alerts(dets)
                    field_results.append({
                        "file": fp.name, "type": "background",
                        "windows": len(wins), "detections": int(dets.sum()),
                        "alerts": alerts, "max_score": float(scores.max()),
                    })

                # Alerts
                alert_dir = FIELD_DIR / "alerts"
                if alert_dir.exists():
                    for d in sorted(alert_dir.iterdir()):
                        if not d.is_dir():
                            continue
                        for fp in sorted([f for f in d.glob("*.wav") if f.stat().st_size > 0]):
                            try:
                                audio, sr_ = torchaudio.load(str(fp))
                            except Exception:
                                continue
                            audio = audio.mean(dim=0).numpy().astype(np.float32)
                            wins = split_into_windows(audio, clip_s)
                            if not wins:
                                continue
                            scores = predict(np.stack(wins))
                            dets = apply_hysteresis(scores, sigma)
                            alerts = count_alerts(dets)
                            field_results.append({
                                "file": f"{d.name}/{fp.name}", "type": "alert",
                                "windows": len(wins), "detections": int(dets.sum()),
                                "alerts": alerts, "max_score": float(scores.max()),
                            })

                # Recordings (continuous segments, unknown ground truth)
                rec_dir = FIELD_DIR / "recordings"
                rec_total_windows = 0
                rec_detections = 0
                rec_alerts = 0
                rec_files = []
                if rec_dir.exists():
                    rec_files = sorted(rec_dir.glob("*.flac"))
                    rec_total_windows = 0
                    rec_detections = 0
                    rec_alerts = 0
                    for fp in rec_files:
                        try:
                            audio, sr_ = torchaudio.load(str(fp))
                        except Exception:
                            continue
                        audio = audio.mean(dim=0).numpy().astype(np.float32)
                        wins = split_into_windows(audio, clip_s)
                        if not wins:
                            continue
                        scores = predict(np.stack(wins))
                        dets = apply_hysteresis(scores, sigma)
                        alerts = count_alerts(dets)
                        rec_total_windows += len(wins)
                        rec_detections += int(dets.sum())
                        rec_alerts += alerts
                        field_results.append({
                            "file": fp.name, "type": "recording",
                            "windows": len(wins), "detections": int(dets.sum()),
                            "alerts": alerts, "max_score": float(scores.max()),
                        })

                field_df = pd.DataFrame(field_results)
                bg_fp = field_df[(field_df["type"] == "background") & (field_df["alerts"] > 0)]
                bg_ok = field_df[(field_df["type"] == "background") & (field_df["alerts"] == 0)]
                alert_tp = field_df[(field_df["type"] == "alert") & (field_df["alerts"] > 0)]
                alert_fn = field_df[(field_df["type"] == "alert") & (field_df["alerts"] == 0)]

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("BG False Positives", len(bg_fp))
                col2.metric("BG Clean", len(bg_ok))
                col3.metric("Alert TP", len(alert_tp))
                col4.metric("Alert FN", len(alert_fn))

                if rec_total_windows > 0:
                    st.metric("Recordings", f"{rec_alerts} alerts across {len(rec_files)} segments ({rec_detections} det / {rec_total_windows} windows)")

                if len(bg_fp) > 0:
                    st.warning(f"BG False Positives ({len(bg_fp)}):")
                    st.dataframe(bg_fp[["file", "detections", "alerts", "max_score"]], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: SCATTER COMPARE
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    df = load_data()
    x_level = st.selectbox("X-axis (cov%)", LEVELS, index=6, key="scatter_x")  # default P90
    y_level = st.selectbox("Y-axis (cov%)", LEVELS, index=0, key="scatter_y")  # default P50

    scatter_df = df[[f"{x_level}_cov", f"{y_level}_cov", "model", "sweep"]].dropna()
    scatter_df = scatter_df.rename(columns={
        f"{x_level}_cov": f"{x_level} coverage",
        f"{y_level}_cov": f"{y_level} coverage",
    })

    fig3 = px.scatter(
        scatter_df,
        x=f"{x_level} coverage", y=f"{y_level} coverage",
        hover_data=["model", "sweep"],
        title=f"{y_level} vs {x_level} Coverage",
        template="plotly_dark", height=600,
    )
    fig3.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                   line=dict(dash="dash", color="gray"))
    fig3.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    st.caption("Points above diagonal = models that lose more coverage at higher precision. Below diagonal = unusual (should be rare).")
