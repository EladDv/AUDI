from __future__ import annotations

import csv
import datetime as dt
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from datasets import load_from_disk

# NOTE: this file references audi.training.config which no longer exists.
# Replace with: from audi.config import ... (review manually)
from audi.training.dataset import (
    _decode_audio_value,
    _decode_hf_audio_row,
    _fit_length_by_loop,
)
from audi.training.hearability import (
    HearabilityTemplateBank,
    scale_drone_to_hearability_margin,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_hearability_bins_text() -> str:
    return ",".join(
        f"{b.name}:{b.low_db:g}:{b.high_db:g}:{b.probability:g}"
        for b in MixConfig().hearability_bins
    )


@dataclass(frozen=True)
class AppConfig:
    noise_dataset_path: str
    drone_dataset_path: str
    hearability_template_path: str
    split: str = TrainConfig.eval_split
    target_sr: int = MixConfig.sample_rate
    segment_sec: float = MixConfig.duration_sec
    n_tests: int = 100
    hearability_bins: tuple[HearabilityBinConfig, ...] = MixConfig.hearability_bins
    seed: int = 123


def _decode_screening_audio(row: dict[str, Any]) -> tuple[np.ndarray, int | None]:
    """Decode HF-style rows the same way as ``audi.training`` (``audio`` or ``waveform``)."""
    if "audio" in row:
        arr, sr = _decode_hf_audio_row(row, key="audio")
        return arr, sr
    if "waveform" in row:
        arr, sr = _decode_audio_value(row["waveform"])
        return arr, sr
    raise KeyError("Row must contain 'audio' or 'waveform' (see HF chunked / sharded exports)")


def _get_split(dataset_obj: Any, split: str) -> Any:
    if isinstance(dataset_obj, dict) or hasattr(dataset_obj, "keys"):
        if split not in dataset_obj:
            keys = list(dataset_obj.keys()) if hasattr(dataset_obj, "keys") else []
            raise KeyError(f"Split {split!r} missing; available: {keys}")
        return dataset_obj[split]
    return dataset_obj


def _resample_if_needed(y: np.ndarray, sr: int | None, target_sr: int) -> np.ndarray:
    if sr is None or int(sr) == int(target_sr):
        return np.asarray(y, dtype=np.float32).reshape(-1)
    return librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=int(sr), target_sr=int(target_sr))


def _safe_mix(background: np.ndarray, scaled_drone: np.ndarray) -> np.ndarray:
    """Peak-limit like training ``peak_target`` ≈ 0.99 on the sum."""
    mix = (background + scaled_drone).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.99:
        mix = (mix * (0.99 / peak)).astype(np.float32, copy=False)
    return mix


def _parse_hearability_bins(text: str) -> tuple[HearabilityBinConfig, ...]:
    bins: list[HearabilityBinConfig] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) != 4:
            raise ValueError(f"Invalid hearability bin {part!r}; expected name:low:high:weight")
        name = pieces[0].strip()
        low, high, prob = (float(x) for x in pieces[1:])
        if not name:
            raise ValueError("Hearability bin name must not be empty")
        if prob < 0.0:
            raise ValueError(f"Hearability bin {name!r} has negative weight")
        bins.append(HearabilityBinConfig(name=name, low_db=low, high_db=high, probability=prob))
    if not bins:
        raise ValueError("At least one hearability bin is required")
    total = sum(float(b.probability) for b in bins)
    if total <= 0.0:
        raise ValueError("Hearability bin weights must sum to > 0")
    return tuple(bins)


def _make_trials(cfg: AppConfig) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)
    bins = tuple(cfg.hearability_bins)
    weights = np.asarray([float(b.probability) for b in bins], dtype=np.float64)
    weights = weights / float(weights.sum())
    n_total = max(len(bins), int(cfg.n_tests))
    raw_counts = weights * float(n_total)
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(n_total - int(counts.sum()))
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        for i in order[:remainder]:
            counts[int(i)] += 1
    for i in range(len(counts)):
        if counts[i] <= 0:
            counts[i] = 1
    trials: list[dict[str, Any]] = []
    for b, c in zip(bins, counts, strict=False):
        lo = float(min(b.low_db, b.high_db))
        hi = float(max(b.low_db, b.high_db))
        for _ in range(int(c)):
            target = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
            trials.append(
                {
                    "hearability_difficulty": str(b.name),
                    "hearability_target_margin_db": target,
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "bin_probability": float(b.probability),
                }
            )
    rng.shuffle(trials)
    for i, t in enumerate(trials):
        t["trial_idx"] = i
    return trials


@st.cache_resource
def _load_datasets(noise_path: str, drone_path: str) -> tuple[Any, Any]:
    noise_ds = load_from_disk(noise_path)
    drone_ds = load_from_disk(drone_path)
    return noise_ds, drone_ds


@st.cache_resource
def _load_hearability_bank(path: str) -> HearabilityTemplateBank:
    return HearabilityTemplateBank.load(path)


def _spectrogram_figure(y: np.ndarray, sr: int, *, title: str) -> plt.Figure:
    """Mel preview using ``SpecConfig`` FFT/hop/mel counts (matches training STFT geometry)."""
    spec = SpecConfig()
    y = np.asarray(y, dtype=np.float32)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=int(spec.n_fft),
        hop_length=int(spec.hop_length),
        n_mels=int(spec.n_mels),
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=float(spec.top_db))
    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=160)
    im = ax.imshow(S_db, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Frames", fontsize=12)
    ax.set_ylabel("Mel bins", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(y, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _apply_playback_gain_db(y: np.ndarray, gain_db: float) -> np.ndarray:
    """Linear gain in dB for browser playback; clip to ±1 before WAV encode."""
    x = np.asarray(y, dtype=np.float32)
    lin = np.float32(10.0 ** (float(gain_db) / 20.0))
    return np.clip(x * lin, -1.0, 1.0).astype(np.float32, copy=False)


def _fit_logistic_torch(x_values: np.ndarray, heard: np.ndarray) -> tuple[float, float]:
    """Fit: p(heard|x) = sigmoid(a*(x - b)). Returns (a, b)."""
    x = torch.tensor(x_values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(heard.astype(np.float32), dtype=torch.float32).view(-1, 1)
    a = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float32))
    b = torch.nn.Parameter(torch.tensor([-5.0], dtype=torch.float32))
    opt = torch.optim.Adam([a, b], lr=0.05)
    for _ in range(600):
        opt.zero_grad(set_to_none=True)
        p = torch.sigmoid(a * (x - b))
        loss = torch.nn.functional.binary_cross_entropy(p, y)
        loss.backward()
        opt.step()
    return float(a.detach().cpu().item()), float(b.detach().cpu().item())


def _session_dir() -> Path:
    base = _repo_root() / "artifacts" / "hearing_sessions"
    base.mkdir(parents=True, exist_ok=True)
    if "session_id" not in st.session_state:
        st.session_state.session_id = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
    d = base / f"hearing_test_{st.session_state.session_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _persist(cfg: AppConfig) -> None:
    d = _session_dir()
    (d / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    meta = {
        "created_at_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "hearability_bins": [asdict(b) for b in cfg.hearability_bins],
        "n_tests_requested": int(cfg.n_tests),
        "n_tests_actual": int(st.session_state.get("n_tests_actual", cfg.n_tests)),
        "mixing": "background_noise_plus_drone_scaled_to_hearability_margin",
        "reference": "audi.training.hearability.scale_drone_to_hearability_margin",
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def main() -> None:
    st.set_page_config(page_title="Drone hearability test", layout="wide")

    st.markdown(
        """
<style>
  html, body, [class*="css"]  { font-size: 18px !important; }
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }
  div[data-testid="stHorizontalBlock"] { gap: 1rem; }
  h1 { font-size: 2.4rem !important; }
  h2 { font-size: 1.9rem !important; }
  h3 { font-size: 1.35rem !important; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  .muted { color: rgba(250,250,250,0.7); font-size: 1.05rem; }
  .instruction { color: #ff4b4b; font-weight: 900; font-size: 1.25rem; }
  .card {
    padding: 1.25rem 1.3rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
    box-shadow: 0 10px 26px rgba(0,0,0,0.25);
  }
  .card h3 { margin: 0 0 .35rem 0; font-size: 1.05rem; }
  .answer-row div[data-testid="column"] button {
    height: 92px !important;
    width: 100% !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    border-radius: 18px !important;
  }
  .answer-row div[data-testid="stBaseButton-primary"] > button {
    background: linear-gradient(135deg, #2F9E44, #1B5E20) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
  }
  .answer-row div[data-testid="stBaseButton-secondary"] > button {
    background: linear-gradient(135deg, #E03131, #A61E4D) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
  }
</style>
""",
        unsafe_allow_html=True,
    )

    st.title("Drone audibility vs hearability")
    st.caption(
        "Mixture matches training: **background (noise) + drone scaled to an ERB-band "
        "hearability target** using drone templates."
    )

    st.sidebar.header("Dataset inputs")
    noise_path = st.sidebar.text_input(
        "Noise dataset (`datasets.load_from_disk`)",
        value=TrainConfig.noise_dataset_path,
        help="Default matches `TrainConfig.noise_dataset_path` (e.g. HF export `hf_dataset_sharded`).",
    )
    drone_path = st.sidebar.text_input(
        "Drone dataset (`datasets.load_from_disk`)",
        value=TrainConfig.drone_dataset_path,
        help="Default matches `TrainConfig.drone_dataset_path` (e.g. `hf_dataset_chunked_drone`).",
    )
    hearability_template_path = st.sidebar.text_input(
        "Hearability template (`.npz`)",
        value=MixConfig.hearability_template_path,
        help="Build with `scripts/build_hearability_templates.py`.",
    )
    _split_opts = ("train", "validation", "test")
    _split_default = TrainConfig.eval_split if TrainConfig.eval_split in _split_opts else "validation"
    split = st.sidebar.selectbox(
        "Split",
        options=list(_split_opts),
        index=list(_split_opts).index(_split_default),
    )

    st.sidebar.header("Test configuration")
    target_sr = int(
        st.sidebar.number_input(
            "Target sample rate",
            min_value=8000,
            max_value=48000,
            value=int(MixConfig.sample_rate),
            step=1000,
        )
    )
    segment_sec = float(
        st.sidebar.number_input(
            "Segment length (s)",
            min_value=0.5,
            max_value=12.0,
            value=float(MixConfig.duration_sec),
            step=0.01,
        )
    )
    n_tests = int(st.sidebar.number_input("Number of trials", min_value=10, max_value=500, value=100, step=10))
    bins_text = st.sidebar.text_area(
        "Hearability bins",
        value=_default_hearability_bins_text(),
        help="Format: name:low_db:high_db:weight, one or more separated by commas.",
    )
    seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=10_000_000, value=123, step=1))

    st.sidebar.header("Playback")
    playback_gain_db = float(
        st.sidebar.slider(
            "Volume boost (dB)",
            min_value=-24.0,
            max_value=30.0,
            value=0.0,
            step=0.5,
            help="Applied only in the browser audio player. Does not change the hearability-scaled mixture; "
            "samples are clipped to ±1 before WAV encoding.",
        )
    )

    try:
        hearability_bins = _parse_hearability_bins(bins_text)
    except ValueError as exc:
        st.error(str(exc))
        return

    cfg = AppConfig(
        noise_dataset_path=noise_path,
        drone_dataset_path=drone_path,
        hearability_template_path=hearability_template_path,
        split=split,
        target_sr=target_sr,
        segment_sec=segment_sec,
        n_tests=n_tests,
        hearability_bins=hearability_bins,
        seed=seed,
    )

    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        if st.button("Start / Reset session", type="primary"):
            trials = _make_trials(cfg)
            st.session_state.trials = trials
            st.session_state.responses = []
            st.session_state.trial_ptr = 0
            st.session_state.session_id = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
            st.session_state.n_tests_actual = len(trials)
            _persist(cfg)
    with cols[1]:
        st.caption("Tip: use headphones for consistency.")
    with cols[2]:
        reveal = st.toggle("Reveal hearability target (for debugging)", value=False)
    with cols[3]:
        st.caption(f"Session dir: `{_session_dir().as_posix()}`")

    if "trials" not in st.session_state:
        st.info("Click **Start / Reset session** to begin.")
        return

    trials: list[dict[str, Any]] = st.session_state.trials
    responses: list[dict[str, Any]] = st.session_state.responses
    ptr: int = int(st.session_state.trial_ptr)

    n_actual = int(st.session_state.get("n_tests_actual", len(trials)))
    if n_actual != int(cfg.n_tests):
        st.caption(f"Using **{n_actual}** trials to keep bins perfectly balanced.")

    top_l, top_r = st.columns([3, 2])
    with top_l:
        st.markdown(
            f"""
<div class="card">
  <h3>Progress</h3>
  <div class="muted">Trial <b>{min(ptr + 1, len(trials))}</b> / <b>{len(trials)}</b></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, ptr / max(1, len(trials))), text="")
    with top_r:
        st.markdown(
            f"""
<div class="card">
  <h3>Session</h3>
  <div class="muted">Outputs: <code>{_session_dir().as_posix()}</code></div>
</div>
""",
            unsafe_allow_html=True,
        )

    if ptr >= len(trials):
        st.success("All trials complete.")
        if not responses:
            st.warning("No responses recorded.")
            return

        out_dir = _session_dir()
        csv_path = out_dir / "responses.csv"
        fieldnames = sorted({k for r in responses for k in r.keys()})
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(responses)
        csv_text = csv_buf.getvalue()
        csv_path.write_text(csv_text)

        st.subheader("Results")
        st.dataframe(responses, use_container_width=True)
        st.download_button("Download CSV", data=csv_text, file_name="responses.csv", mime="text/csv")

        bins = tuple(cfg.hearability_bins)
        centers = np.array(
            [(float(b.low_db) + float(b.high_db)) / 2.0 for b in bins],
            dtype=np.float64,
        )
        p_hat = np.full(len(bins), np.nan, dtype=np.float64)
        n_bin = np.zeros(len(bins), dtype=np.int64)
        margins_all = np.asarray(
            [float(r["hearability_margin_db"]) for r in responses],
            dtype=np.float64,
        )
        heard_all = np.asarray([1 if bool(r["heard_drone"]) else 0 for r in responses], dtype=np.int64)
        for bi, b in enumerate(bins):
            lo = float(min(b.low_db, b.high_db))
            hi = float(max(b.low_db, b.high_db))
            if bi == len(bins) - 1:
                m = (margins_all >= lo) & (margins_all <= hi)
            else:
                m = (margins_all >= lo) & (margins_all < hi)
            if bool(m.any()):
                n_bin[bi] = int(m.sum())
                p_hat[bi] = float(np.mean(heard_all[m]))

        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
        ax.plot(centers, p_hat, "o-", label="Observed (bin mean)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Hearability margin (dB)")
        ax.set_ylabel("P(heard drone)")
        ax.grid(True, alpha=0.3)

        a, b = _fit_logistic_torch(margins_all, heard_all)
        xs = np.linspace(float(np.min(margins_all)), float(np.max(margins_all)), 200)
        ys = 1.0 / (1.0 + np.exp(-(a * (xs - b))))
        ax.plot(xs, ys, "-", label=f"Logistic fit (a={a:.2f}, b={b:.2f})")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        st.caption(f"Saved: `{csv_path.as_posix()}`")
        return

    trial = trials[ptr]

    noise_ds_obj, drone_ds_obj = _load_datasets(cfg.noise_dataset_path, cfg.drone_dataset_path)
    noise_ds = _get_split(noise_ds_obj, cfg.split)
    drone_ds = _get_split(drone_ds_obj, cfg.split)
    try:
        hearability_bank = _load_hearability_bank(cfg.hearability_template_path)
    except Exception as exc:
        st.error(f"Could not load hearability template: {exc}")
        return
    if int(hearability_bank.sample_rate) != int(cfg.target_sr):
        st.error(
            "Hearability template sample rate must match Target sample rate: "
            f"{hearability_bank.sample_rate} != {cfg.target_sr}"
        )
        return

    rng = np.random.default_rng(int(cfg.seed) + int(trial["trial_idx"]) * 1_000_003)
    noise_row = noise_ds[int(rng.integers(0, len(noise_ds)))]
    drone_row = drone_ds[int(rng.integers(0, len(drone_ds)))]
    y_noise, sr_noise = _decode_screening_audio(noise_row)
    y_drone, sr_drone = _decode_screening_audio(drone_row)

    y_noise = _resample_if_needed(y_noise, sr_noise, cfg.target_sr)
    y_drone = _resample_if_needed(y_drone, sr_drone, cfg.target_sr)

    seg_len = int(round(cfg.segment_sec * cfg.target_sr))
    y_noise = _fit_length_by_loop(y_noise, seg_len, rng=rng, mix_cfg=None, training=False)
    y_drone = _fit_length_by_loop(y_drone, seg_len, rng=rng, mix_cfg=None, training=False)

    drone_label = str(drone_row.get("label", ""))
    target_margin_db = float(trial["hearability_target_margin_db"])
    y_drone_scaled, h_metrics = scale_drone_to_hearability_margin(
        y_drone,
        y_noise,
        label=drone_label,
        target_margin_db=target_margin_db,
        bank=hearability_bank,
    )
    y_mix = _safe_mix(y_noise, y_drone_scaled)

    st.subheader("Listen and answer")
    if reveal:
        st.write(
            {
                "difficulty": trial["hearability_difficulty"],
                "target_margin_db": target_margin_db,
                "measured_margin_db": float(h_metrics["hearability_margin_db"]),
                "gain_db": float(h_metrics["hearability_gain_db"]),
                "bin": [trial["bin_lo"], trial["bin_hi"]],
            }
        )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Audio (mixture)")
    y_play = _apply_playback_gain_db(y_mix, playback_gain_db)
    if playback_gain_db > 0.0:
        peak_u = float(np.max(np.abs(np.asarray(y_mix, dtype=np.float64))))
        lin = float(10.0 ** (playback_gain_db / 20.0))
        if peak_u > 1e-8 and peak_u * lin > 1.0 + 1e-7:
            st.caption(
                "Playback boost exceeds ±1 before clipping—reduce **Volume boost (dB)** or raise OS/system volume instead."
            )
    st.audio(_wav_bytes(y_play, cfg.target_sr), format="audio/wav")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Reference: drone only (dry, **not** hearability-scaled)")
    st.markdown(
        "<div class='muted'>Same cropped segment as in this trial—the **original** drone RMS before scaling into the mixture. Same **Volume boost (dB)** as the mixture player.</div>",
        unsafe_allow_html=True,
    )
    y_drone_dry_play = _apply_playback_gain_db(np.clip(np.asarray(y_drone, dtype=np.float32), -1.0, 1.0), playback_gain_db)
    if playback_gain_db > 0.0:
        peak_d = float(np.max(np.abs(np.asarray(y_drone, dtype=np.float64))))
        lin = float(10.0 ** (playback_gain_db / 20.0))
        if peak_d > 1e-8 and peak_d * lin > 1.0 + 1e-7:
            st.caption("(Drone reference may be clipping—lower **Volume boost (dB)** if distorted.)")
    st.audio(_wav_bytes(y_drone_dry_play, cfg.target_sr), format="audio/wav")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Do you hear a drone in the **mixture** (first player)?")
    st.markdown(
        "<div class='instruction'>Judge only the **mixture**—the dry clip is optional practice. Ignore the spectrograms. Keep volume constant.</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="answer-row">', unsafe_allow_html=True)
    b1, b2 = st.columns([1, 1])
    with b1:
        yes = st.button("YES — I hear it", type="primary", use_container_width=True)
    with b2:
        no = st.button("NO — I don't", type="secondary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if yes or no:
        resp = {
            "trial_idx": int(trial["trial_idx"]),
            "hearability_difficulty": str(trial["hearability_difficulty"]),
            "hearability_target_margin_db": float(target_margin_db),
            "hearability_margin_db": float(h_metrics["hearability_margin_db"]),
            "hearability_gain_db": float(h_metrics["hearability_gain_db"]),
            "erb_weighted_snr_db": float(h_metrics["erb_weighted_snr_db"]),
            "erb_peak_snr_db": float(h_metrics["erb_peak_snr_db"]),
            "audible_band_fraction": float(h_metrics["audible_band_fraction"]),
            "bin_lo": float(trial["bin_lo"]),
            "bin_hi": float(trial["bin_hi"]),
            "heard_drone": bool(yes),
            "noise_label": str(noise_row.get("label", "")),
            "drone_label": drone_label,
        }
        responses.append(resp)
        st.session_state.responses = responses
        st.session_state.trial_ptr = ptr + 1
        st.rerun()

    st.markdown("### Spectrograms")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.pyplot(
            _spectrogram_figure(y_mix, cfg.target_sr, title="Mixture (noise + scaled drone)"),
        clear_figure=True,
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.pyplot(
            _spectrogram_figure(
                y_drone_scaled,
                cfg.target_sr,
                title="Drone (hearability-scaled)",
            ),
            clear_figure=True,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.pyplot(
            _spectrogram_figure(y_noise, cfg.target_sr, title="Background noise"),
            clear_figure=True,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
