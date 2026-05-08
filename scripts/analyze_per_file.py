"""Per-file attack analysis for production convnext_small.

Shows which attack files/segments the model misses and which
background files trigger false alarms.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# ── Project setup ──
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from audi.checkpoint import strip_compile_prefix
from audi.config import MelConfig, ModelConfig, OptimizerConfig
from audi.training.detector import DroneDetector
from audi.hysteresis import apply_hysteresis

CKPT_PATH = (
    PROJECT / "checkpoints/production_20260513_080048/"
    "02_convnext_small/checkpoints/epoch=12-step=3250.ckpt"
)
ATTACK_DIR = PROJECT / "data" / "attack_runs"
SR = MelConfig().sample_rate
CLIP_S = 5.12
CLIP_SAMPLES = int(SR * CLIP_S)
STRIDE = 0.125
SIGMA = 0.7084  # P90 threshold from production eval


def split_by_zero_gaps(audio, sr, min_dur=3.0, min_gap_s=0.5):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    exact_zero = audio == 0.0
    zero_runs = []
    in_zero, start = False, 0
    for i in range(len(exact_zero) + 1):
        z = bool(exact_zero[i]) if i < len(exact_zero) else False
        if z and not in_zero:
            start = i
            in_zero = True
        elif not z and in_zero:
            if (i - start) / sr >= min_gap_s:
                zero_runs.append((start, i))
            in_zero = False
    if not zero_runs:
        return [audio] if len(audio) / sr >= min_dur else []
    segments, prev = [], 0
    for zs, ze in zero_runs:
        if (zs - prev) / sr >= min_dur:
            segments.append(audio[prev:zs].copy())
        prev = ze
    if (len(audio) - prev) / sr >= min_dur:
        segments.append(audio[prev:].copy())
    return segments


def split_into_windows(audio, sr):
    win = int(sr * CLIP_S)
    step = int(win * STRIDE)
    return [audio[i : i + win] for i in range(0, len(audio) - win + 1, step)]


@torch.no_grad()
def predict_windows(model, windows, device):
    scores = []
    for i in range(0, len(windows), 32):
        batch = torch.as_tensor(windows[i : i + 32], dtype=torch.float32).to(device)
        logits = model(batch).cpu().numpy()
        scores.append(1.0 / (1.0 + np.exp(-logits)))
    return np.concatenate(scores) if scores else np.array([])


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    model_hp = hp.get("model", {})
    if isinstance(model_hp, dict):
        model_cfg = ModelConfig(arch=model_hp.get("arch", "cnn14"), pretrained=True, compile=False)
    else:
        model_cfg = ModelConfig(arch=model_hp.arch, pretrained=model_hp.pretrained, compile=False)
    mel_hp = hp.get("mel", {})
    mel_cfg = MelConfig(
        n_mels=mel_hp.get("n_mels", 128) if isinstance(mel_hp, dict) else mel_hp.n_mels,
        n_fft=mel_hp.get("n_fft", 1024) if isinstance(mel_hp, dict) else mel_hp.n_fft,
    )
    opt_cfg = OptimizerConfig(lr=1e-4, schedule="cosine", warmup_epochs=3, max_epochs=30)
    model = DroneDetector(model=model_cfg, mel=mel_cfg, optimizer=opt_cfg,
                          loss_type=hp.get("loss_type", "bce"),
                          label_smoothing=hp.get("label_smoothing", 0.0))
    model.load_state_dict(strip_compile_prefix(ckpt["state_dict"]), strict=False)
    return model.eval()


def main():
    print(f"Model: {CKPT_PATH}")
    print(f"Sigma: {SIGMA} (P90)")
    print(f"Clip: {CLIP_S}s, Stride: {STRIDE}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(CKPT_PATH)).to(device)

    # ── Attack files ──
    atk_files = sorted([f for f in ATTACK_DIR.glob("*.wav") if not f.name.startswith("background")])
    bg_files = sorted([f for f in ATTACK_DIR.glob("*.wav") if f.name.startswith("background")])

    print("=" * 80)
    print("ATTACK FILES (sorted by coverage, worst first)")
    print("=" * 80)
    print(f"{'file':<40} {'segs':>5} {'cov%':>7} {'1st%':>7} {'dur':>7}")
    print("-" * 62)

    atk_results = []
    for fp in atk_files:
        audio, sr = torchaudio.load(str(fp))
        audio = audio.mean(dim=0).numpy().astype(np.float32).reshape(-1)
        segs = split_by_zero_gaps(audio, sr)
        if not segs:
            continue
        seg_covs, seg_firsts, seg_durs = [], [], []
        for seg in segs:
            wins = split_into_windows(seg, sr)
            if not wins:
                seg_covs.append(0.0)
                seg_firsts.append(100.0)
                seg_durs.append(0)
                continue
            scores = predict_windows(model, np.stack(wins), device)
            dets = apply_hysteresis(scores, SIGMA)
            cov = 100.0 * dets.sum() / len(dets)
            det_idx = np.where(dets)[0]
            first = 100.0 * det_idx[0] / len(dets) if len(det_idx) > 0 else 100.0
            seg_covs.append(cov)
            seg_firsts.append(first)
            seg_durs.append(len(seg) / sr)
        avg_cov = np.mean(seg_covs)
        med_first = np.median(seg_firsts)
        atk_results.append((fp.name, len(segs), seg_covs, seg_firsts, seg_durs, avg_cov, med_first))

    for name, n_segs, covs, firsts, durs, avg_cov, med_first in sorted(atk_results, key=lambda x: x[5]):
        print(f"{name:<40} {n_segs:>5} {avg_cov:>6.1f}% {med_first:>6.1f}% {sum(durs):>6.0f}s")

    # ── Worst attack segments (per-file detail) ──
    print(f"\n{'=' * 80}")
    print("WORST ATTACK SEGMENTS (coverage < 20%)")
    print("=" * 80)
    found = False
    for name, n_segs, covs, firsts, durs, avg_cov, med_first in atk_results:
        for i, (cov, first, dur) in enumerate(zip(covs, firsts, durs)):
            if cov < 20:
                found = True
                print(f"  {name} seg{i}: cov={cov:.1f}%  first={first:.1f}%  dur={dur:.1f}s")
    if not found:
        print("  (none — all segments >20% coverage)")

    # ── Background files ──
    print(f"\n{'=' * 80}")
    print("BACKGROUND FILES (false alarm sources)")
    print("=" * 80)
    print(f"{'file':<40} {'windows':>7} {'detections':>10} {'det%':>7}")
    print("-" * 67)

    total_bg_wins = 0
    total_bg_dets = 0
    for fp in bg_files:
        audio, sr = torchaudio.load(str(fp))
        audio = audio.mean(dim=0).numpy().astype(np.float32).reshape(-1)
        wins = split_into_windows(audio, sr)
        if not wins:
            continue
        scores = predict_windows(model, np.stack(wins), device)
        dets = apply_hysteresis(scores, SIGMA)
        n_dets = int(dets.sum())
        total_bg_wins += len(wins)
        total_bg_dets += n_dets
        print(f"{fp.name:<40} {len(wins):>7} {n_dets:>10} {100*n_dets/len(wins):>6.1f}%")

    print(f"{'TOTAL':<40} {total_bg_wins:>7} {total_bg_dets:>10} {100*total_bg_dets/total_bg_wins:>6.1f}%")

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    avg_cov_all = np.mean([r[5] for r in atk_results])
    print(f"  Mean attack coverage:  {avg_cov_all:.1f}%")
    print(f"  Background detections: {total_bg_dets}/{total_bg_wins} ({100*total_bg_dets/total_bg_wins:.1f}%)")
    files_100_first = sum(1 for r in atk_results if r[6] >= 100)
    files_under_20 = sum(1 for r in atk_results if r[5] < 20)
    print(f"  Files with 100% first (no detection): {files_100_first}/{len(atk_results)}")
    print(f"  Files with <20% coverage: {files_under_20}/{len(atk_results)}")


if __name__ == "__main__":
    main()
