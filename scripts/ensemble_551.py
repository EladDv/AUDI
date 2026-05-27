#!/usr/bin/env python3
"""Ensemble 07_all + 12_proj128 — full-file sliding windows, matching attack_runs eval."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audi.checkpoint import load_model_from_checkpoint
from audi.hysteresis import apply_hysteresis

ROOT = Path(__file__).resolve().parents[1]
ATTACK_DIR = ROOT / "data/attack_runs"

CKPT_A = ROOT / "checkpoints/dsp_sweep_20260522_082951/07_all/checkpoints/epoch=23-step=6000.ckpt"
CKPT_B = ROOT / "checkpoints/dsp_sweep_20260522_082951/12_proj128/checkpoints/epoch=23-step=6000.ckpt"

SR = 16000; CLIP_S = 5.12; CHUNK_N = int(CLIP_S * SR)
HOP_N = int(CHUNK_N * 0.25); BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def get_scores(wav_path: Path, model) -> np.ndarray:
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1: audio = audio.mean(axis=1)
    rms = np.sqrt(np.mean(audio**2) + 1e-8)
    if rms > 0: audio = audio / rms
    scores = []
    n = max(1, (len(audio) - CHUNK_N) // HOP_N + 1)
    for b in range(0, n, BATCH_SIZE):
        be = min(b + BATCH_SIZE, n)
        chunks = [audio[i*HOP_N:i*HOP_N+CHUNK_N] for i in range(b, be)]
        chunks = [np.pad(c, (0, CHUNK_N-len(c))) if len(c) < CHUNK_N else c for c in chunks]
        t = torch.from_numpy(np.stack(chunks)).float().to(device)
        p = torch.sigmoid(model(t)).detach().squeeze(-1).cpu().numpy()
        if p.ndim == 0:
            scores.append(float(p))
        else:
            scores.extend(p.tolist())
    return np.array(scores)


def segment_by_zeros(audio: np.ndarray) -> list[tuple[int, int]]:
    """Split audio into non-silent segments using zero-gap detection (like attack_runs)."""
    bs = 0.05; bn = int(bs * SR)
    nb = len(audio) // bn
    sil = np.abs(audio[:nb*bn].reshape(nb, bn)).max(axis=1) < 1e-5
    segs = []; in_seg = False; start = 0
    for i, s in enumerate(sil):
        if not s and not in_seg:
            in_seg = True; start = i * bn
        elif s and in_seg:
            in_seg = False
            if (i * bn - start) / SR >= 0.5:
                segs.append((start, i * bn))
    if in_seg and (nb * bn - start) / SR >= 0.5:
        segs.append((start, nb * bn))
    # Merge adjacent (<0.5s gap)
    merged = [segs[0]] if segs else []
    for s, e in segs[1:]:
        if (s - merged[-1][1]) / SR < 0.5:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def main() -> int:
    print("Loading models...")
    model_a = load_model_from_checkpoint(CKPT_A, device=device); model_a.eval()
    model_b = load_model_from_checkpoint(CKPT_B, device=device); model_b.eval()

    atk_files = sorted(f for f in ATTACK_DIR.glob("*.wav")
                       if not f.name.startswith("background"))
    bg_files = sorted(ATTACK_DIR.glob("background*.wav"))

    # ── Run inference on all files ──
    print(f"Running inference: {len(atk_files)} attack files + {len(bg_files)} BG files...")
    file_scores_a, file_scores_b = {}, {}
    for f in atk_files:
        file_scores_a[f.stem] = get_scores(f, model_a)
        file_scores_b[f.stem] = get_scores(f, model_b)

    bg_scores_a, bg_scores_b = [], []
    for f in bg_files:
        s = get_scores(f, model_a); bg_scores_a.extend(s.tolist())
        s = get_scores(f, model_b); bg_scores_b.extend(s.tolist())
    bg_scores_a = np.array(bg_scores_a)
    bg_scores_b = np.array(bg_scores_b)

    # ── Segment attack files into drone runs ──
    drone_segs_a, drone_segs_b = [], []
    for f in atk_files:
        audio, _ = sf.read(str(f))
        if audio.ndim > 1: audio = audio.mean(axis=1)
        rms = np.sqrt(np.mean(audio**2) + 1e-8)
        if rms > 0: audio = audio / rms
        segs = segment_by_zeros(audio)
        sa = file_scores_a[f.stem]
        sb = file_scores_b[f.stem]
        for s_start, s_end in segs:
            # Map sample positions to score indices
            t_per_window = HOP_N / SR
            first_win = int(s_start / SR / t_per_window)
            last_win = int(s_end / SR / t_per_window)
            if first_win < len(sa) and last_win > first_win:
                drone_segs_a.append(sa[first_win:last_win+1])
                drone_segs_b.append(sb[first_win:last_win+1])

    print(f"  {len(drone_segs_a)} drone segments, {len(bg_scores_a)} BG windows")

    # ── Evaluate ──
    sigmas_a = [0.55, 0.58, 0.60, 0.62, 0.65]
    sigmas_b = [0.55, 0.58, 0.60, 0.62, 0.65]
    fusions = ["OR", "AND", "AVG", "A_ONLY", "B_ONLY"]

    print(f"\n{'='*95}")
    print(f"  Ensemble: 07_all × 12_proj128 at multiple thresholds")
    print(f"{'='*95}")

    for fusion in fusions:
        print(f"\n── {fusion} ──")
        print(f"  {'σ_a':>6s} {'σ_b':>6s}  {'cov':>5s}  {'1st%':>5s}  {'BG':>5s}")
        print(f"  {'-'*6} {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}")

        for sa in sigmas_a:
            for sb in sigmas_b:
                covs, firsts = [], []
                for scores_a, scores_b in zip(drone_segs_a, drone_segs_b):
                    if len(scores_a) == 0:
                        covs.append(0.0); firsts.append(100.0); continue
                    dets_a = apply_hysteresis(scores_a, sa)
                    dets_b = apply_hysteresis(scores_b, sb)
                    if fusion == "AND": dets = dets_a & dets_b
                    elif fusion == "OR": dets = dets_a | dets_b
                    elif fusion == "AVG":
                        avg = (scores_a + scores_b) / 2
                        dets = apply_hysteresis(avg, (sa + sb) / 2)
                    elif fusion == "A_ONLY": dets = dets_a
                    elif fusion == "B_ONLY": dets = dets_b
                    else: dets = dets_a | dets_b

                    covs.append(100.0 * dets.sum() / len(dets))
                    di = np.where(dets)[0]
                    firsts.append(100.0 * di[0] / len(dets) if len(di) else 100.0)

                # BG
                det_a = apply_hysteresis(bg_scores_a, sa)
                det_b = apply_hysteresis(bg_scores_b, sb)
                if fusion == "AND": bg = int((det_a & det_b).sum())
                elif fusion == "OR": bg = int((det_a | det_b).sum())
                elif fusion == "AVG":
                    avg = (bg_scores_a + bg_scores_b) / 2
                    bg = int(apply_hysteresis(avg, (sa+sb)/2).sum())
                elif fusion == "A_ONLY": bg = int(det_a.sum())
                elif fusion == "B_ONLY": bg = int(det_b.sum())
                else: bg = int((det_a | det_b).sum())

                print(f"  {sa:5.2f}  {sb:5.2f}  {np.mean(covs):4.1f}%  {np.median(firsts):4.1f}%  {bg:>5d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
