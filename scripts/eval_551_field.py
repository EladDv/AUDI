#!/usr/bin/env python3
"""Field eval: multiple models against 551 drone recordings.

Reports per-model: incidents detected, coverage, first-chunk fraction, BG alarms.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audi.checkpoint import load_model_from_checkpoint

ROOT = Path("/home/elad/projects/AUDI")
REC_DIR = ROOT / "data/551/Device_1_MultiMicRecorder_8_5-11_5"
TAG_DIR = ROOT / "data/551/TAGS_PFK_Device_1_MultiMicRecorder_11.05"

CLIP_S = 5.12
OVERLAP = 0.75
SR = 16000
CHUNK_N = int(CLIP_S * SR)
HOP_N = int(CHUNK_N * (1 - OVERLAP))
BATCH_SIZE = 32

HYST_W = 8
HYST_R = 0.6
HYST_M = 0.05

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_time(t: str) -> float:
    if ":" not in t:
        return float(t)
    mm, ss = t.split(":")
    return int(mm) * 60 + float(ss)


def parse_tags(tag_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """{wav_stem: [(start_s, end_s), ...]} drone segments."""
    segments: dict[str, list[tuple[float, float]]] = {}
    for csv_path in sorted(tag_dir.glob("*.csv")):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        wav_stem = None
        for row in rows:
            name = row.get("Name", "").strip()
            dur = parse_time(row.get("Duration", "0:00.000"))
            desc = row.get("Description", "").strip()
            if (desc == "Device1" or "rec_" in name) and dur == 0:
                wav_stem = name.replace(".wav", "")
                break
        if not wav_stem:
            stem = csv_path.stem
            wav_stem = stem[len("Device1_"):] if stem.startswith("Device1_") else stem
        drones = []
        for row in rows:
            name = row.get("Name", "").strip()
            if "רחפן" in name:
                s = parse_time(row["Start"])
                d = parse_time(row["Duration"])
                if d > 0:
                    drones.append((s, s + d))
        if drones:
            segments[wav_stem] = drones
    return segments


def hysteresis(scores: np.ndarray, sigma: float) -> np.ndarray:
    dets = np.zeros(len(scores), dtype=bool)
    state = False
    for i in range(len(scores)):
        w = scores[max(0, i - HYST_W + 1): i + 1]
        if not state:
            if np.sum(w >= sigma + HYST_M) >= HYST_R * HYST_W:
                state = True
        else:
            if np.sum(w <= sigma - HYST_M) >= HYST_R * HYST_W:
                state = False
        dets[i] = state
    return dets


@torch.no_grad()
def infer(wav_path: Path, model) -> tuple[np.ndarray, np.ndarray]:
    audio, sr = sf.read(str(wav_path))
    if sr != SR:
        raise ValueError(f"Bad sr: {sr}")
    rms = np.sqrt(np.mean(audio**2) + 1e-8)
    if rms > 0:
        audio = audio / rms
    scores = []
    n = max(1, (len(audio) - CHUNK_N) // HOP_N + 1)
    for b in range(0, n, BATCH_SIZE):
        be = min(b + BATCH_SIZE, n)
        chunks = []
        for i in range(b, be):
            c = audio[i * HOP_N: i * HOP_N + CHUNK_N]
            if len(c) < CHUNK_N:
                c = np.pad(c, (0, CHUNK_N - len(c)))
            chunks.append(c)
        t = torch.from_numpy(np.stack(chunks)).float().to(device)
        p = torch.sigmoid(model(t)).squeeze(-1).cpu().numpy()
        scores.extend(float(p) if p.ndim == 0 else p.tolist())
    scores = np.array(scores)
    times = np.arange(len(scores)) * (HOP_N / SR) + (CHUNK_N / 2 / SR)
    return scores, times


def eval_one(scores: np.ndarray, times: np.ndarray,
             drones: list[tuple[float, float]], sigma: float) -> dict:
    dets = hysteresis(scores, sigma)

    n_detected = 0
    first_fractions = []  # 0=instant, 1=never

    for seg_s, seg_e in drones:
        mask = (times >= seg_s) & (times <= seg_e)
        sd = dets[mask]
        if sd.any():
            n_detected += 1
            # find first detection index within segment
            idx = np.argmax(sd)  # first True
            det_time = times[mask][idx]
            frac = (det_time - seg_s) / max(seg_e - seg_s, 0.001)
            first_fractions.append(min(frac, 1.0))
        else:
            first_fractions.append(1.0)

    # BG: detections outside all drone intervals
    bg_mask = np.ones(len(times), dtype=bool)
    for seg_s, seg_e in drones:
        bg_mask &= ~((times >= seg_s) & (times <= seg_e))
    bg_dets = dets[bg_mask]
    transitions = 0
    prev = False
    for d in bg_dets:
        if d and not prev:
            transitions += 1
        prev = d

    return {
        "n_detected": n_detected,
        "n_total": len(drones),
        "first_fracs": first_fractions,
        "bg": transitions,
    }


# ── globals ──
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Models to test ──
MODELS = [
    # (label, ckpt_path)
    ("wd_0.03_warmup8", "checkpoints/bce_wd_sweep_20260518_122516/01_wd_003/checkpoints/epoch=12-step=3250.ckpt"),
    ("wd_0.02_warmup8", "checkpoints/bce_wd_sweep_20260518_122516/02_wd_002/checkpoints/epoch=12-step=3250.ckpt"),
    ("wd_0.015_warmup8", "checkpoints/bce_wd_sweep_20260518_122516/03_wd_0015/checkpoints/epoch=12-step=3250.ckpt"),
    ("wd_0.03_nowarmup", "checkpoints/bce_wd_warmup_20260517_175750/01_wd_constant/checkpoints/epoch=12-step=3250.ckpt"),
    ("wd_0.03_warmup5", "checkpoints/bce_wd_warmup_20260517_175750/02_wd_warmup5/checkpoints/epoch=12-step=3250.ckpt"),
    ("focal_smooth_ep15", "checkpoints/convnext_bfr_20260514_114930/02_focal_smooth/checkpoints/epoch=15-step=4000.ckpt"),
    ("focal_aug_1e5", "checkpoints/prod_focal_20260516_160031/04_aug_1e5/checkpoints/epoch=12-step=3250.ckpt"),
]


def main() -> int:
    print(f"Device: {device}")
    print(f"Loading tags...")
    all_segs = parse_tags(TAG_DIR)
    total_drones = sum(len(s) for s in all_segs.values())
    print(f"  {len(all_segs)} files, {total_drones} drone incidents\n")

    # Pre-infer all WAVs once, cache scores per file
    print("Pre-computing inference on all 11 WAVs (shared across models)...")
    t0 = time.time()
    # Use a dummy model to get shapes, then replace per-model
    wav_scores = {}
    wav_times = {}
    wav_drones = {}

    # We need a model for first pass — use first model
    first_model = load_model_from_checkpoint(MODELS[0][1], device=device)
    first_model.eval()

    for wav_stem, drones in sorted(all_segs.items()):
        wav_path = REC_DIR / f"{wav_stem}.wav"
        scores, times = infer(wav_path, first_model)
        wav_scores[wav_stem] = scores
        wav_times[wav_stem] = times
        wav_drones[wav_stem] = drones
        print(f"  {wav_stem}: {len(drones):2d} drones, {len(scores)} windows")

    print(f"  inference cache done in {time.time() - t0:.1f}s\n")

    # ── Evaluate each model ──
    print(f"{'='*90}")
    print(f"  FIELD EVAL — Multi-model comparison on 551 May-11 recordings")
    print(f"  {len(all_segs)} files, {total_drones} drone incidents")
    print(f"{'='*90}")

    header = f"{'Model':<24s} {'σ':>6s} {'Detected':>8s} {'Cov%':>6s} {'1stFrac':>7s} {'BG':>4s}"
    print(header)
    print("-" * 90)

    for label, ckpt_rel in MODELS:
        ckpt_path = ROOT / ckpt_rel
        print(f"\n── {label} ──")
        model = load_model_from_checkpoint(ckpt_path, device=device)
        model.eval()

        for wav_stem in wav_scores:
            scores, times = infer(REC_DIR / f"{wav_stem}.wav", model)
            wav_scores[wav_stem] = scores

        for sigma in THRESHOLDS:
            tot_detected = 0
            tot_total = 0
            all_first_fracs = []
            tot_bg = 0

            for wav_stem in wav_scores:
                scores = wav_scores[wav_stem]
                times = wav_times[wav_stem]
                drones = wav_drones[wav_stem]
                r = eval_one(scores, times, drones, sigma)
                tot_detected += r["n_detected"]
                tot_total += r["n_total"]
                all_first_fracs.extend(r["first_fracs"])
                tot_bg += r["bg"]

            cov = tot_detected / tot_total * 100
            avg_frac = np.mean(all_first_fracs) * 100  # % of incident elapsed
            print(
                f"  {label:<24s} {sigma:5.2f}  "
                f"{tot_detected:>4}/{tot_total:<4} {cov:5.1f}%  {avg_frac:5.1f}%  {tot_bg:>4}"
            )

    print(f"\n{'='*90}")
    print("  1stFrac = mean % of drone incident elapsed before first detection")
    print("  Lower 1stFrac = faster detection. 100% = never detected.")
    print(f"{'='*90}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
