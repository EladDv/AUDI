#!/usr/bin/env python3
"""Run wd_003 model on all field alert 120s WAVs at thresholds 0.5 / 0.75 / 0.9.

Usage:
    uv run python scripts/eval_alerts_wd003.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audi.checkpoint import load_model_from_checkpoint

ROOT = Path(__file__).resolve().parent.parent
ALERTS_DIR = ROOT / "data/field_recordings_20260514/alerts"
LABELS_CSV = ROOT / "data/field_recordings_20260514/labels.csv"
CKPT = "checkpoints/bce_wd_sweep_20260518_122516/01_wd_003/checkpoints/epoch=12-step=3250.ckpt"
THRESHOLDS = [0.5, 0.75, 0.9]
HOP_S = 0.32
BATCH_SIZE = 32
SR = 16000


def find_120s_wavs() -> list[Path]:
    paths = [p for p in sorted(ALERTS_DIR.glob("*/full_120s.wav"))
             if p.stat().st_size > 0]
    return paths


@torch.no_grad()
def infer(wav_path: Path, model, clip_s: float) -> np.ndarray:
    """Return raw scores (sigmoid applied) for a WAV file."""
    audio, sr = sf.read(str(wav_path))
    if sr != SR:
        raise ValueError(f"Bad sr: {sr}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # RMS normalise (same as training/eval)
    rms = np.sqrt(np.mean(audio**2) + 1e-8)
    if rms > 0:
        audio = audio / rms

    chunk_n = int(clip_s * SR)
    hop_n = int(HOP_S * SR)
    n = max(1, (len(audio) - chunk_n) // hop_n + 1)

    scores = []
    for b in range(0, n, BATCH_SIZE):
        be = min(b + BATCH_SIZE, n)
        chunks = []
        for i in range(b, be):
            c = audio[i * hop_n : i * hop_n + chunk_n]
            if len(c) < chunk_n:
                c = np.pad(c, (0, chunk_n - len(c)))
            chunks.append(c)
        t = torch.from_numpy(np.stack(chunks)).float().to(device)
        p = torch.sigmoid(model(t)).squeeze(-1).cpu().numpy()
        if p.ndim == 0:
            scores.append(float(p))
        else:
            scores.extend(p.tolist())
    return np.array(scores)


def main() -> int:
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    wavs = find_120s_wavs()
    print(f"Found {len(wavs)} full_120s.wav files\n")

    # Load model
    ckpt_path = ROOT / CKPT
    print(f"Loading: {CKPT}")
    model = load_model_from_checkpoint(ckpt_path, device=device)
    model.eval()

    # Get clip seconds from checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    clip_s = float(ckpt["hyper_parameters"].get("clip_seconds", 5.12))
    del ckpt
    print(f"Clip length: {clip_s}s\n")

    # Run inference on all files
    results = []  # (wav_name, max_score, mean_score, p90_score)
    all_scores = {}  # alert_dir -> np.ndarray of per-frame scores
    t0 = time.time()
    for i, wav in enumerate(wavs):
        scores = infer(wav, model, clip_s)
        alert_dir = wav.parent.name
        results.append((alert_dir, scores.max(), scores.mean(),
                        np.percentile(scores, 90)))
        all_scores[alert_dir] = scores.astype(np.float16)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(wavs)}] {alert_dir} max={scores.max():.4f} ...")

    elapsed = time.time() - t0
    print(f"\nInference done in {elapsed:.1f}s ({elapsed/len(wavs):.1f}s/file)\n")

    # ── Results ──
    print("=" * 72)
    print(f"  wd_003 on {len(wavs)} field alert 120s recordings")
    print("=" * 72)

    for thresh in THRESHOLDS:
        # Files where max score >= threshold
        passed = [r for r in results if r[1] >= thresh]
        n_pass = len(passed)
        pct = n_pass / len(results) * 100

        print(f"\n── threshold = {thresh:.2f} ──")
        print(f"  Passed: {n_pass}/{len(results)} ({pct:.1f}%)")

        if n_pass > 0:
            maxes = np.array([r[1] for r in passed])
            means = np.array([r[2] for r in passed])
            p90s = np.array([r[3] for r in passed])
            print(f"  Max score stats:  mean={maxes.mean():.4f}  "
                  f"min={maxes.min():.4f}  max={maxes.max():.4f}")
            print(f"  Mean score stats: mean={means.mean():.4f}  "
                  f"min={means.min():.4f}  max={means.max():.4f}")
            print(f"  P90 score stats:  mean={p90s.mean():.4f}  "
                  f"min={p90s.min():.4f}  max={p90s.max():.4f}")

        if n_pass < len(results):
            failed = [r for r in results if r[1] < thresh]
            print(f"  Failed ({len(failed)}):")
            for r in failed[:10]:
                print(f"    {r[0]:30s} max={r[1]:.4f}  mean={r[2]:.4f}  p90={r[3]:.4f}")
            if len(failed) > 10:
                print(f"    ... and {len(failed)-10} more")

    # ── Per-file detail table ──
    # Load manual labels for TP/FP reporting
    alert_labels = {}
    if LABELS_CSV.exists():
        import csv
        with open(LABELS_CSV) as f:
            for r in csv.DictReader(f):
                alert_labels[r["alert_dir"]] = r["label"]

    print(f"\n{'='*72}")
    print(f"  Per-file detail (sorted by max score)")
    print(f"{'='*72}")
    results.sort(key=lambda r: r[1], reverse=True)
    print(f"{'Alert':<30s} {'Max':>8s} {'Mean':>8s} {'P90':>8s}  "
          f"{' '.join(f'@{t:.2f}' for t in THRESHOLDS)}")
    print("-" * 72)
    for alert_dir, max_s, mean_s, p90_s in results:
        label = alert_labels.get(alert_dir, "?")
        marks = " ".join(" ✓" if max_s >= t else " ✗" for t in THRESHOLDS)
        print(f"{alert_dir:<30s} {max_s:8.4f} {mean_s:8.4f} {p90_s:8.4f}  {label:<8s} {marks}")

    # TP/FP summary per threshold
    if alert_labels:
        print(f"\n{'='*72}")
        print("  TP/FP breakdown (using manual labels)")
        print(f"{'='*72}")
        for thresh in THRESHOLDS:
            tp = fp = fn = tn = 0
            for alert_dir, max_s, mean_s, p90_s in results:
                label = alert_labels.get(alert_dir, "?")
                detected = max_s >= thresh
                if label == "drone":
                    if detected: tp += 1
                    else: fn += 1
                elif label == "nodrone":
                    if detected: fp += 1
                    else: tn += 1
            total = tp + fp + fn + tn
            print(f"  @{thresh:.2f}: TP={tp} FP={fp} FN={fn} TN={tn}  "
                  f"recall={100*tp/max(1,tp+fn):.1f}%  precision={100*tp/max(1,tp+fp):.1f}%")

    # ── Save scores JSON for annotation app ──
    scores_json = ROOT / "data/field_recordings_20260514/wd003_scores.json"
    payload = {
        "model": "wd_003",
        "checkpoint": CKPT,
        "clip_s": clip_s,
        "hop_s": HOP_S,
        "alerts": [
            {
                "alert_dir": alert_dir,
                "wav_rel": f"{alert_dir}/full_120s.wav",
                "max_score": float(max_s),
                "mean_score": float(mean_s),
                "p90_score": float(p90_s),
            }
            for alert_dir, max_s, mean_s, p90_s in results
        ],
    }
    import json
    scores_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved scores to {scores_json}")

    # Save per-frame scores as .npz
    npz_path = ROOT / "data/field_recordings_20260514/wd003_frames.npz"
    np.savez_compressed(npz_path, **all_scores)
    print(f"Saved per-frame scores to {npz_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
