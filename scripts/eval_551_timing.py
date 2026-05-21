#!/usr/bin/env python3
"""551 field eval — min 5s, strong vs weak, calibration sigmas."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audi.checkpoint import load_model_from_checkpoint

ROOT = Path("/home/elad/projects/AUDI")
REC_DIR = ROOT / "data/551/Device_1_MultiMicRecorder_8_5-11_5"
TAG_DIR = ROOT / "data/551/TAGS_PFK_Device_1_MultiMicRecorder_11.05"
CALIB = ROOT / "checkpoints/bce_wd_sweep_20260518_122516/01_wd_003/eval_data/curves_best.npz"

SR = 16000
CLIP_S = 5.12
CHUNK_N = int(CLIP_S * SR)
HOP_N = int(320 / 1000 * SR)  # 320ms deployment hop
BATCH_SIZE = 32
HYST_W, HYST_R, HYST_M = 8, 0.6, 0.05
MIN_DUR = 5.0

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_time(t: str) -> float:
    if ":" not in t:
        return float(t)
    mm, ss = t.split(":")
    return int(mm) * 60 + float(ss)


def parse_tags() -> dict[str, list[tuple[float, float, str]]]:
    raw: dict[str, list] = {}
    for csv_path in sorted(TAG_DIR.glob("*.csv")):
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
        for row in rows:
            name = row.get("Name", "").strip()
            if "רחפן" in name:
                s = parse_time(row["Start"])
                d = parse_time(row["Duration"])
                if d > 0:
                    raw.setdefault(wav_stem, []).append((s, s + d, name))
    out: dict[str, list] = {}
    for stem, segs in raw.items():
        segs.sort()
        merged = [segs[0]]
        for s, e, label in segs[1:]:
            if s - merged[-1][1] < 2.0:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e),
                              merged[-1][2] + " + " + label)
            else:
                merged.append((s, e, label))
        out[stem] = merged
    return out


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
def get_scores(wav_path: Path, model) -> np.ndarray:
    audio, sr = sf.read(str(wav_path))
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
    return np.array(scores)


def window_times(n: int) -> np.ndarray:
    return np.arange(n) * (HOP_N / SR) + (CHUNK_N / 2 / SR)


def get_calib_sigmas() -> dict[str, float]:
    c = np.load(CALIB)
    fpr = c["overall/fpr"]
    th = c["overall/thresholds"]
    sigmas = {}
    for label, target_fpr in [
        ("P50", 0.25), ("P75", 0.10), ("P80", 0.05),
        ("P85", 0.02), ("P90", 0.01),
    ]:
        idx = np.searchsorted(-fpr, -target_fpr)
        if idx >= len(th):
            idx = len(th) - 1
        sigmas[label] = float(th[idx])
    return sigmas


CKPT = ROOT / "checkpoints/bce_wd_sweep_20260518_122516/01_wd_003/checkpoints/epoch=12-step=3250.ckpt"


def eval_group(
    group_segs: dict[str, list[tuple[float, float, str]]],
    all_segs: dict[str, list[tuple[float, float, str]]],
    file_scores: dict[str, np.ndarray],
    file_times: dict[str, np.ndarray],
    sigma: float,
) -> dict:
    """Evaluate one group at one sigma."""
    tot_det = 0
    tot_n = 0
    all_rem = []
    tot_bg = 0

    for stem in sorted(file_scores):
        scores = file_scores[stem]
        times = file_times[stem]
        dets = hysteresis(scores, sigma)

        # BG mask: all drone segments >=5s in this file
        bg_mask = np.ones(len(times), dtype=bool)
        file_all = all_segs.get(stem, [])
        for seg_s, seg_e, _ in file_all:
            bg_mask &= ~((times >= seg_s) & (times <= seg_e))

        # Group segments
        file_group = group_segs.get(stem, [])
        for seg_s, seg_e, _ in file_group:
            tot_n += 1
            seg_len = seg_e - seg_s
            mask = (times >= seg_s) & (times <= seg_e)
            seg_dets = dets[mask]
            seg_times = times[mask]
            if seg_dets.any():
                tot_det += 1
                first_idx = np.argmax(seg_dets)
                dt = seg_times[first_idx]
                all_rem.append(seg_len - (dt - seg_s))
            else:
                all_rem.append(0.0)

        # BG transitions
        bg_d = dets[bg_mask]
        prev = False
        for d in bg_d:
            if d and not prev:
                tot_bg += 1
            prev = d

    rem = np.array(all_rem)
    return {
        "n_det": tot_det,
        "n_tot": tot_n,
        "cov": tot_det / tot_n * 100 if tot_n else 0,
        "rem_mean": float(rem.mean()),
        "rem_p10": float(np.percentile(rem, 10)),
        "rem_p25": float(np.percentile(rem, 25)),
        "rem_p50": float(np.median(rem)),
        "bg": tot_bg,
        "_all_rem": rem.tolist(),
    }


def main() -> int:
    print("Loading model + calibration...")
    model = load_model_from_checkpoint(CKPT, device=device)
    model.eval()
    sigmas = get_calib_sigmas()
    for k, v in sigmas.items():
        print(f"  {k}: σ={v:.4f}")

    all_segs = parse_tags()
    raw_count = sum(len(s) for s in all_segs.values())
    print(f"\nRaw segments: {raw_count}")

    # Filter to >=5s and split strong/weak
    all_5s: dict[str, list] = {}
    strong: dict[str, list] = {}
    weak: dict[str, list] = {}

    for stem, segs in all_segs.items():
        for s, e, label in segs:
            if e - s < MIN_DUR:
                continue
            all_5s.setdefault(stem, []).append((s, e, label))
            if "חלש" in label:
                weak.setdefault(stem, []).append((s, e, label))
            else:
                strong.setdefault(stem, []).append((s, e, label))

    n_all = sum(len(s) for s in all_5s.values())
    n_strong = sum(len(s) for s in strong.values())
    n_weak = sum(len(s) for s in weak.values())
    print(f"Segments >= {MIN_DUR}s:  {n_all}  (strong={n_strong}, weak={n_weak})")
    print(f"Excluded (<{MIN_DUR}s):  {raw_count - n_all}")

    # Run inference
    print("\nRunning inference on 11 files...")
    file_scores: dict[str, np.ndarray] = {}
    file_times: dict[str, np.ndarray] = {}
    for stem in sorted(all_segs):
        file_scores[stem] = get_scores(REC_DIR / f"{stem}.wav", model)
        file_times[stem] = window_times(len(file_scores[stem]))

    # ── Print tables ──
    header = (f"  {'σ':>8s}  {'Det':>5s}  {'Cov':>5s}  "
              f"{'Mean':>7s}  {'P10':>6s}  {'P25':>6s}  {'P50':>6s}  {'BG':>4s}")

    # Add raw score thresholds alongside calibration sigmas
    all_thresholds = dict(sigmas)
    for label, val in [("σ=0.3", 0.3), ("σ=0.4", 0.4), ("σ=0.5", 0.5),
                        ("σ=0.6", 0.6), ("σ=0.7", 0.7)]:
        all_thresholds[label] = val

    for group_name, group_data in [
        ("All >=5s", all_5s),
        ("Strong (no חלש)", strong),
        ("Weak (חלש only)", weak),
    ]:
        n = sum(len(s) for s in group_data.values())
        print(f"\n{'='*75}")
        print(f"  {group_name}  ({n} segments)")
        print(f"{'='*75}")
        print(header)
        print(f"  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*4}")

        for sname, sigma in sorted(all_thresholds.items(), key=lambda x: x[1]):
            r = eval_group(group_data, all_5s, file_scores, file_times, sigma)
            rem = np.array(r.get("_all_rem", []))
            # Percentiles on detected-only
            rem_det = rem[rem > 0] if len(rem) > 0 else np.array([])
            p10 = np.percentile(rem_det, 10) if len(rem_det) > 0 else 0
            p25 = np.percentile(rem_det, 25) if len(rem_det) > 0 else 0
            p50 = np.median(rem_det) if len(rem_det) > 0 else 0
            mean_rem = rem_det.mean() if len(rem_det) > 0 else 0
            print(
                f"  {sname:>8s}  {r['n_det']:>3}/{r['n_tot']:<3} {r['cov']:4.0f}%  "
                f"{mean_rem:6.1f}s  {p10:5.1f}s  "
                f"{p25:5.1f}s  {p50:5.1f}s  {r['bg']:>4}"
            )

    print(f"\n  Remaining = seconds of drone audio left after first alert")
    print(f"  P10 = slowest 10% of detections (least remaining time)")
    print(f"  P25 = slowest 25%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
