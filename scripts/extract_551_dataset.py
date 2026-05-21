#!/usr/bin/env python3
"""Extract 551 field data into a drone/background dataset.

Output:
  data/551_dataset/
    drone/         # extracted drone segment WAVs
    background/    # 30s background chunks
    manifest.csv   # all files with metadata + wd_003 detections
"""

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
OUT_DIR = ROOT / "data/551_dataset"
CKPT_PATH = ROOT / "checkpoints/bce_wd_sweep_20260518_122516/01_wd_003/checkpoints/epoch=12-step=3250.ckpt"

SR = 16000
CLIP_S = 5.12
CHUNK_N = int(CLIP_S * SR)
HOP_N = int(CHUNK_N * 0.25)  # 75% overlap
BATCH_SIZE = 32
BG_CHUNK_S = 30
BG_CHUNK_N = BG_CHUNK_S * SR
DRONE_MARGIN_S = 1.0  # margin around drone segments when extracting bg

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_time(t: str) -> float:
    if ":" not in t:
        return float(t)
    mm, ss = t.split(":")
    return int(mm) * 60 + float(ss)


def parse_tags() -> dict[str, list[tuple[float, float]]]:
    """{wav_stem: [(start_s, end_s), ...]} drone segments, merged if close."""
    raw: dict[str, list[tuple[float, float]]] = {}
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

        drones = []
        for row in rows:
            name = row.get("Name", "").strip()
            if "רחפן" in name:
                s = parse_time(row["Start"])
                d = parse_time(row["Duration"])
                if d > 0:
                    drones.append((s, s + d))
        if drones:
            drones.sort()
            # merge overlapping / adjacent (<2s gap)
            merged = [drones[0]]
            for s, e in drones[1:]:
                if s - merged[-1][1] < 2.0:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            raw[wav_stem] = merged
    return raw


@torch.no_grad()
def get_scores(wav_path: Path, model) -> np.ndarray:
    """Return per-window detection scores for a WAV file."""
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
    return np.array(scores)


def window_times(n_windows: int) -> np.ndarray:
    return np.arange(n_windows) * (HOP_N / SR) + (CHUNK_N / 2 / SR)


def extract_drone_segments(audio: np.ndarray, segs: list[tuple[float, float]],
                            wav_stem: str, out_dir: Path) -> list[dict]:
    """Extract drone segments as WAVs. Returns manifest rows."""
    rows = []
    for idx, (s_s, e_s) in enumerate(segs):
        s_sample = int(s_s * SR)
        e_sample = int(e_s * SR)
        seg_audio = audio[s_sample:e_sample]
        if len(seg_audio) < SR:  # skip < 1s
            continue
        fname = f"{wav_stem}_drone_{idx:02d}_{s_s:.1f}s-{e_s:.1f}s.wav"
        out_path = out_dir / fname
        sf.write(str(out_path), seg_audio, SR)
        rows.append({
            "filename": f"drone/{fname}",
            "type": "drone",
            "source": wav_stem,
            "start_s": round(s_s, 1),
            "end_s": round(e_s, 1),
            "duration_s": round(e_s - s_s, 1),
        })
    return rows


def extract_background_chunks(audio: np.ndarray, drone_segs: list[tuple[float, float]],
                               wav_stem: str, out_dir: Path) -> list[dict]:
    """Extract 30s non-overlapping background chunks. Skips drone regions + margin."""
    rows = []
    total_s = len(audio) / SR
    margin_n = int(DRONE_MARGIN_S * SR)

    # Mark drone regions + margin as off-limits
    drone_mask = np.zeros(len(audio), dtype=bool)
    for s, e in drone_segs:
        ss = max(0, int(s * SR) - margin_n)
        ee = min(len(audio), int(e * SR) + margin_n)
        drone_mask[ss:ee] = True

    chunk_idx = 0
    pos = 0
    while pos + BG_CHUNK_N <= len(audio):
        # Skip chunks that overlap drone regions
        if not drone_mask[pos:pos + BG_CHUNK_N].any():
            chunk = audio[pos:pos + BG_CHUNK_N]
            fname = f"{wav_stem}_bg_{chunk_idx:03d}_{pos/SR:.0f}s.wav"
            out_path = out_dir / fname
            sf.write(str(out_path), chunk, SR)
            rows.append({
                "filename": f"background/{fname}",
                "type": "background",
                "source": wav_stem,
                "start_s": round(pos / SR, 1),
                "end_s": round((pos + BG_CHUNK_N) / SR, 1),
                "duration_s": BG_CHUNK_S,
            })
            chunk_idx += 1
            pos += BG_CHUNK_N
        else:
            # Step forward in 1s increments to find clean bg
            pos += SR
    return rows


def main() -> int:
    print("Loading wd_003 model...")
    model = load_model_from_checkpoint(CKPT_PATH, device=device)
    model.eval()

    print("Parsing tags...")
    all_segs = parse_tags()
    total = sum(len(s) for s in all_segs.values())
    print(f"  {len(all_segs)} files, {total} drone segments (merged)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "drone").mkdir(exist_ok=True)
    (OUT_DIR / "background").mkdir(exist_ok=True)

    all_rows: list[dict] = []
    manifest_path = OUT_DIR / "manifest.csv"

    FIELD_NAMES = [
        "filename", "type", "source", "start_s", "end_s", "duration_s",
        "score_max", "score_mean", "score_median", "score_p90",
        "n_windows", "n_above_05", "n_above_07",
    ]

    for wav_stem, drone_segs in sorted(all_segs.items()):
        wav_path = REC_DIR / f"{wav_stem}.wav"
        print(f"\n{wav_stem}: {len(drone_segs)} drone segs")
        audio, sr = sf.read(str(wav_path))

        # ── Run inference ──
        scores = get_scores(wav_path, model)
        times = window_times(len(scores))
        print(f"  inference: {len(scores)} windows")

        # ── Extract drone segments ──
        drone_rows = extract_drone_segments(audio, drone_segs, wav_stem, OUT_DIR / "drone")
        print(f"  drone WAVs: {len(drone_rows)}")

        # ── Extract background chunks ──
        bg_rows = extract_background_chunks(audio, drone_segs, wav_stem, OUT_DIR / "background")
        print(f"  bg WAVs: {len(bg_rows)}")

        # ── Attach detection scores to each segment ──
        for row in drone_rows + bg_rows:
            s_s = row["start_s"]
            e_s = row["end_s"]
            mask = (times >= s_s) & (times <= e_s)
            seg_scores = scores[mask]
            if len(seg_scores) > 0:
                row["score_max"] = round(float(seg_scores.max()), 4)
                row["score_mean"] = round(float(seg_scores.mean()), 4)
                row["score_median"] = round(float(np.median(seg_scores)), 4)
                row["score_p90"] = round(float(np.percentile(seg_scores, 90)), 4)
                row["n_windows"] = len(seg_scores)
                row["n_above_05"] = int(np.sum(seg_scores > 0.5))
                row["n_above_07"] = int(np.sum(seg_scores > 0.7))
            else:
                for k in ["score_max", "score_mean", "score_median", "score_p90",
                           "n_windows", "n_above_05", "n_above_07"]:
                    row[k] = ""

            all_rows.append(row)

    # ── Write manifest ──
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        w.writeheader()
        w.writerows(all_rows)

    n_drone = sum(1 for r in all_rows if r["type"] == "drone")
    n_bg = sum(1 for r in all_rows if r["type"] == "background")
    print(f"\n{'='*60}")
    print(f"  Saved to {OUT_DIR}")
    print(f"  Drone segments:  {n_drone}")
    print(f"  Background 30s:  {n_bg}")
    print(f"  Manifest:        {manifest_path}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
