#!/usr/bin/env python3
"""Mine model false positives from raw field recordings.

Usage:
  uv run audi-data mine-field-hard-negatives --checkpoint <path>

Default input is limited to ``data/field_recordings_20260514/recordings``.
The script writes:
  - WAV clips under ``<output>/clips``
  - ``<output>/manifest.csv``
  - optional HF DatasetDict at ``<output>/hf_dataset``
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, Dataset, DatasetDict, Features, Value

from audi.checkpoint import get_clip_seconds, load_model_from_checkpoint
from audi.hard_negative_mining import (
    build_alert_exclusions,
    clip_with_padding,
    discover_field_recordings,
    extract_alert_runs,
    is_allowed,
    write_manifest,
)


def _load_mono(path: Path, target_sr: int) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy().astype(np.float32)


def _window_starts(n_samples: int, win_samples: int, hop_samples: int) -> list[int]:
    if n_samples < win_samples:
        return [0]
    return list(range(0, n_samples - win_samples + 1, hop_samples))


@torch.no_grad()
def _predict_scores(
    model,
    audio: np.ndarray,
    *,
    win_samples: int,
    hop_samples: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    starts = _window_starts(len(audio), win_samples, hop_samples)
    scores: list[np.ndarray] = []
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        windows = [
            clip_with_padding(audio, start_sample=s, length_samples=win_samples)
            for s in batch_starts
        ]
        wav = torch.as_tensor(np.stack(windows), dtype=torch.float32, device=device)
        logits = model(wav).detach().cpu().numpy().reshape(-1)
        scores.append(1.0 / (1.0 + np.exp(-logits)))
    all_scores = np.concatenate(scores) if scores else np.array([], dtype=np.float32)
    centers_s = (np.asarray(starts, dtype=np.float32) + win_samples / 2) / 16000.0
    return all_scores, centers_s


def _load_clip_seconds(ckpt_path: Path) -> float:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return get_clip_seconds(ckpt["hyper_parameters"])


def _write_hf_dataset(rows: list[dict], output_dir: Path, *, val_ratio: float) -> None:
    if not rows:
        return
    records = [
        {
            "audio": {"path": str(Path(row["clip_path"]).resolve())},
            "label": "hard_fp",
        }
        for row in rows
    ]
    features = Features({"audio": Audio(sampling_rate=16000), "label": Value("string")})
    ds = Dataset.from_list(records, features=features)
    if len(ds) > 1 and val_ratio > 0:
        split = ds.train_test_split(test_size=val_ratio, seed=42)
        dd = DatasetDict({"train": split["train"], "validation": split["test"]})
    else:
        dd = DatasetDict({"train": ds, "validation": Dataset.from_list([], features=features)})
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dd.save_to_disk(str(output_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--field-dir", type=Path, default=Path("data/field_recordings_20260514"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/field_recordings_20260514/mined_hard_negatives"),
    )
    parser.add_argument("--threshold", type=float, default=0.67)
    parser.add_argument("--min-windows", type=int, default=3)
    parser.add_argument("--stride-ratio", type=float, default=0.125)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exclusion-buffer-s", type=float, default=10.0)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--no-hf-dataset", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    sample_rate = 16000
    clip_s = _load_clip_seconds(args.checkpoint)
    win_samples = int(round(clip_s * sample_rate))
    hop_samples = max(1, int(round(win_samples * args.stride_ratio)))

    with contextlib.redirect_stdout(io.StringIO()):
        model = load_model_from_checkpoint(args.checkpoint, device=device, quiet=True)
    exclusions = build_alert_exclusions(args.field_dir, buffer_s=args.exclusion_buffer_s)
    recordings = discover_field_recordings(args.field_dir)
    if not recordings:
        raise SystemExit(f"No recordings found under {args.field_dir / 'recordings'}")

    clips_dir = args.output_dir / "clips"
    if clips_dir.exists():
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for rec_idx, recording in enumerate(recordings, start=1):
        print(f"[{rec_idx}/{len(recordings)}] {recording.path.name}")
        audio = _load_mono(recording.path, sample_rate)
        scores, times_s = _predict_scores(
            model,
            audio,
            win_samples=win_samples,
            hop_samples=hop_samples,
            batch_size=args.batch_size,
            device=device,
        )
        runs = extract_alert_runs(
            scores,
            times_s,
            threshold=args.threshold,
            min_windows=args.min_windows,
        )
        for run_idx, run in enumerate(runs):
            center_s = (run.start_s + run.end_s) / 2.0
            start_s = max(0.0, center_s - clip_s / 2.0)
            end_s = start_s + clip_s
            if recording.start_epoch is not None:
                abs_start = recording.start_epoch + start_s
                abs_end = recording.start_epoch + end_s
                if not is_allowed(abs_start, abs_end, exclusions):
                    continue
            clip = clip_with_padding(
                audio,
                start_sample=int(round(start_s * sample_rate)),
                length_samples=win_samples,
            )
            clip_name = f"{recording.path.stem}_fp_{run_idx:04d}_{start_s:.2f}s.wav"
            clip_path = clips_dir / clip_name
            sf.write(str(clip_path), clip, sample_rate, subtype="PCM_16")
            recording_start_epoch = (
                recording.start_epoch if recording.start_epoch is not None else ""
            )
            rows.append(
                {
                    "clip_path": str(clip_path),
                    "source_path": str(recording.path),
                    "recording_start_epoch": recording_start_epoch,
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "run_start_s": run.start_s,
                    "run_end_s": run.end_s,
                    "max_score": run.max_score,
                    "mean_score": run.mean_score,
                    "n_windows": run.n_windows,
                    "label": "hard_fp",
                }
            )
            if args.max_clips and len(rows) >= args.max_clips:
                break
        if args.max_clips and len(rows) >= args.max_clips:
            break
        write_manifest(rows, args.output_dir / "manifest.csv")

    write_manifest(rows, args.output_dir / "manifest.csv")
    if not args.no_hf_dataset:
        _write_hf_dataset(rows, args.output_dir / "hf_dataset", val_ratio=args.val_ratio)

    print(f"Mined {len(rows)} hard-negative clips")
    print(f"Manifest: {args.output_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
