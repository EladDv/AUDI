#!/usr/bin/env python3
"""Preprocess field recordings into a blue (5-inch) vs red (10-inch) classification dataset.

Pipeline:
  1. Chop each session WAV into non-overlapping 2.56s clips
  2. Remove speech clips via Silero VAD
  3. Remove rooster calls via AudioSet-pretrained MN05 model
  4. Save as HF DatasetDict (train/validation, 80/20 stratified)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, Dataset, DatasetDict, Features, Value, concatenate_datasets

# AudioSet class indices for bird/rooster detection (from EfficientAT metadata)
# Verified against EfficientAT/metadata/class_labels_indices.csv
LIVESTOCK_IDX = 86    # Livestock, farm animals, working animals
ROOSTER_IDX = 99      # Chicken, rooster
BIRD_IDX = 111        # Bird
BIRD_VOCAL_IDX = 112  # Bird vocalization, bird call, bird song
BIRD_FLIGHT_IDX = 121 # Bird flight, flapping wings

BIRD_CLASSES = {LIVESTOCK_IDX, ROOSTER_IDX, BIRD_IDX, BIRD_VOCAL_IDX, BIRD_FLIGHT_IDX}

SR = 16000
CLIP_S = 2.56
CLIP_SAMPLES = int(SR * CLIP_S)  # 40960


def manifest_label(manifest_path: Path) -> tuple[str | None, str | None]:
    """Return (class_label, class_id) or (None, None) for non-blue/red sessions."""
    with open(manifest_path) as f:
        m = json.load(f)
    size = m["size_inches"]
    if size == 5:
        return "blue", 0
    elif size == 10:
        return "red", 1
    return None, None


def load_wav(path: Path) -> np.ndarray:
    """Load mono WAV as float32 numpy array (no resampling, already 16kHz)."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SR, f"Expected {SR} Hz, got {sr}"
    return audio.astype(np.float32)


def chop_clips(audio: np.ndarray) -> np.ndarray:
    """Chop audio into non-overlapping CLIP_SAMPLES-length clips."""
    n_clips = len(audio) // CLIP_SAMPLES
    if n_clips == 0:
        return np.empty((0, CLIP_SAMPLES), dtype=np.float32)
    clipped = audio[:n_clips * CLIP_SAMPLES].reshape(n_clips, CLIP_SAMPLES)
    return clipped


def vad_filter(clips: np.ndarray, vad_model, frame_ratio: float = 0.3) -> np.ndarray:
    """Remove clips where >frame_ratio fraction of frames are speech.

    Uses silero-vad get_speech_timestamps which returns list of
    {'start': samples, 'end': samples} dicts for speech segments.
    """
    from silero_vad import get_speech_timestamps

    if len(clips) == 0:
        return clips

    keep = []
    for clip in clips:
        tensor = torch.from_numpy(clip)
        speech_ts = get_speech_timestamps(tensor, vad_model, sampling_rate=SR)
        total_speech = sum(ts["end"] - ts["start"] for ts in speech_ts)
        speech_frac = total_speech / CLIP_SAMPLES
        if speech_frac <= frame_ratio:
            keep.append(clip)

    if not keep:
        return np.empty((0, CLIP_SAMPLES), dtype=np.float32)
    return np.stack(keep)


def load_as_model():
    """Load mn05_as with its 527-class AudioSet head."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.mn.model import get_model

    model = get_model(
        num_classes=527,
        pretrained_name="mn05_as",
        width_mult=0.5,
        head_type="mlp",
        input_dim_f=128,
        input_dim_t=1000,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_mel_for_mn(wav: torch.Tensor) -> torch.Tensor:
    """Compute 128-bin mel spectrogram shaped for MN backbone: [1, 1, 128, T_mel]."""
    import torchaudio.transforms as T
    mel_transform = T.MelSpectrogram(
        sample_rate=SR, n_fft=1024, hop_length=160, n_mels=128,
    )
    to_db = T.AmplitudeToDB()
    mel = to_db(mel_transform(wav))  # (1, 128, T_mel)
    mean = mel.mean(dim=(1, 2), keepdim=True)
    std = mel.std(dim=(1, 2), keepdim=True) + 1e-8
    mel = (mel - mean) / std
    return mel.unsqueeze(1)  # (1, 1, 128, T_mel)


def rooster_filter(clips: np.ndarray, as_model, device: str = "cpu",
                   confidence_threshold: float = 0.3) -> np.ndarray:
    """Remove clips where AudioSet model fires on bird/rooster classes."""
    if len(clips) == 0:
        return clips

    as_model = as_model.to(device)
    keep = []

    for clip in clips:
        wav = torch.from_numpy(clip).unsqueeze(0).to(device)  # (1, T)
        mel = compute_mel_for_mn(wav).to(device)              # (1, 1, 128, T_mel)
        # MN model returns (logits, features); we only need logits
        with torch.no_grad():
            result = as_model(mel)
            logits = result[0] if isinstance(result, tuple) else result
        probs = torch.sigmoid(logits.squeeze(0))  # (527,)
        top5_idx = torch.topk(probs, 5).indices.cpu().numpy()

        bird_hits = set(top5_idx) & BIRD_CLASSES
        top_probs_bird = {int(idx): float(probs[idx].item()) for idx in bird_hits}

        is_rooster = any(p > confidence_threshold for p in top_probs_bird.values())
        if not is_rooster:
            keep.append(clip)

    if not keep:
        return np.empty((0, CLIP_SAMPLES), dtype=np.float32)
    return np.stack(keep)


def build_dataset(records: list[dict]) -> Dataset:
    """Build a HF Dataset from records list."""
    features = Features({
        "audio": Audio(sampling_rate=SR),
        "label": Value("string"),
        "label_id": Value("int8"),
        "session_id": Value("string"),
        "offset_s": Value("float32"),
    })
    return Dataset.from_list(records, features=features)


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess field recordings for blue/red classification")
    ap.add_argument("--recordings-dir", type=Path,
                    default=Path("data/recordings_20260501T061603"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("data/hf_blue_red"))
    ap.add_argument("--skip-vad", action="store_true")
    ap.add_argument("--skip-rooster", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sessions_dir = args.recordings_dir / "sessions"
    if not sessions_dir.exists():
        print(f"ERROR: {sessions_dir} not found")
        return 1

    # ── Load VAD model ─────────────────────────────────────────
    vad_model = None
    if not args.skip_vad:
        print("Loading Silero VAD...")
        from silero_vad import load_silero_vad
        vad_model = load_silero_vad()

    # ── Load AudioSet rooster model ────────────────────────────
    as_model = None
    if not args.skip_rooster:
        print("Loading mn05_as AudioSet model for rooster detection...")
        as_model = load_as_model()
        as_model = as_model.to(args.device)

    # ── Process each session ───────────────────────────────────
    all_records = []
    stats = {"blue": {"total_clips": 0, "post_vad": 0, "post_rooster": 0, "sessions": 0},
             "red": {"total_clips": 0, "post_vad": 0, "post_rooster": 0, "sessions": 0},
             "skipped": 0}

    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue

        manifest_path = session_dir / "manifest.json"
        wav_path = session_dir / "alsa_default.wav"
        if not manifest_path.exists() or not wav_path.exists():
            continue

        cls_label, cls_id = manifest_label(manifest_path)
        if cls_label is None:
            print(f"  SKIP {session_dir.name}: not 5-inch or 10-inch")
            stats["skipped"] += 1
            continue

        session_name = session_dir.name
        print(f"\n[{cls_label.upper()}] {session_name}")

        audio = load_wav(wav_path)
        clips = chop_clips(audio)
        stats[cls_label]["total_clips"] += len(clips)
        print(f"  Raw clips: {len(clips)}")

        # Step 1: VAD
        if vad_model is not None:
            clips = vad_filter(clips, vad_model)
            stats[cls_label]["post_vad"] += len(clips)
            print(f"  After VAD:  {len(clips)}")

        # Step 2: Rooster
        if as_model is not None:
            clips = rooster_filter(clips, as_model, device=args.device)
            stats[cls_label]["post_rooster"] += len(clips)
            print(f"  After rooster: {len(clips)}")

        stats[cls_label]["sessions"] += 1

        # Build records
        for i, clip in enumerate(clips):
            offset_s = i * CLIP_S
            all_records.append({
                "audio": {"array": clip, "sampling_rate": SR},
                "label": cls_label,
                "label_id": cls_id,
                "session_id": session_name,
                "offset_s": float(offset_s),
            })

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    for cls in ("blue", "red"):
        s = stats[cls]
        print(f"  {cls}: {s['sessions']} sessions, {s['total_clips']} raw → "
              f"{s['post_vad']} post-VAD → {s['post_rooster']} post-rooster")
    print(f"  Total records: {len(all_records)}")
    if stats["skipped"]:
        print(f"  Skipped: {stats['skipped']} sessions")

    if len(all_records) == 0:
        print("ERROR: No records generated!")
        return 1

    # ── Train/val split (stratified) ───────────────────────────
    full_ds = build_dataset(all_records)
    rng = np.random.RandomState(args.seed)

    blue_indices = [i for i, r in enumerate(full_ds) if r["label"] == "blue"]
    red_indices = [i for i, r in enumerate(full_ds) if r["label"] == "red"]
    rng.shuffle(blue_indices)
    rng.shuffle(red_indices)

    n_val_blue = max(1, int(len(blue_indices) * args.val_split))
    n_val_red = max(1, int(len(red_indices) * args.val_split))

    val_idx = set(blue_indices[:n_val_blue] + red_indices[:n_val_red])
    train_idx = [i for i in range(len(full_ds)) if i not in val_idx]

    ds_dict = DatasetDict({
        "train": full_ds.select(train_idx),
        "validation": full_ds.select(sorted(val_idx)),
    })

    print(f"\nTrain: {len(ds_dict['train'])} ({sum(1 for r in ds_dict['train'] if r['label']=='blue')} blue, "
          f"{sum(1 for r in ds_dict['train'] if r['label']=='red')} red)")
    print(f"Val:   {len(ds_dict['validation'])} ({sum(1 for r in ds_dict['validation'] if r['label']=='blue')} blue, "
          f"{sum(1 for r in ds_dict['validation'] if r['label']=='red')} red)")

    ds_dict.save_to_disk(str(args.output_dir))
    print(f"\nSaved: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
