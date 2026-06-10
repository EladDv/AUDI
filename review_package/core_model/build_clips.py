#!/usr/bin/env python3
"""
build_clips.py - WAV files -> clips.npz (feature-agnostic, built ONCE).

For every labelled WAV:
  * resample to 16 kHz, mix to mono (4-ch array -> broadside delay-and-sum)
  * slice into overlapping 2-second clips
  * estimate per-clip SNR (drone band vs noise floor)
Stores the mono audio plus enough metadata to recover the raw 4-ch clip later
(source path + start sample) so spatial features (gcc_spatial) can re-read it.

The train/val/test split is assigned per SESSION (file), stratified by
(class, domain), so clips from one recording never leak across splits.

Usage:
  python build_clips.py            # build
  python build_clips.py --dry-run  # show label/domain inventory only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np

from config import (CFG, CLIPS, DATA_ROOTS, CLASSES, DOMAIN,
                    infer_label, infer_domain, infer_subtype)


def list_wavs():
    files = []
    for root in DATA_ROOTS:
        if root.exists():
            files.extend(sorted(root.rglob("*.wav")))
    return files


def load_mono(path, sr):
    import librosa
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        print(f"  ! skip {path.name}: {exc}")
        return None
    return y.astype(np.float32) if y.size >= sr // 2 else None


def band_snr(clip, sr, lo=150, hi=4000):
    """Crude per-clip SNR (dB): in-band energy vs the quiet-frame floor."""
    import numpy as np
    f = np.fft.rfftfreq(len(clip), 1 / sr)
    P = np.abs(np.fft.rfft(clip)) ** 2
    band = (f >= lo) & (f <= hi)
    sig = P[band].mean() + 1e-12
    # noise floor: 10th-percentile of short-frame energies
    blk = sr // 16
    en = [np.mean(clip[i:i + blk] ** 2) for i in range(0, len(clip) - blk, blk)]
    noise = np.percentile(en, 10) + 1e-12 if en else 1e-12
    return float(10 * np.log10(sig / noise))


def session_split(file_ids, classes, domains, seed):
    """Stratified per-session split. Anything tagged 'ext_test' is FORCED to
    the test split (2) and never used for train/val."""
    rng = np.random.default_rng(seed)
    split = {}
    buckets = {}
    for fid, c, d in zip(file_ids, classes, domains):
        if d == "ext_test":
            split[fid] = 2
            continue
        buckets.setdefault((c, d), []).append(fid)
    for ids in buckets.values():
        ids = list(ids); rng.shuffle(ids); n = len(ids)
        n_tr = max(1, round(0.70 * n)); n_va = round(0.15 * n)
        for i, fid in enumerate(ids):
            split[fid] = 0 if i < n_tr else (1 if i < n_tr + n_va else 2)
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(CLIPS))
    ap.add_argument("--hop-s", type=float, default=CFG.hop_s,
                    help="seconds between clips (=clip_s means no overlap)")
    ap.add_argument("--max-per-file", type=int, default=0,
                    help="cap clips taken from any single file (0 = no cap)")
    args = ap.parse_args()

    wavs = list_wavs()
    print(f"Found {len(wavs)} WAV files.\n")

    keep, fcls, fdom, fsub = [], [], [], []
    for w in wavs:
        cls = infer_label(str(w))
        if cls is None:
            continue
        keep.append(w); fcls.append(cls)
        fdom.append(infer_domain(str(w))); fsub.append(infer_subtype(str(w)))

    inv = Counter(zip(fcls, fdom))
    print("Files by (class, domain):")
    for (c, d), n in sorted(inv.items()):
        print(f"  {n:4d}  {c:13s} [{d}]")
    sub = Counter(s for s in fsub if s)
    print(f"  drone subtypes: {dict(sub)}")
    print()

    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    file_ids = list(range(len(keep)))
    split_map = session_split(file_ids, fcls, fdom, CFG.seed)

    if args.dry_run:
        print("dry-run: not extracting clips.")
        return 0

    from tqdm import tqdm
    n = int(CFG.clip_s * CFG.sr); hop = int(args.hop_s * CFG.sr)
    audio, y, dom, spl, grp, snr, starts = [], [], [], [], [], [], []
    src_paths = [str(p) for p in keep]
    for fid, w in enumerate(tqdm(keep, desc="files")):
        yk = load_mono(w, CFG.sr)
        if yk is None:
            continue
        if len(yk) < n:
            yk = np.pad(yk, (0, n - len(yk)))
        taken = 0
        for s in range(0, len(yk) - n + 1, hop):
            if args.max_per_file and taken >= args.max_per_file:
                break
            clip = yk[s:s + n]
            audio.append(clip)
            y.append(cls_to_idx[fcls[fid]])
            dom.append(DOMAIN[fdom[fid]])
            spl.append(split_map[fid]); grp.append(fid); starts.append(s)
            snr.append(band_snr(clip, CFG.sr)); taken += 1

    if not audio:
        print("No clips produced."); return 1

    # store audio as int16 (half the size, much faster) - clips.npz is a
    # local-only intermediate, so we skip slow compression too.
    audio16 = (np.clip(np.stack(audio), -1.0, 1.0) * 32767).astype(np.int16)
    np.savez(
        args.out,
        audio=audio16,
        y=np.asarray(y, np.int64), domain=np.asarray(dom, np.int64),
        split=np.asarray(spl, np.int64), group=np.asarray(grp, np.int64),
        start=np.asarray(starts, np.int64), snr_db=np.asarray(snr, np.float32),
        src=np.array(src_paths), subtype=np.array(fsub),
        label_names=np.array(CLASSES))

    with open(str(args.out).replace(".npz", ".json"), "w") as f:
        json.dump({"config": CFG.__dict__, "labels": CLASSES,
                   "n_clips": len(audio)}, f, indent=2)

    yv = np.asarray(y)
    print(f"\nSaved {args.out}  clips={len(audio)}")
    print("Clips per class:", {CLASSES[i]: int((yv == i).sum())
                               for i in range(len(CLASSES))})
    spv = np.asarray(spl); dv = np.asarray(dom)
    print("Clips per split [train/val/test]:",
          [int((spv == s).sum()) for s in (0, 1, 2)])
    print("Clips per domain:", {name: int((dv == code).sum())
                                for name, code in DOMAIN.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
