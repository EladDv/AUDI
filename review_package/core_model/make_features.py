#!/usr/bin/env python3
"""
make_features.py - clips.npz -> features/<name>.npz for any feature variant.

Every model reads these .npz files. Because all variants come from the same
clips with the same split, model comparisons are fair.

  python make_features.py --name mel
  python make_features.py --name mel2
  python make_features.py --all                 # every non-4ch feature
  python make_features.py --name gcc_spatial    # needs 4-ch array clips

Output X has shape (N, C, F, T). y/domain/split/group/snr are copied through.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from config import CFG, CLIPS, FEATURES_DIR
from features import FEATURES
from augment import mix_at_snr, field_augment


def load_clips():
    d = np.load(CLIPS, allow_pickle=True)
    return d


def recover_4ch(src_path, start, sr, n):
    """Re-read the raw 4-channel clip for spatial features."""
    import librosa
    y, _ = librosa.load(src_path, sr=sr, mono=False)
    if y.ndim == 1:
        y = y[None]
    seg = y[:, start:start + n]
    if seg.shape[1] < n:
        seg = np.pad(seg, ((0, 0), (0, n - seg.shape[1])))
    return seg.astype(np.float32)


def _save(out, X, y, dom, spl, grp, snr, label_names, dtype):
    FEATURES_DIR.mkdir(exist_ok=True)
    n = len(X)
    feat_shape = np.asarray(X[0]).shape
    # Build the stacked array in RAM in chunks, freeing the list as we go.
    # Avoids both: (a) np.stack full-copy peak, (b) large mmap temp file on disk.
    Xa = np.empty((n,) + feat_shape, dtype=dtype)
    chunk = 512
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        Xa[i:end] = np.stack(X[i:end]).astype(dtype)
        for j in range(i, end):
            X[j] = None  # free processed entries immediately
    np.savez_compressed(
        out, X=Xa, y=np.asarray(y), domain=np.asarray(dom),
        split=np.asarray(spl), group=np.asarray(grp),
        snr_db=np.asarray(snr, np.float32), label_names=label_names)
    print(f"  saved {out}  shape={Xa.shape}")


def _subsample_keep(keep, d, budget, seed=1337):
    """Shrink the kept set to ~budget clips for a low-memory build.
    ALWAYS keeps every array clip (domain==1, the few precious EVO/FPV); samples
    the big mono pile stratified by class so the classes stay balanced."""
    if not budget or budget >= int(keep.sum()):
        return keep
    yv = d["y"]; domv = d["domain"]
    rng = np.random.default_rng(seed)
    idx = np.where(keep)[0]
    arr_idx = idx[domv[idx] == 1]
    mono_idx = idx[domv[idx] != 1]
    room = max(0, budget - len(arr_idx))
    classes = np.unique(yv[mono_idx]) if len(mono_idx) else np.array([], int)
    per = max(1, room // max(1, len(classes)))
    chosen = []
    for c in classes:
        ci = mono_idx[yv[mono_idx] == c]
        chosen.append(rng.choice(ci, min(len(ci), per), replace=False))
    new_idx = np.concatenate([arr_idx] + chosen) if chosen else arr_idx
    newkeep = np.zeros_like(keep)
    newkeep[new_idx] = True
    print(f"  subsample: {int(keep.sum())} -> {int(newkeep.sum())} clips "
          f"(kept all {len(arr_idx)} array, sampled mono ~{per}/class)")
    return newkeep


def build_one(name, d, dtype=np.float16, domain_filter=None, suffix="",
              snr_aug=0, snr_range=(-5.0, 20.0), subsample=0, field_aug=0):
    spec = FEATURES[name]
    fn = spec["fn"]
    audio = d["audio"]
    if audio.dtype == np.int16:               # stored as int16 -> back to float
        audio = audio.astype(np.float32) / 32768.0
    n = int(CFG.clip_s * CFG.sr)
    from tqdm import tqdm

    keep = np.ones(len(audio), bool)
    if domain_filter is not None:             # e.g. drop ext_test before featurising
        keep &= domain_filter(d["domain"])
    keep = _subsample_keep(keep, d, subsample, CFG.seed)
    out = FEATURES_DIR / f"{name}{suffix}.npz"

    # ---- spatial (4-ch) path: no audio-domain noise aug ------------------ #
    if spec["needs_4ch"]:
        X = []
        src = d["src"]; grp = d["group"]; start = d["start"]; dom = d["domain"]
        for i in tqdm(range(len(audio)), desc=name):
            if not keep[i]:
                continue
            if dom[i] != 1:
                keep[i] = False; continue
            try:
                y4 = recover_4ch(str(src[grp[i]]), int(start[i]), CFG.sr, n)
                X.append(fn(y4, CFG))
            except Exception as exc:
                keep[i] = False
                print(f"  ! {name} clip {i}: {exc}")
        if not X:
            print(f"  (no clips for {name})"); return
        _save(out, X, d["y"][keep], d["domain"][keep], d["split"][keep],
              d["group"][keep], d["snr_db"][keep], d["label_names"], dtype)
        return

    # ---- mono path (optional real background-noise augmentation) --------- #
    yv = d["y"]; splv = d["split"]; domv = d["domain"]
    grpv = d["group"]; snrv = d["snr_db"]
    labels = [str(x) for x in d["label_names"]]
    bg_idx = labels.index("background") if "background" in labels else None
    noise_pool = ([audio[i] for i in range(len(audio))
                   if keep[i] and bg_idx is not None and yv[i] == bg_idx]
                  if (snr_aug > 0 or field_aug > 0) else [])
    rng = np.random.default_rng(CFG.seed)
    if snr_aug > 0:
        print(f"  noise-aug: {snr_aug}x per TRAIN clip, SNR {snr_range} dB, "
              f"pool={len(noise_pool)} background clips")
    if field_aug > 0:
        print(f"  field-aug: {field_aug}x per TRAIN clip (ALL classes: drones AND "
              f"negatives) - flyby Doppler / distance LPF / reverb / clip / dropout")

    X, Y, DOM, SPL, GRP, SNR = [], [], [], [], [], []
    for i in tqdm(range(len(audio)), desc=name):
        if not keep[i]:
            continue
        base = audio[i]
        X.append(fn(base, CFG)); Y.append(yv[i]); DOM.append(domv[i])
        SPL.append(splv[i]); GRP.append(grpv[i]); SNR.append(snrv[i])
        # augment ONLY training clips (split==0), only non-background, only if pool
        if (snr_aug > 0 and splv[i] == 0 and noise_pool
                and (bg_idx is None or yv[i] != bg_idx)):
            for _ in range(snr_aug):
                bg = noise_pool[rng.integers(len(noise_pool))]
                snr = float(rng.uniform(*snr_range))
                mixed = mix_at_snr(base, bg, snr, rng)
                X.append(fn(mixed, CFG)); Y.append(yv[i]); DOM.append(domv[i])
                SPL.append(0); GRP.append(grpv[i]); SNR.append(snr)
        # field-condition aug on ALL TRAIN clips (drones AND negatives)
        if field_aug > 0 and splv[i] == 0:
            for _ in range(field_aug):
                aug = field_augment(base, CFG.sr, rng, backgrounds=noise_pool)
                X.append(fn(aug, CFG)); Y.append(yv[i]); DOM.append(domv[i])
                SPL.append(0); GRP.append(grpv[i]); SNR.append(snrv[i])

    if not X:
        print(f"  (no clips for {name})"); return
    _save(out, X, Y, DOM, SPL, GRP, SNR, d["label_names"], dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=list(FEATURES))
    ap.add_argument("--all", action="store_true",
                    help="build every feature that does not need 4 channels")
    ap.add_argument("--float32", action="store_true",
                    help="store full float32 instead of the default float16")
    ap.add_argument("--exclude-ext", action="store_true",
                    help="drop ext_test clips (domain 2) - slim set to UPLOAD")
    ap.add_argument("--only-ext", action="store_true",
                    help="keep ONLY ext_test clips - built LOCALLY for evaluation")
    ap.add_argument("--snr-aug", type=int, default=0,
                    help="N extra background-noise-mixed copies per TRAIN clip "
                         "(real noise robustness; 0 = off)")
    ap.add_argument("--snr-min", type=float, default=-5.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--field-aug", type=int, default=0,
                    help="N extra FIELD-condition copies per TRAIN clip, applied to "
                         "ALL classes (drones AND negatives): flyby Doppler, distance "
                         "low-pass, reverb, saturation, dropout. 0 = off")
    ap.add_argument("--subsample", type=int, default=0,
                    help="cap to ~N clips (keeps all array, samples mono "
                         "stratified) - low-memory / fast build. 0 = all")
    args = ap.parse_args()
    dtype = np.float32 if args.float32 else np.float16

    from config import DOMAIN
    domain_filter, suffix = None, ""
    if args.exclude_ext:
        domain_filter = lambda dom: dom != DOMAIN["ext_test"]
    elif args.only_ext:
        domain_filter = lambda dom: dom == DOMAIN["ext_test"]
        suffix = "_ext"

    if not CLIPS.exists():
        print("clips.npz not found - run build_clips.py first."); return 1
    d = load_clips()

    names = ([n for n, s in FEATURES.items() if not s["needs_4ch"]]
             if args.all else [args.name])
    if not names or names == [None]:
        print("pass --name <feature> or --all. features:", list(FEATURES))
        return 1
    for nm in names:
        build_one(nm, d, dtype=dtype, domain_filter=domain_filter, suffix=suffix,
                  snr_aug=args.snr_aug, snr_range=(args.snr_min, args.snr_max),
                  subsample=args.subsample, field_aug=args.field_aug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
