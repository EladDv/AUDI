#!/usr/bin/env python3
"""
train_stage2.py - EVO vs FPV subtype classifier (the cascade's Stage 2).

Stage 1 (the existing experiments/) already answers DRONE vs NO-DRONE and works
well. This script ONLY learns to tell your two drone types apart, and is meant
to run *after* Stage 1 says "drone". Combined, you get EVO / FPV / no-drone
without retraining the detector.

Tiny-data design (we have ~6 EVO + ~5 FPV flights), built to resist overfit:
  * uses your 4-mic (array) EVO/FPV clips ONLY  -> same mic, honest signal
  * a FROZEN Stage-1 CNN backbone (mel2_cnn body) as a fixed feature extractor
  * a small Logistic-Regression head on 128-d embeddings (few params)
  * Leave-One-Flight-Out cross-validation, scored at the FILE level
  * probes the others' 10/13-inch drones (a healthy model calls them FPV/not-EVO)

Nothing here modifies Stage-1. Outputs go to stage2_evo_fpv/.

  python train_stage2.py
  python train_stage2.py --backbone mel2_cnn --aug 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import CFG, CLIPS, EXPERIMENTS_DIR, PROJECT  # noqa: E402
from features import FEATURES, TYPE_STAT_NAMES, type_physics_stats  # noqa: E402
from models import MODELS                                    # noqa: E402
from augment import spec_augment_batch                       # noqa: E402
from scd_probe import scd_alpha_profile                      # noqa: E402


def fuse(emb, scd, stats, use_scd=True, use_stats=True):
    """Concatenate the TYPE views into one vector per clip:
       [ CNN embedding (spectral shape) ‖ SCD profile (rhythm) ‖ physics stats ].
    The physics stats give the head a DIRECT read on the mel3 type cues
    (>4 kHz comb + rotor rhythm) instead of only the pooled 128-d summary.

    use_scd / use_stats toggle the extra views for the CNN-vs-9-cues A/B:
      * default (both True)  -> fused (current behavior)
      * --emb-only           -> CNN embedding alone (pure CNN typer)
      * --no-stats           -> embedding + SCD, but drop the 9 physics cues
    """
    parts = [emb]
    if use_scd:
        parts.append(scd)
    if use_stats:
        parts.append(stats)
    return np.concatenate(parts, axis=1).astype(np.float32)

OUT = HERE / "stage2_evo_fpv"
SUBTYPES = ["evo", "fpv"]           # class 0 = evo, 1 = fpv
_DV2 = PROJECT / "data_other" / "raw" / "dataset_v2" / "dataset_v2"
PROBE_DIRS = [_DV2 / "10inchP", _DV2 / "13inchP"]


# --------------------------------------------------------------------------- #
def load_backbone(name: str):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(EXPERIMENTS_DIR / name / "model.pt",
                    map_location=device, weights_only=False)
    model = MODELS[ck["model"]](in_ch=ck["in_ch"], n_classes=len(ck["labels"]))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    if not hasattr(model, "body"):
        raise SystemExit(f"backbone '{name}' has no .body to freeze; use a cnn/tinyai model")
    print(f"backbone device: {device}")
    return model, ck


def embed(model, Xn, batch_size=512):
    """Frozen backbone -> 128-d embedding (global-avg-pooled body output).
    Runs on GPU when available; batched to keep VRAM manageable."""
    import torch
    device = next(model.parameters()).device
    parts = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch_size):
            xb = torch.tensor(Xn[i:i + batch_size].astype(np.float32)).to(device)
            f = model.body(xb).mean(dim=(2, 3))
            parts.append(f.cpu().numpy())
    return np.concatenate(parts, axis=0)


def featurize(clips, feat_name):
    fn = FEATURES[feat_name]["fn"]
    return np.stack([fn(c, CFG) for c in clips]).astype(np.float32)


# --------------------------------------------------------------------------- #
def build_arrayset():
    """Return (clips, labels, group_ids, file_names) for your 4-mic EVO/FPV."""
    d = np.load(CLIPS, allow_pickle=True)
    audio = d["audio"]          # keep int16 until after filtering
    grp = d["group"]; dom = d["domain"]; src = d["src"]; subf = d["subtype"]
    clip_sub = np.array([str(subf[g]) for g in grp])

    keep = (dom == 1) & np.isin(clip_sub, SUBTYPES)     # array + evo/fpv
    idx = np.where(keep)[0]
    # Convert to float32 ONLY for the ~3k kept clips, not all 52k.
    # Avoids materialising ~18 GB float32 for the full 4-mic array.
    raw = audio[idx]
    del audio  # release the large int16 array immediately
    if raw.dtype == np.int16:
        raw = raw.astype(np.float32) / 32768.0
    clips = [raw[j] for j in range(len(raw))]
    del raw
    labels = np.array([SUBTYPES.index(clip_sub[i]) for i in idx])
    groups = grp[idx]
    names = {int(g): Path(str(src[g])).name for g in set(groups.tolist())}
    return clips, labels, groups, names


def load_extra_fpv(existing_groups):
    """Fold the others' 10"/13" drones (PROBE_DIRS) in as SUPPLEMENTARY fpv
    training flights. Different mics (mono) than your array, so this is a
    domain-mismatched booster for the tiny FPV class - use it to bridge until
    your own 10" recordings land. Each wav becomes its own LOSO flight."""
    import librosa
    clips, labels, groups, names = [], [], [], {}
    gid = (max(existing_groups) + 1) if len(existing_groups) else 0
    n = int(CFG.clip_s * CFG.sr); hop = int(CFG.hop_s * CFG.sr)
    for pd in PROBE_DIRS:
        for w in sorted(Path(pd).glob("*.wav")) if Path(pd).exists() else []:
            yv, _ = librosa.load(str(w), sr=CFG.sr, mono=True)
            cl = [yv[s:s + n] for s in range(0, max(1, len(yv) - n + 1), hop)] \
                 or [np.pad(yv, (0, n))[:n]]
            for c in cl:
                clips.append(c.astype(np.float32))
                labels.append(SUBTYPES.index("fpv"))
                groups.append(gid)
            names[gid] = "borrowed:" + w.name
            gid += 1
    return clips, np.array(labels, int), np.array(groups, int), names


def augment_feats(Xn, rng, n):
    """n spec-augmented copies per sample (helps tiny data)."""
    if n <= 0:
        return Xn
    extra = []
    for _ in range(n):
        extra.append(np.stack([spec_augment_batch(t, rng) for t in Xn]))
    return np.concatenate([Xn] + extra, 0)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mel3_cnn",
                    help="Stage-1 experiment to use as frozen feature extractor")
    ap.add_argument("--aug", type=int, default=3,
                    help="spec-augment copies per training clip")
    ap.add_argument("--emb-only", action="store_true",
                    help="A/B: type from the CNN embedding ALONE (no SCD, no 9 cues) "
                         "- tests whether the CNN is enough on its own")
    ap.add_argument("--no-stats", action="store_true",
                    help="A/B: keep embedding+SCD but DROP the 9 physics cues "
                         "- isolates the contribution of the hand features")
    ap.add_argument("--extra-fpv", action="store_true",
                    help="fold the borrowed 10\"/13\" drones in as supplementary "
                         "fpv training flights (different mics; bridges tiny FPV data)")
    args = ap.parse_args()
    use_scd = not args.emb_only
    use_stats = not (args.emb_only or args.no_stats)
    mode = "emb_only" if args.emb_only else ("no_stats" if args.no_stats else "fused")
    out_suffix = "" if mode == "fused" else f"_{mode}"

    if not CLIPS.exists():
        raise SystemExit("clips.npz not found - run build_clips.py first.")
    outdir = HERE / ("stage2_evo_fpv" + out_suffix)
    outdir.mkdir(exist_ok=True)
    print(f"MODE: {mode}  (use_scd={use_scd}, use_stats={use_stats})  -> {outdir.name}")

    from sklearn.linear_model import LogisticRegression

    model, ck = load_backbone(args.backbone)
    feat_name = ck["features"]
    mean, std = ck["mean"], ck["std"]
    print(f"backbone={args.backbone}  feature={feat_name}")

    clips, y, groups, names = build_arrayset()
    if args.extra_fpv:
        ec, ey, eg, en = load_extra_fpv(groups.tolist())
        if ec:
            clips = clips + ec
            y = np.concatenate([y, ey]); groups = np.concatenate([groups, eg])
            names.update(en)
            print(f"+extra-fpv: folded {len(ec)} borrowed clips from "
                  f"{len(en)} files into the fpv class")
    flights = sorted(set(groups.tolist()))
    print(f"EVO/FPV clips: {len(clips)} from {len(flights)} flights")
    print("per-class clips:", {SUBTYPES[k]: int((y == k).sum()) for k in (0, 1)})
    print("per-flight:", {names[g]: int((groups == g).sum()) for g in flights})
    if len(flights) < 3 or len(set(y.tolist())) < 2:
        raise SystemExit("Not enough flights/classes for EVO-vs-FPV. Record more.")

    # featurize + normalize once, then embed once (frozen backbone)
    X = featurize(clips, feat_name)
    Xn = (X - mean) / std
    del X
    # cheap rhythm signal: scd profile per clip (1-D), fused into the head
    scd = np.stack([scd_alpha_profile(c, CFG.sr) for c in clips]).astype(np.float32)
    # explicit, named type cues (the >4 kHz comb + rotor rhythm you read by eye)
    stats = np.stack([type_physics_stats(c, CFG) for c in clips]).astype(np.float32)

    rng = np.random.default_rng(CFG.seed)

    # Pre-compute ALL augmented embeddings in ONE GPU pass.
    # This is 26× faster than re-augmenting+embedding per LOSO fold.
    print("Pre-computing augmented embeddings on GPU (one pass)...")
    Xn_aug = augment_feats(Xn, rng, args.aug)          # (N*(aug+1), 3, H, W)
    emb_aug = embed(model, Xn_aug)                      # GPU — fast
    del Xn_aug
    emb = emb_aug[:len(Xn)]                             # original (no aug)
    y_aug = np.tile(y, args.aug + 1)
    groups_aug = np.tile(groups, args.aug + 1)
    scd_aug = np.tile(scd, (args.aug + 1, 1))
    stats_aug = np.tile(stats, (args.aug + 1, 1))

    print(f"fusing TYPE views: embedding {emb.shape[1]}-d + scd {scd.shape[1]}-d "
          f"+ physics {stats.shape[1]}-d ({', '.join(TYPE_STAT_NAMES)})")

    def fit_head(Ftr, ytr):
        """z-score (train stats) then logistic regression. Returns (clf, mu, sd)."""
        mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        clf.fit((Ftr - mu) / sd, ytr)
        return clf, mu, sd

    # ---- Leave-One-Flight-Out CV ----------------------------------------- #
    clip_true, clip_pred = [], []
    file_true, file_pred, file_name = [], [], []
    for hold in flights:
        te = groups == hold
        tr_aug = groups_aug != hold      # augmented training clips, held-out excluded
        if len(set(y[~te].tolist())) < 2:
            print(f"  (skip fold {names[hold]}: training set has one class)")
            continue
        Ftr = fuse(emb_aug[tr_aug], scd_aug[tr_aug], stats_aug[tr_aug], use_scd, use_stats)
        clf, mu, sd = fit_head(Ftr, y_aug[tr_aug])
        pr = clf.predict((fuse(emb[te], scd[te], stats[te], use_scd, use_stats) - mu) / sd)
        clip_true += y[te].tolist(); clip_pred += pr.tolist()
        vote = int(round(pr.mean()))                 # majority over the flight
        file_true.append(int(y[te][0])); file_pred.append(vote)
        file_name.append(names[hold])

    clip_true = np.array(clip_true); clip_pred = np.array(clip_pred)
    file_true = np.array(file_true); file_pred = np.array(file_pred)

    clip_acc = float((clip_true == clip_pred).mean()) if len(clip_true) else 0.0
    file_acc = float((file_true == file_pred).mean()) if len(file_true) else 0.0
    cm = np.zeros((2, 2), int)
    for t, p in zip(file_true, file_pred):
        cm[t, p] += 1
    print(f"\nLOSO clip-level acc = {clip_acc:.3f}")
    print(f"LOSO file-level acc = {file_acc:.3f}  ({len(file_true)} flights)")
    print("file confusion [true x pred] (rows/cols = evo,fpv):\n", cm)
    for n, t, p in zip(file_name, file_true, file_pred):
        flag = "ok" if t == p else "WRONG"
        print(f"  {flag:5s} {n[:48]:48s} true={SUBTYPES[t]} pred={SUBTYPES[p]}")

    # ---- final model on ALL flights -------------------------------------- #
    Fall = fuse(emb_aug, scd_aug, stats_aug, use_scd, use_stats)
    final, fmu, fsd = fit_head(Fall, y_aug)

    # ---- probe the others' 10/13-inch drones ----------------------------- #
    import librosa
    probe = {}
    for pd in PROBE_DIRS:
        wavs = sorted(str(f) for f in Path(pd).glob("*.wav")) if pd.exists() else []
        preds = []
        for w in wavs:
            yv, _ = librosa.load(w, sr=CFG.sr, mono=True)
            n = int(CFG.clip_s * CFG.sr); hop = int(CFG.hop_s * CFG.sr)
            cl = [yv[s:s + n] for s in range(0, max(1, len(yv) - n + 1), hop)] or [np.pad(yv, (0, n))[:n]]
            Xp = (featurize(cl, feat_name) - mean) / std
            scdp = np.stack([scd_alpha_profile(c, CFG.sr) for c in cl]).astype(np.float32)
            statsp = np.stack([type_physics_stats(c, CFG) for c in cl]).astype(np.float32)
            pr = final.predict((fuse(embed(model, Xp), scdp, statsp, use_scd, use_stats) - fmu) / fsd)
            preds.append(SUBTYPES[int(round(pr.mean()))])
        probe[Path(pd).name] = dict(files=len(wavs), votes=dict(Counter(preds)))
        print(f"\nprobe {Path(pd).name}: {len(wavs)} files -> {dict(Counter(preds))}")

    # ---- save ------------------------------------------------------------ #
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(SUBTYPES)
        ax.set_yticks([0, 1]); ax.set_yticklabels(SUBTYPES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        ax.set_title(f"EVO/FPV LOSO (file acc {file_acc:.2f})")
        fig.tight_layout(); fig.savefig(outdir / "confusion_loso.png", dpi=120); plt.close(fig)
    except Exception as exc:
        print("plot skipped:", exc)

    np.savez(outdir / "stage2_model.npz",
             coef=final.coef_, intercept=final.intercept_,
             classes=np.array(SUBTYPES), backbone=args.backbone,
             feature=feat_name, mean=mean, std=std,
             use_scd=bool(use_scd), use_stats=bool(use_stats),
             emb_dim=int(emb.shape[1]), scd_dim=int(scd.shape[1]),
             stat_names=np.array(TYPE_STAT_NAMES), feat_mean=fmu, feat_std=fsd)
    json.dump(dict(backbone=args.backbone, feature=feat_name, mode=mode,
                   use_scd=bool(use_scd), use_stats=bool(use_stats),
                   extra_fpv=bool(args.extra_fpv),
                   n_clips=len(clips), n_flights=len(flights),
                   clip_acc=clip_acc, file_acc=file_acc,
                   confusion_evo_fpv=cm.tolist(),
                   per_flight=[dict(file=n, true=SUBTYPES[int(t)], pred=SUBTYPES[int(p)])
                               for n, t, p in zip(file_name, file_true, file_pred)],
                   probe=probe),
              open(outdir / "metrics.json", "w"), indent=2)
    print(f"\nSaved -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
