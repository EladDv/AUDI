#!/usr/bin/env python3
"""
detect_experiment.py - DRONE vs NO-DRONE: CNN vs the 9 physics cues vs fusion.

Answers the question "is the CNN enough for DETECTION on its own, or do the 9
hand cues help?" by scoring three detectors on the SAME train/test split:

  1. CNN-only      : the trained mel3_cnn, P(drone)=P(target)+P(other)   (no retrain)
  2. physics-only  : LogisticRegression on the 9 type_physics_stats cues
  3. fusion        : LogisticRegression on [CNN embedding (128) + 9 cues]

For each it reports ROC-AUC, average precision, and (at the best-F1 threshold)
drone recall + false-alarm rate. Binary label: drone = target_drone|other_drone.

  python detect_experiment.py
  python detect_experiment.py --backbone mel3_cnn --max-per-class 1500
  python detect_experiment.py --full            # use every clip (slow)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import CFG, CLIPS, CLASSES, EXPERIMENTS_DIR   # noqa: E402
from features import FEATURES, type_physics_stats          # noqa: E402
from models import MODELS                                   # noqa: E402

TARGET = CLASSES.index("target_drone")
OTHER  = CLASSES.index("other_drone")


def load_backbone(name):
    import torch
    ck = torch.load(EXPERIMENTS_DIR / name / "model.pt",
                    map_location="cpu", weights_only=False)
    model = MODELS[ck["model"]](in_ch=ck["in_ch"], n_classes=len(ck["labels"]))
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck


def cnn_forward(model, ck, clips, feat_name, bs=256):
    """Batched -> (P(drone) per clip, 128-d embedding per clip)."""
    import torch
    fn = FEATURES[feat_name]["fn"]
    mean, std = ck["mean"], ck["std"]
    pdrone, embs = [], []
    for i in range(0, len(clips), bs):
        chunk = clips[i:i + bs]
        X = np.stack([fn(c, CFG) for c in chunk]).astype(np.float32)
        Xn = ((X - mean) / std).astype(np.float32)
        with torch.no_grad():
            t = torch.tensor(Xn)
            p = torch.softmax(model(t), 1).numpy()
            e = model.body(t).mean(dim=(2, 3)).numpy()
        pdrone.append(p[:, TARGET] + p[:, OTHER]); embs.append(e)
        print(f"  cnn {min(i + bs, len(clips))}/{len(clips)}", end="\r")
    print()
    return np.concatenate(pdrone), np.concatenate(embs)


def metrics(name, ytrue, score):
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(ytrue, score)
    ap  = average_precision_score(ytrue, score)
    # best-F1 threshold sweep
    ths = np.linspace(score.min(), score.max(), 200)
    best = (0.0, 0.5, 0.0, 0.0)
    for th in ths:
        pred = score >= th
        tp = int((pred & (ytrue == 1)).sum()); fp = int((pred & (ytrue == 0)).sum())
        fn = int((~pred & (ytrue == 1)).sum()); tn = int((~pred & (ytrue == 0)).sum())
        rec = tp / max(1, tp + fn); prec = tp / max(1, tp + fp)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        fa = fp / max(1, fp + tn)
        if f1 > best[0]:
            best = (f1, th, rec, fa)
    f1, th, rec, fa = best
    print(f"  {name:14s}  AUC={auc:.3f}  AP={ap:.3f}  | @bestF1={f1:.3f} "
          f"thr={th:.2f}  recall={rec:.3f}  false_alarm={fa:.3f}")
    return dict(auc=float(auc), ap=float(ap), f1=float(f1),
                thr=float(th), recall=float(rec), false_alarm=float(fa))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mel3_cnn")
    ap.add_argument("--max-per-class", type=int, default=1500,
                    help="cap clips per (split, drone/not) for speed")
    ap.add_argument("--full", action="store_true", help="use every clip (slow)")
    args = ap.parse_args()

    if not CLIPS.exists():
        raise SystemExit("clips.npz not found - run build_clips.py first.")
    d = np.load(CLIPS, allow_pickle=True)
    audio = d["audio"]
    audio = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio
    y4 = d["y"]; spl = d["split"]
    ybin = np.isin(y4, [TARGET, OTHER]).astype(int)

    rng = np.random.default_rng(CFG.seed)

    def pick(split_val):
        idx = np.where(spl == split_val)[0]
        if args.full:
            return idx
        keep = []
        for cls in (0, 1):
            ci = idx[ybin[idx] == cls]
            if len(ci) > args.max_per_class:
                ci = rng.choice(ci, args.max_per_class, replace=False)
            keep.append(ci)
        return np.concatenate(keep) if keep else idx

    tr_idx, te_idx = pick(0), pick(2)
    if len(te_idx) == 0:                      # some builds fold test into val
        te_idx = pick(1)
    print(f"train clips={len(tr_idx)}  test clips={len(te_idx)}")
    print(f"train drone/not = {int(ybin[tr_idx].sum())}/{int((ybin[tr_idx]==0).sum())}"
          f"   test drone/not = {int(ybin[te_idx].sum())}/{int((ybin[te_idx]==0).sum())}")

    model, ck = load_backbone(args.backbone)
    feat_name = ck["features"]
    print(f"backbone={args.backbone} feature={feat_name}")

    # ---- features for both splits --------------------------------------- #
    tr_clips = [audio[i] for i in tr_idx]; te_clips = [audio[i] for i in te_idx]
    print("computing 9 physics cues...")
    Str = np.stack([type_physics_stats(c, CFG) for c in tr_clips]).astype(np.float32)
    Ste = np.stack([type_physics_stats(c, CFG) for c in te_clips]).astype(np.float32)
    print("running CNN (probs + embeddings)...")
    _,        Etr = cnn_forward(model, ck, tr_clips, feat_name)
    pdrone_te, Ete = cnn_forward(model, ck, te_clips, feat_name)

    ytr, yte = ybin[tr_idx], ybin[te_idx]
    from sklearn.linear_model import LogisticRegression

    def fit_score(Ftr, Fte):
        mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        clf.fit((Ftr - mu) / sd, ytr)
        return clf.predict_proba((Fte - mu) / sd)[:, 1]

    print("\n=== DETECTION: drone vs no-drone (test split) ===")
    res = {}
    res["cnn_only"]     = metrics("CNN-only", yte, pdrone_te)
    res["physics_only"] = metrics("physics-only(9)", yte, fit_score(Str, Ste))
    res["fusion"]       = metrics("fusion(cnn+9)", yte,
                                  fit_score(np.hstack([Etr, Str]), np.hstack([Ete, Ste])))

    import json
    out = HERE / "detect_experiment_metrics.json"
    json.dump(dict(backbone=args.backbone, n_train=len(tr_idx), n_test=len(te_idx),
                   results=res), open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")
    print("\nRead: higher AUC/AP = better separation. A low false_alarm with high "
          "recall is the goal. Compare physics-only vs CNN-only to see if the 9 cues "
          "alone detect drones, and fusion to see if they ADD to the CNN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
