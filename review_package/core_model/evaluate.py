#!/usr/bin/env python3
"""
evaluate.py - score a trained experiment and write plots + metrics.json.

Produces, for experiments/<features>_<model>/:
  * confusion_matrix.png
  * roc_pr.png            (ROC + AUC, Precision-Recall + AP) for target-vs-rest
  * detection_vs_snr.png  (recall as a function of mixed SNR in dB)
  * curves.png            (train vs val accuracy -> overfitting check)
  * metrics.json          (acc, precision, recall, false-alarm, AUC, AP,
                           min detectable SNR, latency ms on this CPU)

  python evaluate.py --exp mel2_cnn
  python evaluate.py --all
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from config import FEATURES_DIR, EXPERIMENTS_DIR, CLIPS, CFG, CLASSES
from models import MODELS
from features import FEATURES
from augment import mix_at_snr


def load_model(exp_dir, device):
    import torch
    ck = torch.load(exp_dir / "model.pt", map_location=device, weights_only=False)
    model = MODELS[ck["model"]](in_ch=ck["in_ch"], n_classes=len(ck["labels"]))
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    return model, ck


def predict(model, X, device, bs=128):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs]).to(device)
            out.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out)


def plot_confusion(cm, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_roc_pr(y_true_bin, p_target, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    fpr, tpr, _ = roc_curve(y_true_bin, p_target)
    pr, rc, _ = precision_recall_curve(y_true_bin, p_target)
    roc_auc = auc(fpr, tpr); ap = average_precision_score(y_true_bin, p_target)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0].set_title(f"ROC (AUC={roc_auc:.3f})"); ax[0].set_xlabel("false-alarm rate"); ax[0].set_ylabel("true-positive rate")
    ax[1].plot(rc, pr); ax[1].set_title(f"PR (AP={ap:.3f})"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return float(roc_auc), float(ap)


def plot_curves(log, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not log:
        return
    ep = range(1, len(log) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, [r["train_acc"] for r in log], label="train acc")
    ax.plot(ep, [r["val_acc"] for r in log], label="val acc")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend()
    ax.set_title("overfitting check (train vs val)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def detection_vs_snr(model, ck, device, exp_dir):
    """Mix clean target clips over background at known SNRs, featurize, predict."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not CLIPS.exists():            # clips.npz is local-only; skip on the cloud
        return None
    try:
        import librosa  # noqa: F401  (feature fns need it; may be absent on cloud)
    except Exception:
        return None
    d = np.load(CLIPS, allow_pickle=True)
    audio16, y, spl = d["audio"], d["y"], d["split"]
    audio = audio16.astype(np.float32) / 32768.0 if audio16.dtype == np.int16 else audio16
    ti = CLASSES.index("target_drone"); bi = CLASSES.index("background")
    tgt = audio[(y == ti) & (spl == 2)]
    bg = audio[y == bi]
    if len(tgt) == 0 or len(bg) == 0:
        return None
    feat = FEATURES[ck["features"]]
    if feat["needs_4ch"]:
        return None
    fn = feat["fn"]; rng = np.random.default_rng(0)
    snrs = [-10, -5, 0, 5, 10, 15, 20]; rates = []
    for snr in snrs:
        X = []
        for clip in tgt:
            mixed = mix_at_snr(clip, bg[rng.integers(len(bg))], snr, rng)
            X.append(fn(mixed, CFG))
        X = (np.stack(X) - ck["mean"]) / ck["std"]
        p = predict(model, X.astype(np.float32), device)
        rates.append(float((p.argmax(1) == ti).mean()))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(snrs, rates, "o-"); ax.axhline(0.5, ls="--", c="grey")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("detection rate"); ax.set_ylim(0, 1.02)
    ax.set_title("detection vs SNR")
    fig.tight_layout(); fig.savefig(exp_dir / "detection_vs_snr.png", dpi=110); plt.close(fig)
    min_det = next((s for s, r in zip(snrs, rates) if r >= 0.5), None)
    return {"snrs": snrs, "rates": rates, "min_snr_50pct_db": min_det}


def latency_ms(model, ck, device, in_ch):
    import torch, time
    x = torch.randn(1, in_ch, CFG.F, CFG.T).to(device)
    for _ in range(3):
        model(x)
    t = time.time()
    for _ in range(20):
        with torch.no_grad():
            model(x)
    return (time.time() - t) / 20 * 1000


def eval_one(exp_name):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = EXPERIMENTS_DIR / exp_name
    model, ck = load_model(exp_dir, device)
    X = np.load(FEATURES_DIR / f"{ck['features']}.npz", allow_pickle=True)
    Xte_all, y, dom, spl = X["X"], X["y"], X["domain"], X["split"]
    Xn = ((Xte_all - ck["mean"]) / ck["std"]).astype(np.float32)
    ti = CLASSES.index("target_drone")

    metrics = {"experiment": exp_name, "features": ck["features"], "model": ck["model"]}
    from sklearn.metrics import precision_score, recall_score, f1_score

    for tag, mask in [("all_test", spl == 2),
                      ("array_test", (spl == 2) & (dom == 1)),
                      ("mono_test", (spl == 2) & (dom == 0)),
                      ("ext_test", (spl == 2) & (dom == 2))]:
        if not mask.any():
            continue
        p = predict(model, Xn[mask], device)
        pred = p.argmax(1); yt = y[mask]
        ybin = (yt == ti).astype(int); pbin = (pred == ti).astype(int)
        m = dict(
            n=int(mask.sum()),
            accuracy=float((pred == yt).mean()),
            target_recall=float(recall_score(ybin, pbin, zero_division=0)),
            precision=float(precision_score(ybin, pbin, zero_division=0)),
            f1=float(f1_score(ybin, pbin, zero_division=0)),
            false_alarm=float((pred[yt != ti] == ti).mean()) if (yt != ti).any() else 0.0)
        metrics[tag] = m
        if tag == "all_test":
            cm = np.zeros((len(CLASSES), len(CLASSES)), int)
            for t, q in zip(yt, pred):
                cm[t, q] += 1
            plot_confusion(cm, exp_dir / "confusion_matrix.png")
            roc, ap = plot_roc_pr(ybin, p[:, ti], exp_dir / "roc_pr.png")
            m["roc_auc"] = roc; m["average_precision"] = ap

    log = []
    p = exp_dir / "train_log.json"
    if p.exists():
        log = json.load(open(p)); plot_curves(log, exp_dir / "curves.png")
        if log:
            metrics["final_gap_train_minus_val"] = log[-1]["gap"]

    dvs = detection_vs_snr(model, ck, device, exp_dir)
    if dvs:
        metrics["detection_vs_snr"] = dvs
    metrics["latency_ms_cpu" if device == "cpu" else "latency_ms_gpu"] = \
        latency_ms(model, ck, device, ck["in_ch"])

    json.dump(metrics, open(exp_dir / "metrics.json", "w"), indent=2)
    print(f"\n[{exp_name}]")
    for k in ("all_test", "array_test", "ext_test"):
        if k in metrics:
            mm = metrics[k]
            print(f"  {k:11s} acc={mm['accuracy']:.3f} recall={mm['target_recall']:.3f} "
                  f"FA={mm['false_alarm']:.3f}" +
                  (f" AUC={mm.get('roc_auc', float('nan')):.3f}" if 'roc_auc' in mm else ""))
    if dvs and dvs["min_snr_50pct_db"] is not None:
        print(f"  detects down to {dvs['min_snr_50pct_db']} dB SNR")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        exps = [d.name for d in EXPERIMENTS_DIR.iterdir()
                if (d / "model.pt").exists()] if EXPERIMENTS_DIR.exists() else []
    else:
        exps = [args.exp]
    if not exps or exps == [None]:
        print("pass --exp <name> or --all"); return 1
    summary = [eval_one(e) for e in exps]
    json.dump(summary, open(EXPERIMENTS_DIR / "summary.json", "w"), indent=2)
    print(f"\nWrote {EXPERIMENTS_DIR/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
