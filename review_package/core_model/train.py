#!/usr/bin/env python3
"""
train.py - train ONE model on ONE feature variant.

  python train.py --features mel2 --model cnn
  python train.py --features scalogram --model mobilenet --epochs 40

Two stages:
  PRETRAIN  on the mono pile (general drone/noise features)
  FINE-TUNE on your array data (adapt to your INMP441 mic signature)

Training stability ("small jumps"): Adam + cosine LR with warmup, gradient
clipping, class weights, SpecAugment, dropout, and best-checkpoint on val.

Saves experiments/<features>_<model>/model.pt and also records the train-vs-val
gap each epoch so evaluate.py can flag overfitting.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from config import FEATURES_DIR, EXPERIMENTS_DIR, CLASSES
from models import MODELS
from augment import spec_augment_batch


def load_feature(name):
    d = np.load(FEATURES_DIR / f"{name}.npz", allow_pickle=True)
    # Keep as float16 in RAM to halve memory footprint (~13 GB vs ~27 GB).
    # Conversion to float32 happens per-batch in make_batches / evaluate.
    return d["X"], d["y"], d["domain"], d["split"]


def normalize(X, mean=None, std=None):
    # Chunked normalize: never materializes a full float32 copy of X.
    # Each chunk is ~800 MB float32 instead of 27 GB all at once.
    chunk = 2000
    if mean is None:
        sample = X[:min(10000, len(X))].astype(np.float32)
        mean = sample.mean((0, 2, 3), keepdims=True)
        std  = sample.std((0, 2, 3), keepdims=True) + 1e-6
        del sample
    Xn = np.empty_like(X, dtype=np.float16)
    for i in range(0, len(X), chunk):
        Xn[i:i+chunk] = ((X[i:i+chunk].astype(np.float32) - mean) / std).astype(np.float16)
    return Xn, mean, std


def cosine_warmup(opt, step, total, warmup, base_lr):
    import math
    if step < warmup:
        lr = base_lr * step / max(1, warmup)
    else:
        p = (step - warmup) / max(1, total - warmup)
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * p))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def make_batches(X, y, bs, shuffle, rng, augment):
    idx = np.arange(len(X))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        xb = X[j].astype(np.float32)  # float16 → float32 per batch
        if augment:
            xb = np.stack([spec_augment_batch(t, rng) for t in xb])
        yield xb, y[j]


def evaluate(model, X, y, device, target_idx, batch_size=256):
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size].astype(np.float32)).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    nt = y != target_idx
    it = y == target_idx
    fa = float((pred[nt] == target_idx).mean()) if nt.any() else 0.0
    rec = float((pred[it] == target_idx).mean()) if it.any() else 0.0
    return float((pred == y).mean()), rec, fa


def fit(model, Xtr, ytr, Xva, yva, device, epochs, base_lr, bs, target_idx,
        tag, augment, log):
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)
    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(float)
    w = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(0)
    total = epochs * (len(Xtr) // bs + 1)
    step = 0
    best = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        tr_correct = tr_total = 0
        for xb, yb in make_batches(Xtr, ytr, bs, True, rng, augment):
            xb = torch.tensor(xb).to(device); yb = torch.tensor(yb).to(device)
            cosine_warmup(opt, step, total, total // 20 + 1, base_lr); step += 1
            opt.zero_grad()
            out = model(xb); loss = loss_fn(out, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_correct += int((out.argmax(1) == yb).sum()); tr_total += len(yb)
        tr_acc = tr_correct / max(tr_total, 1)
        va_acc, rec, fa = evaluate(model, Xva, yva, device, target_idx)
        score = rec - fa
        flag = ""
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " *"
        log.append(dict(stage=tag, epoch=ep + 1, train_acc=tr_acc,
                        val_acc=va_acc, recall=rec, false_alarm=fa,
                        gap=tr_acc - va_acc))
        print(f"[{tag}] ep{ep+1:02d} train={tr_acc:.3f} val={va_acc:.3f} "
              f"recall={rec:.3f} FA={fa:.3f} gap={tr_acc-va_acc:+.3f}{flag}")
    if best_state:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-finetune", action="store_true")
    ap.add_argument("--ft-array-only", action="store_true",
                    help="OLD behavior: fine-tune on array clips only. Default "
                         "now folds mono NO-DRONE clips into fine-tune so the "
                         "detector keeps its drone/no-drone boundary (otherwise "
                         "the array set is 100%% drone and the model collapses "
                         "to 'always drone').")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_idx = CLASSES.index("target_drone")
    print(f"device={device}  features={args.features}  model={args.model}")

    X, y, dom, spl = load_feature(args.features)
    X, mean, std = normalize(X)
    print(f"X={X.shape}  classes={CLASSES}")

    model = MODELS[args.model](in_ch=X.shape[1], n_classes=len(CLASSES)).to(device)

    mono, arr = dom == 0, dom == 1
    tr, va, te = spl == 0, spl == 1, spl == 2
    log: list[dict] = []

    m_tr, m_va = mono & tr, mono & va
    print(f"\n=== PRETRAIN (mono) train={m_tr.sum()} val={m_va.sum()} ===")
    model = fit(model, X[m_tr], y[m_tr], X[m_va], y[m_va], device,
                args.epochs, args.lr, args.batch, target_idx, "pre",
                not args.no_augment, log)

    if not args.no_finetune and arr.any():
        # Array clips are 100% drone (the mic only ever recorded drones). If we
        # fine-tune on them alone the loss is minimised by predicting "drone"
        # for everything and the pretrain boundary is wiped (recall=1, FA=0).
        # So we fold the mono NO-DRONE clips (background / false_alarm /
        # other_drone) into the fine-tune set, keeping the array drones to adapt
        # the mic signature. class weights in fit() handle the imbalance.
        nondrone = y != target_idx
        if args.ft_array_only:
            a_tr = arr & (tr | va); a_va = arr & te
            tag_extra = "array-only"
        else:
            a_tr = (arr & (tr | va)) | (mono & (tr | va) & nondrone)
            a_va = (arr & te) | (mono & te & nondrone)
            tag_extra = "array drones + mono no-drone"
        if a_tr.sum() >= 4:
            import collections
            comp = dict(collections.Counter(
                [CLASSES[c] for c in y[a_tr]]))
            print(f"\n=== FINE-TUNE ({tag_extra}) train={a_tr.sum()} "
                  f"val={a_va.sum()} ===\n    train classes: {comp}")
            model = fit(model, X[a_tr], y[a_tr],
                        X[a_va] if a_va.any() else X[a_tr],
                        y[a_va] if a_va.any() else y[a_tr], device,
                        args.ft_epochs, args.ft_lr, max(8, args.batch // 2),
                        target_idx, "ft", not args.no_augment, log)
        else:
            print("\n(skip fine-tune: <4 array clips - record more 4-ch data)")

    exp = EXPERIMENTS_DIR / f"{args.features}_{args.model}"
    exp.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": CLASSES,
                "mean": mean, "std": std, "in_ch": int(X.shape[1]),
                "features": args.features, "model": args.model}, exp / "model.pt")
    with open(exp / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved {exp/'model.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
