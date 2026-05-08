"""cmd_fpr_multi eval subcommand."""
from __future__ import annotations

from pathlib import Path

from audi.checkpoint import strip_compile_prefix, get_clip_seconds


def run_multi(noise_path: str | None, drone_path: str | None) -> None:
    import argparse

    import lightning as L
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from audi.cli_utils import NUM_WORKERS
    from audi.config import MelConfig, MixConfig, ModelConfig, parse_snr_bins
    from audi.training.dataset import make_dataset
    from audi.training.detector import DroneDetector
    from audi.training.validation import (
        compute_roc_values,
        split_by_bin,
        tpr_at_fpr,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fpr-targets", default="0.01,0.02,0.05,0.10,0.20,0.50",
                    help="Comma-separated FPR targets")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("run_name", nargs="?")
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu"
    )
    SR = MelConfig().sample_rate
    FPR_TARGETS = [float(x) for x in args.fpr_targets.split(",")]
    _DEFAULT_CKPT = Path(
        "checkpoints_v2/full_train/b_256s_focal_specaug/lightning_logs/version_0/checkpoints/epoch=11-step=1200.ckpt"
    )
    CKPT = args.ckpt or _DEFAULT_CKPT

    L.seed_everything(42)
    ckpt_data = torch.load(CKPT, map_location="cpu", weights_only=False)
    hp = ckpt_data["hyper_parameters"]
    model = DroneDetector(
        model=ModelConfig(
            arch=hp.get("model_arch", "cnn14"),
            pretrained=hp.get("pretrained_backbone", True),
        ),
        bin_names=hp.get("bin_names", []),
        loss_type=hp.get("loss_type", "bce"),
    )
    model.load_state_dict(strip_compile_prefix(ckpt_data["state_dict"]), strict=False)
    model = model.to(device).eval()
    
    snr_bins = parse_snr_bins(
        [
            "easy:-5:0:0.20",
            "medium:-10:-5:0.20",
            "hard:-15:-10:0.20",
            "very_hard:-20:-15:0.15",
            "extreme:-25:-20:0.15",
            "far_field:-30:-25:0.10",
        ]
    )
    bin_names = [b.name for b in snr_bins]
    clip = int(SR * get_clip_seconds(hp))
    _ = noise_path, drone_path
    mix_cfg = MixConfig(
        noise_path=noise_path,
        drone_path=drone_path,
        snr_bins=snr_bins,
        target_length_samples=clip,
        dataset_length=16 * 200,
    )
    ds = make_dataset(cfg=mix_cfg, split="validation", return_components=True)
    dl = DataLoader(ds, batch_size=16, num_workers=NUM_WORKERS, pin_memory=True)
    
    logits, labels, bin_idx = [], [], []
    for batch in dl:
        wav, label, bi, *_ = batch
        with torch.no_grad():
            logit = model(wav.to(device))
        logits.append(logit.cpu().numpy().flatten())
        labels.append(label.numpy().flatten())
        bin_idx.append(bi.numpy().flatten())
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    bin_idx = np.concatenate(bin_idx).astype(int)
    bin_names_list = [bin_names[i] if i >= 0 else "" for i in bin_idx]
    
    def find_thresh(bl, ll, tfpr):
        fpr, tpr, th, _ = compute_roc_values(bl, ll)
        idx = np.searchsorted(-fpr, -tfpr, side="left")
        idx = min(idx, len(fpr) - 1)
        if idx > 0:
            fl, fh = fpr[idx - 1], fpr[idx]
            tl, th_h = th[idx - 1], th[idx]
            a = (tfpr - fl) / (fh - fl) if fh != fl else 0
            return tl + a * (th_h - tl), tpr_at_fpr(fpr, tpr, tfpr)
        return th[0], tpr_at_fpr(fpr, tpr, tfpr)
    
    per_bin = split_by_bin(logits, labels, bin_names_list)
    print(f"\n{'Bin':<15}", end="")
    for fpr in FPR_TARGETS:
        print(f"  thr@{fpr * 100:>3.0f}%FPR(prob)", end=" ")
    print(" | TPR@Prec90")
    print("=" * (15 + 21 * len(FPR_TARGETS) + 8))
    
    for bn in ["far_field", "extreme", "very_hard", "hard", "medium", "easy"]:
        if bn not in per_bin:
            continue
        bl, ll = per_bin[bn]
        print(f"{bn:<15}", end="")
        for fpr in FPR_TARGETS:
            th, tv = find_thresh(bl, ll, fpr)
            tl = float(np.log(th / (1 - th + 1e-10))) if 0 < th < 1 else -99
            print(f"  {th:.4f}({tl:<+6.3f})", end=" ")
        _, t10 = find_thresh(bl, ll, 0.10)
        print(f" | {t10:.4f}")
    
    print(f"\n{'OVERALL':<15}", end="")
    for fpr in FPR_TARGETS:
        th, tv = find_thresh(logits, labels, fpr)
        tl = float(np.log(th / (1 - th + 1e-10))) if 0 < th < 1 else -99
        print(f"  {th:.4f}({tl:<+6.3f})", end=" ")
    _, t10 = find_thresh(logits, labels, 0.10)
    print(f" | {t10:.4f}")
    
    hardness = ["easy", "medium", "hard", "very_hard", "extreme", "far_field"]
    print(f"\n\n{'Composite ≤':<20}", end="")
    for fpr in FPR_TARGETS:
        print(f"  thr@{fpr * 100:>3.0f}%FPR(prob)", end=" ")
    print(" | TPR@Prec90")
    print("=" * (20 + 21 * len(FPR_TARGETS) + 8))
    
    for i in range(1, len(hardness) + 1):
        inc = hardness[:i]
        label = f"<= {hardness[i - 1]}"
        pos = labels > 0.5
        ba = np.array(bin_names_list)
        mask = (~pos) | (pos & np.isin(ba, inc))
        cl, cll = logits[mask], labels[mask]
        print(f"{label:<20}", end="")
        for fpr in FPR_TARGETS:
            th, tv = find_thresh(cl, cll, fpr)
            tl = float(np.log(th / (1 - th + 1e-10))) if 0 < th < 1 else -99
            print(f"  {th:.4f}({tl:<+6.3f})", end=" ")
        _, t10 = find_thresh(cl, cll, 0.10)
        print(f" | {t10:.4f}")
    
    
    # ====================================================================
    # operational — Operational metrics
    # ====================================================================
    
    
