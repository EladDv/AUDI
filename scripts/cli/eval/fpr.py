"""cmd_fpr_thresholds eval subcommand."""
from __future__ import annotations

from pathlib import Path

from audi.checkpoint import strip_compile_prefix, get_clip_seconds


def run(noise_path: str | None, drone_path: str | None) -> None:
    import argparse

    import lightning as L
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from audi.cli_utils import NUM_WORKERS
    from audi.config import MixConfig, ModelConfig, parse_snr_bins
    from audi.training.dataset import make_dataset
    from audi.training.detector import DroneDetector
    from audi.training.validation import (
        compute_roc_values,
        split_by_bin,
        tpr_at_fpr,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("run_name", nargs="?")
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu"
    )
    SR = MelConfig().sample_rate
    _DEFAULT_CKPT = Path(
        "checkpoints_v2/full_train/b_256s_focal_specaug/lightning_logs/version_0/checkpoints/epoch=11-step=1200.ckpt"
    )
    BEST_CKPT = args.ckpt or _DEFAULT_CKPT

    L.seed_everything(42)
    ckpt = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    model = DroneDetector(
        model=ModelConfig(
            arch=hp.get("model_arch", "cnn14"),
            pretrained=hp.get("pretrained_backbone", True),
        ),
        bin_names=hp.get("bin_names", []),
        loss_type=hp.get("loss_type", "bce"),
    )
    model.load_state_dict(strip_compile_prefix(ckpt["state_dict"]), strict=False)
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
    print(f"Bin order (easiest→hardest): {bin_names}")
    
    clip_samples = int(SR * get_clip_seconds(hp))
    _ = noise_path, drone_path
    mix_cfg = MixConfig(
        noise_path=noise_path,
        drone_path=drone_path,
        snr_bins=snr_bins,
        target_length_samples=clip_samples,
        dataset_length=16 * 200,
    )
    val_ds = make_dataset(
        cfg=mix_cfg, split="validation", return_components=True
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    
    all_logits, all_labels, all_bins = [], [], []
    for batch in val_dl:
        wav, label, bin_idx, drone, noise, snr_val = batch
        with torch.no_grad():
            logit = model(wav.to(device))
        all_logits.append(logit.cpu().numpy().flatten())
        all_labels.append(label.numpy().flatten())
        all_bins.append(bin_idx.numpy().flatten())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    bin_idx = np.concatenate(all_bins).astype(int)
    bin_names_list = [bin_names[i] if i >= 0 else "" for i in bin_idx]
    per_bin = split_by_bin(logits, labels, bin_names_list)
    
    TARGET_FPR = 0.10
    
    def find_threshold_at_fpr(bin_logits, bin_labels, target_fpr):
        fpr, tpr, thresholds, auc = compute_roc_values(bin_logits, bin_labels)
        idx = np.searchsorted(-fpr, -target_fpr, side="left")
        idx = min(idx, len(fpr) - 1)
        if idx > 0 and idx < len(fpr):
            f_low, f_high = fpr[idx - 1], fpr[idx]
            t_low, t_high = thresholds[idx - 1], thresholds[idx]
            if f_high != f_low:
                alpha = (target_fpr - f_low) / (f_high - f_low)
                thresh = t_low + alpha * (t_high - t_low)
            else:
                thresh = thresholds[idx]
        elif idx == 0:
            thresh = thresholds[0]
        else:
            thresh = thresholds[-1]
        tpr_at_target = tpr_at_fpr(fpr, tpr, target_fpr)
        return thresh, tpr_at_target, fpr[idx] if idx < len(fpr) else 0.0
    
    print(f"\n{'=' * 70}")
    print(
        f"THRESHOLDS AT {TARGET_FPR * 100:.0f}% FPR — Per Bin (Model B: Focal+SpecAug)"
    )
    print(f"{'=' * 70}")
    print(
        f"{'Bin':<15} {'Thresh (prob)':>15} {'Thresh (logit)':>15} {'TPR':>10} {'n_pos':>8}"
    )
    print(f"{'-' * 63}")
    for bn in ["far_field", "extreme", "very_hard", "hard", "medium", "easy"]:
        if bn in per_bin:
            bl, ll = per_bin[bn]
            n_pos = int((ll > 0.5).sum())
            th_prob, tpr_val, actual_fpr = find_threshold_at_fpr(
                bl, ll, TARGET_FPR
            )
            th_logit = (
                float(np.log(th_prob / (1 - th_prob + 1e-10)))
                if 0 < th_prob < 1
                else float("nan")
            )
            print(
                f"{bn:<15} {th_prob:>15.4f} {th_logit:>15.4f} {tpr_val:>10.4f} {n_pos:>8}"
            )
    print(f"\n{'─' * 63}")
    th_prob_all, tpr_all, _ = find_threshold_at_fpr(logits, labels, TARGET_FPR)
    th_logit_all = (
        float(np.log(th_prob_all / (1 - th_prob_all + 1e-10)))
        if 0 < th_prob_all < 1
        else float("nan")
    )
    n_pos_all = int((labels > 0.5).sum())
    print(
        f"{'OVERALL':<15} {th_prob_all:>15.4f} {th_logit_all:>15.4f} {tpr_all:>10.4f} {n_pos_all:>8}"
    )
    
    print(f"\n\n{'=' * 70}")
    print('COMPOSITE — All bins "easier than" threshold')
    print(f"{'=' * 70}")
    print(
        f"{'Easier than':<20} {'Bins included':<35} {'Thresh (prob)':>15} {'Thresh (logit)':>15} {'TPR':>10}"
    )
    print(f"{'-' * 95}")
    hardness_order = [
        "easy",
        "medium",
        "hard",
        "very_hard",
        "extreme",
        "far_field",
    ]
    for i in range(1, len(hardness_order) + 1):
        included = hardness_order[:i]
        label = f"≤ {hardness_order[i - 1]}"
        pos_mask = labels > 0.5
        bin_array = np.array(bin_names_list)
        in_mask = np.isin(bin_array, included)
        composite_mask = (~pos_mask) | (pos_mask & in_mask)
        comp_logits = logits[composite_mask]
        comp_labels = labels[composite_mask]
        th_prob, tpr_val, _ = find_threshold_at_fpr(
            comp_logits, comp_labels, TARGET_FPR
        )
        th_logit = (
            float(np.log(th_prob / (1 - th_prob + 1e-10)))
            if 0 < th_prob < 1
            else float("nan")
        )
        n_total = len(comp_logits)
        n_pos_comp = int(comp_labels.sum())
        bins_str = ", ".join(included)
        print(
            f"{label:<20} {bins_str:<35} {th_prob:>15.4f} {th_logit:>15.4f} {tpr_val:>10.4f}"
        )
        print(f"{'':>20} {'':>15} n_samples={n_total}, n_pos={n_pos_comp}")
    
    
    # ====================================================================
    # fpr-multi — Thresholds at multiple FPR targets
    # ====================================================================
    
    
