"""cmd_operational eval subcommand."""
from __future__ import annotations

from pathlib import Path

from audi.checkpoint import strip_compile_prefix, get_clip_seconds


def run(noise_path: str | None, drone_path: str | None) -> None:
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
    
    SR = MelConfig().sample_rate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT = Path(
        "checkpoints_v2/full_train/b_256s_focal_specaug/lightning_logs/version_0/checkpoints/epoch=11-step=1200.ckpt"
    )
    FPR_TARGETS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    
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
            "very_hard:-20:-15:0.20",
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
        dataset_length=16 * 300,
    )
    ds = make_dataset(cfg=mix_cfg, split="validation", return_components=True)
    dl = DataLoader(ds, batch_size=16, num_workers=NUM_WORKERS, pin_memory=True)
    
    logits, labels, bin_idx = [], [], []
    for batch in dl:
        wav, label, bi, _, _, _ = batch
        with torch.no_grad():
            logit = model(wav.to(device))
        logits.append(logit.cpu().numpy().flatten())
        labels.append(label.numpy().flatten())
        bin_idx.append(bi.numpy().flatten())
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    bin_idx = np.concatenate(bin_idx).astype(int)
    bin_names_list = [bin_names[i] if i >= 0 else "" for i in bin_idx]
    
    def get_tpr_at_fpr(bl, ll, tfpr):
        fpr, tpr, _, _ = compute_roc_values(bl, ll)
        return tpr_at_fpr(fpr, tpr, tfpr)
    
    per_bin = split_by_bin(logits, labels, bin_names_list)
    DRONES_PER_BIN = 100
    BACKGROUNDS = 500
    SLIDING_WINDOW_SECONDS = 15
    bin_labels = ["easy", "medium", "hard", "very_hard", "extreme", "far_field"]
    
    print(
        "Scenario: 100 drones per bin, 500 background 15s recordings, model clips at 2.56s\n"
    )
    
    for fpr_target in FPR_TARGETS:
        fp_count = int(BACKGROUNDS * fpr_target)
        fp_per_hour = (3600 / SLIDING_WINDOW_SECONDS) * fpr_target
        print(f"{'=' * 65}")
        print(f"THRESHOLD @ {fpr_target * 100:.0f}% FPR")
        print(f"{'=' * 65}")
        print(
            f"  False positives: {fp_count} / {BACKGROUNDS} backgrounds ({fp_per_hour:.1f}/hour at 15s windows)"
        )
        print()
        print(f"  {'Bin':<15} {'Detected':>10} {'Missed':>10} {'TP:FP':>10}")
        print(f"  {'─' * 45}")
        total_detected = 0
        for bn in bin_labels:
            if bn in per_bin:
                bl, ll = per_bin[bn]
                tpr = get_tpr_at_fpr(bl, ll, fpr_target)
                detected = int(DRONES_PER_BIN * tpr)
                missed = DRONES_PER_BIN - detected
                total_detected += detected
                _ratio = detected / max(fp_count, 1)
                print(
                    f"  {bn:<15} {detected:>10} {missed:>10} 1:{max(1, fp_count // max(detected, 1)) if detected > 0 and fp_count > 0 else '∞'}"
                )
        print(f"  {'─' * 45}")
        print(
            f"  {'TOTAL':<15} {total_detected:>10} {DRONES_PER_BIN * 6 - total_detected:>10} 1:{max(1, fp_count // max(total_detected, 1)):.0f} (TP:FP = {total_detected}:{fp_count})"
        )
        print()
    
    
    # ====================================================================
    # attack-runs — Attack-run evaluation at calibrated precision
    # ====================================================================
    
    
