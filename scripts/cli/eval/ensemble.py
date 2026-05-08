"""cmd_ensemble eval subcommand."""
from __future__ import annotations

from pathlib import Path

from audi.checkpoint import strip_compile_prefix, get_clip_seconds


def run(noise_path: str | None, drone_path: str | None) -> None:
    import lightning as L
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    
    from audi.cli_utils import NUM_WORKERS
    from audi.config import (
        MelConfig,
        MixConfig,
        ModelConfig,
        parse_snr_bins,
    )
    from audi.training.dataset import make_dataset
    from audi.training.detector import DroneDetector
    from audi.training.validation import (
        compute_calibration,
        compute_precision,
        compute_roc_values,
        find_threshold_at_precision,
        split_by_bin,
    )
    
    SR = MelConfig().sample_rate
    BATCH_SIZE = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_A = Path(
        "checkpoints_v2/full_train/a_256s_bce/lightning_logs/version_0/checkpoints/epoch=5-step=600.ckpt"
    )
    CKPT_B = Path(
        "checkpoints_v2/full_train/b_256s_focal_specaug/lightning_logs/version_0/checkpoints/epoch=11-step=1200.ckpt"
    )
    
    def build_model(ckpt_path: Path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt["hyper_parameters"]
        model_cfg = ModelConfig(
            arch=hp.get("model_arch", "cnn14"),
            pretrained=hp.get("pretrained_backbone", True),
            compile=False,
        )
        mel_cfg = MelConfig(
            n_mels=hp.get("n_mels", 128),
            n_fft=hp.get("n_fft", 1024),
            hop_length=hp.get("hop_length", 160),
            mean_db=hp.get("mel_mean"),
            std_db=hp.get("mel_std"),
        )
        opt_cfg = OptimizerConfig(
            lr=hp.get("lr", 1e-3),
            weight_decay=hp.get("weight_decay", 0.01),
        )
        model = DroneDetector(
            model=model_cfg,
            mel=mel_cfg,
            optimizer=opt_cfg,
            bin_names=hp.get("bin_names", []),
            loss_type=hp.get("loss_type", "bce"),
            label_smoothing=hp.get("label_smoothing", 0.0),
            per_bin_weights=hp.get("per_bin_weights", False),
            spec_augment_prob=float(hp.get("spec_augment_prob", 0.0)),
            bn_momentum=hp.get("bn_momentum", 0.1),
        )
        sd = ckpt["state_dict"]
        model.load_state_dict(strip_compile_prefix(sd), strict=False)
        return model.to(device).eval()
    
    @torch.no_grad()
    def compute_all_metrics(logits, label, per_bin, name):
        print(f"\n{'=' * 60}")
        print(f"{name}")
        print(f"{'=' * 60}")
        fpr, tpr, th, auc = compute_roc_values(logits, label)
        prec = compute_precision(logits, label, th)
        _, tpr_at_p90, _ = find_threshold_at_precision(prec, tpr, th, 0.90)
        ece = compute_calibration(logits, label, n_bins=15)[3]
        from audi.training.validation import compute_pr_curve
    
        _, _, _, ap = compute_pr_curve(logits, label)
        acc = ((logits > 0).astype(np.float32) == label).mean()
        print(f"  TPR@Prec90: {tpr_at_p90:.4f}")
        print(f"  AP:      {ap:.4f}")
        print(f"  AUC:     {auc:.4f}")
        print(f"  Acc:     {acc:.4f}")
        print(f"  ECE:     {ece:.4f}")
        if per_bin:
            print("\n  Per-bin TPR@Prec90:")
            for bin_name in [
                "far_field",
                "extreme",
                "very_hard",
                "hard",
                "medium",
                "easy",
            ]:
                if bin_name in per_bin:
                    bin_logits, bin_labels = per_bin[bin_name]
                    fpr_b, tpr_b, th_b, _ = compute_roc_values(
                        bin_logits, bin_labels
                    )
                    prec_b = compute_precision(bin_logits, bin_labels, th_b)
                    _, tpr_p90_b, _ = find_threshold_at_precision(
                        prec_b, tpr_b, th_b, 0.90
                    )
                    print(f"    {bin_name:<12} TPR@Prec90={tpr_p90_b:.4f}")
        return tpr_at_p90, ap, auc
    
    L.seed_everything(42)
    print(f"Device: {device}")
    print("Loading model A (BCE, 2.56s)...")
    model_a = build_model(CKPT_A)
    print(
        f"  Done — loss_type={model_a._loss_type}, n_params={sum(p.numel() for p in model_a.parameters()):,}"
    )
    print("Loading model B (Focal+SpecAug, 2.56s)...")
    model_b = build_model(CKPT_B)
    print(
        f"  Done — loss_type={model_b._loss_type}, n_params={sum(p.numel() for p in model_b.parameters()):,}"
    )
    
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
    _ckpt = torch.load(CKPT_A, map_location="cpu", weights_only=False)
    clip_samples = int(SR * get_clip_seconds(_ckpt["hyper_parameters"]))
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
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )
    print(f"Validation dataset: {len(val_ds)} samples ({len(val_dl)} batches)")
    
    all_logits_a, all_logits_b, all_labels = [], [], []
    all_bin_names, all_drone, all_noise, all_snr = [], [], [], []
    bin_distribution = {name: 0 for name in bin_names}
    bin_distribution["neg"] = 0
    for batch in val_dl:
        if len(batch) == 6:
            wav, label, bin_idx, drone, noise, snr_val = batch
            flat_names = [
                bin_names[int(i)] if i >= 0 else "neg" for i in bin_idx.numpy()
            ]
            for n in flat_names:
                bin_distribution[n] = bin_distribution.get(n, 0) + 1
            all_bin_names.append(flat_names)
            all_drone.append(drone)
            all_noise.append(noise)
            all_snr.append(snr_val)
        else:
            wav, label = batch
        wav = wav.to(device)
        logit_a = model_a(wav)
        logit_b = model_b(wav)
        all_logits_a.append(logit_a.cpu())
        all_logits_b.append(logit_b.cpu())
        all_labels.append(label)
    
    logits_a = torch.cat(all_logits_a).numpy()
    logits_b = torch.cat(all_logits_b).numpy()
    labels = torch.cat(all_labels).numpy()
    logits_ens = (logits_a + logits_b) / 2.0
    
    print(f"\n  Bin distribution: {bin_distribution}")
    print(f"  Total samples: {sum(bin_distribution.values())}")
    
    _per_bin_labels = {}
    if all_bin_names:
        flat_bin_names = [name for batch in all_bin_names for name in batch]
        per_bin_ens = split_by_bin(logits_ens, labels, np.array(flat_bin_names))
        per_bin_a = split_by_bin(logits_a, labels, np.array(flat_bin_names))
        per_bin_b = split_by_bin(logits_b, labels, np.array(flat_bin_names))
    else:
        per_bin_ens = per_bin_a = per_bin_b = {}
    
    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS — A (BCE) + B (Focal+SpecAug)")
    print("=" * 70)
    
    m_a = compute_all_metrics(
        logits_a, labels, per_bin_a, "Model A alone (BCE, 2.56s)"
    )
    m_b = compute_all_metrics(
        logits_b, labels, per_bin_b, "Model B alone (Focal, 2.56s)"
    )
    m_ens = compute_all_metrics(
        logits_ens, labels, per_bin_ens, "ENSEMBLE A+B (avg logits)"
    )
    
    print(f"\n{'=' * 70}")
    print(
        f"{'Metric':<20} {'A (BCE)':>12} {'B (Focal)':>12} {'ENSEMBLE':>12} {'Δ vs best':>10}"
    )
    print(f"{'=' * 70}")
    print(
        f"{'TPR@Prec90':<20} {m_a[0]:>12.4f} {m_b[0]:>12.4f} {m_ens[0]:>12.4f} {m_ens[0] - max(m_a[0], m_b[0]):>+10.4f}"
    )
    print(
        f"{'AP':<20} {m_a[1]:>12.4f} {m_b[1]:>12.4f} {m_ens[1]:>12.4f} {m_ens[1] - max(m_a[1], m_b[1]):>+10.4f}"
    )
    print(
        f"{'AUC':<20} {m_a[2]:>12.4f} {m_b[2]:>12.4f} {m_ens[2]:>12.4f} {m_ens[2] - max(m_a[2], m_b[2]):>+10.4f}"
    )
    
    print(
        f"\n{'Per-bin TPR@Prec90':<20} {'A (BCE)':>12} {'B (Focal)':>12} {'ENSEMBLE':>12}"
    )
    print(f"{'-' * 56}")
    for bin_name in [
        "far_field",
        "extreme",
        "very_hard",
        "hard",
        "medium",
        "easy",
    ]:
    
        def get_tpr(per_bin, bn):
            if bn in per_bin:
                bl, ll = per_bin[bn]
                fpr_b, tpr_b, th_b, _ = compute_roc_values(bl, ll)
                prec_b = compute_precision(bl, ll, th_b)
                _, tp_p90, _ = find_threshold_at_precision(
                    prec_b, tpr_b, th_b, 0.90
                )
                return tp_p90
            return float("nan")
    
        va = get_tpr(per_bin_a, bin_name)
        vb = get_tpr(per_bin_b, bin_name)
        vens = get_tpr(per_bin_ens, bin_name)
        print(f"{bin_name:<20} {va:>12.4f} {vb:>12.4f} {vens:>12.4f}")
    
    
    # ====================================================================
    # Entry point
    # ====================================================================
    
    
