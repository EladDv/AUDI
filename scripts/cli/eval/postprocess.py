"""cmd_postprocess eval subcommand."""
from __future__ import annotations

import sys
from pathlib import Path

from audi.checkpoint import strip_compile_prefix, get_clip_seconds


def run(noise_path: str | None, drone_path: str | None) -> None:
    import argparse
    import shutil

    import lightning as L
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from audi.cli_utils import NUM_WORKERS
    from audi.config import (
        AugmentationConfig,
        MelConfig,
        MixConfig,
        ModelConfig,
        parse_snr_bins,
    )
    from audi.training.dataset import make_dataset
    from audi.training.detector import DroneDetector
    from audi.training.validation import (
        compute_precision,
        compute_roc_values,
        find_threshold_at_precision,
        split_by_bin,
        tpr_at_fpr,
    )

    SR = MelConfig().sample_rate
    HARDNESS = ["easy", "medium", "hard", "very_hard", "extreme", "far_field"]
    GROUPINGS = {
        "all": HARDNESS,
        "medium_and_easier": ["medium", "easy"],
        "hard_and_easier": ["hard", "medium", "easy"],
        "extra_hard_and_easier": ["very_hard", "hard", "medium", "easy"],
    }

    ap = argparse.ArgumentParser(description="Post-process sweep results.")
    ap.add_argument("--device", default="auto",
                    help='Device ("cuda", "cpu", or "auto" to detect)')
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fpr-targets", default="0.01,0.05,0.10,0.20",
                    help="Comma-separated FPR targets")
    ap.add_argument("--all", dest="all_runs", action="store_true",
                    help="Reprocess all runs (even already-postprocessed ones)")
    ap.add_argument("sweep_dir", type=Path, nargs="?")
    ap.add_argument("run_name", nargs="?")
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu"
    )
    FPR_TARGETS = [float(x) for x in args.fpr_targets.split(",")]
    BATCH_SIZE = args.batch_size

    SWEEP_DIR = args.sweep_dir
    RUN_NAME = args.run_name
    
    def _build_detector(ckpt: dict, bin_names: list[str]) -> DroneDetector:
        hp = ckpt["hyper_parameters"]
        # Handle both old flat-param and new ModelConfig checkpoints
        model_hp = hp.get("model", {})
        if isinstance(model_hp, dict):
            model_cfg = ModelConfig(
                arch=model_hp.get("arch", hp.get("model_arch", "cnn14")),
                pretrained=model_hp.get(
                    "pretrained", hp.get("pretrained_backbone", True)
                ),
                compile=model_hp.get("compile", False),
            )
        else:
            # Clone config but force compile=False for eval
            model_cfg = ModelConfig(
                arch=model_hp.arch,
                num_classes=model_hp.num_classes,
                pretrained=model_hp.pretrained,
                compile=False,
            )
        mel_hp = hp.get("mel", {})
        if isinstance(mel_hp, dict):
            mel_cfg = MelConfig(
                n_mels=mel_hp.get("n_mels", hp.get("n_mels", 128)),
                n_fft=mel_hp.get("n_fft", hp.get("n_fft", 1024)),
                hop_length=mel_hp.get("hop_length", hp.get("hop_length", 160)),
                mean_db=mel_hp.get("mean_db", hp.get("mel_mean")),
                std_db=mel_hp.get("std_db", hp.get("mel_std")),
            )
        else:
            mel_cfg = mel_hp  # already a MelConfig object
        opt_hp = hp.get("optimizer", {})
        if isinstance(opt_hp, dict):
            opt_cfg = OptimizerConfig(
                lr=opt_hp.get("lr", hp.get("lr", 1e-3)),
                weight_decay=opt_hp.get(
                    "weight_decay", hp.get("weight_decay", 0.01)
                ),
                schedule=opt_hp.get(
                    "schedule", hp.get("lr_schedule", "constant")
                ),
                warmup_epochs=opt_hp.get(
                    "warmup_epochs", hp.get("warmup_epochs", 0)
                ),
            )
        else:
            opt_cfg = opt_hp  # already an OptimizerConfig object
        model = DroneDetector(
            model=model_cfg,
            mel=mel_cfg,
            optimizer=opt_cfg,
            bin_names=bin_names,
            loss_type=hp.get("loss_type", "bce"),
            label_smoothing=hp.get("label_smoothing", 0.0),
            per_bin_weights=hp.get("per_bin_weights", False),
            spec_augment_prob=float(hp.get("spec_augment_prob", 0.0)),
            mixup_alpha=hp.get("mixup_alpha", 0.0),
            cutmix_alpha=hp.get("cutmix_alpha", 0.0),
            dropout=hp.get("dropout", 0.0),
            bn_momentum=hp.get("bn_momentum", 0.1),
        )
        return model
    
    def process_run(run_dir: Path, bin_names, clip_seconds: float, force_all: bool = False) -> dict:
        name = run_dir.name
        # Try default Lightning path first, fall back to direct checkpoints/ dir
        ckpt_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
        if not ckpt_dir.exists():
            ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            print(f"  SKIP: no checkpoints in {run_dir}")
            return {}
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            print(f"  SKIP: no .ckpt files in {ckpt_dir}")
            return {}
        out_dir = run_dir / "eval_data"
        out_dir.mkdir(exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"{name}  ({len(ckpts)} checkpoints)")
        print(f"{'=' * 60}")

        # Build per-run val_dl using this run's clip_seconds
        clip_samples = int(SR * clip_seconds)
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
        best_auc = -1
        best_tag = None
        best_metrics = {}
        for ckpt_path in ckpts:
            epoch = int(ckpt_path.stem.split("epoch=")[1].split("-")[0])
            tag = f"epoch_{epoch:02d}"
            if not force_all and (out_dir / f"predictions_{tag}.pt").exists():
                print(f"  Already done: {tag}")
                continue
            print(f"  Processing: {tag} ({ckpt_path.name})")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = _build_detector(ckpt, bin_names)
            model.load_state_dict(
                strip_compile_prefix(ckpt["state_dict"]), strict=False
            )
            model = model.to(device).eval()
            all_logits, all_labels, all_bins = [], [], []
            for batch in val_dl:
                wav, label, bi, *_ = batch
                with torch.no_grad():
                    logit = model(wav.to(device))
                all_logits.append(logit.cpu())
                all_labels.append(label)
                all_bins.append(bi)
            logits = torch.cat(all_logits).numpy().flatten()
            labels = torch.cat(all_labels).numpy().flatten()
            bin_idx = torch.cat(all_bins).numpy().flatten().astype(int)
            bin_names_list = np.array(
                [bin_names[i] if i >= 0 else "" for i in bin_idx]
            )
            torch.save(
                {
                    "logits": logits,
                    "labels": labels,
                    "bin_idx": bin_idx,
                    "bin_names": bin_names_list,
                },
                out_dir / f"predictions_{tag}.pt",
            )
            per_bin = split_by_bin(logits, labels, bin_names_list.tolist())
            curves = {}
            for bn in HARDNESS:
                if bn in per_bin:
                    bl, ll = per_bin[bn]
                    fpr, tpr, th, auc = compute_roc_values(bl, ll)
                    curves[bn] = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": th,
                        "auc": auc,
                    }
            fpr, tpr, th, auc = compute_roc_values(logits, labels)
            curves["overall"] = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": th,
                "auc": auc,
            }
            np.savez_compressed(
                out_dir / f"curves_{tag}.npz",
                **{
                    f"{bn}/{k}": v
                    for bn, data in curves.items()
                    for k, v in data.items()
                },
            )
            metrics = {}
            for group_name, included_bins in GROUPINGS.items():
                pos_mask = labels > 0.5
                in_mask = np.isin(bin_names_list, included_bins)
                mask = (~pos_mask) | (pos_mask & in_mask)
                gl, gll = logits[mask], labels[mask]
                gfpr, gtpr, _, _ = compute_roc_values(gl, gll)
                for tf in FPR_TARGETS:
                    metrics[
                        f"group_{group_name}/tpr_at_fpr_{int(tf * 100):02d}"
                    ] = tpr_at_fpr(gfpr, gtpr, tf)
            _, _, roc_th, _ = compute_roc_values(logits, labels)
            prec = compute_precision(logits, labels, roc_th)
            recall = tpr
            for pt in [0.99, 0.95, 0.90, 0.80]:
                if pt < prec.min() or pt > prec.max():
                    continue
                th_pt, tp_at_pt, _ = find_threshold_at_precision(
                    prec, recall, roc_th, pt
                )
                metrics[f"precision_{int(pt * 100):02d}/threshold"] = th_pt
                metrics[f"precision_{int(pt * 100):02d}/recall"] = tp_at_pt
            if auc > best_auc:
                best_auc = auc
                best_tag = tag
                tpr_at_p90 = 0.0
                for pt in [0.99, 0.95, 0.90, 0.80]:
                    th_pt = metrics.get(
                        f"precision_{int(pt * 100):02d}/threshold"
                    )
                    tp_at_pt = metrics.get(
                        f"precision_{int(pt * 100):02d}/recall"
                    )
                    if pt == 0.90 and tp_at_pt is not None:
                        tpr_at_p90 = tp_at_pt
                best_metrics = {
                    "epoch": epoch,
                    "auc": auc,
                    "tpr_at_precision_90": tpr_at_p90,
                    **metrics,
                }
            bin_line = "  ".join(
                f"{bn}:TPR@P90=---" for bn in HARDNESS if bn in per_bin
            )
            print(f"    AUC={auc:.4f}  |  {bin_line}")
        if best_tag:
            for suffix in ["predictions_", "curves_"]:
                ext = ".pt" if suffix == "predictions_" else ".npz"
                src = out_dir / f"{suffix}{best_tag}{ext}"
                dst = out_dir / f"{suffix}best{ext}"
                if src.exists():
                    shutil.copy2(src, dst)
        last_ckpt = ckpts[-1]
        last_epoch = int(last_ckpt.stem.split("epoch=")[1].split("-")[0])
        last_tag = f"epoch_{last_epoch:02d}"
        for suffix in ["predictions_", "curves_"]:
            ext = ".pt" if suffix == "predictions_" else ".npz"
            src = out_dir / f"{suffix}{last_tag}{ext}"
            dst = out_dir / f"{suffix}last{ext}"
            if src.exists():
                shutil.copy2(src, dst)
        print(f"  ✓ Best AUC={best_auc:.4f} ({best_tag})")
        return best_metrics
    
    if SWEEP_DIR is None:
        print("ERROR: No sweep directory provided.")
        print(
            "Usage: uv run python scripts/evaluate.py postprocess <sweep_dir> [run_name]"
        )
        print(
            "Example: uv run python scripts/evaluate.py postprocess checkpoints_v2/sweep12_20260506_200208"
        )
        sys.exit(1)
    
    L.seed_everything(42)
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
    _ = noise_path, drone_path
    print(f"Sweep dir: {SWEEP_DIR}")
    
    if RUN_NAME:
        run_dir = SWEEP_DIR / RUN_NAME
        if run_dir.exists():
            # Read clip_seconds from run's first checkpoint
            _cs = 2.56
            for _cp in sorted(run_dir.rglob("*.ckpt")):
                try:
                    _d = torch.load(str(_cp), map_location="cpu", weights_only=False)
                    _cs = get_clip_seconds(_d["hyper_parameters"])
                    break
                except Exception:
                    continue
            process_run(run_dir, bin_names, _cs, force_all=args.all_runs)
        else:
            print(f"Run not found: {RUN_NAME}")
    else:
        run_dirs = sorted([d for d in SWEEP_DIR.iterdir() if d.is_dir()])
        run_dirs = [
            d
            for d in run_dirs
            if (d / "lightning_logs").exists() or (d / "checkpoints").exists()
        ]
        if not run_dirs and (
            (SWEEP_DIR / "lightning_logs").exists()
            or (SWEEP_DIR / "checkpoints").exists()
        ):
            run_dirs = [SWEEP_DIR]
        print(f"Found {len(run_dirs)} runs to process")
        all_metrics = []
        skipped_count = 0
        for run_dir in run_dirs:
            if not args.all_runs and (run_dir / "eval_data" / "predictions_best.pt").exists():
                skipped_count += 1
                continue
            # Read clip_seconds from run's first checkpoint
            _cs = 2.56
            for _cp in sorted(run_dir.rglob("*.ckpt")):
                try:
                    _d = torch.load(str(_cp), map_location="cpu", weights_only=False)
                    _cs = get_clip_seconds(_d["hyper_parameters"])
                    break
                except Exception:
                    continue
            m = process_run(run_dir, bin_names, _cs, force_all=args.all_runs)
            if m:
                m["name"] = run_dir.name
                all_metrics.append(m)
        if skipped_count:
            print(f"(Skipped {skipped_count} already-postprocessed runs — use --all to force)")
        if all_metrics:
            print(f"\n{'=' * 80}")
            print("SUMMARY — Best AUC checkpoint per run")
            print(f"{'=' * 80}")
            print(f"{'name':<35} {'epoch':>5} {'AUC':>7} {'TPR@P90':>9}")
            print(f"{'-' * 80}")
            for m in sorted(
                all_metrics,
                key=lambda x: x.get("tpr_at_precision_90", 0),
                reverse=True,
            ):
                print(
                    f"{m['name']:<35} {m['epoch']:>5} {m['auc']:>7.4f} {m.get('tpr_at_precision_90', 0):>9.4f}"
                )
    print("\nDone!")
    
    
    # ====================================================================
    # calibrate — Calibrate hearability-bin estimator
    # ====================================================================
    
    
