"""cmd_postprocess eval subcommand."""
from __future__ import annotations

import sys
from pathlib import Path

from audi.checkpoint import get_clip_seconds, strip_compile_prefix


def run(noise_path: str | None, drone_path: str | None) -> None:
    import argparse
    import csv
    import json
    import shutil

    import lightning as L
    import numpy as np
    import torch
    import torchaudio
    from datasets import load_from_disk
    from torch.utils.data import DataLoader

    from audi.cli_utils import NUM_WORKERS
    from audi.config import (
        MelConfig,
        ModelConfig,
        OptimizerConfig,
    )
    from audi.evaluation.deployment import (
        deployment_score,
        detection_precision_curve_points,
        find_threshold_at_min_precision,
        mix_config_from_run,
    )
    from audi.evaluation.field_mix import COLOR_NAMES, SOURCE_NAMES, FieldMixDataset
    from audi.hysteresis import apply_hysteresis
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
    ap.add_argument(
        "--selection-metric",
        choices=["deployment", "auc"],
        default="deployment",
        help="Checkpoint selector for predictions_best.pt",
    )
    ap.add_argument(
        "--skip-deployment-validation",
        action="store_true",
        help="Only compute classic synthetic validation metrics",
    )
    ap.add_argument("--field-bg-path", type=Path, default=Path("data/HF_dataset_v7_background"))
    ap.add_argument("--blue-red-drone-path", type=Path, default=Path("data/hf_blue_red"))
    ap.add_argument(
        "--field-hard-negative-path",
        type=Path,
        default=Path("data/field_recordings_20260514/mined_hard_negatives/hf_dataset"),
    )
    ap.add_argument("--field-mix-samples-per-color-bin", type=int, default=24)
    ap.add_argument("--field-mix-background-negatives", type=int, default=160)
    ap.add_argument("--field-mix-hard-negatives", type=int, default=160)
    ap.add_argument("--field-mix-seed", type=int, default=42)
    ap.add_argument("--attack-validation-dir", type=Path, default=Path("data/attack_runs"))
    ap.add_argument(
        "--deployment-precision-target",
        type=float,
        default=0.80,
        help="Validation precision target used to choose the deployment threshold",
    )
    ap.add_argument(
        "--deployment-precision-targets",
        default="0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95,0.99",
        help="Comma-separated precision targets to save as detection curve points",
    )
    ap.add_argument("sweep_dir", type=Path, nargs="?")
    ap.add_argument("run_name", nargs="?")
    args = ap.parse_args()
    selection_metric = args.selection_metric
    if args.skip_deployment_validation and selection_metric == "deployment":
        selection_metric = "auc"

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu"
    )
    FPR_TARGETS = [float(x) for x in args.fpr_targets.split(",")]
    BATCH_SIZE = args.batch_size
    DEPLOYMENT_PRECISION_TARGETS = [
        float(x) for x in str(args.deployment_precision_targets).split(",") if x
    ]

    SWEEP_DIR = args.sweep_dir
    RUN_NAME = args.run_name
    fallback_noise_path = Path(noise_path) if noise_path else None
    fallback_drone_path = Path(drone_path) if drone_path else None
    field_mix_cache: dict[tuple[int, tuple[str, ...]], DataLoader] = {}
    attack_audio_cache: dict[str, np.ndarray] | None = None
    
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

    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -10.0, 10.0)))

    def _count_alerts(dets: np.ndarray) -> int:
        if len(dets) == 0:
            return 0
        padded = np.pad(dets.astype(np.int8), (1, 0), constant_values=0)
        return int(np.sum((padded[1:] == 1) & (padded[:-1] == 0)))

    def _split_into_windows(audio: np.ndarray, sr: int, clip_s: float) -> list[np.ndarray]:
        win = int(sr * clip_s)
        step = max(1, int(win * 0.125))
        if len(audio) < win:
            return []
        return [audio[i : i + win] for i in range(0, len(audio) - win + 1, step)]

    def _split_by_zero_gaps(
        audio: np.ndarray, sr: int, min_dur: float = 3.0, min_gap_s: float = 0.5
    ) -> list[np.ndarray]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        exact_zero = audio == 0.0
        zero_runs = []
        in_zero, start = False, 0
        for i in range(len(exact_zero) + 1):
            z = bool(exact_zero[i]) if i < len(exact_zero) else False
            if z and not in_zero:
                start, in_zero = i, True
            elif not z and in_zero:
                if (i - start) / sr >= min_gap_s:
                    zero_runs.append((start, i))
                in_zero = False
        if not zero_runs:
            return [audio] if len(audio) / sr >= min_dur else []
        segments, prev = [], 0
        for zs, ze in zero_runs:
            if (zs - prev) / sr >= min_dur:
                segments.append(audio[prev:zs].copy())
            prev = ze
        if (len(audio) - prev) / sr >= min_dur:
            segments.append(audio[prev:].copy())
        return segments

    @torch.no_grad()
    def _predict_windows(model: DroneDetector, windows: list[np.ndarray]) -> np.ndarray:
        if not windows:
            return np.array([])
        scores = []
        for i in range(0, len(windows), BATCH_SIZE):
            batch = torch.as_tensor(
                np.stack(windows[i : i + BATCH_SIZE]), dtype=torch.float32
            ).to(device)
            logits = model(batch).cpu().numpy().reshape(-1)
            scores.append(_sigmoid(logits))
        return np.concatenate(scores) if scores else np.array([])

    def _load_attack_audio() -> dict[str, np.ndarray]:
        nonlocal attack_audio_cache
        if attack_audio_cache is not None:
            return attack_audio_cache
        attack_audio_cache = {}
        if not args.attack_validation_dir.exists():
            return attack_audio_cache
        for fp in sorted(args.attack_validation_dir.glob("*.wav")):
            audio, _sr = torchaudio.load(str(fp))
            attack_audio_cache[fp.name] = audio.mean(dim=0).numpy().astype(np.float32)
        return attack_audio_cache

    def _field_mix_loader(clip_samples: int, snr_bins) -> DataLoader | None:
        cache_key = (clip_samples, tuple(b.name for b in snr_bins))
        if cache_key in field_mix_cache:
            return field_mix_cache[cache_key]
        required = [args.field_bg_path, args.blue_red_drone_path]
        if any(not p.exists() for p in required):
            return None
        background = load_from_disk(str(args.field_bg_path))["validation"]
        drones = load_from_disk(str(args.blue_red_drone_path))["validation"]
        hard_ds = None
        if args.field_hard_negative_path.exists():
            hard_dd = load_from_disk(str(args.field_hard_negative_path))
            hard_ds = hard_dd["validation"] if "validation" in hard_dd else None
        ds = FieldMixDataset(
            background_ds=background,
            drone_ds=drones,
            hard_negative_ds=hard_ds,
            snr_bins=list(snr_bins),
            target_length_samples=clip_samples,
            samples_per_color_bin=args.field_mix_samples_per_color_bin,
            background_negatives=args.field_mix_background_negatives,
            hard_negatives=args.field_mix_hard_negatives,
            seed=args.field_mix_seed,
        )
        dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
        field_mix_cache[cache_key] = dl
        return dl

    def _evaluate_field_mix(
        model: DroneDetector, clip_samples: int, snr_bins
    ) -> dict[str, float]:
        dl = _field_mix_loader(clip_samples, snr_bins)
        if dl is None:
            return {}
        logits_all, labels_all, sources_all, colors_all, bins_all = [], [], [], [], []
        for batch in dl:
            wav, label, source_idx, color_idx, _snr_db, bin_idx = batch
            with torch.no_grad():
                logits = model(wav.to(device)).cpu().numpy().reshape(-1)
            logits_all.append(logits)
            labels_all.append(label.numpy().reshape(-1))
            sources_all.append(source_idx.numpy().reshape(-1))
            colors_all.append(color_idx.numpy().reshape(-1))
            bins_all.append(bin_idx.numpy().reshape(-1))
        logits = np.concatenate(logits_all)
        labels = np.concatenate(labels_all)
        sources = np.concatenate(sources_all)
        colors = np.concatenate(colors_all)
        bin_idx = np.concatenate(bins_all)
        _fpr, tpr, th, auc = compute_roc_values(logits, labels)
        precision = compute_precision(logits, labels, th)
        sigma, actual_precision, recall_at_precision = find_threshold_at_min_precision(
            precision, tpr, th, args.deployment_precision_target
        )
        probs = _sigmoid(logits)
        pred = probs >= sigma
        pos = labels > 0.5

        def _recall(mask: np.ndarray) -> float:
            denom = int((pos & mask).sum())
            if denom == 0:
                return 0.0
            return float((pred & pos & mask).sum() / denom)

        def _fp_rate(mask: np.ndarray) -> float:
            denom = int(((~pos) & mask).sum())
            if denom == 0:
                return 0.0
            return float((pred & (~pos) & mask).sum() / denom)

        source_names = np.array(SOURCE_NAMES, dtype=object)
        color_names = np.array(COLOR_NAMES, dtype=object)
        source_labels = source_names[sources]
        color_labels = color_names[colors]
        metrics = {
            "field_mix_auc": float(auc),
            "field_mix_sigma": float(sigma),
            "field_mix_target_precision": float(args.deployment_precision_target),
            "field_mix_precision": float(actual_precision),
            "field_mix_tpr": float(recall_at_precision),
            "field_mix_fnr": float(1.0 - recall_at_precision),
            "field_mix_recall": float(recall_at_precision),
            "field_mix_blue_recall": _recall(color_labels == "blue"),
            "field_mix_red_recall": _recall(color_labels == "red"),
            "field_mix_background_fp_rate": _fp_rate(source_labels == "field_background"),
            "field_mix_hard_fp_rate": _fp_rate(source_labels == "field_hard_negative"),
        }
        for point in detection_precision_curve_points(
            logits, labels, precision_targets=DEPLOYMENT_PRECISION_TARGETS
        ):
            suffix = f"{int(round(point['target_precision'] * 100)):02d}"
            metrics[f"field_mix_threshold_at_precision_{suffix}"] = point["threshold"]
            metrics[f"field_mix_tpr_at_precision_{suffix}"] = point["tpr"]
            metrics[f"field_mix_fnr_at_precision_{suffix}"] = point["fnr"]
            metrics[f"field_mix_actual_precision_at_precision_{suffix}"] = point[
                "precision"
            ]
        for i, snr_bin in enumerate(snr_bins):
            metrics[f"field_mix_recall_{snr_bin.name}"] = _recall(bin_idx == i)
        return metrics

    def _evaluate_attack_validation(
        model: DroneDetector, clip_seconds: float, sigma: float
    ) -> dict[str, float]:
        audio_waveforms = _load_attack_audio()
        if not audio_waveforms:
            return {}
        segment_coverages, segment_first_pct = [], []
        bg_windows: list[np.ndarray] = []
        for name, audio in audio_waveforms.items():
            if name.startswith("background"):
                bg_windows.extend(_split_into_windows(audio, SR, clip_seconds))
                continue
            for seg in _split_by_zero_gaps(audio, SR):
                scores = _predict_windows(model, _split_into_windows(seg, SR, clip_seconds))
                if len(scores) == 0:
                    segment_coverages.append(0.0)
                    segment_first_pct.append(100.0)
                    continue
                dets = apply_hysteresis(scores, sigma)
                coverage = 100.0 * dets.sum() / len(dets)
                det_idx = np.where(dets)[0]
                first_pct = 100.0 * det_idx[0] / len(dets) if len(det_idx) else 100.0
                segment_coverages.append(float(coverage))
                segment_first_pct.append(float(first_pct))
        bg_scores = _predict_windows(model, bg_windows)
        bg_dets = apply_hysteresis(bg_scores, sigma)
        return {
            "attack_cov_pct": float(np.mean(segment_coverages)) if segment_coverages else 0.0,
            "attack_first_pct": float(np.median(segment_first_pct)) if segment_first_pct else 100.0,
            "attack_bg_windows": float(bg_dets.sum()),
            "attack_bg_alerts": float(_count_alerts(bg_dets)),
        }

    def _evaluate_deployment(
        model: DroneDetector,
        clip_samples: int,
        clip_seconds: float,
        snr_bins,
        *,
        classic_auc: float,
    ) -> dict[str, float]:
        field_metrics = _evaluate_field_mix(model, clip_samples, snr_bins)
        if not field_metrics:
            return {}
        attack_metrics = _evaluate_attack_validation(
            model, clip_seconds, field_metrics["field_mix_sigma"]
        )
        if not attack_metrics:
            return field_metrics
        score = deployment_score(
            classic_auc=classic_auc,
            field_mix_auc=field_metrics["field_mix_auc"],
            field_mix_recall=field_metrics["field_mix_recall"],
            field_mix_red_recall=field_metrics["field_mix_red_recall"],
            field_mix_blue_recall=field_metrics["field_mix_blue_recall"],
            field_mix_hard_fp_rate=field_metrics["field_mix_hard_fp_rate"],
            attack_coverage_pct=attack_metrics["attack_cov_pct"],
            attack_first_pct=attack_metrics["attack_first_pct"],
            attack_bg_alerts=int(attack_metrics["attack_bg_alerts"]),
        )
        return {**field_metrics, **attack_metrics, "deployment_score": score}
    
    def process_run(run_dir: Path, clip_seconds: float, force_all: bool = False) -> dict:
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
        mix_cfg, bin_names = mix_config_from_run(
            run_dir,
            fallback_noise_path=fallback_noise_path,
            fallback_drone_path=fallback_drone_path,
            clip_seconds=clip_seconds,
        )
        snr_bins = mix_cfg.snr_bins
        val_ds = make_dataset(
            cfg=mix_cfg, split="validation", return_components=True
        )
        val_dl = DataLoader(
            val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
        )
        best_auc = -1
        best_selector = -1e18
        best_tag = None
        best_metrics = {}
        deployment_rows = []
        for ckpt_path in ckpts:
            epoch = int(ckpt_path.stem.split("epoch=")[1].split("-")[0])
            tag = f"epoch_{epoch:02d}"
            deployment_json = out_dir / f"deployment_metrics_{tag}.json"
            if (
                not force_all
                and (out_dir / f"predictions_{tag}.pt").exists()
                and (
                    args.skip_deployment_validation
                    or args.selection_metric == "auc"
                    or deployment_json.exists()
                )
            ):
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
            for bn in bin_names:
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
            deployment_metrics = {}
            if not args.skip_deployment_validation:
                deployment_metrics = _evaluate_deployment(
                    model,
                    clip_samples,
                    clip_seconds,
                    snr_bins,
                    classic_auc=auc,
                )
                if deployment_metrics:
                    metrics.update(
                        {f"deployment/{k}": v for k, v in deployment_metrics.items()}
                    )
                    deployment_json.write_text(
                        json.dumps(deployment_metrics, indent=2, sort_keys=True)
                    )
                    deployment_rows.append(
                        {
                            "run": name,
                            "tag": tag,
                            "epoch": epoch,
                            **{
                                k: round(float(v), 6)
                                for k, v in deployment_metrics.items()
                            },
                        }
                    )
            selector = (
                deployment_metrics.get("deployment_score", auc)
                if selection_metric == "deployment"
                else auc
            )
            if auc > best_auc:
                best_auc = auc
            if selector > best_selector:
                best_selector = float(selector)
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
                    "selector": best_selector,
                    "selection_metric": selection_metric,
                    "tpr_at_precision_90": tpr_at_p90,
                    **metrics,
                }
            bin_line = "  ".join(
                f"{bn}:n={len(per_bin[bn][0])}" for bn in bin_names if bn in per_bin
            )
            deploy_line = ""
            if deployment_metrics:
                deploy_line = (
                    f"  DEP={deployment_metrics.get('deployment_score', 0.0):.2f}"
                    f"  atk={deployment_metrics.get('attack_cov_pct', 0.0):.1f}%"
                    f"  detP={deployment_metrics.get('field_mix_precision', 0.0):.3f}"
                    f"  detTPR={deployment_metrics.get('field_mix_tpr', 0.0):.3f}"
                    f"  red={deployment_metrics.get('field_mix_red_recall', 0.0):.3f}"
                    f"  blue={deployment_metrics.get('field_mix_blue_recall', 0.0):.3f}"
                    f"  hardFP={deployment_metrics.get('field_mix_hard_fp_rate', 0.0):.3f}"
                )
            print(f"    AUC={auc:.4f}{deploy_line}  |  {bin_line}")
        if deployment_rows:
            dep_csv = out_dir / "deployment_metrics.csv"
            fieldnames = sorted({k for row in deployment_rows for k in row})
            with open(dep_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(deployment_rows)
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
        print(
            f"  ✓ Best AUC={best_auc:.4f}; "
            f"best {selection_metric}={best_selector:.4f} ({best_tag})"
        )
        return best_metrics
    
    if SWEEP_DIR is None:
        print("ERROR: No sweep directory provided.")
        print(
            "Usage: uv run audi-eval postprocess <sweep_dir> [run_name]"
        )
        print(
            "Example: uv run audi-eval postprocess "
            "checkpoints_v2/sweep12_20260506_200208"
        )
        sys.exit(1)
    
    L.seed_everything(42)
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
            process_run(run_dir, _cs, force_all=args.all_runs)
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
            m = process_run(run_dir, _cs, force_all=args.all_runs)
            if m:
                m["name"] = run_dir.name
                all_metrics.append(m)
        if skipped_count:
            print(f"(Skipped {skipped_count} already-postprocessed runs — use --all to force)")
        if all_metrics:
            print(f"\n{'=' * 80}")
            print(f"SUMMARY — Best {selection_metric} checkpoint per run")
            print(f"{'=' * 80}")
            print(
                f"{'name':<35} {'epoch':>5} {'AUC':>7} "
                f"{'Prec':>7} {'TPR':>7} {'selector':>10}"
            )
            print(f"{'-' * 80}")
            for m in sorted(
                all_metrics,
                key=lambda x: x.get("selector", x.get("tpr_at_precision_90", 0)),
                reverse=True,
            ):
                print(
                    f"{m['name']:<35} {m['epoch']:>5} {m['auc']:>7.4f} "
                    f"{m.get('deployment/field_mix_precision', 0):>7.3f} "
                    f"{m.get('deployment/field_mix_tpr', 0):>7.3f} "
                    f"{m.get('selector', 0):>10.3f}"
                )
            csv_path = SWEEP_DIR / "postprocess_deployment_summary.csv"
            fieldnames = sorted({k for row in all_metrics for k in row})
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_metrics)
            print(f"\nDeployment summary: {csv_path}")
    print("\nDone!")
    
    
    # ====================================================================
    # calibrate — Calibrate hearability-bin estimator
    # ====================================================================
    
    
