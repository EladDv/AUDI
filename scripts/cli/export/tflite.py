#!/usr/bin/env python3
"""Export model to FP32 TFLite via litert-torch and evaluate on validation.

Usage:
    uv run --extra export audi-export-tflite \
        --ckpt checkpoints/my_run/checkpoints/best.ckpt \
        --noise-path data/HF_dataset_v2_background \
        --drone-path data/HF_dataset_v2_drone \
        --output-dir artifacts/tflite/my_run
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from audi.checkpoint import load_model_from_checkpoint
from audi.config import (
    MixConfig,
    parse_snr_bins,
)
from audi.training.dataset import make_dataset
from audi.training.validation import (
    compute_pr_curve,
    compute_precision,
    compute_roc_values,
    find_threshold_at_precision,
)


def _evaluate_tflite(tflite_path: str, val_specs, val_labels) -> dict:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    all_logits = []
    for i in range(len(val_specs)):
        spec = val_specs[i].astype(np.float32)
        if spec.ndim == 3:
            spec = spec[np.newaxis, ...]  # add batch dim
        interpreter.set_tensor(inp["index"], spec)
        interpreter.invoke()
        logit = interpreter.get_tensor(out["index"])
        all_logits.append(logit)

    logits = np.concatenate(all_logits).flatten()
    n = min(len(logits), len(val_labels))
    logits, labels = logits[:n], val_labels[:n]

    fpr, tpr, th, auc = compute_roc_values(logits, labels)
    prec = compute_precision(logits, labels, th)
    _, tpr_p90, _ = find_threshold_at_precision(prec, tpr, th, 0.90)
    _, _, _, ap = compute_pr_curve(logits, labels)

    return {"auc": float(auc), "ap": float(ap), "tpr_at_p90": float(tpr_p90)}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--noise-path", required=True)
    ap.add_argument("--drone-path", required=True)
    ap.add_argument("--output-dir", default="artifacts/tflite/model")
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    ckpt_path = Path(args.ckpt)
    device = args.device

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return 1

    model_name = ckpt_path.parent.parent.name
    if output_dir.name != model_name:
        output_dir = output_dir.parent / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "" if args.batch_size == 1 else f"_bs{args.batch_size}"
    fp32_path = output_dir / f"model{suffix}_fp32.tflite"

    # ── Load PyTorch model ───────────────────────────────────────────
    print(f"Loading: {ckpt_path}")
    model = load_model_from_checkpoint(ckpt_path, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    arch = model._model_cfg.arch if hasattr(model, "_model_cfg") else "?"
    print(f"  Architecture: {arch}, {n_params:,} params")
    model.eval()

    # Move to CPU for export
    model = model.cpu()

    # ── Shared validation config ─────────────────────────────────────
    snr_bins = parse_snr_bins(
        [
            "easy:-5:0:0.20",
            "medium:-10:-5:0.20",
            "hard:-15:-10:0.20",
            "extreme:-20:-15:0.15",
        ]
    )

    # ── Export FP32 TFLite ───────────────────────────────────────────
    print("\n── Exporting to TFLite ──")
    import litert_torch

    # Export backbone only — spectrogram [B, 3, n_mels, T] → logits [B, 1]
    # Mel transform (FFT) is not supported by TFLite; keep it as pre-processing.
    backbone = model.backbone
    n_mels = model._mel_transform.n_mels
    # Compute spectrogram shape: T_frames from clip_samples
    clip_samples = int(16000 * 5.12)
    n_frames = clip_samples // 160  # hop_length=160, n_fft=1024
    dummy_spec = torch.randn(args.batch_size, 3, n_mels, n_frames)

    print(
        f"  Spectrogram shape: [{args.batch_size}, 3, {n_mels}, {n_frames}]"
    )

    # fp32
    print("  Converting fp32 ...")
    edge_model_fp32 = litert_torch.convert(backbone, (dummy_spec,))
    edge_model_fp32.export(str(fp32_path))
    size_fp32 = fp32_path.stat().st_size / 1e6
    print(f"  fp32: {fp32_path} ({size_fp32:.1f} MB)")

    if args.skip_validation:
        print("\n── Validation evaluation skipped ──")
        return 0

    # ── Validation ───────────────────────────────────────────────────
    import lightning as L

    print("\n── Validation evaluation ──")
    L.seed_everything(123)
    val_ds = make_dataset(
        cfg=MixConfig(
            noise_path=Path(args.noise_path),
            drone_path=Path(args.drone_path),
            snr_bins=snr_bins,
            target_length_samples=int(16000 * 5.12),
            dataset_length=3200,
        ),
        split="validation",
        return_components=False,
    )
    # Batch access: each val_ds[i] call generates a new random mix,
    # so we must call __getitem__ once to get matched (wav, label).
    samples = [val_ds[i] for i in range(len(val_ds))]
    val_wavs = np.stack([s[0].numpy() for s in samples])
    val_labels = np.array([s[1] for s in samples])
    print(f"  {len(val_wavs)} samples")

    # PyTorch baseline (reload fresh to avoid device issues from export)
    model_pt = load_model_from_checkpoint(ckpt_path, device=device)
    print("  PyTorch ...")
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(val_wavs), 32):
            batch = torch.as_tensor(
                val_wavs[i : i + 32], dtype=torch.float32
            ).to(device)
            all_logits.append(model_pt(batch).cpu().numpy())
    logits_pt = np.concatenate(all_logits).flatten()[: len(val_labels)]
    fpr, tpr, th, auc_pt = compute_roc_values(logits_pt, val_labels)
    prec_pt = compute_precision(logits_pt, val_labels, th)
    _, tpr_p90_pt, _ = find_threshold_at_precision(prec_pt, tpr, th, 0.90)
    ap_pt = float(compute_pr_curve(logits_pt, val_labels)[3])
    print(f"  PyTorch:    AUC={auc_pt:.4f}  TPR@P90={tpr_p90_pt:.4f}")

    # TFLite evaluation (backbone: spec → logit)
    # Pre-compute spectrograms from validation wavs
    print("  Computing spectrograms for TFLite eval ...")
    val_specs = []
    with torch.no_grad():
        for i in range(0, len(val_wavs), 32):
            batch = torch.as_tensor(
                val_wavs[i : i + 32], dtype=torch.float32
            ).to(device)
            specs = model_pt._to_mel(batch)[..., :n_frames]  # [B, 3, n_mels, T]
            val_specs.append(specs.cpu().numpy())
    val_specs = np.concatenate(val_specs)

    # TFLite fp32
    print("  TFLite fp32 ...")
    print(val_specs.shape)
    print(val_labels.shape)
    m_fp32 = _evaluate_tflite(str(fp32_path), val_specs, val_labels)
    print(
        f"  TFLite fp32: AUC={m_fp32['auc']:.4f}  TPR@P90={m_fp32['tpr_at_p90']:.4f}"
    )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Model:   {model_name} ({arch}, {n_params:,} params)")
    print(f"  fp32:    {size_fp32:.1f} MB")
    print(f"\n  {'':<15} {'AUC':>7} {'AP':>7} {'TPR@P90':>9}")
    print(f"  {'PyTorch':<15} {auc_pt:>7.4f} {ap_pt:>7.4f} {tpr_p90_pt:>9.4f}")
    print(
        f"  {'TFLite fp32':<15} {m_fp32['auc']:>7.4f} "
        f"{m_fp32['ap']:>7.4f} {m_fp32['tpr_at_p90']:>9.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
