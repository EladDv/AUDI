#!/usr/bin/env python3
"""Export model to TFLite (fp32 + int8) via litert-torch, evaluate on validation.

Usage:
    uv run python scripts/export_tflite.py \
        --ckpt checkpoints/multinoise_20260510_223154/07_convnext_small_spec/checkpoints/epoch=22-step=5750.ckpt \
        --noise-path data/HF_dataset_v2_background \
        --drone-path data/HF_dataset_v2_drone \
        --output-dir artifacts/tflite/07_convnext_small_spec
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from audi.checkpoint import load_model_from_checkpoint
from audi.config import (  # noqa: E402
    MixConfig,
    parse_snr_bins,
)
from audi.training.dataset import make_dataset  # noqa: E402
from audi.training.detector import DroneDetector  # noqa: E402
from audi.training.validation import (  # noqa: E402
    compute_pr_curve,
    compute_precision,
    compute_roc_values,
    find_threshold_at_precision,
)


def _load_model(ckpt_path: str, device: str) -> DroneDetector:
    """Thin wrapper around shared checkpoint loader for backward compat."""
    return load_model_from_checkpoint(ckpt_path, device=device)


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
    ap.add_argument("--calib-samples", type=int, default=200)
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

    fp32_path = output_dir / "model_fp32.tflite"
    int8_path = output_dir / "model_int8.tflite"

    # ── Load PyTorch model ───────────────────────────────────────────
    print(f"Loading: {ckpt_path}")
    model = _load_model(str(ckpt_path), device)
    n_params = sum(p.numel() for p in model.parameters())
    arch = model._model_cfg.arch if hasattr(model, "_model_cfg") else "?"
    print(f"  Architecture: {arch}, {n_params:,} params")
    model.eval()

    # Move to CPU for export
    model = model.cpu()

    # ── Calibration data ─────────────────────────────────────────────
    import lightning as L

    L.seed_everything(42)
    snr_bins = parse_snr_bins(
        [
            "easy:-5:0:0.20",
            "medium:-10:-5:0.20",
            "hard:-15:-10:0.20",
            "extreme:-20:-15:0.15",
        ]
    )
    mix_cfg_cal = MixConfig(
        noise_path=args.noise_path,
        drone_path=args.drone_path,
        snr_bins=snr_bins,
        target_length_samples=int(16000 * 5.12),
        dataset_length=args.calib_samples,
    )
    calib_ds = make_dataset(
        cfg=mix_cfg_cal, split="validation", return_components=False
    )
    calib_wavs = [
        calib_ds[i][0].numpy().astype(np.float32)
        for i in range(min(len(calib_ds), args.calib_samples))
    ]
    print(f"  Calibration: {len(calib_wavs)} clips")

    # ── Export fp32 and int8 TFLite ──────────────────────────────────
    print("\n── Exporting to TFLite ──")
    import litert_torch

    # Export backbone only — spectrogram [B, 3, n_mels, T] → logits [B, 1]
    # Mel transform (FFT) is not supported by TFLite; keep it as pre-processing.
    backbone = model.backbone
    n_mels = model._mel_transform.n_mels
    # Compute spectrogram shape: T_frames from clip_samples
    clip_samples = int(16000 * 5.12)
    n_frames = clip_samples // 160  # hop_length=160, n_fft=1024
    dummy_spec = torch.randn(1, 3, n_mels, n_frames)

    print(f"  Spectrogram shape: [B, 3, {n_mels}, {n_frames}]")

    # fp32
    print("  Converting fp32 ...")
    edge_model_fp32 = litert_torch.convert(backbone, (dummy_spec,))
    edge_model_fp32.export(str(fp32_path))
    size_fp32 = fp32_path.stat().st_size / 1e6
    print(f"  fp32: {fp32_path} ({size_fp32:.1f} MB)")

    # int8 with calibration — need spectrogram versions of calibration wavs
    print("  Converting int8 ...")
    n_cal = min(len(calib_wavs), 50)
    calib_specs = []
    with torch.no_grad():
        for i in range(n_cal):
            wav_t = torch.as_tensor(calib_wavs[i]).unsqueeze(0)
            spec = model._to_mel(wav_t)[..., :n_frames]  # [1, 3, n_mels, T]
            calib_specs.append(spec.cpu().numpy().astype(np.float32))
    _calib_batches = [{"spec": s} for s in calib_specs]

    from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer
    from litert_torch.quantize.quant_config import QuantConfig

    quantizer = PT2EQuantizer()
    # Use first supported static config (per-tensor symmetric weight, int8 activation)
    static_configs = [
        c
        for c in quantizer.get_supported_quantization_configs()
        if not c.is_dynamic and not c.is_qat
    ]
    print(static_configs)
    quantizer.set_global(static_configs[0])
    quant_cfg = QuantConfig(pt2e_quantizer=quantizer)

    edge_model_int8 = litert_torch.convert(
        backbone,
        (dummy_spec,),
        quant_config=quant_cfg,
    )
    edge_model_int8.export(str(int8_path))
    size_int8 = int8_path.stat().st_size / 1e6
    print(f"  int8: {int8_path} ({size_int8:.1f} MB)")

    # ── Validation ───────────────────────────────────────────────────
    print("\n── Validation evaluation ──")
    L.seed_everything(123)
    val_ds = make_dataset(
        cfg=MixConfig(
            noise_path=args.noise_path,
            drone_path=args.drone_path,
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
    model_pt = _load_model(str(ckpt_path), device)
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

    # TFLite int8
    print("  TFLite int8 ...")
    m_int8 = _evaluate_tflite(str(int8_path), val_specs, val_labels)
    print(
        f"  TFLite int8: AUC={m_int8['auc']:.4f}  TPR@P90={m_int8['tpr_at_p90']:.4f}"
    )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Model:   {model_name} ({arch}, {n_params:,} params)")
    print(f"  fp32:    {size_fp32:.1f} MB")
    print(
        f"  int8:    {size_int8:.1f} MB ({size_fp32 / size_int8:.1f}x smaller)"
    )
    print(f"\n  {'':<15} {'AUC':>7} {'AP':>7} {'TPR@P90':>9}")
    print(f"  {'PyTorch':<15} {auc_pt:>7.4f} {ap_pt:>7.4f} {tpr_p90_pt:>9.4f}")
    print(
        f"  {'TFLite fp32':<15} {m_fp32['auc']:>7.4f} {m_fp32['ap']:>7.4f} {m_fp32['tpr_at_p90']:>9.4f}"
    )
    print(
        f"  {'TFLite int8':<15} {m_int8['auc']:>7.4f} {m_int8['ap']:>7.4f} {m_int8['tpr_at_p90']:>9.4f}"
    )
    print(
        f"  {'int8 vs PT':<15} {m_int8['auc'] - auc_pt:>+7.4f} {m_int8['ap'] - ap_pt:>+7.4f} {m_int8['tpr_at_p90'] - tpr_p90_pt:>+9.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
