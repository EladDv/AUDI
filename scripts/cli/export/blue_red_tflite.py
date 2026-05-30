#!/usr/bin/env python3
"""Export the combined detector + blue/red head to TFLite.

The Pi app computes the same mel spectrogram used by the detector TFLite model.
This exporter therefore writes a spectrogram-in, three-logit-out model:

    [B, 3, 128, 512] -> [B, 3]

Output order is [detector_logit, blue_logit, red_logit]. RED is the positive
class for the color head.
"""

from __future__ import annotations

import argparse
import tempfile
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class CombinedDetectorBlueRed(nn.Module):
    """Spec-to-detector/color wrapper around a trained BlueRedDetector."""

    def __init__(self, model: nn.Module):
        super().__init__()
        if getattr(model, "detector", None) is None:
            raise ValueError(
                "Blue/red checkpoint must be trained with --detector-checkpoint"
            )
        self.backbone = model.detector.backbone.backbone
        self.det_head = model.detector.backbone.classifier
        self.shared_fc = model.shared_fc
        self.cls_head = model.cls_head

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        _, features = self.backbone(spec[:, :1])
        det_logit = self.det_head(features)
        cls_logits = self.cls_head(self.shared_fc(features))
        return torch.cat([det_logit, cls_logits], dim=1)


def _load_blue_red_model(ckpt_path: Path) -> nn.Module:
    from scripts.cli.train_blue_red import BlueRedDetector

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", {})
    teacher_keys = [key for key in state if key.startswith("teacher.")]
    if teacher_keys:
        for key in teacher_keys:
            del state[key]
        with tempfile.NamedTemporaryFile(suffix=".ckpt") as tmp:
            torch.save(ckpt, tmp.name)
            model = BlueRedDetector.load_from_checkpoint(
                tmp.name, map_location="cpu", weights_only=False
            )
    else:
        model = BlueRedDetector.load_from_checkpoint(
            str(ckpt_path), map_location="cpu", weights_only=False
        )
    return model.eval()


def _dynamic_conv_export_forward(self, x, g=None):
    """Export-friendly DynamicConv forward without broadcasted batch matmul."""
    b, c, f, t = x.size()
    if b != 1:
        raise ValueError("DyMN export patch currently supports batch size 1")

    g_c = g[0].view(b, -1)
    residuals = self.residuals(g_c).view(b, self.att_groups, 1, -1)
    attention = F.softmax(residuals / self.temperature, dim=-1)

    # Original: (attention @ self.weight).transpose(1, 2)
    # Export patch: equivalent weighted sum over K kernels, avoiding
    # tfl.batch_matmul lowering issues in DyMN at batch size 1.
    aggregate_weight = (attention.transpose(2, 3) * self.weight).sum(dim=2)
    aggregate_weight = aggregate_weight.reshape(
        b,
        self.out_channels,
        self.in_channels // self.groups,
        self.kernel_size,
        self.kernel_size,
    )
    aggregate_weight = aggregate_weight.view(
        b * self.out_channels,
        self.in_channels // self.groups,
        self.kernel_size,
        self.kernel_size,
    )

    x = x.view(1, -1, f, t)
    if self.bias is not None:
        aggregate_bias = (attention.squeeze(2) @ self.bias).view(-1)
    else:
        aggregate_bias = None
    output = F.conv2d(
        x,
        weight=aggregate_weight,
        bias=aggregate_bias,
        stride=self.stride,
        padding=self.padding,
        dilation=self.dilation,
        groups=self.groups * b,
    )
    return output.view(b, self.out_channels, output.size(-2), output.size(-1))


def _patch_dymn_dynamic_conv_for_export(model: nn.Module) -> int:
    """Patch DyMN DynamicConv modules to avoid LiteRT batch-matmul failure."""
    try:
        from models.dymn.dy_block import DynamicConv
    except Exception:
        return 0

    patched = 0
    for module in model.modules():
        if isinstance(module, DynamicConv):
            module.forward = types.MethodType(_dynamic_conv_export_forward, module)
            patched += 1
    return patched


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export a combined detector + blue/red TFLite spec classifier"
    )
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument(
        "--output",
        default=Path("audi-app/models/model.tflite"),
        type=Path,
    )
    ap.add_argument("--n-mels", default=128, type=int)
    ap.add_argument("--n-frames", default=512, type=int)
    ap.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="Fixed export batch size. DyMN LiteRT export may require 16.",
    )
    ap.add_argument(
        "--patch-dymn-bs1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Patch DyMN DynamicConv to export true batch size 1.",
    )
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)

    print(f"Loading blue/red checkpoint: {args.ckpt}")
    model = _load_blue_red_model(args.ckpt)
    export_model = CombinedDetectorBlueRed(model).eval()
    if args.patch_dymn_bs1:
        patched = _patch_dymn_dynamic_conv_for_export(export_model)
        if patched:
            print(f"Patched {patched} DyMN DynamicConv modules for export")

    dummy_spec = torch.randn(args.batch_size, 3, args.n_mels, args.n_frames)
    with torch.no_grad():
        out = export_model(dummy_spec)
    if tuple(out.shape) != (args.batch_size, 3):
        raise RuntimeError(
            f"Expected [{args.batch_size}, 3] output, got {tuple(out.shape)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting: {args.output}")
    import litert_torch

    edge_model = litert_torch.convert(export_model, (dummy_spec,))
    edge_model.export(str(args.output))
    size_mb = args.output.stat().st_size / 1e6
    print(
        f"Done: {args.output} ({size_mb:.1f} MB), "
        "output order [detector, blue, red]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
