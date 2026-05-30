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
import sys
from pathlib import Path

import torch
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
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_blue_red import BlueRedDetector

    model = BlueRedDetector.load_from_checkpoint(
        str(ckpt_path), map_location="cpu", weights_only=False
    )
    return model.eval()


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
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)

    print(f"Loading blue/red checkpoint: {args.ckpt}")
    model = _load_blue_red_model(args.ckpt)
    export_model = CombinedDetectorBlueRed(model).eval()

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
