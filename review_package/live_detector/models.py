#!/usr/bin/env python3
"""
models.py - the deep-learning model menu + MODELS registry.

Every builder has the same signature:
    build(in_ch: int, n_classes: int) -> torch.nn.Module
so train.py can swap models freely. Each accepts an arbitrary channel count
(features decide C) and outputs n_classes logits.

Implemented now: cnn, tinyai, mobilenet, efficientnet.
Stubs (need extra packages): ast, mamba - raise a clear message until wired.
"""
from __future__ import annotations


def build_cnn(in_ch: int, n_classes: int):
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True), nn.MaxPool2d(2))

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(block(in_ch, 16), block(16, 32),
                                      block(32, 64), block(64, 128),
                                      nn.AdaptiveAvgPool2d((4, 4)))
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                      nn.Linear(128 * 16, 128), nn.ReLU(inplace=True),
                                      nn.Dropout(0.3), nn.Linear(128, n_classes))

        def forward(self, x):
            return self.head(self.body(x))

    return CNN()


def build_tinyai(in_ch: int, n_classes: int):
    """Smallest practical net - max speed / lowest RAM (MCU / Hailo target)."""
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, stride=1),
                             nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.MaxPool2d(2))

    class TinyAI(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(block(in_ch, 8), block(8, 16), block(16, 24),
                                      nn.AdaptiveAvgPool2d((2, 2)))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(24 * 4, n_classes))

        def forward(self, x):
            return self.head(self.body(x))

    return TinyAI()


def _adapt_first_conv(model, in_ch):
    import torch, torch.nn as nn
    # find the first Conv2d and replace it to accept `in_ch` channels
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            new = nn.Conv2d(in_ch, m.out_channels, m.kernel_size, m.stride,
                            m.padding, bias=m.bias is not None)
            with torch.no_grad():
                w = m.weight.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1)
                new.weight.copy_(w)
            parent = model
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, new)
            break
    return model


def build_mobilenet(in_ch: int, n_classes: int):
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small
    m = mobilenet_v3_small(weights=None)
    m = _adapt_first_conv(m, in_ch)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return m


def build_efficientnet(in_ch: int, n_classes: int):
    import torch.nn as nn
    from torchvision.models import efficientnet_b0
    m = efficientnet_b0(weights=None)
    m = _adapt_first_conv(m, in_ch)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return m


def build_ast(in_ch: int, n_classes: int):
    raise NotImplementedError(
        "AST-Tiny needs `pip install timm`. Wire a timm ViT/AST backbone here "
        "(patch-embed adapted to in_ch). Stub kept for the registry.")


def build_mamba(in_ch: int, n_classes: int):
    raise NotImplementedError(
        "Mamba-2 needs `pip install mamba-ssm`. Flatten (C,F) per time step into "
        "a sequence of length T and run an SSM. Stub kept for the registry.")


MODELS = {
    "cnn":          build_cnn,
    "tinyai":       build_tinyai,
    "mobilenet":    build_mobilenet,
    "efficientnet": build_efficientnet,
    "ast":          build_ast,      # stub
    "mamba":        build_mamba,    # stub
}


def list_models() -> list[str]:
    return list(MODELS)
