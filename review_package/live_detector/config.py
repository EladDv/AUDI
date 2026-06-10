#!/usr/bin/env python3
"""Shared config, paths, and label rules for the training pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HERE            = Path(__file__).resolve().parent
PROJECT         = HERE.parent.parent
DATA_ROOTS      = [PROJECT / "06_model_v1" / "data" / "raw", PROJECT / "data_other"]
CLIPS           = HERE / "clips.npz"
FEATURES_DIR    = HERE / "features"
EXPERIMENTS_DIR = HERE / "experiments"


@dataclass(frozen=True)
class Config:
    sr: int = 16000
    clip_s: float = 2.0
    hop_s: float = 1.0          # 50% overlap between consecutive clips
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 250       # 16000*2 / 250 = 128 frames -> T
    F: int = 256                # frequency bins in every feature
    T: int = 128                # time bins in every feature
    fmin: float = 150.0         # keep the low blade-pass fundamental
    fmax: float = 8000.0
    seed: int = 1337


CFG = Config()

# --------------------------------------------------------------------------- #
# Label rules run on the path *relative to the data root* (lowercased, forward
# slashes). Using the relative path is what fixes the old bug where the project
# folder name "drone_nir" matched the "drone" rule. NEGATIVES come first so that
# e.g. "no_drone" is never caught by a later "drone" rule.
# Return None to skip a file.
# --------------------------------------------------------------------------- #
LABEL_RULES: list[tuple[str, str]] = [
    ("false_alarm",             "false_alarm"),
    ("no_drone",                "background"),
    ("urban_background",        "background"),
    ("attack_runs_backgrounds", "background"),
    ("my_room",                 "background"),
    ("/bg/",                    "background"),
    ("background",              "background"),
    ("field_recordings",        "target_drone"),  # external drone alerts (eval-only)
    ("/alerts/",                "target_drone"),
    ("other_drone",             "other_drone"),
    ("dataset_v2",              "other_drone"),
    ("target_drone",            "target_drone"),
    ("live_session",            "target_drone"),
    ("drone_with_voices",       "target_drone"),
    ("/evo",                    "target_drone"),
    ("/fpv",                    "target_drone"),
    ("/drone",                  "target_drone"),
]

CLASSES = ["background", "false_alarm", "other_drone", "target_drone"]

# Sources used ONLY for testing - never train/val/fine-tune on these.
# They are forced into the test split and tagged domain 'ext_test'.
EVAL_ONLY_SUBSTRINGS = ["field_recordings_20260514"]

# domain codes stored in clips.npz / features
DOMAIN = {"mono": 0, "array": 1, "ext_test": 2}


def _rel(path_abs: str) -> str:
    p = path_abs.replace("\\", "/").lower()
    for root in DATA_ROOTS:
        r = str(root).replace("\\", "/").lower()
        if p.startswith(r):
            return p[len(r):]
    return p


def infer_label(path_abs: str) -> str | None:
    rel = _rel(path_abs)
    for sub, cls in LABEL_RULES:
        if sub in rel:
            return cls
    return None


def is_eval_only(path_abs: str) -> bool:
    rel = _rel(path_abs)
    return any(s in rel for s in EVAL_ONLY_SUBSTRINGS)


def infer_domain(path_abs: str) -> str:
    if is_eval_only(path_abs):
        return "ext_test"
    return "array" if "live_session" in _rel(path_abs) else "mono"


def infer_subtype(path_abs: str) -> str:
    """Fine drone subtype for later EVO-vs-FPV analysis ('' if unknown)."""
    rel = _rel(path_abs)
    if "fpv" in rel:
        return "fpv"
    if "evo" in rel:
        return "evo"
    return ""
