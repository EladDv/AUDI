#!/usr/bin/env python3
"""Shared config, paths, and label rules for the training pipeline.

Training reads the CONSOLIDATED, human-readable corpus at
06_model_v1/data/corpus/ (built by hand from every source, split into
mine/ vs external/).  This is the single source of truth so what we train on
== what is documented in corpus/DATA_README.md and shareable.

field_recordings_20260514 is added as a SECOND root but is forced to be
TEST-ONLY (held out, never trained) - it is our external detection check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HERE            = Path(__file__).resolve().parent
PROJECT         = HERE.parent.parent
CORPUS          = PROJECT / "06_model_v1" / "data" / "corpus"
FIELD_HOLDOUT   = PROJECT / "data_other" / "field_recordings_20260514"
DATA_ROOTS      = [CORPUS, FIELD_HOLDOUT]
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
# Folders we must never train OR test on (checked on the absolute path).
# The "dont check" session is your explicit do-not-touch FPV folder; the rest
# are duplicates / scratch dirs that could leak in if roots ever change.
# --------------------------------------------------------------------------- #
# NOTE: keep these specific - substrings like "recordings_202605" would also
# match "field_recordings_20260514" and wrongly drop the held-out test set.
EXCLUDE_SUBSTRINGS = ["dont check", "dont_check"]

# --------------------------------------------------------------------------- #
# Label rules run on the path *relative to the data root* (lowercased, forward
# slashes). NEGATIVES come first so e.g. "background" is never caught by a later
# "drone" rule. Maps every corpus folder -> one of CLASSES. Return None = skip.
#
#   target_drone : drones we operate on  (your EVO + your FPV-10" + untyped real
#                  drone + external field alerts)  -> Stage-1 positives
#   other_drone  : others' size-labelled drones (5/7/10/13")  -> still "a drone"
#                  for Stage-1, and the EVO-vs-FPV probe set for Stage-2
#   background   : no-drone / room / urban / attack-run backgrounds
#   false_alarm  : hard negatives
# --------------------------------------------------------------------------- #
LABEL_RULES: list[tuple[str, str]] = [
    ("false_alarm",            "false_alarm"),
    ("background",             "background"),   # incl. attack_runs_backgrounds, my_room, urban, BG, no_drone
    ("alerts",                 "target_drone"),  # field_recordings (eval-only)
    ("attack",                 "target_drone"),  # attack_runs_untyped (untyped real drone)
    ("red13",                  "other_drone"),
    ("blue7",                  "other_drone"),
    ("fpv_by_size",            "other_drone"),
    ("fpv",                    "target_drone"),  # your fpv_10in_array
    ("evo",                    "target_drone"),  # your evo_array / evo_mono
    ("drone_untyped",          "target_drone"),
    ("array_untyped",          "target_drone"),
    ("drone",                  "target_drone"),
]

CLASSES = ["background", "false_alarm", "other_drone", "target_drone"]

# Sources used ONLY for testing - never train/val/fine-tune on these.
# Checked against the ABSOLUTE path, so it works regardless of which root.
EVAL_ONLY_SUBSTRINGS = ["field_recordings"]

# domain codes stored in clips.npz / features
DOMAIN = {"mono": 0, "array": 1, "ext_test": 2}


def _abs(path_abs: str) -> str:
    return path_abs.replace("\\", "/").lower()


def _rel(path_abs: str) -> str:
    p = _abs(path_abs)
    for root in DATA_ROOTS:
        r = str(root).replace("\\", "/").lower()
        if p.startswith(r):
            return p[len(r):]
    return p


def is_excluded(path_abs: str) -> bool:
    p = _abs(path_abs)
    return any(e in p for e in EXCLUDE_SUBSTRINGS)


def infer_label(path_abs: str) -> str | None:
    if is_excluded(path_abs):
        return None
    rel = _rel(path_abs)
    for sub, cls in LABEL_RULES:
        if sub in rel:
            return cls
    return None


def is_eval_only(path_abs: str) -> bool:
    p = _abs(path_abs)
    return any(s in p for s in EVAL_ONLY_SUBSTRINGS)


def infer_domain(path_abs: str) -> str:
    if is_eval_only(path_abs):
        return "ext_test"
    rel = _rel(path_abs)
    return "array" if ("_array" in rel or "array_untyped" in rel) else "mono"


def infer_subtype(path_abs: str) -> str:
    """Fine drone subtype for EVO-vs-FPV (Stage 2). '' = unknown / untyped."""
    rel = _rel(path_abs)
    if "evo" in rel:
        return "evo"
    if ("fpv" in rel) or ("red13" in rel) or ("blue7" in rel) or ("inch" in rel):
        return "fpv"
    return ""
