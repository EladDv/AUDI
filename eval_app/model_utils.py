"""Checkpoint discovery and model loading for the eval dashboard."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from audi.checkpoint import load_model_from_checkpoint as _load_from_ckpt
from audi.model import SUPPORTED_MODEL_ARCHS
from audi.training.detector import DroneDetector

_SR = 16000
_CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
_SUPPORTED_MODEL_ARCHS = {arch.lower() for arch in SUPPORTED_MODEL_ARCHS}


@st.cache_resource
def discover_checkpoints() -> list[dict]:
    """Scan checkpoints/ for all .ckpt files, keep only latest per experiment."""
    raw = []
    for ckpt_path in sorted(_CHECKPOINTS_DIR.rglob("*.ckpt")):
        rel = ckpt_path.relative_to(_CHECKPOINTS_DIR)
        parts = list(rel.parts)
        exp_dir = None
        sweep_dir = None
        for i, p in enumerate(parts):
            if p == "lightning_logs" and i > 0:
                exp_dir = parts[i - 1]
                sweep_dir = parts[i - 2] if i >= 2 else None
                break
        if not exp_dir:
            for i, p in enumerate(parts):
                if p == "checkpoints" and i >= 2:
                    exp_dir = parts[i - 1]
                    sweep_dir = parts[i - 2] if i >= 2 else None
                    break
        ckpt_name = ckpt_path.stem
        epoch = 0
        if "epoch=" in ckpt_name:
            try:
                epoch = int(ckpt_name.split("epoch=")[1].split("-")[0])
            except ValueError:
                pass
        if exp_dir:
            label = f"{exp_dir}  [{ckpt_name}]"
            run = sweep_dir or exp_dir
        else:
            label = f"{ckpt_name}"
            run = parts[0] if parts else "unknown"
        arch = get_model_arch_from_ckpt(str(ckpt_path))
        if arch is None or arch.lower() not in _SUPPORTED_MODEL_ARCHS:
            continue
        raw.append(
            {
                "label": label,
                "path": str(ckpt_path),
                "run": run,
                "exp_dir": exp_dir,
                "epoch": epoch,
                "arch": arch,
            }
        )
    ckpts = []
    seen = {}
    for c in sorted(raw, key=lambda c: c["epoch"], reverse=True):
        key = (c["run"], c["exp_dir"])
        if key not in seen:
            seen[key] = True
            ckpts.append(c)
    ckpts.sort(key=lambda c: (c["run"], c["exp_dir"] or ""))
    return ckpts


@st.cache_resource
def load_model(ckpt_path: str, device: str) -> DroneDetector | None:
    """Load a checkpoint into eval mode on the given device."""
    return _load_from_ckpt(ckpt_path, device=device, quiet=True)


def get_model_arch_from_ckpt(ckpt_path: str) -> str | None:
    """Extract model_arch from checkpoint without full load."""
    import io
    import pickle
    import zipfile

    try:
        with zipfile.ZipFile(ckpt_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/hyper_parameters"):
                    with zf.open(name) as hp_file:
                        hp = pickle.load(io.BytesIO(hp_file.read()))
                    arch = hp.get("model_arch")
                    if arch:
                        return str(arch)
                    model_hp = hp.get("model", {})
                    if isinstance(model_hp, dict):
                        arch = model_hp.get("arch")
                        if arch:
                            return str(arch)
                    if hasattr(model_hp, "arch"):
                        return str(model_hp.arch)
                    break
    except Exception:
        pass
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt.get("hyper_parameters", {})
        arch = hp.get("model_arch")
        if arch:
            return str(arch)
        model_hp = hp.get("model", {})
        if isinstance(model_hp, dict):
            arch = model_hp.get("arch")
            return str(arch) if arch else None
        arch = getattr(model_hp, "arch", None)
        return str(arch) if arch else None
    except Exception:
        pass
    return None


def find_predictions_file(ckpt_path: str) -> str | None:
    """Find eval_data/predictions_best.pt for a checkpoint."""
    ckpt = Path(ckpt_path)
    run_dir = ckpt.parent
    while run_dir.parent != run_dir:
        pred = run_dir / "eval_data" / "predictions_best.pt"
        if pred.exists():
            return str(pred)
        run_dir = run_dir.parent
    return None


def find_hearability_calib(ckpt_path: str) -> str | None:
    """Look for hearability_calib.npz near a checkpoint's run directory."""
    ckpt = Path(ckpt_path)
    run_dir = ckpt.parent
    while run_dir.parent != run_dir:
        if (run_dir / "eval_data" / "hearability_calib.npz").exists():
            return str(run_dir / "eval_data" / "hearability_calib.npz")
        run_dir = run_dir.parent
    return None


@st.cache_data
def load_precision_thresholds() -> dict[str, dict[str, dict]]:
    """Return {sweep/model: {P_level: {sigma, cov, bg}}} from attack eval CSV."""
    csv_path = _CHECKPOINTS_DIR / "attack_run_precision_eval.csv"
    if not csv_path.exists():
        return {}
    thresholds: dict[str, dict[str, dict]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ref = f"{row.get('sweep','')}/{row['model']}"
            if ref not in thresholds:
                thresholds[ref] = {}
            thresholds[ref][row["precision"]] = {
                "sigma": float(row["sigma"]),
                "cov_pct": float(row["cov_pct"]),
                "first_pct": float(row["first_pct"]),
                "bg": int(row["bg"]),
                "bg_alerts": row.get("bg_alerts", "-") or "-",
            }
    return thresholds


def compute_precision_recall_curve(pred_file: str) -> dict:
    """Compute precision/recall vs threshold from a predictions file."""
    pred_data = torch.load(pred_file, map_location="cpu", weights_only=False)
    val_logits = np.asarray(pred_data["logits"]).flatten()
    val_labels = np.asarray(pred_data["labels"]).flatten()

    thresholds = np.linspace(val_logits.min(), val_logits.max(), 200)
    precisions = []
    recalls = []
    for th in thresholds:
        preds = (val_logits > th).astype(int)
        tp = ((preds == 1) & (val_labels == 1)).sum()
        fp = ((preds == 1) & (val_labels == 0)).sum()
        fn = ((preds == 0) & (val_labels == 1)).sum()
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(tp + fn, 1))
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    return {
        "thresholds": thresholds,
        "precisions": precisions,
        "recalls": recalls,
        "sig_thresholds": 1.0 / (1.0 + np.exp(-thresholds)),
    }
