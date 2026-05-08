"""Checkpoint discovery and model loading for the eval dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from audi.checkpoint import load_model_from_checkpoint as _load_from_ckpt
from audi.training.detector import DroneDetector

_SR = 16000
_CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


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
        raw.append(
            {
                "label": label,
                "path": str(ckpt_path),
                "run": run,
                "exp_dir": exp_dir,
                "epoch": epoch,
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
                        return arch
                    model_hp = hp.get("model", {})
                    if hasattr(model_hp, "arch"):
                        return model_hp.arch
                    return None
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
