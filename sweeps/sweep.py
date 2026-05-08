#!/usr/bin/env python3
"""Sweep runner — pass a YAML config path to run a sweep.

Usage:
    uv run python sweeps/sweep.py configs/arch.yaml
    uv run python sweeps/sweep.py configs/regularization.yaml

YAML format:
    name: str
    base_flags: str            # flags applied to every config
    configs:
      - name: str
        flags: str             # per-config flags, combined with base_flags
    pretrained_checkpoint: str  # optional, finetune sweeps only
"""

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_CMD = "uv run audi-train detect"
BASE_FLAGS = ""


def load_sweep_config(yaml_path: str | Path) -> dict[str, Any]:
    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    base = data.get("base_flags", "")
    configs = []
    for c in data["configs"]:
        per_flags = c.get("flags", "")
        combined = f"{base} {per_flags}".strip()
        configs.append({"name": c["name"], "flags": combined})

    return {
        "name": data["name"],
        "configs": configs,
        "noise_path": data.get("noise_path", ""),
        "drone_path": data.get("drone_path", ""),
        "description": data.get("description", ""),
    }


def _find_checkpoint_dir(run_dir: Path) -> Path:
    """Find the actual checkpoint directory inside a run_dir.

    Supports both TB-direct layout (checkpoints/) and lightning_logs layout.
    """
    # TB-direct (current default with name="" version="")
    tb_dir = run_dir / "checkpoints"
    if tb_dir.exists() and list(tb_dir.glob("*.ckpt")):
        return tb_dir
    # lightning_logs layout (fallback for older runs)
    ll_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
    if ll_dir.exists() and list(ll_dir.glob("*.ckpt")):
        return ll_dir
    # No checkpoints yet — default to TB-direct
    return tb_dir


def _save_run_config(
    run_dir: Path,
    name: str,
    flags: str,
    *,
    sweep_name: str = "",
    noise_path: str = "",
    drone_path: str = "",
) -> None:
    """Save the run's config as YAML inside its checkpoint folder."""
    import yaml

    ckpt_dir = _find_checkpoint_dir(run_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_path = ckpt_dir / "sweep_config.yaml"

    config = {
        "sweep": sweep_name,
        "run": name,
        "flags": flags,
        "noise_path": noise_path,
        "drone_path": drone_path,
        "timestamp": datetime.now().isoformat(),
    }
    # Strip empty values for cleaner output
    config = {k: v for k, v in config.items() if v}

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  📄 Config saved → {config_path}")


def run_config(
    name: str, flags: str, sweep_dir: Path,
    noise_path: str = "", drone_path: str = "",
    sweep_name: str = "",
) -> Path | None:
    """Run one training config. Returns run_dir on success, None on failure."""
    run_dir = sweep_dir / name

    # Resume: skip if already completed (either checkpoint layout)
    for ckpt_candidate in (
        run_dir / "checkpoints",
        run_dir / "lightning_logs" / "version_0" / "checkpoints",
    ):
        if ckpt_candidate.exists() and list(ckpt_candidate.glob("*.ckpt")):
            print(f"\n  ⏭  SKIP (already done): {name}")
            # Backfill config if missing (legacy runs)
            if not (ckpt_candidate / "sweep_config.yaml").exists():
                _save_run_config(
                    run_dir, name, flags,
                    sweep_name=sweep_name,
                    noise_path=noise_path, drone_path=drone_path,
                )
            return run_dir

    # Clean partial from previous crash
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)

    # Build command with noise/drone paths
    path_args = ""
    if noise_path:
        path_args += f" --noise-path {noise_path}"
    if drone_path:
        path_args += f" --drone-path {drone_path}"
    cmd = f"{TRAIN_SCRIPT_CMD} {BASE_FLAGS} {flags}{path_args} --output-dir {run_dir}"
    print(f"\n{'=' * 60}")
    print(f"[{datetime.now():%H:%M:%S}] Running: {name}")
    print(f"  {cmd}")
    print(f"{'=' * 60}")

    proc = subprocess.Popen(cmd, shell=True, cwd=str(PROJECT))
    try:
        ret = proc.wait()
    except KeyboardInterrupt:
        print("\n  ⏹  Interrupted by Ctrl+C — killing run, continuing sweep...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return None

    if ret != 0:
        print(f"  ✗ FAILED (exit {ret}) — continuing to next run")
        return None

    # Save config alongside checkpoints
    _save_run_config(
        run_dir, name, flags,
        sweep_name=sweep_name,
        noise_path=noise_path, drone_path=drone_path,
    )
    return run_dir


def extract_metrics(run_dir: Path) -> dict[str, float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        log_dir = run_dir / "lightning_logs" / "version_0"
        if not log_dir.exists():
            return {}

        ea = EventAccumulator(str(log_dir))
        ea.Reload()

        metrics: dict[str, float] = {}
        scalar_tags = ea.Tags().get("scalars", [])
        for tag in [
            "val/tpr_at_precision_90",
            "val/auc",
            "val/ece",
            "val/average_precision",
        ]:
            if tag in scalar_tags:
                events = ea.Scalars(tag)
                if events:
                    metrics[tag.replace("val/", "")] = max(
                        e.value for e in events
                    )
        return metrics
    except Exception as e:
        print(f"  ⚠  Metrics extraction failed: {e}")
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def auto_postprocess(
    sweep_dir: Path, noise_path: str = "", drone_path: str = ""
) -> None:

    extra = []
    if noise_path:
        extra += ["--noise-path", noise_path]
    if drone_path:
        extra += ["--drone-path", drone_path]

    print(f"\n{'─' * 50}")
    print("Running postprocess...")
    subprocess.run(
        [
            "uv",
            "run",
            "audi-eval",
            *extra,
            "postprocess",
            str(sweep_dir),
        ]
    )

    print("Running calibration...")
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "eval_data" / "predictions_best.pt").exists():
            subprocess.run(
                [
                    "uv",
                    "run",
                    "audi-eval",
                    *extra,
                    "calibrate",
                    str(run_dir),
                ],
                capture_output=True,
            )


def sweep_main(
    configs: list[dict],
    sweep_name: str,
    *,
    sweep_dir: Path | None = None,
    do_postprocess: bool = True,
    noise_path: str = "",
    drone_path: str = "",
) -> int:
    if sweep_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = PROJECT / "checkpoints" / f"{sweep_name}_{ts}"

    csv_path = sweep_dir / "results.csv"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    try:
        for cfg in configs:
            run_dir = run_config(
                cfg["name"], cfg["flags"], sweep_dir,
                noise_path=noise_path, drone_path=drone_path,
                sweep_name=sweep_name,
            )
            if run_dir is None:
                results.append({"name": cfg["name"], "status": "failed"})
                write_csv(csv_path, results)
                continue

            metrics = extract_metrics(run_dir)
            results.append({"name": cfg["name"], "status": "ok", **metrics})
            write_csv(csv_path, results)

            best = metrics.get("tpr_at_precision_90", 0)
            print(f"  ✓ TPR@P90={best:.4f}")

    except KeyboardInterrupt:
        print(f"\n\n⚠  Interrupted. Saving {len(results)} results...")
        write_csv(csv_path, results)
        return 1

    print(f"\n{'─' * 50}")
    print(f"Done. {len(results)} configs → {csv_path}")

    if do_postprocess:
        auto_postprocess(
            sweep_dir, noise_path=noise_path, drone_path=drone_path
        )

    return 0


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Parse global args: --noise-path, --drone-path
    noise_path = drone_path = ""
    positional = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--noise-path" and i + 1 < len(sys.argv):
            noise_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--drone-path" and i + 1 < len(sys.argv):
            drone_path = sys.argv[i + 1]
            i += 2
        else:
            positional.append(sys.argv[i])
            i += 1

    if not positional:
        print(
            "Usage: uv run python sweeps/sweep.py [--noise-path <path>] [--drone-path <path>] configs/<name>.yaml"
        )
        sys.exit(1)

    yaml_path = Path(positional[0])
    cfg = load_sweep_config(yaml_path)

    # CLI overrides YAML
    noise_path = noise_path or cfg.get("noise_path", "")
    drone_path = drone_path or cfg.get("drone_path", "")

    raise SystemExit(
        sweep_main(
            cfg["configs"],
            cfg["name"],
            noise_path=noise_path,
            drone_path=drone_path,
        )
    )
