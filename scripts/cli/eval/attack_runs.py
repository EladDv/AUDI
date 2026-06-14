"""cmd_attack_runs eval subcommand.

Runs attack evaluation on all checkpoints. By default:
  - Auto-postprocesses any checkpoints that don't have eval_data yet
  - Auto-calibrates any checkpoints that don't have hearability_calib.npz yet
  - Only evaluates checkpoints NOT already in the CSV (appends new results)
  - Saves CSV incrementally after each checkpoint (crash-resilient)
  - Skips checkpoints that OOM or fail (auto-recovery)
Use --all to force reprocessing of everything.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torchaudio

from audi.checkpoint import load_model_from_checkpoint
from audi.hysteresis import apply_hysteresis
from audi.training.validation import (
    compute_precision,
    compute_roc_values,
    find_threshold_at_precision,
)


def _run_stage(
    stage: str, args: list[str], noise_path: str | None, drone_path: str | None
) -> bool:
    """Run an eval stage via subprocess. Returns True on success."""
    cmd = ["uv", "run", "audi-eval"]
    if noise_path:
        cmd += ["--noise-path", noise_path]
    if drone_path:
        cmd += ["--drone-path", drone_path]
    cmd += [stage] + args
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run(
    noise_path: str | None,
    drone_path: str | None,
    *,
    _all: bool = False,
    _skip_postprocess: bool = False,
    _skip_calibrate: bool = False,
) -> None:
    # --- argparse (manual, since audi-eval dispatches via sys.argv) ---
    rest = sys.argv[1:] if len(sys.argv) > 1 else []
    if any(arg in {"-h", "--help"} for arg in rest):
        print(
            "usage: audi-eval attack-runs [--all] "
            "[--skip-postprocess] [--skip-calibrate] [--sweep <name>]"
        )
        return

    all_flag = _all
    skip_pp = _skip_postprocess
    skip_cal = _skip_calibrate
    sweep_filter = None
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--all":
            all_flag = True
        elif a == "--skip-postprocess":
            skip_pp = True
        elif a == "--skip-calibrate":
            skip_cal = True
        elif a == "--sweep":
            if i + 1 >= len(rest):
                raise SystemExit("--sweep requires a sweep name")
            sweep_filter = rest[i + 1]
            i += 1
        i += 1

    _STRIDE = 0.125
    _PROJECT = Path(__file__).resolve().parents[3]  # project root
    _ATTACK_DIR = _PROJECT / "data" / "attack_runs"
    _CHECKPOINTS_DIRS = [_PROJECT / "checkpoints"]
    _CSV_PATH = _PROJECT / "checkpoints" / "attack_run_precision_eval.csv"
    PRECISION_LEVELS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    _INFER_BATCH_SIZE = 16  # lower batch to avoid OOM on large models

    FIELD_NAMES = [
        "model",
        "sweep",
        "precision",
        "sigma",
        "cov_pct",
        "first_pct",
        "bg",
        "bg_alerts",
    ]

    def _count_alerts(dets: np.ndarray) -> int:
        """Count contiguous alert runs (0→1 transitions)."""
        if len(dets) == 0:
            return 0
        padded = np.pad(dets.astype(np.int8), (1, 0), constant_values=0)
        return int(np.sum((padded[1:] == 1) & (padded[:-1] == 0)))

    # ── incremental CSV helpers ─────────────────────────────────────────
    def _load_csv() -> list[dict]:
        """Load current CSV rows, or empty list."""
        if not _CSV_PATH.exists():
            return []
        with open(_CSV_PATH) as f:
            return list(csv.DictReader(f))

    def _flush_csv(rows: list[dict]) -> None:
        """Atomic write via temp file."""
        if not rows:
            return
        tmp = _CSV_PATH.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(_CSV_PATH)

    # ── helpers ──────────────────────────────────────────────────────────
    def split_by_zero_gaps(audio, sr, min_dur=3.0, min_gap_s=0.5):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        exact_zero = audio == 0.0
        zero_runs = []
        in_zero, start = False, 0
        for i in range(len(exact_zero) + 1):
            z = bool(exact_zero[i]) if i < len(exact_zero) else False
            if z and not in_zero:
                start, in_zero = i, True
            elif not z and in_zero:
                if (i - start) / sr >= min_gap_s:
                    zero_runs.append((start, i))
                in_zero = False
        if not zero_runs:
            return [audio] if len(audio) / sr >= min_dur else []
        segments, prev = [], 0
        for zs, ze in zero_runs:
            if (zs - prev) / sr >= min_dur:
                segments.append(audio[prev:zs].copy())
            prev = ze
        if (len(audio) - prev) / sr >= min_dur:
            segments.append(audio[prev:].copy())
        return segments

    def split_into_windows(audio, sr, clip_s: float):
        win = int(sr * clip_s)
        step = int(win * _STRIDE)
        return [audio[i : i + win] for i in range(0, len(audio) - win + 1, step)]

    def discover_best_checkpoints():
        raw = []
        for ckpts_dir in _CHECKPOINTS_DIRS:
            if not ckpts_dir.exists():
                continue
            for ckpt_path in sorted(ckpts_dir.rglob("*.ckpt")):
                parts = list(ckpt_path.relative_to(ckpts_dir).parts)
                exp_dir = sweep_dir = None
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
                if not exp_dir:
                    continue
                if not sweep_dir:
                    continue
                epoch = 0
                if "epoch=" in ckpt_path.stem:
                    try:
                        epoch = int(ckpt_path.stem.split("epoch=")[1].split("-")[0])
                    except Exception:
                        pass
                raw.append(
                    {
                        "path": str(ckpt_path),
                        "exp": exp_dir,
                        "sweep": sweep_dir,
                        "epoch": epoch,
                    }
                )
        best = {}
        for c in sorted(raw, key=lambda c: c["epoch"], reverse=True):
            if c["sweep"] + "__" + c["exp"] not in best:
                best[c["sweep"] + "__" + c["exp"]] = c
        return sorted(best.values(), key=lambda c: (c.get("sweep") or "", c["exp"]))

    def _cleanup(*objs):
        """Delete references and clear GPU cache."""
        for o in objs:
            try:
                del o
            except Exception:
                pass
        torch.cuda.empty_cache()

    def load_model(ckpt_path: str):
        model = load_model_from_checkpoint(ckpt_path, device="cpu", quiet=True)
        return model, model._clip_seconds, int(model._mel_cfg.sample_rate)

    def find_predictions_file(ckpt_path):
        ckpt = Path(ckpt_path)
        run_dir = ckpt.parent
        while run_dir.parent != run_dir:
            p = run_dir / "eval_data" / "predictions_best.pt"
            if p.exists():
                return p
            run_dir = run_dir.parent
        return None

    @torch.no_grad()
    def predict_windows(model, windows):
        device = next(model.parameters()).device
        scores = []
        for i in range(0, len(windows), _INFER_BATCH_SIZE):
            batch = torch.as_tensor(
                windows[i : i + _INFER_BATCH_SIZE], dtype=torch.float32
            ).to(device)
            logits = model(batch).cpu().numpy()
            scores.append(1.0 / (1.0 + np.exp(-logits)))
        return np.concatenate(scores) if scores else np.array([])

    # ── Stage 1: Discover checkpoints ──────────────────────────────────
    ckpts = discover_best_checkpoints()
    if sweep_filter is not None:
        ckpts = [c for c in ckpts if c["sweep"] == sweep_filter]
    print(f"Found {len(ckpts)} checkpoints\n")

    # ── Stage 2: Auto-postprocess new checkpoints ──────────────────────
    if not skip_pp:
        sweeps_needing_pp: set[str] = set()
        for ckpt in ckpts:
            run_dir = _PROJECT / "checkpoints" / ckpt["sweep"] / ckpt["exp"]
            pp_file = run_dir / "eval_data" / "predictions_best.pt"
            if all_flag or not pp_file.exists():
                sweeps_needing_pp.add(ckpt["sweep"])
        for sweep in sorted(sweeps_needing_pp):
            sweep_dir = _PROJECT / "checkpoints" / sweep
            if not sweep_dir.exists():
                continue
            runs_needed = 0
            for ckpt in ckpts:
                if ckpt["sweep"] != sweep:
                    continue
                run_dir = sweep_dir / ckpt["exp"]
                pp_file = run_dir / "eval_data" / "predictions_best.pt"
                if all_flag or not pp_file.exists():
                    runs_needed += 1
            if runs_needed == 0:
                continue
            extra = ["--all"] if all_flag else []
            print(f"\n[postprocess] {sweep} ({runs_needed} runs need it)...")
            _run_stage("postprocess", extra + [str(sweep_dir)], noise_path, drone_path)

    # ── Stage 3: Auto-calibrate new checkpoints ────────────────────────
    if not skip_cal:
        runs_needing_cal = []
        for ckpt in ckpts:
            run_dir = _PROJECT / "checkpoints" / ckpt["sweep"] / ckpt["exp"]
            cal_file = run_dir / "eval_data" / "hearability_calib.npz"
            pp_file = run_dir / "eval_data" / "predictions_best.pt"
            if not pp_file.exists():
                continue
            if all_flag or not cal_file.exists():
                runs_needing_cal.append((ckpt["sweep"], ckpt["exp"], run_dir))
        if runs_needing_cal:
            extra = ["--all"] if all_flag else []
            for sweep, exp, run_dir in runs_needing_cal:
                print(f"  [calibrate] {sweep}/{exp} ...")
                _run_stage("calibrate", extra + [str(run_dir)], noise_path, drone_path)

    # ── Stage 4: Load existing CSV (skip already-evaluated) ────────────
    existing: set[tuple[str, str]] = set()
    all_rows = _load_csv()
    for r in all_rows:
        existing.add((r["sweep"], r["model"]))

    # ── Stage 5: Load attack audio once ────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_waveforms: dict[str, tuple[np.ndarray, int]] = {}
    for fp in sorted(_ATTACK_DIR.glob("*.wav")):
        audio, sr = torchaudio.load(str(fp))
        audio_waveforms[fp.name] = (
            audio.mean(dim=0).numpy().astype(np.float32).reshape(-1),
            int(sr),
        )
    print(f"Loaded {len(audio_waveforms)} audio files\n")

    # ── Stage 6: Evaluate checkpoints (incremental, crash-resilient) ──
    to_eval = []
    for ckpt in ckpts:
        key = (ckpt["sweep"], ckpt["exp"])
        if not all_flag and key in existing:
            continue
        run_dir = _PROJECT / "checkpoints" / ckpt["sweep"] / ckpt["exp"]
        pp_file = run_dir / "eval_data" / "predictions_best.pt"
        if not pp_file.exists():
            print(f"[{ckpt['sweep']}/{ckpt['exp']}] ✗ no eval_data — skipping")
            continue
        to_eval.append(ckpt)

    skipped_count = len([c for c in ckpts if (c["sweep"], c["exp"]) in existing])
    if skipped_count:
        print(f"Skipping {skipped_count} already-evaluated checkpoints (use --all to force)\n")
    if not to_eval:
        print("No new checkpoints to evaluate!")
        return

    print(f"Evaluating {len(to_eval)} checkpoints\n")
    failed_ckpts: list[str] = []

    for ci, ckpt in enumerate(to_eval):
        label = ckpt["exp"]
        sweep = ckpt.get("sweep") or ""
        print(f"[{ci + 1}/{len(to_eval)}] {sweep}/{label} ...", end=" ", flush=True)

        model = None
        pred_data = None
        try:
            model, clip_s, model_sample_rate = load_model(ckpt["path"])
            model = model.to(device)

            pred_file = find_predictions_file(ckpt["path"])
            if pred_file is None:
                print("✗ no eval_data — skipping")
                _cleanup(model)
                model = None
                continue

            pred_data = torch.load(pred_file, map_location="cpu", weights_only=False)
            val_logits = np.asarray(pred_data["logits"]).flatten()
            val_labels = np.asarray(pred_data["labels"]).flatten()
            fpr, tpr, th, auc = compute_roc_values(val_logits, val_labels)
            prec = compute_precision(val_logits, val_labels, th)
            prec_thresholds = {}
            for pt in PRECISION_LEVELS:
                if pt < prec.min() or pt > prec.max():
                    continue
                th_pt, _, _ = find_threshold_at_precision(prec, tpr, th, pt)
                prec_thresholds[pt] = 1.0 / (1.0 + np.exp(-th_pt))

            # Build windows using model's actual clip length
            all_atk_segs: list[tuple[str, np.ndarray]] = []
            bg_windows: list[np.ndarray] = []
            for name, (audio, audio_sample_rate) in audio_waveforms.items():
                if audio_sample_rate != model_sample_rate:
                    audio_t = torch.as_tensor(audio, dtype=torch.float32).unsqueeze(0)
                    audio = torchaudio.functional.resample(
                        audio_t, audio_sample_rate, model_sample_rate
                    ).squeeze(0).numpy()
                if name.startswith("background"):
                    bg_windows.extend(
                        split_into_windows(audio, model_sample_rate, clip_s)
                    )
                else:
                    segs = split_by_zero_gaps(audio, model_sample_rate)
                    for i, seg in enumerate(segs):
                        all_atk_segs.append((f"{Path(name).stem}_seg{i}", seg))

            atk_seg_data = []
            for name, seg in all_atk_segs:
                wins = split_into_windows(seg, model_sample_rate, clip_s)
                if not wins:
                    atk_seg_data.append((name, np.array([])))
                else:
                    scores = predict_windows(model, np.stack(wins))
                    atk_seg_data.append((name, scores))
            bg_scores = (
                predict_windows(model, np.stack(bg_windows)) if bg_windows else np.array([])
            )

            checkpoint_rows = []
            for pt, sigma in sorted(prec_thresholds.items()):
                segment_coverages, segment_first_pct = [], []
                for _name, scores in atk_seg_data:
                    if len(scores) == 0:
                        segment_coverages.append(0.0)
                        segment_first_pct.append(100.0)
                        continue
                    dets = apply_hysteresis(scores, sigma)
                    coverage = 100.0 * dets.sum() / len(dets)
                    det_idx = np.where(dets)[0]
                    first_pct = 100.0 * det_idx[0] / len(dets) if len(det_idx) > 0 else 100.0
                    segment_coverages.append(coverage)
                    segment_first_pct.append(first_pct)
                bg_dets = apply_hysteresis(bg_scores, sigma)
                bg_alerts = _count_alerts(bg_dets)
                checkpoint_rows.append({
                    "model": label, "sweep": sweep,
                    "precision": f"P{int(pt * 100)}", "sigma": round(sigma, 4),
                    "cov_pct": round(np.mean(segment_coverages), 1),
                    "first_pct": round(np.median(segment_first_pct), 1),
                    "bg": int(bg_dets.sum()),
                    "bg_alerts": bg_alerts,
                })

            # Flush incrementally
            if checkpoint_rows:
                best_row = max(checkpoint_rows, key=lambda r: r["cov_pct"] - r["bg"] * 0.5)
                print(
                    f"✓ best={best_row['precision']} σ={best_row['sigma']:.3f} "
                    f"cov={best_row['cov_pct']:.0f}% "
                    f"1st={best_row['first_pct']:.0f}% "
                    f"bg={best_row['bg']}/{len(bg_scores)} "
                    f"alerts={best_row.get('bg_alerts', '?')}"
                )
                # Append to all_rows and flush
                all_rows.extend(checkpoint_rows)
                _flush_csv(all_rows)

        except torch.cuda.OutOfMemoryError:
            print("✗ OOM — skipping, will retry on next run")
            failed_ckpts.append(f"{sweep}/{label}")
            _cleanup(model, pred_data)
        except Exception as e:
            traceback.print_exc()
            print(f"✗ {e}")
            failed_ckpts.append(f"{sweep}/{label}")
            _cleanup(model, pred_data)
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()

    # ── Final save (should be redundant with incremental, but safe) ─────
    _flush_csv(all_rows)

    if failed_ckpts:
        print(f"\nFailed ({len(failed_ckpts)}): {', '.join(failed_ckpts[:10])}")

    # ── Summary ─────────────────────────────────────────────────────────
    p90 = [r for r in all_rows if r["precision"] == "P90"]
    p90.sort(key=lambda r: (-float(r["cov_pct"]), float(r["first_pct"]), int(r["bg"])))
    print(f"\n{'=' * 90}")
    print("TOP MODELS at PRECISION=0.90")
    print(f"{'=' * 90}")
    print(
        f"{'#':>3} {'model':<45} {'σ':>7} {'cov%':>6} {'1st%':>6} "
        f"{'bg':>5} {'alerts':>7} {'sweep'}"
    )
    print(f"{'-' * 90}")
    for i, r in enumerate(p90[:15]):
        print(
            f"{i + 1:>3} {r['model']:<45} {float(r['sigma']):>7.4f} "
            f"{float(r['cov_pct']):>6.1f} {float(r['first_pct']):>6.1f} "
            f"{int(r.get('bg', 0)):>5} "
            f"{str(r.get('bg_alerts', '-') or '-'):>7} {r.get('sweep', '')}"
        )
