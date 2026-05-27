"""Batch field-recording evaluation: runs every model against field backgrounds + alerts.

Output: checkpoints/field_eval.csv with columns:
  sweep, model, sigma_P90, bg_fp, bg_total, alert_tp, alert_fn, alert_total, rec_total_windows, rec_detections, rec_alerts

Usage:
    uv run python scripts/eval_field.py
    uv run python scripts/eval_field.py --sweep mel_sweep_20260521_125510
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true", help="Skip already-evaluated models")
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    PROJECT = Path(__file__).resolve().parents[1]
    SR = 32000
    STRIDE = 0.125
    FIELD_DIR = PROJECT / "data" / "field_recordings_20260514"
    CSV_PATH = PROJECT / "checkpoints" / "attack_run_precision_eval.csv"
    OUT_PATH = PROJECT / "checkpoints" / "field_eval.csv"

    # ── Load P90 sigmas from attack CSV ─────────────────────────────────
    sigmas: dict[tuple[str, str], float] = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["precision"] == "P90":
                sigmas[(r["sweep"], r["model"])] = float(r["sigma"])

    # ── Discover checkpoints ────────────────────────────────────────────
    from audi.checkpoint import strip_compile_prefix, get_clip_seconds
    from audi.config import MelConfig, ModelConfig, OptimizerConfig
    from audi.training.detector import DroneDetector
    from audi.hysteresis import apply_hysteresis

    CHECKPOINTS_DIR = PROJECT / "checkpoints"
    models_to_eval: list[tuple[str, str, Path, float]] = []  # (sweep, model, ckpt_path, sigma)

    for sweep_dir in sorted(CHECKPOINTS_DIR.iterdir()):
        if not sweep_dir.is_dir():
            continue
        if args.sweep and sweep_dir.name != args.sweep:
            continue
        for run_dir in sorted(sweep_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            key = (sweep_dir.name, run_dir.name)
            if key not in sigmas:
                continue
            ckpt_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
            if not ckpt_dir.exists():
                ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue
            ckpts = sorted(ckpt_dir.glob("*.ckpt"))
            if not ckpts:
                continue
            models_to_eval.append((sweep_dir.name, run_dir.name, ckpts[-1], sigmas[key]))

    # ── Resume ──────────────────────────────────────────────────────────
    done: set[tuple[str, str]] = set()
    if args.resume and OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for r in csv.DictReader(f):
                done.add((r["sweep"], r["model"]))

    models_to_eval = [m for m in models_to_eval if (m[0], m[1]) not in done]
    print(f"Models to evaluate: {len(models_to_eval)} (already done: {len(done)})")

    # ── Load field audio once ──────────────────────────────────────────
    def load_wavs(glob_pattern: str) -> list[tuple[str, np.ndarray]]:
        files = sorted(Path(p) for p in glob_pattern) if isinstance(glob_pattern, str) else sorted(glob_pattern)
        audio_files = []
        for fp in files:
            # Filter valid files
            if not fp.exists():
                continue
            if fp.stat().st_size == 0:
                continue
            try:
                audio, sr = torchaudio.load(str(fp))
            except Exception:
                continue
            audio_files.append((fp.name, audio.mean(dim=0).numpy().astype(np.float32)))
        return audio_files

    bg_files = load_wavs(list((FIELD_DIR / "backgrounds").glob("*.wav")))

    alert_files: list[tuple[str, np.ndarray]] = []
    alert_dir = FIELD_DIR / "alerts"
    if alert_dir.exists():
        for d in sorted(alert_dir.iterdir()):
            if not d.is_dir():
                continue
            for fp in sorted([f for f in d.glob("*.wav") if f.stat().st_size > 0]):
                try:
                    audio, sr = torchaudio.load(str(fp))
                except Exception:
                    continue
                alert_files.append((f"{d.name}/{fp.name}", audio.mean(dim=0).numpy().astype(np.float32)))

    rec_files: list[tuple[str, np.ndarray]] = []
    rec_dir = FIELD_DIR / "recordings"
    if rec_dir.exists():
        for fp in sorted(rec_dir.glob("*.flac")):
            try:
                audio, sr = torchaudio.load(str(fp))
            except Exception:
                continue
            rec_files.append((fp.name, audio.mean(dim=0).numpy().astype(np.float32)))

    print(f"Field audio: {len(bg_files)} backgrounds, {len(alert_files)} alerts, {len(rec_files)} recordings")

    # ── Helpers ─────────────────────────────────────────────────────────
    def split_into_windows(audio, clip_s):
        win = int(SR * clip_s)
        step = int(win * STRIDE)
        if len(audio) < win:
            return []
        return [audio[i:i+win] for i in range(0, len(audio) - win + 1, step)]

    def count_alerts(dets):
        if len(dets) == 0:
            return 0
        padded = np.pad(dets.astype(np.int8), (1, 0), constant_values=0)
        return int(np.sum((padded[1:] == 1) & (padded[:-1] == 0)))

    @torch.no_grad()
    def predict(model, windows, batch_size=32):
        scores = []
        for i in range(0, len(windows), batch_size):
            batch = torch.as_tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch).cpu().numpy()
            scores.append(1.0 / (1.0 + np.exp(-logits)))
        return np.concatenate(scores).flatten() if scores else np.array([])

    def eval_files(model, clip_s, files, sigma):
        """Run inference on a list of (name, audio) and count alerts."""
        total_alerts = 0
        total_windows = 0
        for name, audio in files:
            wins = split_into_windows(audio, clip_s)
            if not wins:
                continue
            scores = predict(model, np.stack(wins))
            dets = apply_hysteresis(scores, sigma)
            alerts = count_alerts(dets)
            total_alerts += alerts
            total_windows += len(wins)
        # files_with_alerts = number of files that had at least 1 alert
        n_files = len(files)
        return total_alerts, total_windows, n_files

    # ── Evaluate ────────────────────────────────────────────────────────
    results = []
    header = ["sweep", "model", "sigma_P90",
              "bg_fp", "bg_total",
              "alert_tp", "alert_fn", "alert_total",
              "rec_windows", "rec_detections", "rec_alerts", "rec_segments"]

    if args.resume and OUT_PATH.exists():
        with open(OUT_PATH) as f:
            results = list(csv.DictReader(f))

    for idx, (sweep_name, model_name, ckpt_path, sigma) in enumerate(models_to_eval):
        print(f"[{idx+1}/{len(models_to_eval)}] {sweep_name}/{model_name} σ={sigma:.4f} ...", end=" ", flush=True)

        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            hp = ckpt["hyper_parameters"]
            model_hp = hp.get("model", {})
            if isinstance(model_hp, dict):
                model_cfg = ModelConfig(
                    arch=model_hp.get("arch", hp.get("model_arch", "cnn14")),
                    pretrained=model_hp.get("pretrained", hp.get("pretrained_backbone", True)),
                    compile=False,
                )
            else:
                model_cfg = ModelConfig(arch=model_hp.arch, pretrained=model_hp.pretrained, compile=False)
            mel_hp = hp.get("mel", {})
            if isinstance(mel_hp, dict):
                mel_cfg = MelConfig(
                    n_mels=mel_hp.get("n_mels", 128), n_fft=mel_hp.get("n_fft", 1024),
                    hop_length=mel_hp.get("hop_length", 160),
                )
            else:
                mel_cfg = mel_hp
            model = DroneDetector(
                model=model_cfg, mel=mel_cfg, optimizer=OptimizerConfig(),
                bin_names=hp.get("bin_names", []),
                loss_type=hp.get("loss_type", "bce"),
                label_smoothing=hp.get("label_smoothing", 0.0),
                dropout=hp.get("dropout", 0.0),
            )
            model.load_state_dict(strip_compile_prefix(ckpt["state_dict"]), strict=False)
            model = model.to(device).eval()
            clip_s = get_clip_seconds(hp)
            del ckpt
            torch.cuda.empty_cache()

            # Backgrounds — every alert is an FP
            bg_alerts, bg_windows, bg_n = eval_files(model, clip_s, bg_files, sigma)

            # Alerts — every file with 0 alerts is an FN, with >0 alerts is TP
            alert_tp, alert_fn = 0, 0
            for name, audio in alert_files:
                wins = split_into_windows(audio, clip_s)
                if not wins:
                    continue
                scores = predict(model, np.stack(wins))
                dets = apply_hysteresis(scores, sigma)
                a = count_alerts(dets)
                if a > 0:
                    alert_tp += 1
                else:
                    alert_fn += 1

            # Recordings
            rec_alerts = 0
            rec_windows = 0
            rec_segments = len(rec_files)
            for name, audio in rec_files:
                wins = split_into_windows(audio, clip_s)
                if not wins:
                    continue
                scores = predict(model, np.stack(wins))
                dets = apply_hysteresis(scores, sigma)
                a = count_alerts(dets)
                rec_alerts += a
                rec_windows += len(wins)

            result = {
                "sweep": sweep_name, "model": model_name,
                "sigma_P90": sigma,
                "bg_fp": bg_alerts, "bg_total": bg_n,
                "alert_tp": alert_tp, "alert_fn": alert_fn, "alert_total": alert_tp + alert_fn,
                "rec_windows": rec_windows, "rec_detections": rec_alerts,
                "rec_alerts": rec_alerts, "rec_segments": rec_segments,
            }
            results.append(result)

            print(f"bg={bg_alerts}/{bg_n}  alert={alert_tp}/{alert_tp+alert_fn}  rec={rec_alerts}/{rec_segments}")

            # Flush incrementally
            tmp = OUT_PATH.with_suffix(".csv.tmp")
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                w.writerows(results)
            tmp.replace(OUT_PATH)

        except Exception as e:
            print(f"✗ {e}")
            torch.cuda.empty_cache()
            continue

        # Cleanup
        del model
        torch.cuda.empty_cache()

    print(f"\nDone! {len(results)} models written to {OUT_PATH}")


if __name__ == "__main__":
    main()
