"""Evaluate mn sweep models on field recordings at all P levels.

Usage:
    uv run python scripts/eval_field_mn.py
    uv run python scripts/eval_field_mn.py --top 10
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import tqdm


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0, help="Evaluate only top N models by P90 cov")
    ap.add_argument("--sweep", default=None, help="Only evaluate models from this sweep")
    ap.add_argument("--resume", action="store_true", help="Skip already-evaluated models")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    PROJECT = Path(__file__).resolve().parents[1]
    SR = 16000
    STRIDE = 0.125
    FIELD_DIR = PROJECT / "data" / "field_recordings_20260514"
    CSV_PATH = PROJECT / "checkpoints" / "attack_run_precision_eval.csv"

    # ── All sweeps ──────────────────────────────────────────────────────
    # ── Load thresholds from attack CSV ──────────────────────────────────
    LEVELS = ["P50", "P60", "P70", "P75", "P80", "P85", "P90", "P95", "P99"]
    PRECISIONS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

    model_thresholds: dict[str, dict[str, float]] = {}  # "sweep/model" -> {P_level: sigma}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            ref = f"{r['sweep']}/{r['model']}"
            # Filter by sweep if specified
            if args.sweep and r['sweep'] != args.sweep:
                continue
            # Only include models that exist on disk
            sweep_dir = PROJECT / "checkpoints" / r["sweep"]
            run_dir = sweep_dir / r["model"]
            if not run_dir.exists():
                cand = list((PROJECT / "checkpoints").glob(f"*{r['sweep']}*/{r['model']}"))
                if not cand:
                    continue
            ckpt_dir_check = run_dir / "checkpoints"
            if not ckpt_dir_check.exists():
                ckpt_dir_check = run_dir / "lightning_logs" / "version_0" / "checkpoints"
            if not ckpt_dir_check.exists() or not list(ckpt_dir_check.glob("*.ckpt")):
                continue
            if ref not in model_thresholds:
                model_thresholds[ref] = {}
            model_thresholds[ref][r["precision"]] = float(r["sigma"])

    model_order = sorted(model_thresholds, key=lambda m: model_thresholds[m].get("P90", 0), reverse=True)
    if args.top > 0:
        model_order = model_order[:args.top]

    print(f"Models: {len(model_order)}")

    # ── Load field audio once ───────────────────────────────────────────
    def load_audio_files(directory: Path, glob_pattern: str = "*.wav") -> list[tuple[str, np.ndarray]]:
        files = []
        for fp in sorted(directory.glob(glob_pattern)):
            if fp.stat().st_size == 0:
                continue
            try:
                audio, sr = torchaudio.load(str(fp))
            except Exception:
                continue
            files.append((fp.name, audio.mean(dim=0).numpy().astype(np.float32)))
        return files

    # ── Load manual labels ───────────────────────────────────────────────
    labels_csv = FIELD_DIR / "labels.csv"
    alert_labels: dict[str, str] = {}  # alert_dir_name -> "drone" | "nodrone"
    if labels_csv.exists():
        with open(labels_csv) as f:
            for r in csv.DictReader(f):
                alert_labels[r["alert_dir"]] = r["label"]
        print(f"Labels: {sum(1 for v in alert_labels.values() if v=='drone')} drone, "
              f"{sum(1 for v in alert_labels.values() if v=='nodrone')} nodrone")
    else:
        print("WARNING: no labels.csv found, treating all alerts as drone")

    # ── Load alert file paths only (don't pre-load audio to save memory) ──
    alert_file_paths: list[tuple[str, Path, str]] = []  # (dir_name, wav_path, label)
    alert_dir_path = FIELD_DIR / "alerts"
    if alert_dir_path.exists():
        for d in sorted(alert_dir_path.iterdir()):
            if not d.is_dir():
                continue
            fp = d / "full_120s.wav"
            if not fp.exists() or fp.stat().st_size == 0:
                continue
            label = alert_labels.get(d.name, "drone")
            alert_file_paths.append((d.name, fp, label))

    print(f"Field audio: {len(alert_file_paths)} alert recordings")

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

    from audi.hysteresis import apply_hysteresis

    # ── Evaluate each model ─────────────────────────────────────────────
    results = []  # [{ref, model, sweep, P_level, sigma, bg_fp, ...}]

    OUT_PATH = PROJECT / "checkpoints" / "field_eval_all.csv"
    FIELD_NAMES = ["ref", "sweep", "model", "P", "sigma",
                   "alert_tp", "alert_fn", "alert_fp", "alert_tn", "alert_total"]

    # Load existing results for resume
    done_refs: set[str] = set()
    if args.resume and OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for r in csv.DictReader(f):
                done_refs.add(r["ref"])
                results.append(r)

    def flush_results():
        tmp = OUT_PATH.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELD_NAMES)
            w.writeheader()
            w.writerows(results)
        tmp.replace(OUT_PATH)

    for idx, ref in enumerate(model_order):
        if args.resume and ref in done_refs:
            print(f"[{idx+1}/{len(model_order)}] {ref} ... already done, skipping")
            continue
        print(f"[{idx+1}/{len(model_order)}] {ref} ...", end=" ", flush=True)

        # Resolve run dir from "sweep/model" ref
        parts = ref.split("/")
        sweep_name, model_name = parts[0], parts[1]
        sweep_dir = PROJECT / "checkpoints" / sweep_name
        run_dir = sweep_dir / model_name
        if not run_dir.exists():
            # fuzzy match
            cand = list((PROJECT / "checkpoints").glob(f"*{sweep_name}*/{model_name}"))
            if cand:
                run_dir = cand[0]
            else:
                print("not found")
                continue
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            ckpt_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            print("no ckpt")
            continue
        ckpt_path = ckpts[-1]

        try:
            from audi.checkpoint import strip_compile_prefix, get_clip_seconds
            from audi.config import MelConfig, ModelConfig, OptimizerConfig
            from audi.training.detector import DroneDetector

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
                mel_cfg = MelConfig(n_mels=mel_hp.get("n_mels", 128), n_fft=mel_hp.get("n_fft", 1024),
                                    hop_length=mel_hp.get("hop_length", 160))
            else:
                mel_cfg = mel_hp
            model = DroneDetector(model=model_cfg, mel=mel_cfg, optimizer=OptimizerConfig(),
                                  bin_names=hp.get("bin_names", []))
            model.load_state_dict(strip_compile_prefix(ckpt["state_dict"]), strict=False)
            model = model.to(device).eval()
            clip_s = get_clip_seconds(hp)
            del ckpt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"load failed: {e}")
            continue

        # Run inference on alert files only (load audio on-demand to save memory)
        alert_scores_per_file = []
        for dir_name, wav_path, label in tqdm.tqdm(alert_file_paths, desc="Processing alert files"):
            try:
                audio, sr = torchaudio.load(str(wav_path))
            except Exception:
                continue
            audio = audio.mean(dim=0).numpy().astype(np.float32)
            wins = split_into_windows(audio, clip_s)
            if wins:
                scores = predict(model, np.stack(wins))
                alert_scores_per_file.append((dir_name, scores, label))

        # Evaluate at each P level
        thresholds = model_thresholds.get(ref, {})
        for lvl in LEVELS:
            sigma = thresholds.get(lvl)
            if sigma is None:
                continue

            # Alert TP/FP/FN using manual labels
            alert_tp = 0
            alert_fn = 0
            alert_fp = 0
            alert_tn = 0
            for dir_name, scores, label in alert_scores_per_file:
                dets = apply_hysteresis(scores, sigma)
                a = count_alerts(dets)
                has_alert = a > 0
                if label == "drone":
                    if has_alert:
                        alert_tp += 1
                    else:
                        alert_fn += 1
                else:  # nodrone
                    if has_alert:
                        alert_fp += 1
                    else:
                        alert_tn += 1
            alert_total = alert_tp + alert_fn + alert_fp + alert_tn

            results.append({
                "ref": ref, "model": model_name, "sweep": sweep_name, "P": lvl, "sigma": round(sigma, 4),
                "alert_tp": alert_tp, "alert_fn": alert_fn,
                "alert_fp": alert_fp, "alert_tn": alert_tn,
                "alert_total": alert_total,
            })

        del model
        torch.cuda.empty_cache()
        flush_results()
        print(f"done")

    # ── Print results table ─────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("FIELD RECORDING RESULTS — TP/FP at each P level")
    print(f"{'='*120}")
    print(f"{'model':<35}", end="")
    for lvl in LEVELS:
        print(f"  {lvl:>12}  ", end="")
    print()
    print(f"{'':35}", end="")
    for lvl in LEVELS:
        print(f" TP FP FN", end="")
    print()
    print("-" * 80)

    # Group by ref
    from collections import defaultdict
    by_model = defaultdict(dict)
    for r in results:
        p = r["P"]
        by_model[r["ref"]][p] = r

    for ref in model_order:
        print(f"{ref:<35}", end="")
        for lvl in LEVELS:
            r = by_model[ref].get(lvl)
            if r:
                alert_tp = r.get("alert_tp", 0)
                alert_fp = r.get("alert_fp", 0)
                alert_fn = r.get("alert_fn", 0)
                print(f" {alert_tp:>3} {alert_fp:>3} {alert_fn:>3}", end="")
            else:
                print(f" {'--/--/--':>9}", end="")
        print()


if __name__ == "__main__":
    main()
