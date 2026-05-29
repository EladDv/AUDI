"""Explore a model's predictions on attack audio and field recordings.

Usage:
    uv run python scripts/explore_model.py <sweep>/<model> [--sigma 0.7] [--field]
    
Examples:
    uv run python scripts/explore_model.py dsp_sweep_20260525_071739/07_all
    uv run python scripts/explore_model.py mel_sweep_20260521_125510/12_mels96_fft2048 --field
    uv run python scripts/explore_model.py bce_wd_sweep_20260518_162557/01_wd_003 --sigma 0.70
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


def main():
    ap = argparse.ArgumentParser(description="Explore model predictions on attack/field audio")
    ap.add_argument("model_ref", help="sweep/model reference, e.g. 'dsp_sweep_20260525_071739/07_all'")
    ap.add_argument("--sigma", type=float, default=None,
                    help="Hysteresis threshold (probability). Default: auto from P90 threshold")
    ap.add_argument("--field", action="store_true", help="Also evaluate on field recordings")
    ap.add_argument("--all-windows", action="store_true", help="Show every window score (very verbose)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    PROJECT = Path(__file__).resolve().parents[1]
    SR = 32000  # MelConfig default
    STRIDE = 0.125

    # ── resolve model ──────────────────────────────────────────────────
    parts = args.model_ref.split("/")
    if len(parts) != 2:
        print(f"ERROR: Expected 'sweep/model' format, got '{args.model_ref}'")
        sys.exit(1)
    sweep_name, model_name = parts

    sweep_dir = PROJECT / "checkpoints" / sweep_name
    run_dir = sweep_dir / model_name
    if not run_dir.exists():
        print(f"ERROR: Run dir not found: {run_dir}")
        # Try fuzzy match
        candidates = list((PROJECT / "checkpoints").glob(f"*{sweep_name}*/{model_name}"))
        if candidates:
            run_dir = candidates[0]
            print(f"  Using: {run_dir}")

    ckpt_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
    if not ckpt_dir.exists():
        ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        print(f"ERROR: No checkpoints dir in {run_dir}")
        sys.exit(1)
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        print(f"ERROR: No .ckpt files in {ckpt_dir}")
        sys.exit(1)
    ckpt_path = ckpts[-1]  # last epoch
    print(f"Model: {sweep_name}/{model_name}")
    print(f"Checkpoint: {ckpt_path.name}")

    # ── load model ─────────────────────────────────────────────────────
    from audi.checkpoint import strip_compile_prefix, get_clip_seconds
    from audi.config import MelConfig, ModelConfig, OptimizerConfig
    from audi.training.detector import DroneDetector
    from audi.hysteresis import apply_hysteresis

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
    win_samples = int(SR * clip_s)
    step_samples = int(win_samples * STRIDE)
    del ckpt
    torch.cuda.empty_cache()

    print(f"Clip: {clip_s}s, stride: {STRIDE} ({step_samples} samples)")
    print(f"Arch: {model_cfg.arch}, n_mels: {mel_cfg.n_mels}, hop: {mel_cfg.hop_length}")

    # ── determine sigma ────────────────────────────────────────────────
    if args.sigma is None:
        # Try to load P90 sigma from attack CSV
        import csv
        csv_path = PROJECT / "checkpoints" / "attack_run_precision_eval.csv"
        sigma = None
        if csv_path.exists():
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    if row["sweep"] == sweep_name and row["model"] == model_name and row["precision"] == "P90":
                        sigma = float(row["sigma"])
                        break
        if sigma is None:
            sigma = 0.7
            print(f"No P90 sigma found, defaulting to {sigma}")
        else:
            print(f"Sigma: {sigma:.4f} (from P90 threshold)")
    else:
        sigma = args.sigma
        print(f"Sigma: {sigma:.4f} (user-specified)")

    # ── helper functions ───────────────────────────────────────────────
    def split_into_windows(audio, clip_s):
        win = int(SR * clip_s)
        step = int(win * STRIDE)
        if len(audio) < win:
            return []
        return [audio[i:i+win] for i in range(0, len(audio) - win + 1, step)]

    def split_by_zero_gaps(audio, min_dur=3.0, min_gap_s=0.5):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        exact_zero = audio == 0.0
        zero_runs = []
        in_zero, start = False, 0
        for i in range(len(exact_zero) + 1):
            z = bool(exact_zero[i]) if i < len(exact_zero) else False
            if z and not in_zero:
                start, in_zero = i, True
            elif not z and in_zero:
                if (i - start) / SR >= min_gap_s:
                    zero_runs.append((start, i))
                in_zero = False
        if not zero_runs:
            return [audio] if len(audio) / SR >= min_dur else []
        segments, prev = [], 0
        for zs, ze in zero_runs:
            if (zs - prev) / SR >= min_dur:
                segments.append(audio[prev:zs].copy())
            prev = ze
        if (len(audio) - prev) / SR >= min_dur:
            segments.append(audio[prev:].copy())
        return segments

    @torch.no_grad()
    def predict(windows, batch_size=32):
        scores = []
        for i in range(0, len(windows), batch_size):
            batch = torch.as_tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch).cpu().numpy()
            scores.append(1.0 / (1.0 + np.exp(-logits)))
        return np.concatenate(scores).flatten() if scores else np.array([])

    def count_alerts(dets):
        if len(dets) == 0:
            return 0
        padded = np.pad(dets.astype(np.int8), (1, 0), constant_values=0)
        return int(np.sum((padded[1:] == 1) & (padded[:-1] == 0)))

    # ── ATTACK AUDIO ───────────────────────────────────────────────────
    ATTACK_DIR = PROJECT / "data" / "attack_runs"
    print(f"\n{'='*80}")
    print("ATTACK AUDIO ANALYSIS")
    print(f"{'='*80}")

    audio_waveforms = {}
    for fp in sorted(ATTACK_DIR.glob("*.wav")):
        audio, sr = torchaudio.load(str(fp))
        audio_waveforms[fp.name] = audio.mean(dim=0).numpy().astype(np.float32)

    # Separate bg and attack
    bg_names = sorted([n for n in audio_waveforms if n.startswith("background")])
    atk_names = sorted([n for n in audio_waveforms if not n.startswith("background")])

    # Background analysis
    print(f"\n── Background files ({len(bg_names)}) ──")
    all_bg_windows = []
    for name in bg_names:
        audio = audio_waveforms[name]
        windows = split_into_windows(audio, clip_s)
        all_bg_windows.extend(windows)
    if all_bg_windows:
        bg_scores = predict(np.stack(all_bg_windows))
        bg_dets = apply_hysteresis(bg_scores, sigma)
        bg_alerts = count_alerts(bg_dets)
        print(f"  Total windows: {len(bg_scores)}")
        print(f"  Raw scores > 0.5: {(bg_scores > 0.5).sum()}")
        print(f"  Detections (after hysteresis, σ={sigma:.4f}): {bg_dets.sum()} / {len(bg_dets)} ({100*bg_dets.sum()/len(bg_dets):.1f}%)")
        print(f"  Alerts (contiguous runs): {bg_alerts}")
        print(f"  Score range: [{bg_scores.min():.4f}, {bg_scores.max():.4f}]")
        print(f"  Score mean/median: {bg_scores.mean():.4f} / {np.median(bg_scores):.4f}")
        # Distribution
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(bg_scores, bins=bins)
        print(f"  Score distribution:")
        for i in range(len(bins)-1):
            bar = "█" * int(40 * hist[i] / max(hist))
            print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:>5d} {bar}")

        # Per-file breakdown
        print(f"\n  Per-file:")
        offset = 0
        for name in bg_names:
            n_win = len(split_into_windows(audio_waveforms[name], clip_s))
            file_scores = bg_scores[offset:offset+n_win]
            file_dets = apply_hysteresis(file_scores, sigma)
            file_alerts = count_alerts(file_dets)
            print(f"    {name:<40} {n_win:>4d} windows, {file_dets.sum():>4d} det, {file_alerts:>2d} alerts, max={file_scores.max():.3f}")
            offset += n_win

    # Attack analysis
    print(f"\n── Attack files ({len(atk_names)}) ──")
    total_segs = 0
    covered_segs = 0
    all_cov_pcts = []
    all_first_pcts = []
    for name in atk_names:
        audio = audio_waveforms[name]
        segs = split_by_zero_gaps(audio)
        if not segs:
            continue
        seg_covs = []
        seg_firsts = []
        for si, seg in enumerate(segs):
            total_segs += 1
            wins = split_into_windows(seg, clip_s)
            if not wins:
                continue
            scores = predict(np.stack(wins))
            dets = apply_hysteresis(scores, sigma)
            cov = 100.0 * dets.sum() / len(dets)
            det_idx = np.where(dets)[0]
            first = 100.0 * det_idx[0] / len(dets) if len(det_idx) > 0 else 100.0
            seg_covs.append(cov)
            seg_firsts.append(first)
            if cov > 0:
                covered_segs += 1

            if args.all_windows:
                print(f"\n  {name} seg{si}: {len(wins)} windows, cov={cov:.1f}%, first={first:.1f}%")
                for wi, s in enumerate(scores):
                    marker = "▐" if apply_hysteresis(np.array([s]), sigma)[0] else " "
                    bar = "█" * int(30 * s)
                    print(f"    [{wi:>4d}] {marker} {s:.3f} {bar}")

        seg_covs = np.array(seg_covs) if seg_covs else np.array([0.0])
        seg_firsts = np.array(seg_firsts) if seg_firsts else np.array([100.0])
        all_cov_pcts.extend(seg_covs.tolist())
        all_first_pcts.extend(seg_firsts.tolist())
        n_segs = len(segs)
        covered = (seg_covs > 0).sum()
        print(f"  {name:<40} {n_segs:>2d} segs, {covered:>2d} covered, "
              f"cov={seg_covs.mean():.0f}% (median {np.median(seg_covs):.0f}%), "
              f"first={np.median(seg_firsts):.0f}%")

    all_cov = np.array(all_cov_pcts)
    all_first = np.array(all_first_pcts)
    print(f"\n  Summary: {len(all_cov)} attack segments")
    print(f"  Coverage: mean={all_cov.mean():.1f}%, median={np.median(all_cov):.1f}%")
    print(f"  First detection: median={np.median(all_first):.1f}%")
    print(f"  Segments with any detection: {covered_segs}/{total_segs} ({100*covered_segs/max(1,total_segs):.1f}%)")
    print(f"  Segments with 100% coverage: {(all_cov == 100).sum()}")

    # ── FIELD RECORDINGS ───────────────────────────────────────────────
    if args.field:
        FIELD_DIR = PROJECT / "data" / "field_recordings_20260514"
        print(f"\n{'='*80}")
        print("FIELD ALERTS ANALYSIS")
        print(f"{'='*80}")

        # Alert recordings — use manual labels from labels.csv
        alert_dir = FIELD_DIR / "alerts"
        labels_csv = FIELD_DIR / "labels.csv"
        alert_labels = {}
        if labels_csv.exists():
            with open(labels_csv) as f:
                for r in csv.DictReader(f):
                    alert_labels[r["alert_dir"]] = r["label"]

        print(f"\n── Field alerts (manual labels: {sum(1 for v in alert_labels.values() if v=='drone')} drone, {sum(1 for v in alert_labels.values() if v=='nodrone')} nodrone) ──")
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for d in sorted(alert_dir.iterdir()):
            if not d.is_dir():
                continue
            fp_wav = d / "full_120s.wav"
            if not fp_wav.exists() or fp_wav.stat().st_size == 0:
                continue
            label = alert_labels.get(d.name, "drone")
            try:
                audio, sr = torchaudio.load(str(fp_wav))
            except Exception as e:
                print(f"  ⚠ skip {d.name}: {e}")
                continue
            audio = audio.mean(dim=0).numpy().astype(np.float32)
            windows = split_into_windows(audio, clip_s)
            if not windows:
                continue
            scores = predict(np.stack(windows))
            dets = apply_hysteresis(scores, sigma)
            alerts = count_alerts(dets)
            max_score = scores.max()
            has_alert = alerts > 0
            if label == "drone":
                if has_alert:
                    tp += 1
                    status = "✓ TP"
                else:
                    fn += 1
                    status = "✗ FN"
            else:
                if has_alert:
                    fp += 1
                    status = "✗ FP"
                else:
                    tn += 1
                    status = "✓ TN"
            print(f"  {status} {d.name:<30} {len(windows):>4d} windows, "
                  f"{dets.sum():>4d} det, {alerts:>2d} alerts, max_score={max_score:.3f}")
        print(f"\n  Summary: TP={tp} FP={fp} FN={fn} TN={tn}  "
              f"recall={100*tp/max(1,tp+fn):.1f}%  precision={100*tp/max(1,tp+fp):.1f}%")

    print(f"\n{'='*80}")
    print("Done!")


if __name__ == "__main__":
    main()
