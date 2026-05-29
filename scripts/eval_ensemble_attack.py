"""Ensemble evaluation: combine multiple models for better attack detection.

Pipeline:
  1. Load validation logits from predictions_best.pt for selected models
  2. K-fold cross-validation on validation data to fit & calibrate meta-models
  3. Load models, run inference on attack audio for each model
  4. Apply ensemble strategies (avg, max, product, vote, MLP, XGBoost, ...) 
  5. Evaluate across all P levels: cov%, 1st%, bg, alerts

Usage:
    uv run python scripts/eval_ensemble_attack.py \
        mn_sweep_v6_20260527_132707/15_mn05_pcen_aug \
        mn_sweep_v6_20260527_132707/04_mn04_aug_smooth_wd003 \
        mn_sweep_v6_20260527_132707/02_mn04_aug \
        ... (3-10 models)

    uv run python scripts/eval_ensemble_attack.py --top 5  # top 5 by P90 cov
    uv run python scripts/eval_ensemble_attack.py --top 10 --field
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[1]
SR = 32000
STRIDE = 0.125
LEVELS = ["P50", "P60", "P70", "P75", "P80", "P85", "P90", "P95", "P99"]
PRECISION_LEVELS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
K_FOLDS = 5


def main():
    ap = argparse.ArgumentParser(description="Ensemble attack eval")
    ap.add_argument("models", nargs="*", help="Model refs: sweep/model")
    ap.add_argument("--top", type=int, default=0, help="Use top N models by P90 cov (overrides positional)")
    ap.add_argument("--field", action="store_true", help="Also evaluate on field recordings")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for sklearn (-1 = all cores)")
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    # ── Resolve models ──────────────────────────────────────────────────
    from audi.checkpoint import strip_compile_prefix, get_clip_seconds
    from audi.config import MelConfig, ModelConfig, OptimizerConfig
    from audi.training.detector import DroneDetector
    from audi.hysteresis import apply_hysteresis
    from audi.training.validation import (
        compute_precision, compute_roc_values, find_threshold_at_precision,
    )

    CSV_PATH = PROJECT / "checkpoints" / "attack_run_precision_eval.csv"
    model_refs = args.models

    if args.top > 0:
        # Pick top N models by P90 cov from CSV — only models that exist on disk
        best = {}
        with open(CSV_PATH) as f:
            for r in csv.DictReader(f):
                if r["precision"] != "P90":
                    continue
                key = f"{r['sweep']}/{r['model']}"
                # Verify predictions file exists
                sweep_dir = PROJECT / "checkpoints" / r["sweep"]
                run_dir = sweep_dir / r["model"]
                pp_file = run_dir / "eval_data" / "predictions_best.pt"
                if not pp_file.exists():
                    # Try parent search
                    d = run_dir
                    while d.parent != d and d.parent != PROJECT:
                        pf = d / "eval_data" / "predictions_best.pt"
                        if pf.exists():
                            pp_file = pf
                            break
                        d = d.parent
                if pp_file.exists():
                    best[key] = float(r["cov_pct"])
        model_refs = sorted(best, key=best.get, reverse=True)[:args.top]
        print(f"Top {args.top} by P90 cov: {model_refs}")

    if len(model_refs) < 3:
        print("Need at least 3 models")
        sys.exit(1)
    if len(model_refs) > 10:
        print(f"Limiting to 10 models (got {len(model_refs)})")
        model_refs = model_refs[:10]

    # ── Phase 0: Resolve checkpoints and load validation data ────────────
    print(f"\n{'='*60}\nPhase 0: Loading {len(model_refs)} models\n{'='*60}")

    model_info = []  # (sweep, name, ckpt_path, val_logits, val_labels)
    for ref in model_refs:
        parts = ref.split("/")
        if len(parts) != 2:
            print(f"  SKIP invalid ref: {ref}")
            continue
        sweep, name = parts
        run_dir = PROJECT / "checkpoints" / sweep / name
        if not run_dir.exists():
            # try fuzzy
            cand = list((PROJECT / "checkpoints").glob(f"*{sweep}*/{name}"))
            if cand:
                run_dir = cand[0]
            else:
                print(f"  SKIP not found: {ref}")
                continue
        ckpt_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
        if not ckpt_dir.exists():
            ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            print(f"  SKIP no checkpoints: {ref}")
            continue
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            print(f"  SKIP no ckpt: {ref}")
            continue
        ckpt_path = ckpts[-1]

        # Load validation predictions
        pred_file = run_dir / "eval_data" / "predictions_best.pt"
        if not pred_file.exists():
            # Try finding it via parent search
            d = run_dir
            while d.parent != d:
                pf = d / "eval_data" / "predictions_best.pt"
                if pf.exists():
                    pred_file = pf
                    break
                d = d.parent
        if not pred_file.exists():
            print(f"  SKIP no predictions: {ref}")
            continue

        pred_data = torch.load(str(pred_file), map_location="cpu", weights_only=False)
        val_logits = np.asarray(pred_data["logits"]).flatten()
        val_labels = np.asarray(pred_data["labels"]).flatten()

        model_info.append({
            "ref": ref, "sweep": sweep, "name": name,
            "ckpt_path": str(ckpt_path),
            "val_logits": val_logits, "val_labels": val_labels,
        })
        print(f"  {ref}: {len(val_logits)} val samples, ckpt={ckpt_path.name}")

    N = len(model_info)
    print(f"\nLoaded {N} models")

    # ── Phase 1: K-fold CV on validation data ───────────────────────────
    print(f"\n{'='*60}\nPhase 1: K-fold CV (k={K_FOLDS}) on validation data\n{'='*60}")

    # Stack logits into feature matrix
    X = np.column_stack([m["val_logits"] for m in model_info])  # (n_samples, N_models)
    y = model_info[0]["val_labels"]  # all should be identical

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    # Ensemble strategies to evaluate
    strategies = {}

    # Simple averaging of scores
    def make_avg(X_test):
        return X_test.mean(axis=1)
    strategies["avg"] = ("Average score", make_avg, None)

    def make_max(X_test):
        return X_test.max(axis=1)
    strategies["max"] = ("Max score", make_max, None)

    def make_product(X_test):
        return np.prod(X_test, axis=1)
    strategies["product"] = ("Product", make_product, None)

    def make_vote_and(X_test, sigma=0.5):
        return np.all(X_test > sigma, axis=1).astype(float)
    strategies["vote_and"] = ("AND vote (σ=0.5)", lambda x: make_vote_and(x), None)

    def make_vote_or(X_test, sigma=0.5):
        return np.any(X_test > sigma, axis=1).astype(float)
    strategies["vote_or"] = ("OR vote (σ=0.5)", lambda x: make_vote_or(x), None)

    # Learned models with CV calibration
    learned = {
        "lr": LogisticRegression(max_iter=5000, C=1.0),
        "lr_l2": LogisticRegression(max_iter=5000, C=0.1, penalty='l2'),
        "mlp": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, early_stopping=True, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=args.n_jobs),
        "gb": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
    }

    # CV results: strategy -> {P_level: {metric: [fold_values]}}
    cv_results = {name: {lvl: {"tpr": [], "prec": []} for lvl in LEVELS} for name in list(strategies) + list(learned)}

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # ── Simple strategies ──
        for name, (label, fn, _) in strategies.items():
            scores_train = fn(X_train)
            scores_val = fn(X_val)
            # Find thresholds that achieve each precision level on train
            fpr, tpr, th, _ = compute_roc_values(scores_train, y_train)
            prec = compute_precision(scores_train, y_train, th)
            for lvl, pt in zip(LEVELS, PRECISION_LEVELS):
                if pt < prec.min() or pt > prec.max():
                    continue
                th_pt, tp_pt, _ = find_threshold_at_precision(prec, tpr, th, pt)
                # Apply to val fold
                val_preds = (scores_val > th_pt).astype(int)
                cv_results[name][lvl]["tpr"].append(float((val_preds & (y_val > 0.5)).sum() / max(1, (y_val > 0.5).sum())))
                cv_results[name][lvl]["prec"].append(float(tp_pt))

        # ── Learned models ──
        for name, model in learned.items():
            try:
                model_clone = type(model)(**model.get_params())
                model_clone.fit(X_train, y_train)
                scores_val = model_clone.predict_proba(X_val)[:, 1]
                fpr, tpr, th, _ = compute_roc_values(scores_val, y_val)
                prec = compute_precision(scores_val, y_val, th)
                for lvl, pt in zip(LEVELS, PRECISION_LEVELS):
                    if pt < prec.min() or pt > prec.max():
                        continue
                    th_pt, tp_pt, _ = find_threshold_at_precision(prec, tpr, th, pt)
                    val_preds = (scores_val > th_pt).astype(int)
                    cv_results[name][lvl]["tpr"].append(float((val_preds & (y_val > 0.5)).sum() / max(1, (y_val > 0.5).sum())))
                    cv_results[name][lvl]["prec"].append(float(tp_pt))
            except Exception as e:
                pass

    # Print CV summary
    print(f"\n{'Strategy':<20}", end="")
    for lvl in LEVELS:
        print(f"  {lvl:>8}", end="")
    print()
    print("-" * (20 + 10 * len(LEVELS)))
    for name in list(strategies) + list(learned):
        print(f"{name:<20}", end="")
        for lvl in LEVELS:
            tprs = cv_results[name][lvl]["tpr"]
            if tprs:
                mean_tpr = np.mean(tprs)
                print(f"  {mean_tpr:>7.1f}%", end="")
            else:
                print(f"  {'--':>8}", end="")
        print()

    # ── Phase 2: Fit final meta-models on all validation data ────────────
    print(f"\n{'='*60}\nPhase 2: Fit final meta-models + calibrate thresholds\n{'='*60}")

    # Fit learned models on all validation data
    fitted_learned = {}
    for name, model in learned.items():
        try:
            m = type(model)(**model.get_params())
            m.fit(X, y)
            fitted_learned[name] = m
        except Exception as e:
            print(f"  {name}: fit failed - {e}")

    # Compute thresholds for ALL strategies on full validation data
    all_thresholds = {}  # strategy_name -> {P_level: threshold}

    # Simple strategies
    for name, (label, fn, _) in strategies.items():
        scores = fn(X)
        fpr, tpr, th, _ = compute_roc_values(scores, y)
        prec = compute_precision(scores, y, th)
        thresholds = {}
        for lvl, pt in zip(LEVELS, PRECISION_LEVELS):
            if pt < prec.min() or pt > prec.max():
                continue
            th_pt, _, _ = find_threshold_at_precision(prec, tpr, th, pt)
            thresholds[lvl] = th_pt  # logit threshold
        all_thresholds[name] = thresholds
        print(f"  {name}: {len(thresholds)} P levels calibrated")

    # Learned strategies
    for name, model in fitted_learned.items():
        scores = model.predict_proba(X)[:, 1]
        fpr, tpr, th, _ = compute_roc_values(scores, y)
        prec = compute_precision(scores, y, th)
        thresholds = {}
        for lvl, pt in zip(LEVELS, PRECISION_LEVELS):
            if pt < prec.min() or pt > prec.max():
                continue
            th_pt, _, _ = find_threshold_at_precision(prec, tpr, th, pt)
            thresholds[lvl] = th_pt
        all_thresholds[name] = thresholds
        print(f"  {name}: {len(thresholds)} P levels calibrated")

    # ── Phase 3: Run inference on attack audio ──────────────────────────
    print(f"\n{'='*60}\nPhase 3: Attack audio inference\n{'='*60}")

    ATTACK_DIR = PROJECT / "data" / "attack_runs"
    audio_waveforms = {}
    for fp in sorted(ATTACK_DIR.glob("*.wav")):
        audio, sr = torchaudio.load(str(fp))
        audio_waveforms[fp.name] = audio.mean(dim=0).numpy().astype(np.float32)

    bg_names = sorted([n for n in audio_waveforms if n.startswith("background")])
    atk_names = sorted([n for n in audio_waveforms if not n.startswith("background")])

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

    def count_alerts(dets):
        if len(dets) == 0:
            return 0
        padded = np.pad(dets.astype(np.int8), (1, 0), constant_values=0)
        return int(np.sum((padded[1:] == 1) & (padded[:-1] == 0)))

    @torch.no_grad()
    def predict_windows(model, windows, batch_size=32):
        scores = []
        for i in range(0, len(windows), batch_size):
            batch = torch.as_tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch).cpu().numpy()
            scores.append(1.0 / (1.0 + np.exp(-logits)))
        return np.concatenate(scores).flatten() if scores else np.array([])

    # Load each model and run inference
    all_atk_scores = {}  # model_ref -> {seg_name: scores_array}
    all_bg_scores = {}   # model_ref -> bg_scores_array
    clip_s = 5.12  # default

    for mi in model_info:
        ref = mi["ref"]
        print(f"  Loading {ref} ...", end=" ", flush=True)
        ckpt = torch.load(mi["ckpt_path"], map_location="cpu", weights_only=False)
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

        # Attack segments
        atk_data = {}
        for name in atk_names:
            audio = audio_waveforms[name]
            segs = split_by_zero_gaps(audio)
            for si, seg in enumerate(segs):
                wins = split_into_windows(seg, clip_s)
                if not wins:
                    atk_data[f"{name}_seg{si}"] = np.array([])
                else:
                    atk_data[f"{name}_seg{si}"] = predict_windows(model, np.stack(wins))
        all_atk_scores[ref] = atk_data

        # Background
        bg_wins = []
        for name in bg_names:
            bg_wins.extend(split_into_windows(audio_waveforms[name], clip_s))
        all_bg_scores[ref] = predict_windows(model, np.stack(bg_wins)) if bg_wins else np.array([])

        del model
        torch.cuda.empty_cache()
        print(f"done ({len(atk_data)} segs, {len(all_bg_scores[ref])} bg wins)")

    bg_total = len(all_bg_scores[list(all_bg_scores.keys())[0]])
    # Verify all models have same bg length
    for ref in model_refs:
        if ref in all_bg_scores:
            assert len(all_bg_scores[ref]) == bg_total, f"bg length mismatch for {ref}"
        else:
            print(f"  WARNING: {ref} not in all_bg_scores, skipping")
            model_refs = [r for r in model_refs if r in all_bg_scores]

    # ── Phase 4: Apply ensemble strategies to attack data ────────────────
    print(f"\n{'='*60}\nPhase 4: Ensemble evaluation\n{'='*60}")

    seg_names = sorted(all_atk_scores[model_refs[0]].keys())

    def evaluate_strategy(name, get_scores_fn, is_prob=False):
        """Evaluate one ensemble strategy across all P levels."""
        # Get ensemble scores for each segment
        seg_scores = {}
        for sname in seg_names:
            per_model = np.column_stack([
                all_atk_scores[ref].get(sname, np.zeros(1))
                for ref in model_refs
            ])
            if per_model.shape[0] == 0:
                seg_scores[sname] = np.array([])
            else:
                seg_scores[sname] = get_scores_fn(per_model)

        # Background ensemble scores
        bg_per_model = np.column_stack([all_bg_scores[ref] for ref in model_refs])
        bg_scores = get_scores_fn(bg_per_model)

        results = []
        thresholds = all_thresholds.get(name, {})

        for lvl, pt in zip(LEVELS, PRECISION_LEVELS):
            if lvl not in thresholds:
                continue
            th = thresholds[lvl]
            sigma = th if is_prob else 1.0 / (1.0 + np.exp(-th))

            # Background
            bg_dets = apply_hysteresis(bg_scores, sigma)
            bg_alerts = count_alerts(bg_dets)
            bg_count = int(bg_dets.sum())

            # Attack segments
            seg_covs, seg_firsts = [], []
            for sname in seg_names:
                scores = seg_scores[sname]
                if len(scores) == 0:
                    seg_covs.append(0.0)
                    seg_firsts.append(100.0)
                    continue
                dets = apply_hysteresis(scores, sigma)
                cov = 100.0 * dets.sum() / len(dets)
                det_idx = np.where(dets)[0]
                first = 100.0 * det_idx[0] / len(dets) if len(det_idx) > 0 else 100.0
                seg_covs.append(cov)
                seg_firsts.append(first)

            results.append({
                "strategy": name, "precision": lvl, "sigma": round(sigma, 4),
                "cov_pct": round(np.mean(seg_covs), 1),
                "first_pct": round(np.median(seg_firsts), 1),
                "bg": bg_count, "bg_alerts": bg_alerts,
            })

        return results

    all_results = []

    # Simple strategies
    for name, (label, fn, _) in strategies.items():
        print(f"  {name} ...", end=" ", flush=True)
        results = evaluate_strategy(name, fn)
        all_results.extend(results)
        p90 = [r for r in results if r["precision"] == "P90"]
        if p90:
            r = p90[0]
            print(f"P90 cov={r['cov_pct']:.1f}% 1st={r['first_pct']:.1f}% bg={r['bg']}/{bg_total} alerts={r['bg_alerts']}")
        else:
            print("no P90")

    # Learned strategies
    for name, model in fitted_learned.items():
        print(f"  {name} ...", end=" ", flush=True)
        def make_learned_fn(m=model):
            return lambda X: m.predict_proba(X)[:, 1]
        results = evaluate_strategy(name, make_learned_fn(), is_prob=True)
        all_results.extend(results)
        p90 = [r for r in results if r["precision"] == "P90"]
        if p90:
            r = p90[0]
            print(f"P90 cov={r['cov_pct']:.1f}% 1st={r['first_pct']:.1f}% bg={r['bg']}/{bg_total} alerts={r['bg_alerts']}")
        else:
            print("no P90")

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("ENSEMBLE RESULTS (all P levels)")
    print(f"{'='*100}")
    print(f"{'Strategy':<20}", end="")
    for lvl in LEVELS:
        print(f"  {lvl:>8}  ", end="")
    print(f"  {'1st%':>5}  {'bg':>5}  {'alerts':>6}")
    print("-" * (20 + 16 * len(LEVELS) + 25))

    # Collect per-strategy
    strategy_summary = {}
    for r in all_results:
        key = r["strategy"]
        if key not in strategy_summary:
            strategy_summary[key] = {}
        strategy_summary[key][r["precision"]] = r

    for name in strategy_summary:
        print(f"{name:<20}", end="")
        p90 = strategy_summary[name].get("P90", {})
        for lvl in LEVELS:
            r = strategy_summary[name].get(lvl, {})
            cov = r.get("cov_pct", None)
            if cov is not None:
                print(f"  {cov:>5.1f}% {r['bg']:>4}", end="")
            else:
                print(f"  {'--':>6} {'?':>4}", end="")
        print(f"  {p90.get('first_pct', -1):>5.1f}  {p90.get('bg', -1):>5}  {p90.get('bg_alerts', '-'):>6}")


if __name__ == "__main__":
    main()
