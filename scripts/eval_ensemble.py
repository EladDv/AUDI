#!/usr/bin/env python3
"""Meta-model ensemble evaluation — load models, run on shared dataset, ensemble.

Usage:
    uv run python scripts/eval_ensemble.py \\
        --noise-path data/HF_dataset_v2_background \\
        --drone-path data/HF_dataset_v2_drone \\
        checkpoints/multinoise_20260510_223154
"""

from __future__ import annotations

import warnings
from pathlib import Path

import lightning as L
import numpy as np
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader

from audi.checkpoint import strip_compile_prefix  # noqa: E402
from audi.cli_utils import NUM_WORKERS  # noqa: E402
from audi.config import (  # noqa: E402
    MelConfig,
    MixConfig,
    ModelConfig,
    OptimizerConfig,
    parse_snr_bins,
)
from audi.training.dataset import make_dataset  # noqa: E402
from audi.training.detector import DroneDetector  # noqa: E402
from audi.training.validation import (  # noqa: E402
    compute_pr_curve,
    compute_precision,
    compute_roc_values,
    find_threshold_at_precision,
)

_PROJECT = Path(__file__).resolve().parents[1]


def _discover_best_checkpoints(sweep_dir: Path) -> list[dict]:
    """Find the best (highest-epoch) checkpoint for each run in a sweep."""
    raw = []
    for ckpt_path in sorted(sweep_dir.rglob("*.ckpt")):
        parts = list(ckpt_path.relative_to(sweep_dir).parts)
        exp_dir = None
        # TB-direct layout: run/checkpoints/epoch=*.ckpt
        for i, p in enumerate(parts):
            if p == "checkpoints" and i >= 1:
                exp_dir = parts[i - 1]
                break
        if not exp_dir:
            continue
        epoch = 0
        if "epoch=" in ckpt_path.stem:
            try:
                epoch = int(ckpt_path.stem.split("epoch=")[1].split("-")[0])
            except ValueError:
                pass
        raw.append({"path": str(ckpt_path), "exp": exp_dir, "epoch": epoch})
    best = {}
    for c in sorted(raw, key=lambda c: c["epoch"], reverse=True):
        if c["exp"] not in best:
            best[c["exp"]] = c
    return sorted(best.values(), key=lambda c: c["exp"])


def _load_model(ckpt_path: str, device: str) -> DroneDetector:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    model_hp = hp.get("model", {})
    if isinstance(model_hp, dict):
        model_cfg = ModelConfig(
            arch=model_hp.get("arch", hp.get("model_arch", "cnn14")),
            pretrained=model_hp.get(
                "pretrained", hp.get("pretrained_backbone", True)
            ),
            compile=False,
        )
    else:
        model_cfg = ModelConfig(
            arch=model_hp.arch, pretrained=model_hp.pretrained, compile=False,
        )
    mel_hp = hp.get("mel", {})
    if isinstance(mel_hp, dict):
        mel_cfg = MelConfig(
            n_mels=mel_hp.get("n_mels", hp.get("n_mels", 128)),
            n_fft=mel_hp.get("n_fft", hp.get("n_fft", 1024)),
            hop_length=mel_hp.get("hop_length", hp.get("hop_length", 160)),
            mean_db=mel_hp.get("mean_db", hp.get("mel_mean")),
            std_db=mel_hp.get("std_db", hp.get("mel_std")),
        )
    else:
        mel_cfg = MelConfig(
            n_mels=getattr(mel_hp, "n_mels", hp.get("n_mels", 128)),
            n_fft=getattr(mel_hp, "n_fft", hp.get("n_fft", 1024)),
            hop_length=getattr(mel_hp, "hop_length", hp.get("hop_length", 160)),
            mean_db=getattr(mel_hp, "mean_db", hp.get("mel_mean")),
            std_db=getattr(mel_hp, "std_db", hp.get("mel_std")),
        )
    opt_cfg = OptimizerConfig(lr=1e-3)
    model = DroneDetector(
        model=model_cfg, mel=mel_cfg, optimizer=opt_cfg,
        bin_names=hp.get("bin_names", []),
        loss_type=hp.get("loss_type", "bce"),
        label_smoothing=hp.get("label_smoothing", 0.0),
        per_bin_weights=hp.get("per_bin_weights", False),
        spec_augment_prob=float(hp.get("spec_augment_prob", 0.0)),
        mixup_alpha=hp.get("mixup_alpha", 0.0),
        cutmix_alpha=hp.get("cutmix_alpha", 0.0),
        dropout=hp.get("dropout", 0.0),
        bn_momentum=hp.get("bn_momentum", 0.1),
    )
    model.load_state_dict(
        strip_compile_prefix(ckpt["state_dict"]), strict=False,
    )
    return model.to(device).eval()


def _evaluate_model(y_true, y_score, label):
    fpr, tpr, th, auc = compute_roc_values(y_score, y_true)
    prec = compute_precision(y_score, y_true, th)
    _, tpr_p90, th_p90 = find_threshold_at_precision(prec, tpr, th, 0.90)
    _, _, _, ap = compute_pr_curve(y_score, y_true)
    return {
        "label": label, "auc": float(auc), "ap": float(ap),
        "tpr_at_p90": float(tpr_p90), "thresh_p90": float(th_p90),
    }


def _cross_val_evaluate(X, y, clf, label, k=5):
    metrics_all = []
    try:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))
    except ValueError:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        splits = list(kf.split(X))
        warnings.warn(f"KFold fallback for {label}")
    for train_idx, val_idx in splits:
        clf_i = clone(clf)
        clf_i.fit(X[train_idx], y[train_idx])
        y_score = clf_i.predict_proba(X[val_idx])[:, 1]
        metrics_all.append(_evaluate_model(y[val_idx], y_score, label))
    return {
        "label": label,
        "auc": np.mean([m["auc"] for m in metrics_all]),
        "auc_std": np.std([m["auc"] for m in metrics_all]),
        "ap": np.mean([m["ap"] for m in metrics_all]),
        "ap_std": np.std([m["ap"] for m in metrics_all]),
        "tpr_at_p90": np.mean([m["tpr_at_p90"] for m in metrics_all]),
        "tpr_std": np.std([m["tpr_at_p90"] for m in metrics_all]),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-path", required=True)
    ap.add_argument("--drone-path", required=True)
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--dataset-length", type=int, default=3200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--attack-runs", action="store_true",
                    help="Also evaluate on attack-run audio")
    ap.add_argument("--stride", type=float, default=0.3,
                    help="Sliding window stride fraction (default: 0.3)")
    ap.add_argument("sweep_dir")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    device = args.device

    # Discover checkpoints
    ckpts = _discover_best_checkpoints(sweep_dir)
    if len(ckpts) < 2:
        print(f"Need ≥2 checkpoints, found {len(ckpts)}")
        return 1

    m_names = [c["exp"] for c in ckpts]
    print(f"Models: {', '.join(m_names)}")

    # ── Build validation dataset (fixed seed) ───────────────────────
    L.seed_everything(42)
    snr_bins = parse_snr_bins([
        "easy:-5:0:0.20", "medium:-10:-5:0.20", "hard:-15:-10:0.20",
        "very_hard:-20:-15:0.15", "extreme:-25:-20:0.15",
        "far_field:-30:-25:0.10",
    ])
    mix_cfg = MixConfig(
        noise_path=args.noise_path, drone_path=args.drone_path,
        snr_bins=snr_bins,
        target_length_samples=int(16000 * 2.56),
        dataset_length=args.dataset_length,
    )
    val_ds = make_dataset(cfg=mix_cfg, split="validation", return_components=False)
    val_dl = DataLoader(val_ds, batch_size=32, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Validation: {len(val_ds)} samples")

    # ── Run all models on shared batches ────────────────────────────
    all_logits = {name: [] for name in m_names}
    all_labels = []
    models = {}

    for ci, ckpt in enumerate(ckpts):
        name = ckpt["exp"]
        print(f"[{ci+1}/{len(ckpts)}] Loading {name} ...", end=" ", flush=True)
        models[name] = _load_model(ckpt["path"], device)
        print("done")

    print("\nRunning inference on shared validation set ...")
    with torch.no_grad():
        for batch_i, (wav, label) in enumerate(val_dl):
            wav = wav.to(device)
            all_labels.append(label.numpy())
            for name, model in models.items():
                logits = model(wav).cpu().numpy()
                all_logits[name].append(logits)

    y = np.concatenate(all_labels).flatten()
    X = np.column_stack([
        np.concatenate(all_logits[name]).flatten() for name in m_names
    ])
    assert X.shape[0] == y.shape[0], f"{X.shape[0]} != {y.shape[0]}"

    print(f"Dataset: {X.shape[0]} samples, {len(m_names)} features\n")

    # ── Individual models ───────────────────────────────────────────
    print(f"{'='*70}")
    print("INDIVIDUAL MODELS")
    print(f"{'='*70}")
    print(f"{'model':<30} {'AUC':>8} {'AP':>8} {'TPR@P90':>9}")
    print(f"{'-'*56}")
    individual = []
    for j, name in enumerate(m_names):
        m = _evaluate_model(y, X[:, j], name)
        individual.append(m)
        print(f"{name:<30} {m['auc']:>8.4f} {m['ap']:>8.4f} {m['tpr_at_p90']:>9.4f}")

    # ── Simple fusion ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SIMPLE FUSION")
    print(f"{'='*70}")
    print(f"{'method':<30} {'AUC':>8} {'AP':>8} {'TPR@P90':>9}")
    print(f"{'-'*56}")

    fusion = {}
    scores_sig = 1.0 / (1.0 + np.exp(-X))  # sigmoid for vote

    fused = X.max(axis=1)
    fusion["fusion_max"] = _evaluate_model(y, fused, "fusion_max")
    fused = X.mean(axis=1)
    fusion["fusion_avg"] = _evaluate_model(y, fused, "fusion_avg")
    votes = (scores_sig > 0.5).mean(axis=1)
    fusion["fusion_vote"] = _evaluate_model(y, votes, "fusion_vote")

    # Weighted by individual AUC
    weights = np.array([m["auc"] for m in individual])
    weights = np.clip(weights - 0.5, 0, None)  # zero out below-chance
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
    fused = X @ weights
    fusion["fusion_w_auc"] = _evaluate_model(y, fused, "fusion_w_auc")

    for label in ["fusion_max", "fusion_avg", "fusion_vote", "fusion_w_auc"]:
        m = fusion[label]
        print(f"{m['label']:<30} {m['auc']:>8.4f} {m['ap']:>8.4f} {m['tpr_at_p90']:>9.4f}")

    # ── Meta-models (in-sample ref) ─────────────────────────────────
    print(f"\n{'='*70}")
    print("META-MODELS (full train, in-sample reference)")
    print(f"{'='*70}")
    print(f"{'model':<30} {'AUC':>8} {'AP':>8} {'TPR@P90':>9}")
    print(f"{'-'*56}")

    lr = LogisticRegression(max_iter=2000, random_state=42).fit(X, y)
    m = _evaluate_model(y, lr.predict_proba(X)[:, 1], "logistic_regression")
    print(f"{m['label']:<30} {m['auc']:>8.4f} {m['ap']:>8.4f} {m['tpr_at_p90']:>9.4f}")

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1).fit(X, y)
    m = _evaluate_model(y, rf.predict_proba(X)[:, 1], "random_forest")
    print(f"{m['label']:<30} {m['auc']:>8.4f} {m['ap']:>8.4f} {m['tpr_at_p90']:>9.4f}")

    try:
        import xgboost as xgb
        xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                   random_state=42, eval_metric="logloss").fit(X, y)
        m = _evaluate_model(y, xgb_c.predict_proba(X)[:, 1], "xgboost")
        print(f"{m['label']:<30} {m['auc']:>8.4f} {m['ap']:>8.4f} {m['tpr_at_p90']:>9.4f}")
    except ImportError:
        pass

    mlp = MLPClassifier((64, 32), activation="relu", max_iter=1000,
                         random_state=42, early_stopping=True).fit(X, y)
    m = _evaluate_model(y, mlp.predict_proba(X)[:, 1], "mlp")
    print(f"{m['label']:<30} {m['auc']:>8.4f} {m['ap']:>8.4f} {m['tpr_at_p90']:>9.4f}")

    # ── Cross-validated ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"META-MODELS ({args.k_folds}-fold CV)")
    print(f"{'='*70}")
    print(f"{'model':<30} {'AUC':>8} {'±std':>6} {'AP':>8} {'±std':>6} {'TPR@P90':>9} {'±std':>6}")
    print(f"{'-'*72}")

    cv = []
    for clf, label in [
        (LogisticRegression(max_iter=2000, random_state=42), "logistic_regression"),
        (RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1), "random_forest"),
    ]:
        m = _cross_val_evaluate(X, y, clf, label, args.k_folds)
        cv.append(m)
        print(f"{m['label']:<30} {m['auc']:>8.4f} {m['auc_std']:>6.4f} "
              f"{m['ap']:>8.4f} {m['ap_std']:>6.4f} {m['tpr_at_p90']:>9.4f} {m['tpr_std']:>6.4f}")

    try:
        import xgboost as xgb
        m = _cross_val_evaluate(X, y,
            xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                              random_state=42, eval_metric="logloss"),
            "xgboost", args.k_folds)
        cv.append(m)
        print(f"{m['label']:<30} {m['auc']:>8.4f} {m['auc_std']:>6.4f} "
              f"{m['ap']:>8.4f} {m['ap_std']:>6.4f} {m['tpr_at_p90']:>9.4f} {m['tpr_std']:>6.4f}")
    except ImportError:
        pass

    m = _cross_val_evaluate(X, y,
        MLPClassifier((64, 32), activation="relu", max_iter=1000,
                       random_state=42, early_stopping=True),
        "mlp", args.k_folds)
    cv.append(m)
    print(f"{m['label']:<30} {m['auc']:>8.4f} {m['auc_std']:>6.4f} "
          f"{m['ap']:>8.4f} {m['ap_std']:>6.4f} {m['tpr_at_p90']:>9.4f} {m['tpr_std']:>6.4f}")

    # ── Summary ─────────────────────────────────────────────────────
    best_ind = max(individual, key=lambda m: m["tpr_at_p90"])
    best_fus = max(fusion.values(), key=lambda m: m["tpr_at_p90"])
    best_cv = max(cv, key=lambda m: m["tpr_at_p90"]) if cv else None

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Best individual:  {best_ind['label']:<25} TPR@P90={best_ind['tpr_at_p90']:.4f}")
    print(f"  Best fusion:      {best_fus['label']:<25} TPR@P90={best_fus['tpr_at_p90']:.4f}")
    if best_cv:
        print(f"  Best meta (CV):   {best_cv['label']:<25} TPR@P90={best_cv['tpr_at_p90']:.4f} ±{best_cv['tpr_std']:.4f}")

    # ── Attack-run evaluation ────────────────────────────────────────
    if not args.attack_runs:
        return 0

    import torchaudio
    from audi.checkpoint import get_clip_seconds
    _SR = MelConfig().sample_rate
    # Read clip_seconds from the model checkpoint (fallback 2.56)
    _tmp = torch.load(CKPT_A, map_location="cpu", weights_only=False)
    _CLIP_S = get_clip_seconds(_tmp["hyper_parameters"])
    _CLIP_SAMPLES = int(_SR * _CLIP_S)
    _ATK_DIR = _PROJECT / "data" / "attack_runs"

    def _split_by_zero_gaps(audio, sr, min_dur=3.0, min_gap_s=0.5):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        exact_zero = audio == 0.0
        zero_runs, in_zero, start = [], False, 0
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

    def _sliding_windows(audio, win_samples, stride_frac):
        step = int(win_samples * stride_frac)
        if step < 1:
            step = 1
        n = max(1, (len(audio) - win_samples) // step + 1)
        return np.array([audio[i*step:i*step+win_samples] for i in range(n)])

    @torch.no_grad()
    def _predict_windows(model, windows, device, batch_size=64):
        scores = []
        for i in range(0, len(windows), batch_size):
            batch = torch.as_tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch).cpu().numpy()
            scores.append(1.0 / (1.0 + np.exp(-logits)))
        return np.concatenate(scores).flatten() if scores else np.array([])

    from audi.hysteresis import apply_hysteresis

    @torch.no_grad()
    def _predict_windows_multi(models_dict, windows, device, batch_size=64):
        """Returns {name: scores_array} for each model."""
        all_scores = {name: [] for name in models_dict}
        for i in range(0, len(windows), batch_size):
            batch = torch.as_tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            for name, model in models_dict.items():
                logits = model(batch).cpu().numpy()
                all_scores[name].append(1.0 / (1.0 + np.exp(-logits)))
        return {name: np.concatenate(s).flatten() for name, s in all_scores.items()}

    # Load attack audio — scan all WAV files
    all_atk_segs = []
    bg_windows_all = []
    for fp in sorted(_ATK_DIR.glob("*.wav")):
        audio, sr = torchaudio.load(str(fp))
        audio = audio.mean(dim=0).numpy().astype(np.float32).reshape(-1)
        if fp.name.startswith("background"):
            wins = _sliding_windows(audio, _CLIP_SAMPLES, args.stride)
            bg_windows_all.extend(wins)
            print(f"  {fp.name}: {len(wins)} bg windows")
        else:
            segs = _split_by_zero_gaps(audio, sr)
            for i, seg in enumerate(segs):
                all_atk_segs.append((f"{fp.stem}_seg{i}", seg))
            print(f"  {fp.name}: {len(segs)} segments, "
                  f"{sum(_sliding_windows(s, _CLIP_SAMPLES, args.stride).shape[0] for s in segs)} windows")
    bg_windows = np.array(bg_windows_all) if bg_windows_all else np.zeros((0, _CLIP_SAMPLES), dtype=np.float32)
    print(f"\nTotal: {len(all_atk_segs)} attack segments, {len(bg_windows)} bg windows\n")

    # ── Train full-data meta-models for attack eval ──────────────────
    meta_models = {}
    meta_models["logistic_regression"] = LogisticRegression(
        max_iter=2000, random_state=42).fit(X, y)
    meta_models["random_forest"] = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42, n_jobs=-1).fit(X, y)
    try:
        import xgboost as xgb
        meta_models["xgboost"] = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric="logloss").fit(X, y)
    except ImportError:
        pass
    meta_models["mlp"] = MLPClassifier(
        (64, 32), activation="relu", max_iter=1000,
        random_state=42, early_stopping=True).fit(X, y)

    # ── Compute P90 thresholds for all methods ───────────────────────
    all_methods = {}  # {label: (score_array_for_val, y)}

    # Individual models
    for j, name in enumerate(m_names):
        all_methods[name] = (X[:, j], y)
    # Simple fusion
    all_methods["fusion_max"] = (X.max(axis=1), y)
    all_methods["fusion_avg"] = (X.mean(axis=1), y)
    all_methods["fusion_vote"] = ((1/(1+np.exp(-X)) > 0.5).mean(axis=1), y)
    # Meta-models
    for label, clf in meta_models.items():
        all_methods[label] = (clf.predict_proba(X)[:, 1], y)

    thresh_map = {}
    for label, (scores_val, y_val) in all_methods.items():
        fpr, tpr, th, auc = compute_roc_values(scores_val, y_val)
        prec = compute_precision(scores_val, y_val, th)
        for pt in [50, 60, 70, 75, 80, 85, 90, 95, 99]:
            try:
                th_pt, _, _ = find_threshold_at_precision(prec, tpr, th, pt / 100)
                sigma = 1.0 / (1.0 + np.exp(-th_pt))
                thresh_map.setdefault(label, {})[f"P{pt}"] = sigma
            except (ValueError, IndexError):
                pass

    # ── Run attack eval ──────────────────────────────────────────────
    all_atk_windows = {}
    for seg_name, seg in all_atk_segs:
        all_atk_windows[seg_name] = _sliding_windows(seg, _CLIP_SAMPLES, args.stride)

    results = []
    eval_labels = list(all_methods.keys())

    for li, label in enumerate(eval_labels):
        print(f"[{li+1}/{len(eval_labels)}] {label} ...", end=" ", flush=True)

        # Collect scores for all attack segments + background
        atk_scores = {}
        for seg_name, windows in all_atk_windows.items():
            if len(windows) == 0:
                atk_scores[seg_name] = np.array([])
                continue
            if label in m_names:
                # Individual model — direct inference
                atk_scores[seg_name] = _predict_windows(
                    models[label], windows, device)
            elif label.startswith("fusion_"):
                # Simple fusion — run all models, then combine
                scores_dict = _predict_windows_multi(models, windows, device)
                stacked = np.column_stack([scores_dict[n] for n in m_names])
                # Convert sigmoid back to logit for max/avg fusion
                logits = -np.log(1.0 / np.clip(stacked, 1e-12, 1 - 1e-12) - 1.0)
                if label == "fusion_max":
                    fused = logits.max(axis=1)
                elif label == "fusion_avg":
                    fused = logits.mean(axis=1)
                elif label == "fusion_vote":
                    fused = (stacked > 0.5).mean(axis=1)
                else:
                    fused = logits.mean(axis=1)
                atk_scores[seg_name] = (
                    fused if label == "fusion_vote"
                    else 1.0 / (1.0 + np.exp(-fused))
                )
            else:
                # Meta-model — run all models, stack logits, predict
                scores_dict = _predict_windows_multi(models, windows, device)
                stacked_logits = np.column_stack([
                    -np.log(1.0/np.clip(scores_dict[n], 1e-12, 1-1e-12)-1.0)
                    for n in m_names
                ])
                clf = meta_models[label]
                atk_scores[seg_name] = clf.predict_proba(stacked_logits)[:, 1]

        bg_model_scores = {}
        for name, model in models.items():
            bg_model_scores[name] = _predict_windows(model, bg_windows, device)

        # Evaluate at each precision level
        for pt_str, sigma in thresh_map.get(label, {}).items():
            seg_covs, seg_firsts = [], []
            for seg_name in all_atk_windows:
                scores = atk_scores[seg_name]
                if len(scores) == 0:
                    seg_covs.append(0.0)
                    seg_firsts.append(100.0)
                    continue
                dets = apply_hysteresis(scores, sigma)
                cov = 100.0 * dets.sum() / len(dets)
                det_idx = np.where(dets)[0]
                first = (100.0 * det_idx[0] / len(dets)
                         if len(det_idx) > 0 else 100.0)
                seg_covs.append(cov)
                seg_firsts.append(first)

            # Background: use individual model at same sigma
            bg_total = 0
            for name in m_names:
                if name in bg_model_scores:
                    bg_total += int(apply_hysteresis(bg_model_scores[name], sigma).sum())

            results.append({
                "model": label,
                "precision": pt_str,
                "sigma": round(sigma, 4),
                "cov_pct": round(np.mean(seg_covs), 1),
                "first_pct": round(np.median(seg_firsts), 1),
                "bg": bg_total,
            })

        # Show best P90 or P50
        best = max(
            [r for r in results if r["model"] == label],
            key=lambda r: r["cov_pct"] - r["bg"] * 0.5,
            default=None,
        )
        if best:
            print(f"✓ {best['precision']} σ={best['sigma']:.3f} "
                  f"cov={best['cov_pct']:.0f}% 1st={best['first_pct']:.0f}% "
                  f"bg={best['bg']}")
        else:
            print("✗ no threshold")

    # ── Attack-run leaderboard ───────────────────────────────────────
    print(f"\n{'='*85}")
    print("ATTACK-RUN LEADERBOARD (P90)")
    print(f"{'='*85}")
    p90 = [r for r in results if r["precision"] == "P90"]
    p90.sort(key=lambda r: (-r["cov_pct"], r["first_pct"], r["bg"]))
    print(f"{'#':>3} {'model':<40} {'σ':>7} {'cov%':>6} {'1st%':>6} {'bg':>5}")
    print(f"{'-'*72}")
    for i, r in enumerate(p90[:20]):
        print(f"{i+1:>3} {r['model']:<40} {r['sigma']:>7.4f} "
              f"{r['cov_pct']:>6.1f} {r['first_pct']:>6.1f} {r['bg']:>5}")

    # Also show per-model breakdown across precision levels
    print(f"\n{'='*85}")
    print("PRECISION-COVERAGE MATRIX (all methods, all precision levels)")
    print(f"{'='*85}")
    prec_levels = ["P50","P60","P70","P75","P80","P85","P90","P95","P99"]
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], {})[r["precision"]] = r

    for label in eval_labels:
        if label not in by_model:
            continue
        row = by_model[label]
        best_p = max(row.values(),
                     key=lambda r: r["cov_pct"] - r["bg"] * 0.5)
        print(f"\n  {label}  [best={best_p['precision']} "
              f"σ={best_p['sigma']:.3f} cov={best_p['cov_pct']:.0f}% "
              f"1st={best_p['first_pct']:.0f}% bg={best_p['bg']}]")
        print(f"  {'Prec':>5} {'σ':>8} {'cov%':>7} {'1st%':>7} {'bg':>5}")
        print(f"  {'-'*35}")
        for pt in prec_levels:
            r = row.get(pt)
            if r:
                print(f"  {pt:>5} {r['sigma']:>8.4f} {r['cov_pct']:>7.1f} "
                      f"{r['first_pct']:>7.1f} {r['bg']:>5}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
