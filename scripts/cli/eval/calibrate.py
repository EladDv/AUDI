"""cmd_calibrate eval subcommand."""
from __future__ import annotations

import sys
from pathlib import Path


def run(noise_path: str | None, drone_path: str | None) -> None:
    import argparse

    import numpy as np

    from audi.hearability_estimator import HearabilityEstimator

    ap = argparse.ArgumentParser()
    ap.add_argument("--logit", type=float, default=None)
    ap.add_argument("--all", dest="all_runs", action="store_true",
                    help="Recalibrate even if hearability_calib.npz exists")
    ap.add_argument("run_dir", type=Path, nargs="?")
    args = ap.parse_args()
    if args.run_dir is None:
        print(
            "Usage: uv run python scripts/evaluate.py calibrate <run_dir> [--logit <value>]"
        )
        sys.exit(1)

    run_dir = args.run_dir
    calib_path = run_dir / "eval_data" / "hearability_calib.npz"
    if not args.all_runs and calib_path.exists():
        print(f"Already calibrated: {calib_path}")
        print("(use --all to force recalibration)")
        sys.exit(0)

    pred_path = run_dir / "eval_data" / "predictions_best.pt"
    if not pred_path.exists():
        candidates = sorted(
            (run_dir / "eval_data").glob("predictions_epoch_*.pt")
        )
        if not candidates:
            print(
                f"ERROR: No predictions files found in {run_dir / 'eval_data'}"
            )
            sys.exit(1)
        pred_path = candidates[-1]
    
    print(f"Loading: {pred_path}")
    estimator = HearabilityEstimator.from_predictions(pred_path)
    
    print(f"\n{'=' * 55}")
    print(f"Per-bin calibration (from {pred_path.name})")
    print(f"{'=' * 55}")
    print(f"{'bin':<12} {'mean':>7} {'std':>7} {'prior':>7}  {'P at mean':>9}")
    print(f"{'-' * 55}")
    for b in estimator.bins:
        p = estimator.predict(b.mean)[b.name]
        print(
            f"{b.name:<12} {b.mean:>7.3f} {b.std:>7.3f} {b.prior:>7.3f}  {p:>9.3f}"
        )
    
    print("\nDecision boundaries:")
    sorted_bins = sorted(estimator.bins, key=lambda b: b.mean)
    for i in range(len(sorted_bins) - 1):
        a, b_obj = sorted_bins[i], sorted_bins[i + 1]
        v_a, v_b = a.std**2, b_obj.std**2
        m_a, m_b = a.mean, b_obj.mean
        A = 1 / v_a - 1 / v_b
        B = -2 * m_a / v_a + 2 * m_b / v_b
        C = (
            m_a**2 / v_a
            - m_b**2 / v_b
            - 2
            * np.log(
                max(a.prior, 1e-12)
                / max(b_obj.prior, 1e-12)
                * max(b_obj.std, 1e-8)
                / max(a.std, 1e-8)
            )
        )
        if abs(A) < 1e-12:
            boundary = -C / B if abs(B) > 1e-12 else (m_a + m_b) / 2
        else:
            disc = B**2 - 4 * A * C
            if disc < 0:
                boundary = (m_a + m_b) / 2
            else:
                roots = [
                    (-B + np.sqrt(disc)) / (2 * A),
                    (-B - np.sqrt(disc)) / (2 * A),
                ]
                boundary = min(roots, key=lambda r: abs(r - (m_a + m_b) / 2))
        print(f"  {a.name} ↔ {b_obj.name}: logit ≈ {boundary:.3f}")
    
    calib_path = run_dir / "eval_data" / "hearability_calib.npz"
    estimator.save(calib_path)
    print(f"\nSaved: {calib_path}")
    
    if "--logit" in sys.argv:
        idx = sys.argv.index("--logit")
        logit = float(sys.argv[idx + 1])
        probs = estimator.predict(logit)
        best, conf = estimator.classify(logit)
        print(f"\nLogit = {logit:.3f}")
        for bn, p in probs.items():
            bar = "█" * int(p * 40)
            print(f"  {bn:<12} {p:.3f}  {bar}")
        print(f"  → {best} (confidence: {conf:.3f})")
    else:
        print(f"\n{'=' * 55}")
        print("Test sweep:")
        print(f"{'=' * 55}")
        print(f"{'logit':>7}  {'prob':>6}  best_bin")
        print(f"{'-' * 40}")
        for logit in np.linspace(-3, 5, 17):
            prob = 1 / (1 + np.exp(-logit))
            best, _ = estimator.classify(float(logit))
            print(f"{logit:>7.2f}  {prob:>6.3f}  {best}")
