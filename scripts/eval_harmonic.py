#!/usr/bin/env python3
"""Compare harmonic folding detector vs wd_003 on all field alerts."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audi.harmonic_detector import HarmonicFoldingDetector, HarmonicConfig

DATA = ROOT / "data/field_recordings_20260514"
SCORES_JSON = DATA / "wd003_scores.json"
ALERTS_DIR = DATA / "alerts"

SR = 16000


def main():
    scores = json.loads(SCORES_JSON.read_text())
    alerts = scores["alerts"]
    print(f"Evaluating {len(alerts)} alerts\n")

    # Config: slow noise floor, moderate stacking
    cfg = HarmonicConfig(
        sample_rate=SR, n_fft=1024, hop_length=256,
        f0_min=125, f0_max=350, n_harmonics=12,
        noise_beta=0.0001, stack_alpha=0.50,
        score_scale=1000, threshold=0.3,
    )

    results = []
    t0 = time.time()
    for i, a in enumerate(alerts):
        alert_dir = a["alert_dir"]
        wav = ALERTS_DIR / alert_dir / "full_120s.wav"
        if not wav.exists() or wav.stat().st_size == 0:
            continue

        audio, sr = sf.read(str(wav))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        det = HarmonicFoldingDetector(cfg)
        raw, stacked, times = det.process(audio)

        results.append({
            "alert_dir": alert_dir,
            "wd003_max": a["max_score"],
            "harm_raw_max": float(raw.max()),
            "harm_raw_mean": float(raw.mean()),
            "harm_stk_max": float(stacked.max()),
            "harm_stk_mean": float(stacked.mean()),
            "n_raw_det": int((raw > 0.5).sum()),
            "n_stk_det": int((stacked > 0.3).sum()),
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(alerts)}] done")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(alerts)*1000:.0f}ms/alert)\n")

    # ── Summary ──
    wd = np.array([r["wd003_max"] for r in results])
    raw = np.array([r["harm_raw_max"] for r in results])
    stk = np.array([r["harm_stk_max"] for r in results])

    print(f"{'Metric':>15s} {'Mean':>8s} {'Min':>8s} {'Max':>8s}")
    print("-" * 45)
    print(f"{'wd_003 max':>15s} {wd.mean():8.4f} {wd.min():8.4f} {wd.max():8.4f}")
    print(f"{'harm raw max':>15s} {raw.mean():8.4f} {raw.min():8.4f} {raw.max():8.4f}")
    print(f"{'harm stk max':>15s} {stk.mean():8.4f} {stk.min():8.4f} {stk.max():8.4f}")

    # Correlation
    corr_raw = np.corrcoef(wd, raw)[0, 1]
    corr_stk = np.corrcoef(wd, stk)[0, 1]
    print(f"\nCorrelation wd_003 vs harmonic raw:  {corr_raw:.3f}")
    print(f"Correlation wd_003 vs harmonic stk:  {corr_stk:.3f}")

    # Save
    out = DATA / "harmonic_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")

    # Top/bottom comparisons
    print(f"\n{'='*70}")
    print(f"Top 5 harmonic vs wd_003:")
    for r in sorted(results, key=lambda x: -x["harm_stk_max"])[:5]:
        print(f"  {r['alert_dir']:25s}  wd={r['wd003_max']:.4f}  "
              f"raw={r['harm_raw_max']:.4f}  stk={r['harm_stk_max']:.4f}  "
              f"det={r['n_stk_det']}")

    print(f"\nBottom 5 harmonic stk:")
    for r in sorted(results, key=lambda x: x["harm_stk_max"])[:5]:
        print(f"  {r['alert_dir']:25s}  wd={r['wd003_max']:.4f}  "
              f"raw={r['harm_raw_max']:.4f}  stk={r['harm_stk_max']:.4f}  "
              f"det={r['n_stk_det']}")


if __name__ == "__main__":
    main()
