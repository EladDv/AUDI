#!/usr/bin/env python3
"""
feature_probe.py - READ-ONLY: which feature separates two drone types best,
                   and which one HOLDS UP in noise?

For a chosen class pair it:
  1) computes several features per clip (mel, scalogram, pcen, harmonic, modulation, SCD),
  2) ranks them by leakage-safe separability (grouped-CV balanced accuracy + shuffle null),
  3) injects REAL background noise at clean/10/5/0 dB and plots separability-vs-SNR
     -> the curve that stays highest is the noise-robust winner.

No model is saved, no data is modified.

Run:
   python feature_probe.py --pair red_blue
   python feature_probe.py --pair array
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import CFG, DATA_ROOTS                       # noqa: E402
from features import FEATURES                            # noqa: E402
from scd_probe import scd_alpha_profile                  # noqa: E402

OUT = HERE / "feature_probe_out"
MAX_CLIPS_PER_FILE = 8
REG_FEATS = ["mel", "scalogram", "pcen", "harmonic_stack", "modulation"]
SNRS = ["clean", 10.0, 5.0, 0.0]

DATASET = DATA_ROOTS[1] / "raw" / "data_clean_unit_06052026"
NOISE_DIRS = [
    DATA_ROOTS[1] / "raw" / "dataset_v2" / "dataset_v2" / "BG",
    DATASET / "background",
    DATASET / "false_alarms",
]
ARRAY_DIRS = [
    DATA_ROOTS[0] / "live_session_20260528",
    DATA_ROOTS[0] / "live_session_20260601",
]


# --------------------------------------------------------------------------- #
def reduce_map(M):
    """(C,F,T) feature map -> compact L2-normalised vector (fair for all feats)."""
    from scipy.ndimage import zoom
    out = []
    for c in range(M.shape[0]):
        z = zoom(M[c], (32 / M.shape[1], 16 / M.shape[2]), order=1)
        out.append(z.ravel())
    v = np.concatenate(out).astype(np.float64)
    v = v - v.mean()
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def all_features(clip):
    feats = {}
    for name in REG_FEATS:
        try:
            feats[name] = reduce_map(FEATURES[name]["fn"](clip, CFG))
        except Exception as exc:
            feats[name] = None
            print(f"   ! {name} failed: {exc}")
    feats["scd"] = scd_alpha_profile(clip, CFG.sr)
    return feats


def slice_clips(y):
    n = int(CFG.clip_s * CFG.sr); hop = int(CFG.hop_s * CFG.sr)
    if len(y) < n:
        return [np.pad(y, (0, n - len(y)))]
    starts = list(range(0, len(y) - n + 1, hop))
    if len(starts) > MAX_CLIPS_PER_FILE:
        idx = np.linspace(0, len(starts) - 1, MAX_CLIPS_PER_FILE).astype(int)
        starts = [starts[i] for i in idx]
    return [y[s:s + n] for s in starts]


# --------------------------------------------------------------------------- #
def load_pair(pair):
    """Return clips list: each = dict(cls, file, audio(float32))."""
    import librosa
    rows = []
    if pair == "red_blue":
        specs = [("target_drone", "red_13"), ("other_drones", "blue_7")]
        for fld, label in specs:
            d = DATASET / fld
            for w in sorted(d.glob("*.wav")):
                t = w.name.lower().split("_")
                if label == "red_13" and not w.name.lower().startswith("red_13"):
                    continue
                if label == "blue_7" and not w.name.lower().startswith("blue_7"):
                    continue
                y, _ = librosa.load(str(w), sr=CFG.sr, mono=True)
                for c in slice_clips(y.astype(np.float32)):
                    rows.append(dict(cls=label, file=w.name, audio=c))
    else:  # array EVO vs FPV (your mic)
        for root in ARRAY_DIRS:
            for w in root.rglob("*.wav"):
                p = str(w).lower()
                label = "evo" if "evo" in p else ("fpv" if "fpv" in p else None)
                if label is None:
                    continue
                y, _ = librosa.load(str(w), sr=CFG.sr, mono=True)
                for c in slice_clips(y.astype(np.float32)):
                    rows.append(dict(cls=label, file=w.name, audio=c))
    return rows


def load_noise_pool():
    import librosa
    pool = []
    for d in NOISE_DIRS:
        if d.exists():
            for w in sorted(d.glob("*.wav")):
                y, _ = librosa.load(str(w), sr=CFG.sr, mono=True)
                if len(y) > CFG.sr:
                    pool.append(y.astype(np.float32))
    return pool


def mix(clip, noise, snr_db, rng):
    n = len(clip)
    if len(noise) < n:
        noise = np.tile(noise, n // len(noise) + 1)
    s = rng.integers(0, len(noise) - n + 1)
    seg = noise[s:s + n]
    ps = np.mean(clip ** 2) + 1e-12
    pn = np.mean(seg ** 2) + 1e-12
    g = np.sqrt(ps / (pn * (10 ** (snr_db / 10))))
    return (clip + g * seg).astype(np.float32)


# --------------------------------------------------------------------------- #
def grouped_balacc(X, y, groups):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        return float("nan")
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LinearDiscriminantAnalysis(solver="lsqr",
                                                       shrinkage="auto"))
        clf.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)) if accs else float("nan")


def silhouette(X, y):
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    try:
        return float(silhouette_score(StandardScaler().fit_transform(X), y))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=["red_blue", "array"], default="red_blue")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(CFG.seed)

    rows = load_pair(args.pair)
    cls = np.array([r["cls"] for r in rows])
    classes = sorted(set(cls))
    print(f"[{args.pair}] {len(rows)} clips: "
          + ", ".join(f"{c}={int((cls==c).sum())}" for c in classes))
    if len(classes) < 2:
        raise SystemExit("need 2 classes")
    y = (cls == classes[1]).astype(int)
    groups = np.array([r["file"] for r in rows])
    yshuf = rng.permutation(y)
    noise = load_noise_pool()
    print(f"noise pool: {len(noise)} background clips")

    # ---- clean ranking --------------------------------------------------- #
    print("\ncomputing clean features ...")
    clean = [all_features(r["audio"]) for r in rows]
    feat_names = REG_FEATS + ["scd"]
    rank = {}
    for fn in feat_names:
        vecs = [c[fn] for c in clean]
        if any(v is None for v in vecs):
            continue
        X = np.stack(vecs)
        rank[fn] = dict(balacc=grouped_balacc(X, y, groups),
                        silhouette=silhouette(X, y),
                        shuffle=grouped_balacc(X, yshuf, groups))
    print("\n=== CLEAN separability (balanced acc; chance 0.5) ===")
    for fn in sorted(rank, key=lambda k: -rank[k]["balacc"]):
        r = rank[fn]
        print(f"  {fn:16s} balacc={r['balacc']:.3f}  sil={r['silhouette']:+.3f}  "
              f"shuffle_null={r['shuffle']:.3f}")

    # ---- noise robustness curve ------------------------------------------ #
    curve = {fn: [] for fn in feat_names if fn in rank}
    if noise:
        print("\ninjecting noise (clean/10/5/0 dB) ...")
        for snr in SNRS:
            if snr == "clean":
                feats_at = clean
            else:
                feats_at = [all_features(mix(r["audio"],
                            noise[rng.integers(0, len(noise))], snr, rng))
                            for r in rows]
            for fn in curve:
                X = np.stack([c[fn] for c in feats_at])
                curve[fn].append(grouped_balacc(X, y, groups))
            done = {fn: round(curve[fn][-1], 3) for fn in curve}
            print(f"  SNR={str(snr):>5}: {done}")

    # ---- plot ------------------------------------------------------------ #
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(len(SNRS)))
        fig, ax = plt.subplots(figsize=(8, 5))
        for fn in sorted(curve, key=lambda k: -np.nanmean(curve[k])):
            ax.plot(xs, curve[fn], marker="o", label=fn)
        ax.axhline(0.5, ls="--", color="gray", label="chance")
        ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in SNRS])
        ax.set_xlabel("noise level (SNR dB; left=clean, right=loud noise)")
        ax.set_ylabel("balanced accuracy (separability)")
        ax.set_title(f"{args.pair}: which feature separates the types best in noise?")
        ax.legend(); ax.set_ylim(0.4, 1.0); fig.tight_layout()
        fig.savefig(OUT / f"{args.pair}_noise_curve.png", dpi=120); plt.close(fig)
        print(f"\nsaved plot -> {OUT / (args.pair + '_noise_curve.png')}")
    except Exception as exc:
        print("plot skipped:", exc)

    json.dump(dict(pair=args.pair, classes=classes,
                   per_class={c: int((cls == c).sum()) for c in classes},
                   clean_ranking=rank,
                   noise_curve={fn: curve[fn] for fn in curve},
                   snrs=[str(s) for s in SNRS]),
              open(OUT / f"{args.pair}_report.json", "w"), indent=2)
    print(f"saved -> {OUT / (args.pair + '_report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
