#!/usr/bin/env python3
"""
scd_probe.py - READ-ONLY detectability probe: does SCD separate drone types?

This builds NO model and changes NO data. It answers one question objectively:
   "Is there information in the SCD feature that separates the classes?"
...by measuring separability model-free (silhouette) + with a SIMPLE linear
classifier under leakage-safe, group-aware cross-validation, alongside three
honesty controls so we don't fool ourselves:

   1) label-shuffle null   -> should collapse to chance if the pipeline is clean
   2) spectrogram baseline -> SCD only "wins" if it beats mel
   3) distance confound    -> can the feature predict DISTANCE instead of type?

Plus a field-realism test: leave-one-distance-out (train one distance, test the
other) so a "win" can't ride on distance/throttle.

Feature compared:
   * SCD a-profile  = squared-envelope (cyclic) spectrum, L2-normalised.
                      A fast, standard proxy for the SCD integrated over carrier
                      frequency - peaks at blade-pass / rotation harmonics.
   * mel-mean       = time-averaged log-mel spectrum, L2-normalised (baseline).

Run:
   cd 06_model_v1/training
   python scd_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import CFG, PROJECT                          # noqa: E402

OUT = HERE / "scd_probe_out"
DATASET = PROJECT / "data_other" / "raw" / "data_clean_unit_06052026"
FOLDERS = ["target_drone", "other_drones"]               # red_13, blue_7 live here
MAX_CLIPS_PER_FILE = 15


# --------------------------------------------------------------------------- #
# features (both READ-ONLY transforms of the audio)
# --------------------------------------------------------------------------- #
def scd_alpha_profile(y, sr, amin=10.0, amax=2000.0, nbins=256):
    """Squared-envelope (cyclic) spectrum -> the SCD a-profile."""
    e = y.astype(np.float64) ** 2          # instantaneous power (envelope^2)
    e = e - e.mean()                       # drop a=0 DC so it can't dominate
    w = np.hanning(len(e))
    E = np.abs(np.fft.rfft(e * w))
    fa = np.fft.rfftfreq(len(e), 1.0 / sr)
    band = (fa >= amin) & (fa <= amax)
    grid = np.linspace(amin, amax, nbins)
    prof = np.interp(grid, fa[band], E[band])
    prof = np.log1p(prof)
    return (prof / (np.linalg.norm(prof) + 1e-9)).astype(np.float32)


def mel_mean(y, sr, nmels=128):
    """Time-averaged log-mel = spectral-shape baseline, L2-normalised."""
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=CFG.n_fft,
                                        win_length=CFG.win_length,
                                        hop_length=CFG.hop_length, n_mels=nmels,
                                        fmin=CFG.fmin, fmax=CFG.fmax, power=2.0)
    v = np.log1p(S).mean(axis=1)
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


# --------------------------------------------------------------------------- #
def parse(name: str):
    """red_13_3kg_40m_fast_center -> dict, or None if not a size-labelled clip."""
    t = name.lower().replace(".wav", "").split("_")
    if len(t) < 6:
        return None
    try:
        dist = int("".join(c for c in t[3] if c.isdigit()))
    except ValueError:
        return None
    return dict(cls=f"{t[0]}_{t[1]}", weight=t[2], dist=dist,
                throttle=t[4], bearing=t[5])


def slice_clips(y):
    n = int(CFG.clip_s * CFG.sr); hop = int(CFG.hop_s * CFG.sr)
    if len(y) < n:
        return [np.pad(y, (0, n - len(y)))]
    starts = list(range(0, len(y) - n + 1, hop))
    if len(starts) > MAX_CLIPS_PER_FILE:                 # even subsample
        idx = np.linspace(0, len(starts) - 1, MAX_CLIPS_PER_FILE).astype(int)
        starts = [starts[i] for i in idx]
    return [y[s:s + n] for s in starts]


def load_dataset():
    import librosa
    rows = []
    for fld in FOLDERS:
        d = DATASET / fld
        if not d.exists():
            continue
        for w in sorted(d.glob("*.wav")):
            meta = parse(w.name)
            if meta is None:
                continue
            y, _ = librosa.load(str(w), sr=CFG.sr, mono=True)
            for ci, clip in enumerate(slice_clips(y.astype(np.float32))):
                rows.append(dict(meta, file=w.name, clip=ci,
                                 scd=scd_alpha_profile(clip, CFG.sr),
                                 mel=mel_mean(clip, CFG.sr)))
    return rows


# --------------------------------------------------------------------------- #
# evaluation (model-free + simple-linear, all leakage-safe)
# --------------------------------------------------------------------------- #
def grouped_cv_acc(X, y, groups, seed=0):
    """Balanced accuracy (chance = 0.5 even with class imbalance), group-safe."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, groups):
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
    Xs = StandardScaler().fit_transform(X)
    try:
        return float(silhouette_score(Xs, y))
    except Exception:
        return float("nan")


def leave_one_distance_out(X, y, dist):
    """Train on one distance, test on the other - field-realism / confound test."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score
    ds = sorted(set(dist))
    accs = []
    for held in ds:
        tr = dist != held; te = dist == held
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LinearDiscriminantAnalysis(solver="lsqr",
                                                       shrinkage="auto"))
        clf.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)) if accs else float("nan")


def main():
    OUT.mkdir(exist_ok=True)
    print(f"dataset: {DATASET}")
    if not DATASET.exists():
        raise SystemExit("dataset folder not found")

    rows = load_dataset()
    cls = np.array([r["cls"] for r in rows])
    classes = sorted(set(cls))
    print(f"loaded {len(rows)} clips from classes: "
          + ", ".join(f"{c}={int((cls==c).sum())}" for c in classes))
    if len(classes) < 2:
        raise SystemExit("need 2 classes")

    y = (cls == classes[1]).astype(int)
    dist = np.array([int(r["dist"]) for r in rows])
    files = np.array([r["file"] for r in rows])
    # group = one recording pass (all bearings + clips together) -> no leakage
    groups = np.array([f"{r['cls']}|{r['weight']}|{r['dist']}|{r['throttle']}"
                       for r in rows])
    Xscd = np.stack([r["scd"] for r in rows])
    Xmel = np.stack([r["mel"] for r in rows])

    print(f"\ndistances present: {sorted(set(dist.tolist()))}")
    common = sorted(int(d) for d in (set(dist[y == 0]) & set(dist[y == 1])))
    print(f"distances with BOTH classes (used for confound-safe test): {common}")
    rng = np.random.default_rng(CFG.seed)
    yshuf = rng.permutation(y)

    rep = {"classes": classes, "n_clips": len(rows),
           "per_class": {c: int((cls == c).sum()) for c in classes},
           "common_distances": common, "feized": {}}

    def block(name, X):
        sil = silhouette(X, y)
        acc = grouped_cv_acc(X, y, groups)
        acc_sh = grouped_cv_acc(X, yshuf, groups)
        # distance-confound probe (only where both classes share distances)
        if len(common) >= 2:
            mask = np.isin(dist, common)
            ydist = (dist[mask] == common[-1]).astype(int)
            # group by FILE so both distances appear across folds (clip-leak-safe)
            accd = grouped_cv_acc(X[mask], ydist, files[mask])
            lodo = leave_one_distance_out(X[mask], y[mask], dist[mask])
        else:
            accd = lodo = float("nan")
        rep["feized"][name] = dict(silhouette=sil, cv_acc=acc,
                                   cv_acc_shuffled=acc_sh,
                                   distance_probe_acc=accd,
                                   leave_one_distance_out_acc=lodo)
        print(f"\n=== {name} ===  (balanced accuracy; chance = 0.500)")
        print(f"  silhouette (model-free separation)   : {sil:+.3f}  (higher=more separated, 0=none)")
        print(f"  CV bal-acc   (TYPE, leakage-safe)     : {acc:.3f}")
        print(f"  CV bal-acc   (labels SHUFFLED = null) : {acc_sh:.3f}  (should be ~0.5)")
        print(f"  distance-probe bal-acc (confound)     : {accd:.3f}  (high => encodes distance)")
        print(f"  leave-one-distance-out TYPE bal-acc   : {lodo:.3f}  (survives distance change?)")

    block("SCD_alpha_profile", Xscd)
    block("mel_mean_baseline", Xmel)

    s, m = rep["feized"]["SCD_alpha_profile"], rep["feized"]["mel_mean_baseline"]
    verdict = []
    verdict.append(("pipeline clean (shuffle ~chance)",
                    s["cv_acc_shuffled"] < 0.65))
    verdict.append(("SCD separates type (CV > 0.65)",
                    s["cv_acc"] > 0.65))
    verdict.append(("SCD beats mel baseline",
                    s["cv_acc"] >= m["cv_acc"]))
    verdict.append(("SCD survives distance change (LODO > 0.6)",
                    (s["leave_one_distance_out_acc"] or 0) > 0.6))
    verdict.append(("type signal > distance confound",
                    s["cv_acc"] >= s["distance_probe_acc"]))
    print("\n================ VERDICT ================")
    for label, ok in verdict:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    allpass = all(ok for _, ok in verdict)
    print(f"\n  => SCD is {'WORTH building a model on' if allpass else 'NOT clearly worth it yet'}")
    rep["verdict"] = {label: bool(ok) for label, ok in verdict}
    rep["overall_pass"] = bool(allpass)

    json.dump(rep, open(OUT / "scd_probe_report.json", "w"), indent=2)
    print(f"\nsaved -> {OUT / 'scd_probe_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
