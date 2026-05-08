"""cmd_run inference subcommand."""
from __future__ import annotations

from pathlib import Path


def run() -> None:
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    import torch

    from audi.config import MelConfig

    SR = MelConfig().sample_rate
    N_MELS = MelConfig().n_mels
    N_FFT = MelConfig().n_fft
    HOP = MelConfig().hop_length
    CHUNK_S = 2.56; CHUNK_N = int(CHUNK_S * SR)
    OVERLAP = 0.75; HOP_N = int(CHUNK_N * (1 - OVERLAP))
    ROOT = Path(__file__).resolve().parents[2]  # project root
    CKPT = ROOT / "checkpoints_v2/phase3_full/scratch_resnet18/lightning_logs/version_0/checkpoints/epoch=14-step=1500.ckpt"  # noqa: E501
    
    from audi.training.detector import DroneDetector
    
    model = DroneDetector.load_from_checkpoint(
        str(CKPT), map_location="cpu",
        model_arch="resnet18", n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP,
        loss_type="bce", spec_augment_prob=0.0, pretrained_backbone=False,
        mel_mean=None, mel_std=None,
        use_compile=False, bn_momentum=0.1,
    )
    model.eval()
    print(f"Loaded: {CKPT.name}")
    
    bg, _ = sf.read(str(ROOT / "data/background.wav"))
    drone_full, _ = sf.read(str(ROOT / "data/200m_attackrun.wav"))
    
    def rms(x): return np.sqrt(np.mean(x**2) + 1e-8)
    
    @torch.no_grad()
    def predict(chunk):
        r = rms(chunk)
        if r < 1e-8:
            return 0.0, 0.0
        x = torch.from_numpy((chunk / r).astype(np.float32)).unsqueeze(0)
        logit = model(x)
        return torch.sigmoid(logit).item(), logit.item()
    
    def sliding(audio, cn, hn):
        ts, ps, ls = [], [], []
        for s in range(0, max(1, len(audio)-cn), hn):
            c = audio[s:s+cn]
            if len(c) < cn: c = np.pad(c, (0, cn-len(c)))
            p, label = predict(c)
            ts.append((s + cn/2)/SR); ps.append(p); ls.append(label)
        return np.array(ts), np.array(ps), np.array(ls)
    
    # FPR calibration
    print("\n=== FPR Calibration ===")
    bg_t, bg_p, bg_l = sliding(bg, CHUNK_N, HOP_N)
    print(f"  {len(bg_p)} bg chunks: min={bg_p.min():.4f} max={bg_p.max():.4f} mean={bg_p.mean():.4f}")
    thresh_fpr10 = np.percentile(bg_p, 90)
    print(f"  FPR=10% threshold: {thresh_fpr10:.4f}  (actual FPR={(bg_p>=thresh_fpr10).mean():.3%})")
    for fpr in [0.01, 0.05, 0.10, 0.20]:
        t = np.percentile(bg_p, 100*(1-fpr))
        print(f"    FPR={fpr:.0%} → thresh={t:.4f}")
    
    # Drone inference
    print("\n=== Drone Inference ===")
    d_t, d_p, d_l = sliding(drone_full, CHUNK_N, HOP_N)
    print(f"  {len(d_p)} chunks: min={d_p.min():.4f} max={d_p.max():.4f} mean={d_p.mean():.4f}")
    w = 4
    ra = np.convolve(d_p, np.ones(w)/w, mode='same') if len(d_p) >= w else d_p
    
    # Segment runs
    bs, bn = 0.05, int(0.05*SR); nb = len(drone_full)//bn
    sil = np.abs(drone_full[:nb*bn]) < 1e-5
    blk_s = np.array([sil[i*bn:(i+1)*bn].mean()>0.7 for i in range(nb)])
    runs3 = []; in_r = False; r_start = 0.0
    for i, s in enumerate(blk_s):
        t = i*bs
        if not s and not in_r:
            in_r = True; r_start = t
        elif s and in_r:
            dur = t - r_start
            if dur >= 0.5: runs3.append((r_start, t, dur))
            in_r = False
    if in_r:
        dur = nb*bs - r_start
        if dur >= 0.5: runs3.append((r_start, nb*bs, dur))
    merged = []
    for r in runs3:
        if merged and r[0] - merged[-1][1] < 0.5:
            merged[-1] = (merged[-1][0], r[1], r[1]-merged[-1][0])
        else: merged.append(r)
    runs = merged
    
    print(f"\n  {len(runs)} attack runs:")
    for i, (s, e, dur) in enumerate(runs, 1):
        m = (d_t >= s) & (d_t <= e)
        rp = d_p[m]; rt = d_t[m]
        if len(rp):
            hit = (rp >= thresh_fpr10).mean()
            print(
                f"  Run {i}: {s:.1f}s→{e:.1f}s ({dur:.1f}s) det@{thresh_fpr10:.3f}={hit:.1%} max={rp.max():.3f}"  # noqa: E501
            )
    
    # Distance mapping
    dg = np.linspace(0, 200, 201)
    ad = np.full((len(runs), len(dg)), np.nan)
    for j, (rs_s, re_s, dur) in enumerate(runs):
        m = (d_t >= rs_s) & (d_t <= re_s); rt = d_t[m]; rp = d_p[m]
        if len(rt) < 2: continue
        dj = 200 * (1 - (rt - rs_s) / dur); si = np.argsort(dj)
        ad[j] = np.interp(dg, dj[si], rp[si])
    md = np.nanmean(ad, axis=0); sd = np.nanstd(ad, axis=0)
    
    # Mel spectrogram
    fw = torch.from_numpy(drone_full.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        fm = model.to_db(model.mel(fw)).squeeze(0).numpy()
    mt = np.arange(fm.shape[1]) * HOP / SR
    
    # Save
    out = str(ROOT / "artifacts/detection_vs_distance.json")
    with open(out, "w") as f:
        json.dump(dict(
            config={"sr": SR, "chunk_s": CHUNK_S, "overlap": OVERLAP, "thresh_fpr10": thresh_fpr10},
            drone_times=d_t.tolist(), drone_probs=d_p.tolist(), running_avg=ra.tolist(),
            runs=[{"id": i+1, "start": float(s), "end": float(e), "dur": float(d)} for i, (s, e, d) in enumerate(runs)],
            dist_grid=dg.tolist(),
            mean_prob=[float(x) if not np.isnan(x) else None for x in md],
            std_prob=[float(x) if not np.isnan(x) else None for x in sd],
        ), f, indent=2)
    
    # Plot
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, max(len(runs), 1)))
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3, height_ratios=[2.5, 2.3, 2.3, 2.5])
    ax = fig.add_subplot(gs[0, :])
    ax.pcolormesh(mt, np.arange(N_MELS), fm, shading="auto", cmap="magma", vmin=-80, vmax=0)
    for j, (s, e, _d) in enumerate(runs):
        ax.axvspan(s, e, alpha=0.12, color=colors[j])
        ax.text((s+e)/2, N_MELS+2, f"Run {j+1}", fontsize=8, ha="center", color=colors[j], fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
    ax.set_ylabel("Mel bin"); ax.set_title("Mel Spectrogram (RMS-norm'd) + Runs", fontsize=12, fontweight="bold")
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(d_t, d_p, lw=0.3, color="steelblue", alpha=0.5, label="Raw")
    ax.plot(d_t, ra, lw=1.5, color="darkblue", label=f"Running avg (n={w})")
    ax.axhline(y=thresh_fpr10, color="red", ls="--", lw=1, label=f"FPR=10% ({thresh_fpr10:.3f})")
    for j, (s, e, _d) in enumerate(runs): ax.axvspan(s, e, alpha=0.08, color=colors[j])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Detection Prob"); ax.legend(fontsize=7)
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25); ax.set_title("Detection vs Time")
    ax = fig.add_subplot(gs[1, 1])
    for j, (s, e, d) in enumerate(runs):
        m = (d_t >= s) & (d_t <= e); rt, dp = d_t[m], d_p[m]
        if len(rt) < 2: continue
        ax.plot(200*(1-(rt-s)/d), dp, lw=0.5, alpha=0.35, color=colors[j])
    ax.plot(dg, md, lw=2.5, color="#2980b9", label="Mean")
    ax.fill_between(dg, md-sd, md+sd, alpha=0.1, color="#2980b9")
    ax.axhline(y=thresh_fpr10, color="red", ls="--", lw=1, label="FPR=10%")
    ax.invert_xaxis(); ax.set_xlabel("Distance (m) ←"); ax.set_ylabel("Detection Prob")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25); ax.legend(fontsize=7); ax.set_title("Detection vs Distance")
    ax = fig.add_subplot(gs[2, 0])
    marks = [200, 150, 100, 75, 50, 25, 10, 5, 2]
    for j, (s, e, d) in enumerate(runs):
        m = (d_t >= s) & (d_t <= e); rt, dp = d_t[m], d_p[m]
        if len(rt) < 2: continue
        dj = 200 * (1 - (rt - s) / d); pts_d, pts_p = [], []
        for dm in marks:
            idx = np.argmin(np.abs(dj - dm))
            if idx < len(dp): pts_d.append(dm); pts_p.append(dp[idx])
        ax.plot(pts_d, pts_p, "o-", ms=4, lw=1.2, color=colors[j], alpha=0.5)
    ax.plot(dg, md, lw=2.5, color="#2980b9", label="Mean")
    ax.axhline(y=thresh_fpr10, color="red", ls="--", lw=1)
    ax.invert_xaxis(); ax.set_xlabel("Distance (m) ←"); ax.set_ylabel("Detection Prob")
    ax.grid(alpha=0.25); ax.legend(fontsize=7); ax.set_title("Detection at Distance Markers")
    ax = fig.add_subplot(gs[2, 1])
    xp = np.arange(len(runs)); wd = 0.35
    hr, mp = [], []
    for s, e, d in runs:
        m = (d_t >= s) & (d_t <= e); rp = d_p[m]
        hr.append((rp >= thresh_fpr10).mean() if len(rp) else 0)
        mp.append(rp.max() if len(rp) else 0)
    b1 = ax.bar(xp - wd/2, hr, wd, color="steelblue", alpha=0.7, label="Det rate @FPR10")
    b2 = ax.bar(xp + wd/2, mp, wd, color="coral", alpha=0.7, label="Max prob")
    for b, v in zip(b1, hr): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.0%}", ha="center", fontsize=8)
    for b, v in zip(b2, mp): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xp); ax.set_xticklabels([f"Run {i+1}" for i in range(len(runs))])
    ax.set_ylabel("Rate/Prob"); ax.set_ylim(0, 1.15); ax.grid(alpha=0.2, axis="y"); ax.legend(fontsize=8)
    ax.set_title("Per-Run Detection Stats")
    ax = fig.add_subplot(gs[3, 0])
    ax.hist(bg_p, bins=60, color="gray", alpha=0.6, density=True, label="Background")
    ax.hist(d_p, bins=60, color="steelblue", alpha=0.4, density=True, label="Drone recording")
    ax.axvline(x=thresh_fpr10, color="red", ls="--", lw=1.5, label=f"FPR=10% ({thresh_fpr10:.3f})")
    ax.set_xlabel("Detection Prob"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(alpha=0.2, axis="y"); ax.set_title("Prob Distribution: BG vs Drone")
    ax = fig.add_subplot(gs[3, 1])
    ax.pcolormesh(dg, np.arange(1, len(runs)+1), ad, shading="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.invert_xaxis(); ax.set_xlabel("Distance (m) ←"); ax.set_ylabel("Run #")
    ax.set_title("Detection Prob Heatmap")
    fig.suptitle("audi CNN14: Detection vs Distance (RMS-normalized, 200m→0m)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    png = str(ROOT / "artifacts/detection_vs_distance.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\nPlot: {png}")
    print(f"\n{'='*60}")
    print("  DETECTION vs DISTANCE (RMS-normalized)")
    print(f"{'='*60}")
    print(f"  FPR=10% threshold: {thresh_fpr10:.4f}")
    for dm in [200, 175, 150, 125, 100, 75, 50, 25, 10, 5, 2]:
        i = np.argmin(np.abs(dg - dm))
        if not np.isnan(md[i]):
            f = "DETECT" if md[i] >= thresh_fpr10 else ""
            print(f"  {dm:>4}m  {md[i]:.4f} ±{sd[i]:.4f}  {f}")
    
    
    # ====================================================================
    # attackrun — Predict on attack run WAV files
    # ====================================================================
    
