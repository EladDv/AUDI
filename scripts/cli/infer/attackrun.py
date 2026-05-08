"""cmd_attackrun inference subcommand."""
from __future__ import annotations

from pathlib import Path


def run() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    import torch

    from audi.config import MelConfig

    SR = MelConfig().sample_rate
    CLIP_S = 1.28
    CLIP_N = int(SR * CLIP_S); HOP_S = 0.2; HOP_N = int(HOP_S * SR)

    repo = Path(__file__).resolve().parents[2]  # project root

    from audi.training.detector import DroneDetector
    
    model = DroneDetector.load_from_checkpoint(
        str(repo / "checkpoints_v2/full_train/c_128s_focal_specaug/"
               "lightning_logs/version_0/checkpoints/epoch=10-step=1100.ckpt"),
        map_location="cpu")
    model.eval()
    print("Model: cnn14, focal+SpecAug, 1.28s segments")
    
    drone, _ = sf.read(str(repo / "data/200m_attackrun.wav"))
    
    block_s, block_n = 0.05, int(0.05 * SR)
    n_blocks = len(drone) // block_n
    zmask = np.abs(drone[:n_blocks*block_n]) < 1e-5
    bsil = np.array([zmask[i*block_n:(i+1)*block_n].mean() > 0.7 for i in range(n_blocks)])
    
    runs = []; in_run = False; rs = 0.0
    for i, s in enumerate(bsil):
        t = i * block_s
        if not s and not in_run:
            in_run = True; rs = t
        elif s and in_run:
            in_run = False
            if t - rs >= 0.5:
                runs.append((rs, t, t - rs))
    if in_run:
        te = n_blocks * block_s
        if te - rs >= 0.5:
            runs.append((rs, te, te - rs))
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 0.5:
            merged[-1] = (merged[-1][0], r[1], r[1] - merged[-1][0])
        else:
            merged.append(r)
    runs = merged
    print(f"Attack runs: {len(runs)}")
    
    @torch.no_grad()
    def predict(arr):
        a = np.asarray(arr, dtype=np.float32)
        if len(a) < CLIP_N: a = np.pad(a, (0, CLIP_N - len(a)))
        return model(torch.from_numpy(a[:CLIP_N]).unsqueeze(0)).item()
    
    all_data = []
    for rid, (rs_s, re_s, dur) in enumerate(runs, 1):
        ss, es = int(rs_s * SR), int(re_s * SR)
        rd = drone[ss:es]
        times, logits = [], []
        for idx in range(0, max(1, len(rd)-CLIP_N), HOP_N):
            ch = rd[idx:idx+CLIP_N]
            if len(ch) >= CLIP_N // 2:
                logits.append(predict(ch))
                times.append(rs_s + idx / SR + CLIP_S / 2)
        all_data.append(dict(
            run_id=rid, start_s=rs_s, end_s=re_s, duration_s=dur,
            times=times, logits=logits))
        print(f"  Run {rid}: {len(logits)} windows, "
              f"logit [{min(logits):+.2f}, {max(logits):+.2f}]")
    
    dgrid = np.linspace(0, 200, 201)
    all_ip = np.full((len(all_data), len(dgrid)), np.nan)
    for j, rd in enumerate(all_data):
        if not rd["times"]: continue
        dj = 200 * (1 - (np.array(rd["times"]) - rd["start_s"]) / rd["duration_s"])
        sj = np.array(rd["logits"]); si = np.argsort(dj)
        if len(si) > 1: all_ip[j] = np.interp(dgrid, dj[si], sj[si])
    ml = np.nanmean(all_ip, axis=0); sl = np.nanstd(all_ip, axis=0)
    
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(all_data)))
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, height_ratios=[2.2, 1.8, 1.8])
    ax0 = fig.add_subplot(gs[0, :])
    nfft = 1024; noverlap = 768
    spec, fs, ts, _ = plt.specgram(drone, NFFT=nfft, Fs=SR, noverlap=noverlap, window=np.hanning(nfft), mode='magnitude')
    spec_db = 20 * np.log10(spec + 1e-12)
    im = ax0.pcolormesh(ts, fs, spec_db, shading='gouraud', cmap='inferno', vmin=-70, vmax=-20)
    for rd in all_data:
        ax0.axvspan(rd["start_s"], rd["end_s"], alpha=0.1, color=colors[rd["run_id"]-1])
        ax0.text((rd["start_s"]+rd["end_s"])/2, 7000, f"Run {rd['run_id']}", fontsize=8, ha='center', color='white', fontweight='bold')
    ax0.set_ylabel("Frequency (Hz)")
    ax0.set_title("Spectrogram — 200m_attackrun.wav")
    plt.colorbar(im, ax=ax0, label="dB", shrink=0.8)
    ax1 = fig.add_subplot(gs[1, 0])
    for rd, c in zip(all_data, colors):
        ax1.plot(rd["times"], rd["logits"], lw=1.5, color=c, label=f"Run {rd['run_id']} ({rd['duration_s']:.1f}s)")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Detection Logit")
    ax1.set_title("Detection Logit vs Time"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.25)
    ax2 = fig.add_subplot(gs[1, 1])
    for rd, c in zip(all_data, colors):
        if not rd["times"]: continue
        dj = [200*(1-(t-rd["start_s"])/rd["duration_s"]) for t in rd["times"]]
        ax2.plot(dj, rd["logits"], lw=1.5, color=c, label=f"Run {rd['run_id']}")
    ax2.invert_xaxis()
    ax2.set_xlabel("Distance (m)  ← approaching"); ax2.set_ylabel("Detection Logit")
    ax2.set_title("Detection Logit vs Distance"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.25)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.fill_between(dgrid, ml-sl, ml+sl, alpha=0.15, color="#3498db", label="±1σ")
    ax3.plot(dgrid, ml, lw=2.5, color="#2980b9", label="Mean logit")
    for j in range(len(all_data)):
        ax3.plot(dgrid, all_ip[j], lw=0.4, alpha=0.25, color=colors[j])
    for dm in [200, 150, 100, 50, 25, 10, 5]:
        i = np.argmin(np.abs(dgrid - dm))
        if not np.isnan(ml[i]):
            ax3.annotate(f"{ml[i]:+.1f}", xy=(dm, ml[i]), fontsize=8, ha='center', va='bottom', color="#2980b9", bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    ax3.invert_xaxis()
    ax3.set_xlabel("Distance (m)  ← approaching"); ax3.set_ylabel("Detection Logit")
    ax3.set_title("Aligned Detection Logit vs Distance"); ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.25)
    ax4 = fig.add_subplot(gs[2, 1])
    markers = [200, 150, 100, 75, 50, 25, 10, 5]
    for rd in all_data:
        if not rd["times"]: continue
        dj = np.array([200*(1-(t-rd["start_s"])/rd["duration_s"]) for t in rd["times"]])
        pts_d, pts_l = [], []
        for dm in markers:
            i = np.argmin(np.abs(dj - dm))
            if i < len(rd["logits"]): pts_d.append(dm); pts_l.append(rd["logits"][i])
        ax4.plot(pts_d, pts_l, 'o-', ms=4, lw=1, color=colors[rd["run_id"]-1], alpha=0.7, label=f"Run {rd['run_id']}")
    ax4.set_xlabel("Distance (m)"); ax4.set_ylabel("Detection Logit")
    ax4.set_title("Logit at Distance Markers"); ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.25)
    fig.suptitle("Drone Detection: 200m → 0m Attack Runs  |  CNN14, 1.28s, focal+SpecAug", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = repo / "artifacts/detection_vs_distance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot: {out}")
    
    print(f"\n{'='*55}")
    print("  DISTANCE → DETECTION LOGIT (CNN14, 1.28s)")
    print(f"{'='*55}")
    print(f"  {'Dist':>6}  {'Logit':>8}  {'±1σ':>6}  {'Prob':>12}")
    print(f"  {'':->6}  {'':->8}  {'':->6}  {'':->12}")
    for dm in [200, 175, 150, 125, 100, 75, 50, 25, 10, 5, 2]:
        i = np.argmin(np.abs(dgrid - dm))
        if not np.isnan(ml[i]):
            p = 1/(1+np.exp(-ml[i]))
            print(f"  {dm:>5}m  {ml[i]:>+8.2f}  {sl[i]:>5.2f}  {p:>12.6f}")
    print("\n  Per-run peak logit (near ~20m, before acoustic shift):")
    for rd in all_data:
        valid = [v for v in rd["logits"] if v > -20]
        peak = max(valid) if valid else max(rd["logits"])
        print(f"    Run {rd['run_id']}: logit={peak:+.2f}  prob={1/(1+np.exp(-peak)):.6f}")
    print("\n  NOTE: ~4m chunks produce very negative logits (−28 to −32).")
    print("  This is likely the drone landing/motor-off at point-blank range.")
    print("  Peak detection occurs at ~15-25m, not at the closest point.")
    
    
    # ====================================================================
    # Entry point
    # ====================================================================
    
