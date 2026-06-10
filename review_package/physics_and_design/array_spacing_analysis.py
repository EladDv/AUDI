#!/usr/bin/env python3
"""array_spacing_analysis.py - how mic spacing affects the 4-mic array.

Plots the two things spacing actually controls:
  (1) f_max = c/(2d)  -> highest aliasing-free frequency (how high a harmonic
      you can still localize / beamform without direction ambiguity).
  (2) beamwidth ~ lambda / aperture  -> angular resolution at a reference freq
      (aperture = (N-1)*d for a uniform line array).
Read-only, just makes a chart.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = 343.0            # speed of sound, m/s
N = 4                # mics
OUT = Path(__file__).resolve().parent / "feature_probe_out" / "array_spacing.png"

d_cm = np.linspace(0.5, 12, 300)
d = d_cm / 100.0
f_max = C / (2 * d)                       # aliasing-free max freq (Hz)
aperture = (N - 1) * d                     # line-array aperture
f_ref = 1000.0                             # ref freq for resolution (Hz)
beamwidth_deg = np.degrees(C / (f_ref * aperture))   # ~ lambda/L (rad) -> deg

markers = [2, 4, 7, 10]                    # candidate spacings (cm)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(d_cm, f_max / 1000.0, color="tab:blue")
ax[0].axhspan(0.1, 6.0, color="tab:orange", alpha=0.15,
              label="typical drone harmonic band (~0.1-6 kHz)")
for m in markers:
    fm = C / (2 * m / 100.0) / 1000.0
    ax[0].plot(m, fm, "o", color="tab:red")
    ax[0].annotate(f"{m} cm -> {fm:.1f} kHz", (m, fm),
                   textcoords="offset points", xytext=(6, 6), fontsize=9)
ax[0].set_xlabel("mic spacing d (cm)")
ax[0].set_ylabel("aliasing-free max frequency  f_max = c/(2d)  (kHz)")
ax[0].set_title("Smaller spacing -> handles higher harmonics spatially")
ax[0].grid(alpha=0.3); ax[0].legend()

ax[1].plot(d_cm, beamwidth_deg, color="tab:green")
for m in markers:
    bw = np.degrees(C / (f_ref * (N - 1) * m / 100.0))
    ax[1].plot(m, bw, "o", color="tab:red")
    ax[1].annotate(f"{m} cm -> {bw:.0f}deg", (m, bw),
                   textcoords="offset points", xytext=(6, 6), fontsize=9)
ax[1].set_xlabel("mic spacing d (cm)")
ax[1].set_ylabel(f"beamwidth at {f_ref:.0f} Hz (deg, smaller=sharper)")
ax[1].set_title("Larger spacing -> sharper direction resolution (but aliases)")
ax[1].grid(alpha=0.3)

fig.suptitle("4-mic array: the spacing trade-off (c=343 m/s)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("saved ->", OUT)

print("\n f_max table (aliasing-free):")
for m in [1, 2, 3, 4, 5, 7, 10]:
    print(f"  d={m:>2} cm  ->  lambda/2 match at  f_max = {C/(2*m/100.0)/1000:5.2f} kHz")
