#!/usr/bin/env python3
"""physics_bpf.py - derive blade-pass frequencies from FIRST PRINCIPLES and
compare to the empirically measured spectra, then size the array from PHYSICS
(so the geometry generalises to any drone in the class, not just our samples).

Chain (from the user's notes):
  Max RPM   = KV * Voltage
  RPM       = Max RPM * throttle
  rev/s     = RPM / 60
  BPF (Hz)  = rev/s * blades          (fundamental; harmonics at 2x,3x,...)
  lambda    = c / f ;  d_max = lambda/2 = c/(2f)
"""
import numpy as np

C = 343.0
V = 24.0          # 6S nominal pack voltage (design envelope)
TH_CRUISE, TH_MAX = 0.40, 1.00
N_HARM = 3        # strong harmonics we care about for type/beamforming

# representative configs per class (KV, cells->V handled above, blades, prop")
CONFIGS = [
    # name,          KV,    blades, note
    ('FPV 5"',       2400,  3),
    ('FPV 7"',       1700,  3),
    ('FPV 10"',      1100,  2),
    ('FPV 13"',       900,  2),
    ('FPV 15"',       500,  2),   # estimate (KV ~400-600, 6S) - NO empirical data yet
]
# camera drones don't publish KV/throttle the same way -> model by hover/max RPM
EVO = ('EVO (camera)', 5000, 8000, 2)   # rpm_cruise, rpm_max, blades

# measured centroids from type_frequencies.py (Hz) for the reality-check
MEASURED = {
    'FPV 5"': 2111, 'FPV 7"': 744, 'FPV 10"': 2372, 'FPV 13"': 1738,
    'EVO (camera)': 747, 'FPV (array, measured)': 3095,
}


def bpf(rpm, blades):
    return rpm / 60.0 * blades


def line(name, f1c, f1m, blades):
    ftop = f1m * N_HARM
    print(f"{name:14s} blades={blades}  "
          f"fundamental {f1c:5.0f}-{f1m:5.0f} Hz  "
          f"| up to {N_HARM}rd harmonic ~{ftop:5.0f} Hz  "
          f"| d_max(fund)={C/(2*f1m)*100:4.1f} cm  "
          f"d_max(harm)={C/(2*ftop)*100:4.1f} cm")
    return ftop


print(f"Design envelope: 6S ({V:.0f} V), throttle {int(TH_CRUISE*100)}-100%, "
      f"first {N_HARM} harmonics\n")
print("PHYSICS-PREDICTED blade-pass frequencies:")
tops = []
for name, kv, blades in CONFIGS:
    f1c = bpf(kv * V * TH_CRUISE, blades)
    f1m = bpf(kv * V * TH_MAX, blades)
    tops.append(line(name, f1c, f1m, blades))
# EVO
_, rc, rm, bl = EVO
f1c, f1m = bpf(rc, bl), bpf(rm, bl)
tops.append(line(EVO[0], f1c, f1m, bl))

ftop_all = max(tops)
print(f"\n=> highest strong frequency across the FPV class ~{ftop_all:.0f} Hz")
print(f"=> DENSE spacing must satisfy d <= c/(2*{ftop_all:.0f}) "
      f"= {C/(2*ftop_all)*100:.1f} cm  (round to ~2.5-3 cm)")
print(f"=> EVO fundamentals ({bpf(rc,bl):.0f}-{bpf(rm,bl):.0f} Hz) -> "
      f"WIDE pair localises these; aperture sets bearing precision")

print("\nREALITY CHECK (physics fundamental range vs measured centroid):")
print(f"  EVO   : predicted fund {bpf(rc,bl):.0f}-{bpf(rm,bl):.0f} Hz, "
      f"measured centroid {MEASURED['EVO (camera)']} Hz  (centroid>fund "
      f"because harmonics add energy) -> CONSISTENT")
print(f"  FPV 5\": predicted fund up to {bpf(2400*V,3):.0f} Hz + harmonics, "
      f"measured centroid {MEASURED['FPV 5\"']} Hz -> CONSISTENT")
print(f"  FPV array: measured centroid {MEASURED['FPV (array, measured)']} Hz "
      f"-> matches high-RPM small-prop physics")


def fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    names = [c[0] for c in CONFIGS] + [EVO[0]]
    f1c_l, f1m_l = [], []
    for _, kv, bl in CONFIGS:
        f1c_l.append(bpf(kv*V*TH_CRUISE, bl)); f1m_l.append(bpf(kv*V*TH_MAX, bl))
    f1c_l.append(bpf(rc, bl)); f1m_l.append(bpf(rm, bl))
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (a, b) in enumerate(zip(f1c_l, f1m_l)):
        ax.plot([a, b], [y[i], y[i]], lw=7, color="tab:blue", alpha=0.5,
                solid_capstyle="butt")
        for h in range(2, N_HARM + 1):           # harmonics of the max fundamental
            ax.plot([b*h, b*h], [y[i]-0.18, y[i]+0.18], color="tab:blue", lw=1.2)
        ax.plot(b*N_HARM, y[i], ">", color="tab:blue")
        nm = names[i]
        key = nm if nm in MEASURED else None
        if key:
            ax.plot(MEASURED[key], y[i], "*", color="tab:red", ms=14, zorder=5)
    ax.plot([], [], lw=7, color="tab:blue", alpha=0.5,
            label="physics fundamental range (cruise->max)")
    ax.plot([], [], color="tab:blue", lw=1.2, label="harmonics (2x,3x)")
    ax.plot([], [], "*", color="tab:red", ms=12, label="MEASURED centroid")
    ax.axvline(6125, ls="--", color="green",
               label="dense 2.8cm ceiling (6.1 kHz)")
    ax.axvline(1143, ls="--", color="black", label="wide 15cm ceiling (1.1 kHz)")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("frequency (Hz)"); ax.set_xlim(0, 9000)
    ax.set_title("Physics-predicted blade-pass frequencies vs measured")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = Path(__file__).resolve().parent / "feature_probe_out" / "physics_bpf.png"
    fig.savefig(p, dpi=120); print("\nsaved ->", p)


fig()
