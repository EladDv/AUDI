# Microphone Array Design — Drone Detection, Classification & Tracking

Reference for sizing the mic array from **physics** (so it generalises to any drone
in a class, not just our recordings) and from our **empirical** measurements.
All numbers use the speed of sound **c = 343 m/s**.

---

## 1. The 4 questions you must know per drone

To predict a drone's tone from first principles you need **all four** (KV alone is
not enough):

| # | Question | Used for |
|---|----------|----------|
| 1 | **KV** (motor RPM per volt) | `Max RPM = KV × V` |
| 2 | **Pack voltage** (cell count, e.g. 6S ≈ 24 V) | `Max RPM = KV × V` |
| 3 | **Throttle %** (cruise ≈ 40 %, max = 100 %) | `Actual RPM = Max RPM × throttle` |
| 4 | **Blade count** (2 or 3) | `BPF = (RPM/60) × blades` |

**Commercial drones (e.g. Autel EVO Max 4T "T4"):** KV / throttle are not
published. Use the **alternate path**: blade count + a **measured / known RPM
range** (telemetry, teardown, or the recorded spectrum). For our EVO we used the
measured spectrum (fundamentals ~167–267 Hz, 2 blades).

---

## 2. Frequency → wavelength → spacing (the core chain)

```
1. Max RPM    = KV × Voltage
2. Actual RPM = Max RPM × throttle%
3. rev/s      = Actual RPM / 60
4. BPF (Hz)   = rev/s × blades          # fundamental; harmonics at 2×, 3×, ...
5. wavelength λ = c / f                  # f = highest harmonic of interest
6. max spacing  d ≤ λ/2 = c / (2·f)      # avoids spatial aliasing at f
```

**Worked example — FPV 5" (KV 2400, 6S = 24 V, 3 blades, throttle 40–100 %):**
```
Max RPM    = 2400 × 24            = 57,600 RPM
Actual RPM = 57,600 × 0.4–1.0     = 23,040 – 57,600 RPM
rev/s      = 384 – 960
BPF        = ×3 blades            = 1152 – 2880 Hz (fundamental)
harmonics                          → strong to ~8640 Hz (3rd)
λ          = 343 / 8640           = 0.0397 m
d ≤ λ/2    = 343/(2·8640)         = 0.0198 m ≈ 2.0 cm
```

---

## 3. Two independent ways we derived the band (and they agree)

**Method A — Empirical** (`type_frequencies.py`, `discrim_band.py`):
measure the recorded spectrum → read the energy band / centroid / top harmonic →
λ = c/f → d = λ/2.
Sources: `data_other/raw/data_clean_unit_06052026` (red_13 13", blue_7 7"),
`data_other/raw/dataset_v2` (5/7/10/13"), `06_model_v1/data/raw/live_session_*`
(your array EVO/FPV).

**Method B — Physics** (`physics_bpf.py`):
the 4 questions → BPF → λ → d, independent of any recording.

**Cross-check (figure `feature_probe_out/physics_bpf.png`):** measured centroids
fall inside the physics-predicted ranges → the empirical result is **explained by
physics**, not an artifact of our specific clips.

| Class | Physics fundamental | Top harmonic | Measured centroid | d = λ/2 (top harmonic) |
|-------|--------------------|--------------|-------------------|------------------------|
| FPV 5" (3-bl)  | 1152–2880 Hz | ~8640 Hz | 2111 Hz | **2.0 cm** |
| FPV 7" (3-bl)  | 816–2040 Hz  | ~6120 Hz | 744 Hz* | **2.8 cm** |
| FPV 10" (2-bl) | 352–880 Hz   | ~2640 Hz | 2372 Hz | **6.5 cm** |
| FPV 13" (2-bl) | 288–720 Hz   | ~2160 Hz | 1738 Hz | **7.9 cm** |
| FPV 15" (2-bl, ~500 KV est) | 160–400 Hz | ~1200 Hz | — (no data) | **14.3 cm** |
| EVO T4 (2-bl)  | 167–267 Hz   | ~800 Hz  | 747 Hz  | **21–64 cm** |

\* 7" measured from only 2 files → low confidence; physics fills the gap.
**15" has no recording → physics-only, not validated; record one to confirm.
Its band (~160–400 Hz) overlaps EVO → separate them by harmonic structure / rhythm,
not by pitch.**

**Design takeaway:** worst case is FPV 5" → a **dense spacing ≈ 2–2.8 cm** covers
the whole FPV class up to ~6–8 kHz. EVO's low fundamentals want a **wide** pair.

---

## 4. SNR boost & beamforming

- **Coherent (delay-and-sum) array gain** with N mics, uncorrelated noise:
  `G = 10·log10(N)` → +6 dB (4 mics), +9 dB (8), +11.5 dB (14).
  **~Independent of spacing** — spacing does **not** add summation gain.
- **Where spacing matters = grating lobes.** Clean (alias-free) beamforming only
  **below** `f_max = c/(2d)`:
  - 2.8 cm → clean to **6125 Hz** (covers FPV harmonics)
  - 7 cm → 2450 Hz, 15 cm → 1143 Hz (aliases the FPV band)
- Above `f_max` the beam grows phantom lobes (see `feature_probe_out/beampattern.png`)
  → noise from other directions leaks in → **lost SNR / interference rejection**.

**Rule:** to beamform a band cleanly, spacing must satisfy `d ≤ c/(2·f_top)`.

---

## 5. The conflict, and how it resolves

| Job | Frequencies it uses | Wants spacing | Why |
|-----|--------------------|---------------|-----|
| **Detection** (drone / no-drone) | low fundamentals (few 100 Hz) | any (15 cm fine) | strong low tones |
| **Classification** (type/size) | **high harmonics 2–6 kHz** | **small (~2–3 cm)** | beamform highs in noise |
| **Localization / tracking** | broadband / low | **large (~0.5–1 m)** | bigger delay = sharper bearing |

Classification and localization pull **opposite** directions → solve with **TWO
separate 4-mic arrays, each on its own clock** (fits the "max 4 mics per clock"
hardware limit — the limit only bans >4 mics on *one* clock, not two independent
arrays):

| Array | Mics / clock | Size | Job | Output |
|-------|--------------|------|-----|--------|
| **Small (dense)** | 4 / ESP32 #1 | ~2.8 cm spacing | beamform 2–6 kHz → **detect + classify** | DRONE? + EVO/FPV/size |
| **Large (wide)**  | 4 / ESP32 #2 | ~30 cm spacing (~0.9 m span) | **localize** | bearing / direction |

= 8 mics on 2 clocks (you have 14 → 6 spare for redundancy or to widen the large
array). Sizes chosen so the small array's `f_max = c/(2·0.028) ≈ 6 kHz` covers the
FPV harmonics, while the large array's aperture maximises bearing precision.

Empirical basis: `discrim_band.py` showed ~71 % of the FPV-size separation lives
**above** 1143 Hz; `feature_probe.py` showed rhythm/harmonic features (`scd`,
`modulation`, `harmonic_stack`) survive noise better than raw `mel`.

**Note:** a single wide array gives **direction (bearing)**, not a full נ.צ
coordinate. Exact position needs a 2-D layout or a second array location
(triangulation) — an optional later extension.

---

## 6. Localization & tracking — what to expect (honest)

- **A line array gives azimuth only**, with a **front/back mirror ambiguity** — not
  unambiguous 360°, no elevation.
- **Full 360° azimuth + elevation** → arrange the wide mics as a **2-D array**
  (cross / L / grid), not a single line.
- **One array gives DIRECTION, not POSITION** (no range). For an exact (x,y,z) point:
  - use **2–3 array nodes** separated by tens of metres and **triangulate** bearings, or
  - fuse with another sensor (radar / camera / RF).
- **Bearing precision:** with ~0.9 m aperture and good SNR, a few degrees (sharper
  for FPV/high-freq, coarser for EVO/low-freq). Cross-range error ≈ `R·Δθ`
  (e.g. 100 m × 3° ≈ ±5 m); range unknown from a single array.

---

## 7. Raspberry Pi integration, timing & outputs

**Pipeline (per frame):**
```
N-ch capture @16 kHz (synced)
 → VAD / energy gate
 → GCC-PHAT on WIDE sub-array       → bearing θ
 → delay-and-sum beamform DENSE     → enhanced mono (steered to θ)
 → features (harmonic + scd / mel2) → Stage-1 (drone?) → Stage-2 (type)
 → tracker smooths θ over time      → track
```

**Timing on RPi 5:**

| Output | Window / hop | Update | Latency |
|--------|--------------|--------|---------|
| Bearing (track) | 50–100 ms | ~10–20 Hz | ~0.1 s |
| Detection (drone?) | ~1 s / 0.5 s hop | ~1–2 Hz | ~1 s |
| Classification (type) | 1–2 s | ~1 Hz | ~1–2 s |

GCC-PHAT and beamforming are a few ms; CNN is tens of ms (with warmup).
Classification latency is set by the analysis **window** (need ~1–2 s of audio).

**Outputs:** detector → `NO-DRONE / DRONE` + conf; classifier → `EVO / FPV / size`
+ P(type); tracker → bearing (+ elevation if 2-D) + track id.

---

## 8. Capture hardware constraint (current vs target)

- **Current:** 4× INMP441 on a **shared clock**, built by combining the ESP32 I2S
  protocol with the INMP441 (2 mics per I2S bus via L/R word-select, ESP32 has
  2 I2S peripherals → **~4 mics max**). INMP441 is **L/R only — no TDM**, so you
  cannot daisy-chain more than 2 per data line.
- **To reach 8–14 mics you need a different front-end**, all on **one shared clock**:
  - a **TDM-capable multichannel ADC / codec** (e.g. analog mics + multi-ch ADC), or
  - a dedicated mic-array board / FPGA / multichannel USB interface.
- **Hard requirement:** every mic shares **BCLK + WS (frame clock)** — otherwise
  the inter-mic phase/TDOA is meaningless and beamforming/localization fail.

---

## 9. Anti-overfitting principles

- **Size the array from the physics *envelope*** (lowest EVO fundamental ~150 Hz →
  highest FPV harmonic ~6–8 kHz), not from specific recordings → swapping drone
  models changes your **data**, not your **hardware design**.
- **BPF moves with throttle**, so models keyed on *absolute* frequency are fragile.
  Classify on **harmonic structure / blade-count comb / rhythm (SCD)** — these are
  throttle-invariant and physical.
- **Validate leave-one-DRONE-out** (not just leave-one-flight-out); augment
  (level, noise, Doppler, distance); collect diverse drones per class.

---

## 10. Scripts (sources)

| Script | Produces |
|--------|----------|
| `training/type_frequencies.py` | per-type spectra, centroid, peaks → `feature_probe_out/type_frequencies.png` |
| `training/discrim_band.py` | which frequencies separate types → `discrim_band.png` |
| `training/feature_probe.py` | feature ranking + noise-robustness curves |
| `training/physics_bpf.py` | physics BPF vs measured → `physics_bpf.png` |
| `training/array_spacing.py` | f_max & resolution vs spacing → `array_spacing.png` |
| `training/beampattern_demo.py` | grating-lobe beam pattern → `beampattern.png` |
| `training/array_layout.py` | nested 14-mic layout → `array_layout.png` |
