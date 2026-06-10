# How to Run — Drone Detection System Review

## Prerequisites (install once)
```
pip install numpy scipy librosa soundfile scikit-learn torch tqdm matplotlib
```

---

## 1. Run the model on your own WAV file

```
cd core_model
python evaluate.py --wav path/to/your_audio.wav
```

This loads the trained model from `../trained_models/stage1_cnn/model.pt`,
computes the mel3 feature (PCEN + scalogram + modulation), runs the CNN,
and prints P(drone) for each 2-second window with a summary verdict.

---

## 2. See the features visually (spectrograms, scalogram, modulation)

```
cd core_model
python feature_gallery.py path/to/your_audio.wav
```

Saves a PNG showing all 3 channels of the mel3 feature side by side:
- Channel 0: PCEN-mel spectrogram (detection cue)
- Channel 1: Scalogram (frequency structure — EVO low vs FPV high)
- Channel 2: Modulation spectrogram (blade rhythm / RPM)

---

## 3. Run the full A/B experiment (CNN vs physics vs fusion)

```
cd core_model
python detect_experiment.py
```

Requires `clips.npz` + `features/mel3.npz` (the training dataset).
Outputs: AUC, recall, false alarm rate for 3 methods.

---

## 4. Understand the physics derivation

Read `physics_and_design/ARRAY_DESIGN.md` — this explains:
- How BPF and harmonics are computed from drone motor specs
- Why the small array uses 2.8 cm spacing (covers FPV 6 kHz harmonics)
- Why the large array uses ~40 cm spacing (EVO localization)
- All numbers validated both from physics and empirical measurements

Run the physics scripts to reproduce:
```
cd physics_and_design
python physics_bpf.py          # BPF vs measured frequencies
python type_frequencies.py     # measured EVO/FPV spectra from recordings
python discrim_band.py         # which frequencies separate drone types
```

---

## Folder structure

```
review_package/
├── HOW_TO_RUN.md               ← this file
├── core_model/                 ← full training pipeline (11 scripts)
│   ├── config.py               config, labels, data paths
│   ├── features.py             all features: mel3, scalogram, modulation, SCD, physics
│   ├── models.py               CNN architecture
│   ├── augment.py              field + spec augmentations
│   ├── scd_probe.py            SCD harmonic/rhythm feature
│   ├── build_clips.py          WAV → 2-sec clips
│   ├── make_features.py        clips → mel3 tensor features
│   ├── train.py                Stage 1: drone/no-drone CNN training
│   ├── train_stage2.py         Stage 2: EVO vs FPV classifier
│   ├── detect_experiment.py    A/B: CNN vs physics vs fusion
│   └── evaluate.py             evaluation and metrics
├── live_detector/              ← real-time 4-mic array detector
│   └── detect_server.py        Flask server + web UI on :8766
├── trained_models/             ← weights from the Jun 8 2026 training run
│   ├── stage1_cnn/
│   │   ├── model.pt            Stage 1 CNN weights (~1.4 MB)
│   │   └── train_log.json      per-epoch metrics
│   └── stage2_evo_fpv/
│       ├── stage2_model.npz    EVO/FPV classifier weights
│       ├── metrics.json        LOSO per-flight results
│       └── confusion_loso.png  confusion matrix
├── results/
│   ├── README.md               full training run summary with all metrics
│   ├── train_out.log           full training output
│   └── detect_experiment_metrics.json   CNN/physics/fusion comparison
└── physics_and_design/
    ├── ARRAY_DESIGN.md         array sizing from physics (the key design doc)
    ├── physics_bpf.py          BPF derivation script
    ├── type_frequencies.py     measured EVO/FPV spectra
    ├── discrim_band.py         frequency discrimination analysis
    └── feature_probe.py        feature noise-robustness ranking
```

---

## Key results (Jun 8, 2026 training run)

| Metric | Value |
|---|---|
| Stage 1 recall (drone detection) | **97.8%** |
| Stage 1 false alarm rate | **6.4%** |
| Stage 1 AUC (held-out test) | **0.961** |
| Stage 2 EVO/FPV file accuracy | **92.3%** (24/26 flights correct) |
| Training data | 877 files, ~372 min, 52,204 clips after slicing |
| Augmentations | Doppler, atmospheric low-pass, reverb, clipping, dropout |
