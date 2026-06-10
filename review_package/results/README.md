# Training Run: mel3_cnn — 2026-06-08

## Overview
Full end-to-end training of the 2-stage drone detection and classification pipeline.
Trained on a RunPod RTX 4090 server (124 GB RAM, pod `clever_aquamarine_dragon`).

---

## Pipeline Summary

```
clips.npz (52,204 clips, 3.2 GB)
    └─► make_features.py --field-aug 1
         └─► mel3.npz (87,035 augmented clips, 8.7 GB)
              └─► train.py (Stage 1 CNN)
                   └─► experiments/mel3_cnn/model.pt
                        └─► train_stage2.py (Stage 2 EVO/FPV)
                        └─► detect_experiment.py (A/B comparison)
```

---

## Stage 1 — Drone / No-drone CNN (mel3_cnn)

**Features:** MEL3 (3-channel: PCEN-mel + Scalogram + Modulation spectrogram)  
**Augmentations:** field-aug (Doppler flyby, atmospheric low-pass, reverb, clipping, dropout)  
**Architecture:** CNN with frozen body + classification head  
**Training:** 30 pre-train epochs → 20 fine-tune epochs on array data  

| Metric | Pre-train (best) | Fine-tune (final) |
|---|---|---|
| Val accuracy | 86.1% | 89.3% |
| Recall (drone) | 78.6% | **97.8%** |
| False alarm rate | 5.9% | **6.4%** |

**Saved model:** `stage1_cnn/model.pt`  
**Training log:** `stage1_cnn/train_out.log`  
**Epoch-by-epoch:** `stage1_cnn/train_log.json`

---

## Stage 2 — EVO vs FPV Classifier (A/B comparison)

Leave-One-Flight-Out cross-validation on 3,875 clips from 26 flights (24 EVO, 2 FPV).

| Variant | Clip acc | **File acc (26 flights)** | Features used |
|---|---|---|---|
| **fused** | 73.7% | **92.3%** | CNN emb (128d) + SCD (256d) + physics (9d) |
| emb_only | 73.4% | **92.3%** | CNN emb (128d) only |
| no_stats | 74.2% | **92.3%** | CNN emb (128d) + SCD (256d) |

**Conclusion:** All three variants tie at 92.3% file-level accuracy.  
The CNN embedding alone is sufficient — SCD and physics cues add no measurable benefit for EVO/FPV classification with this dataset.

**Active model (live detector uses):** `stage2_fused/` (fused variant)

---

## Detection Experiment — CNN vs Physics vs Fusion

Evaluated on test split: 2,000 drone + 1,117 no-drone clips.

| Method | AUC | AP | Recall | False Alarm |
|---|---|---|---|---|
| **CNN only** | **0.961** | **0.973** | 93.1% | **8.4%** |
| Physics only (9 cues) | 0.656 | 0.749 | 93.7% | 67.3% |
| CNN + physics fusion | 0.949 | 0.968 | 93.0% | 14.8% |

**Conclusion:** CNN alone is the best detector.  
Physics cues alone are unreliable (67% false alarm). Fusing them with the CNN degrades performance (8.4% → 14.8% false alarm). Use CNN only for Stage 1.

---

## Data

| File | Size | Description |
|---|---|---|
| `clips.npz` | 3.2 GB | 52,204 raw audio clips (4-mic array + external datasets) |
| `mel3.npz` | 8.7 GB | MEL3 features with field augmentations (87k clips) |

**Clip breakdown (approximate):**
- EVO drone (array): ~24 flights
- FPV drone (array): ~2 flights  
- Negatives: quiet room, wind, speech, music, vehicles, gunfire, aircraft, helicopters

---

## Files in This Run

```
run_20260608_mel3_cnn/
├── README.md                         ← this file
├── stage1_cnn/
│   ├── model.pt                      ← Stage 1 CNN weights (used by live detector)
│   ├── train_log.json                ← per-epoch metrics
│   └── train_out.log                 ← full training output
├── stage2_fused/
│   ├── stage2_model.npz              ← EVO/FPV weights (fused: emb+SCD+9stats)
│   ├── metrics.json                  ← LOSO results per flight
│   └── confusion_loso.png            ← confusion matrix plot
├── stage2_emb_only/
│   ├── stage2_model.npz              ← EVO/FPV weights (CNN emb only)
│   ├── metrics.json
│   └── confusion_loso.png
├── stage2_no_stats/
│   ├── stage2_model.npz              ← EVO/FPV weights (emb + SCD, no physics)
│   ├── metrics.json
│   └── confusion_loso.png
├── detect_experiment/
│   ├── metrics.json                  ← CNN vs physics vs fusion comparison
│   └── detect_exp.log
└── logs/
    ├── stage2_fused.log
    ├── stage2_emb.log
    └── stage2_nostats.log
```

---

## Live Deployment

Copy to the live detector directory:
- `stage1_cnn/model.pt` → `experiments/mel3_cnn/model.pt`
- `stage2_fused/stage2_model.npz` → `stage2_evo_fpv/stage2_model.npz`

Recommended live settings:
- `threshold`: 0.49 (CNN best-F1 threshold from detect_experiment)
- `min_hits`: 3–5 consecutive detections before alerting
