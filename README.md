# AUDI

**A**coustic **U**AV **D**etection and **I**dentification

A deep learning pipeline for real-time acoustic drone detection. Built on PyTorch Lightning with EfficientAT MN/DyMN backbones, audio augmentation, SNR-bin evaluation, and edge deployment (Raspberry Pi via TFLite).

---

## Features

- **EfficientAT training** — maintained MN/DyMN backbones for compact drone detection and edge export
- **SNR-bin evaluation** — measure performance across six signal-to-noise bins: easy (-5 to 0 dB) through far-field (-30 to -25 dB)
- **Rich augmentation** — MixUp, CutMix, SpecAugment, gain jitter, multi-noise background, atmospheric absorption filtering, Doppler shift
- **Crash-resilient sweeps** — Ctrl+C kills only the current run, not the whole sweep. Results stream to CSV incrementally
- **Bayesian hearability calibration** — per-bin Gaussian calibration maps logits to calibrated probabilities
- **Attack-run evaluation** — real-world detection metrics at calibrated precision thresholds with OOM recovery and incremental CSV saving
- **Interactive dashboard** — Streamlit app for model exploration, bin analysis, and attack-run diagnosis
- **Edge deployment** — FP32 TFLite export + Docker-based Raspberry Pi service with web UI, ring buffer storage, and GPIO alerting
- **Schmitt-trigger hysteresis** — stable detection state with configurable on/off ratios for deployment

---

## Quick Start

```bash
# Install dependencies
uv sync

# Train a single model
uv run audi-train \
    --noise-path data/my_background \
    --drone-path data/my_drone \
    --arch mn10_as \
    --lr 1e-4 \
    --mixup-alpha 0.2 \
    --epochs 15 \
    --patience 0 \
    --output-dir checkpoints/my_run

# Run a maintained sweep
uv run python sweeps/sweep.py sweeps/configs/mn10_06_new_tricks_finetune.yaml

# Postprocess + calibrate a sweep
uv run audi-eval postprocess checkpoints/<sweep_dir>
uv run audi-eval calibrate checkpoints/<sweep_dir>/<run_name>

# Run attack evaluation on all checkpoints
uv run audi-eval --noise-path data/my_background --drone-path data/my_drone attack-runs

# Compare DOA algorithms on the FPV flyover recording
uv run audi-eval doa-compare data/FPV_flyover_with_attacks.wav

# Launch the eval dashboard
uv run --extra eval streamlit run eval_app/

# Run tests
uv run pytest -q
```

---

## Requirements

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- CUDA-capable GPU recommended (8+ GB VRAM for most MN/DyMN runs)
- Audio data: drone recordings + background noise (see [Data Pipeline](#data-pipeline))

```bash
uv sync                    # core deps
uv sync --group dev        # + pytest, ruff, app-test audio frontend deps
uv sync --extra eval       # + streamlit, plotly dashboards
uv sync --extra export     # + TFLite export tooling
```

---

## Project Structure

```
src/audi/
  __init__.py              # Package metadata
  config.py                # Immutable dataclasses (ModelConfig, MelConfig, OptimizerConfig)
  augment.py               # Audio augmentation transforms
  checkpoint.py            # Checkpoint loading utilities
  hysteresis.py            # Schmitt-trigger hysteresis for deployment
  frontend.py              # Mel and STFT frontend variants
  hard_negative_mining.py  # Field false-positive mining helpers
  model/
    __init__.py            # build_model() factory + arch registry
    efficientat.py         # MN/DyMN/EfficientAT backbones
  training/
    dataset.py             # MixedDataset + binned SNR sampling
    detector.py            # DroneDetector LightningModule
    hearability.py         # ERB-band SNR scaling
    validation.py          # ROC, precision, threshold computation

scripts/
  cli/                     # Console entry points and maintained command modules
    _dispatch.py           # audi-eval and audi-data dispatch helpers
    train_detect.py        # audi-train detector training command
    export/                # FP32 audi-export-tflite and blue/red export

sweeps/
  sweep.py                 # YAML-driven sweep runner
  configs/                 # Sweep configuration YAML files (arch, regularization, etc.)

tests/                     # pytest test suite

audi-app/                  # Edge deployment (Raspberry Pi Docker service)
```

---

## Running Experiments

### Single Model Training

Train a detection model with `audi-train`:

```bash
uv run audi-train \
    --noise-path data/HF_dataset_v2_background \
    --drone-path data/HF_dataset_v2_drone \
    --arch mn10_as \
    --clip-seconds 5.12 \
    --lr 1e-4 \
    --lr-schedule linear \
    --warmup-epochs 8 \
    --epochs 25 \
    --batch-size 24 \
    --loss bce \
    --label-smoothing 0.1 \
    --augment \
    --output-dir checkpoints/my_experiment
```

Training produces:
- Checkpoints in `checkpoints/<run>/checkpoints/epoch=N-step=M.ckpt`
- TensorBoard logs in `checkpoints/<run>/lightning_logs/`
- A `sweep_config.yaml` with the full config

### Sweep Infrastructure

Sweeps are defined as YAML configs under `sweeps/configs/`. Each config specifies a `base_flags` shared across all runs plus per-config `flags` variations:

```yaml
# sweeps/configs/my_sweep.yaml
name: my_sweep
noise_path: data/my_background
drone_path: data/my_drone
description: My sweep description
base_flags: --arch mn10_as --mixup-alpha 0.2 --epochs 15 --patience 0
configs:
  - name: "01_baseline"
    flags: --lr 1e-4
  - name: "02_low_lr"
    flags: --lr 5e-5
  - name: "03_high_lr"
    flags: --lr 2e-4
```

Run the sweep:

```bash
uv run python sweeps/sweep.py sweeps/configs/my_sweep.yaml
```

Each sweep automatically:
1. Runs configs **sequentially** with crash resilience — Ctrl+C kills only the current run, saves partial results
2. Extracts validation metrics from TensorBoard event files after each run
3. Writes incremental `results.csv` with TPR@P90, AUC, and ECE per config
4. Runs `audi-eval postprocess` + `audi-eval calibrate` on completion
5. Creates a timestamped directory under `checkpoints/`

The sweep runner also supports `--no-postprocess` and `--no-calibrate` flags to skip post-sweep evaluation.

#### Available Sweep Configs

| Config | What it tests |
|--------|--------------|
| `blue_red_mn10_mined_hardneg_classifier.yaml` | Blue/red classifier follow-up on mined hard negatives |
| `efficientat_v7_noisier.yaml` | EfficientAT/MN size and noise coverage |
| `mn10_06_new_tricks_finetune.yaml` | MN10 mined-hard-negative finetune used as the deployment detector source |
| `mel_preprocessing_sweep.yaml` | Mel geometry and preprocessing research |
| `audio_resample_frontend_sweep.yaml` | 8 kHz 128-mel and 4 kHz linear-STFT frontend research |

Blue/red training and export are maintained commands:

```bash
uv run audi-train-blue-red --help
uv run --extra export audi-export-blue-red-tflite --help
```

---

## Training Reference

### Detection Training Flags

**Data:**
| Flag | Default | Description |
|------|---------|-------------|
| `--noise-path` | (required) | Background noise dataset directory |
| `--drone-path` | (required) | Drone audio dataset directory |
| `--noise2` | `None` | Secondary noise dataset for multi-noise training |
| `--snr-bin` | easy/medium/hard | SNR bins: `name:min:max:ratio`. Repeat for multiple bins |
| `--clip-seconds` | `1.28` | Audio clip length in seconds (1.28, 2.56, 5.12, 7.68, 10.24) |
| `--highpass-hz` | `125.0` | High-pass filter cutoff frequency |
| `--positive-probability` | `0.5` | Probability a training sample contains drone |

**Model:**
| Flag | Default | Description |
|------|---------|-------------|
| `--arch` | `mn10_as` | EfficientAT backbone: MN, DyMN, or static-DyMN variant |
| `--no-pretrained` | `False` | Train from scratch (no AudioSet pretrained weights) |
| `--no-compile` | `False` | Disable `torch.compile` |
| `--dropout` | `0.0` | Dropout rate (0.2 recommended for calibration) |
| `--bn-momentum` | `0.1` | Batch norm momentum |
| `--mel-preset` | `default` | Mel spectrogram preset: `default` (128 mels) or `custom` |
| `--n-fft` | preset | FFT size when `--mel-preset custom` is used |
| `--win-length` | `n_fft` | STFT analysis window length when `--mel-preset custom` is used |
| `--hop-length` | preset | Hop length when `--mel-preset custom` is used |

**Optimizer:**
| Flag | Default | Description |
|------|---------|-------------|
| `--lr` | `1e-3` | Learning rate |
| `--weight-decay` | `0.01` | AdamW weight decay (0.03 helps small datasets) |
| `--lr-schedule` | `constant` | LR schedule: `constant`, `cosine`, or `linear` |
| `--warmup-epochs` | `0` | LR warmup epochs (3–8 recommended with cosine/linear) |

**Training loop:**
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `30` | Maximum training epochs |
| `--batch-size` | `32` | Per-GPU batch size |
| `--steps-per-epoch` | `250` | Training steps per epoch (limits dataset passes) |
| `--val-steps-per-epoch` | `200` | Validation steps per epoch |
| `--patience` | `5` | Early stopping patience (0 = disable) |
| `--seed` | `42` | Random seed |
| `--output-dir` | `experiments` | Output directory |
| `--save-top-k` | `1` | Keep N best checkpoints |
| `--accumulate-grad-batches` | `1` | Gradient accumulation steps |
| `--num-workers` | `4` | Data loader worker processes |

**Regularization:**
| Flag | Default | Description |
|------|---------|-------------|
| `--loss` | `bce` | Loss: `bce` or `focal` |
| `--label-smoothing` | `0.0` | Label smoothing factor (0.1 recommended) |
| `--per-bin-weights` | `False` | Weight loss by SNR bin difficulty |
| `--spec-augment-prob` | `0.0` | SpecAugment probability (0.3 recommended) |
| `--mixup-alpha` | `0.0` | MixUp α (0.1–0.2 recommended) |
| `--cutmix-alpha` | `0.0` | CutMix α |
| `--augment` | `False` | Enable waveform augmentations such as Doppler, pitch, stretch, reverb, EQ, injected noise, masks, lowpass, and atmospheric filtering |

**Finetuning:**
| Flag | Default | Description |
|------|---------|-------------|
| `--finetune-from` | `None` | Path to checkpoint for full finetuning |

### Best-Practice Configs

**Quick baseline (15 epochs, good calibration):**
```bash
--arch mn10_as --lr 1e-4 --mixup-alpha 0.2 --epochs 15 --patience 0
```

**Extended training (50 epochs, best attack-run coverage):**
```bash
--arch mn10_as --lr 1e-4 --mixup-alpha 0.2 --epochs 50 --patience 0 --save-top-k 1
```

**Best calibration (dropout 0.2):**
```bash
--arch mn10_as --lr 1e-4 --dropout 0.2 --epochs 15 --patience 0
```

**Cosine schedule with warmup:**
```bash
--arch mn10_as --lr 1e-4 --lr-schedule cosine --warmup-epochs 3 --epochs 15 --patience 0
```

**Production MN10 with long clips:**
```bash
--arch mn10_as --clip-seconds 5.12 --lr 1e-4 --lr-schedule linear --warmup-epochs 8 \
    --loss bce --label-smoothing 0.1 --augment --epochs 25 --patience 0
```

---

## Attack-Run Evaluation

The attack-run evaluator scores every trained checkpoint on real drone flyover recordings. It measures **how quickly and reliably** a model detects actual drone approaches — the operational metric that matters most.

### How It Works

1. **Discover checkpoints** — scans `checkpoints/` for all `.ckpt` files and picks the best (highest epoch) per experiment
2. **Auto-postprocess** — runs `postprocess` on any checkpoint missing `eval_data/predictions_best.pt`
3. **Auto-calibrate** — runs `calibrate` on any checkpoint missing `eval_data/hearability_calib.npz`
4. **Precision thresholds** — computes per-model thresholds at P50, P60, P70, P75, P80, P85, P90, P95, P99 from validation ROC
5. **Attack evaluation** — loads each model, runs sliding-window inference on attack-run audio segments, applies Schmitt-trigger hysteresis at each precision threshold
6. **Incremental save** — writes results to `checkpoints/attack_run_precision_eval.csv` after each checkpoint (crash-resilient)

### Running

```bash
# Full auto: postprocess, calibrate, and evaluate all new checkpoints
uv run audi-eval \
    --noise-path data/HF_dataset_v2_background \
    --drone-path data/HF_dataset_v2_drone \
    attack-runs

# Skip auto-postprocess/calibrate (already done)
uv run audi-eval \
    --noise-path data/HF_dataset_v2_background \
    --drone-path data/HF_dataset_v2_drone \
    attack-runs --skip-postprocess --skip-calibrate

# Force re-evaluation of everything
uv run audi-eval \
    --noise-path data/HF_dataset_v2_background \
    --drone-path data/HF_dataset_v2_drone \
    attack-runs --all
```

### Output

Results are saved to `checkpoints/attack_run_precision_eval.csv`:

| Column | Description |
|--------|-------------|
| `model` | Experiment name within the sweep |
| `sweep` | Sweep directory name |
| `precision` | Precision target (P50–P99) |
| `sigma` | Detection threshold (probability) derived from validation |
| `cov_pct` | Mean % of attack windows above threshold — **higher is better** |
| `first_pct` | Median % of segment before first detection — **lower is better** |
| `bg` | Number of background windows that trigger false alarm — **lower is better** |

### Interpreting Results

A good model at P90 has:
- **cov% > 50** — detects drone in most attack windows
- **1st% < 30** — detects early in the approach
- **bg < 100** — minimal false alarms on 710 background windows

The script prints a ranked leaderboard sorted by coverage (minus bg penalty):

```
TOP MODELS at PRECISION=0.90
  # model                                          σ    cov%   1st%    bg sweep
  1 06_wd                                         0.7206   60.5   15.2    67 bce_push_20260517_083908
  2 03_wd_warmup8                                 0.7202   52.7   15.2    44 bce_wd_warmup_20260517_175750
  ...
```

### Field Evaluation

```bash
# Regenerate field alert TP/FP/FN table from attack-run thresholds
uv run audi-eval field

# Limit to one sweep directory
uv run audi-eval field \
    --sweep <sweep-name>
```

Results are saved to `checkpoints/field_eval_all.csv`.

---

## Data Pipeline

The `audi-data` command exposes the maintained preprocessing utilities used by detector training:

### Dataset Building Subcommands

#### `precompute-waveforms` / `precompute-features`

Precompute detection training shards for `audi-train --precomputed-*` runs. `precompute-waveforms` stores mixed waveform shards; `precompute-features` converts those shards to normalized frontend tensors and uses CUDA when available, otherwise CPU:

```bash
uv run audi-data precompute-waveforms \
    --noise-path data/HF_dataset_v2_background \
    --drone-path data/HF_dataset_v2_drone \
    --split train --num-examples 50000 \
    --output-dir data/precomputed/waveforms/train

uv run audi-data precompute-features \
    --waveform-path data/precomputed/waveforms/train \
    --split train \
    --output-dir data/precomputed/features/train
```

#### Current Field Utilities

```bash
uv run audi-data field-bg
uv run --with silero-vad audi-data blue-red-recordings
uv run audi-data mine-field-hard-negatives --checkpoint checkpoints/my_run/best.ckpt
```

### Data Directory Layout

Expected dataset structure under `data/`:

```
data/
  dataset_v2/                  # Raw dataset v2 for chunking
  attack_runs/                 # Real drone flyover recordings (*.wav)
  HF_dataset_v2_background/    # Background noise (train/val/test splits)
  HF_dataset_v2_drone/         # Drone audio (train/val/test splits)
  HF_dataset_v7_background/    # Field background windows
  field_hard_negatives/        # Mined field false-positive clips
  precomputed/                 # Optional waveform/frontend training shards
```

All `data/` and `checkpoints/` directories are git-ignored.

---

## Evaluation Workflow

After training, the standard evaluation pipeline:

### 1. Postprocess

Generates predictions and ROC curves for every checkpoint in a sweep:

```bash
uv run audi-eval postprocess checkpoints/<sweep_dir>
# Or for a specific run:
uv run audi-eval postprocess checkpoints/<sweep_dir> <run_name>
```

Saves to `eval_data/` inside each run directory:
- `predictions_best.pt` — validation logits, labels, bin indices
- `curves_best.npz` — per-bin ROC curves, thresholds, AUC

### 2. Calibrate

Fits a Bayesian SNR-bin estimator on positive-sample logits:

```bash
uv run audi-eval calibrate checkpoints/<sweep_dir>/<run_name>
```

Saves `eval_data/hearability_calib.npz` — per-bin Gaussian means, stds, priors, and decision boundaries.

### 3. Field Table

```bash
uv run audi-eval field
```

Writes the compact field alert table to `checkpoints/field_eval_all.csv`.

---

## Edge Deployment (Raspberry Pi)

The `audi-app/` directory contains a complete Docker-based deployment:

- Real-time audio capture via ALSA (`arecord`)
- TFLite FP32 inference at 320 ms intervals
- Schmitt-trigger hysteresis for stable YES/NO detection plus RED/BLUE typing
- GPIO alert outputs for configured alert levels (RED by default)
- Physical buttons (reset, record toggle, pause)
- Touch-friendly web UI on port 8080
- 32 GB ring buffer with automatic FLAC compression and eviction
- systemd service for auto-start on boot

See `audi-app/README.md` for full setup instructions.

```bash
# Export a detector-only FP32 TFLite model
uv run --extra export audi-export-tflite \
    --ckpt checkpoints/my_run/best.ckpt \
    --noise-path data/my_background \
    --drone-path data/my_drone

# Export the combined detector + blue/red classifier used by audi-app
uv run --extra export audi-export-blue-red-tflite \
    --ckpt checkpoints/my_blue_red_run/best.ckpt \
    --output audi-app/models/model_combined_mn10_mined_hardneg_blue_red.tflite
```

---

## License

GNU AGPLv3
GNU Affero General Public License v3.0
