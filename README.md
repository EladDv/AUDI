# AUDI

**A**coustic **U**AV **D**etection and **I**dentification

A deep learning pipeline for real-time acoustic drone detection. Built on PyTorch Lightning with support for multiple architectures, audio augmentation, SNR-bin evaluation, and edge deployment (Raspberry Pi via TFLite).

---

## Features

- **Multi-architecture training** — PANNs (CNN8/10/14), ResNet (18/34/50), ConvNeXt (tiny/small/base), EfficientNet, MobileViT, MobileNetV4, and more
- **SNR-bin evaluation** — measure performance across six signal-to-noise bins: easy (-5 to 0 dB) through far-field (-30 to -25 dB)
- **Rich augmentation** — MixUp, CutMix, SpecAugment, gain jitter, multi-noise background, atmospheric absorption filtering, Doppler shift
- **Crash-resilient sweeps** — Ctrl+C kills only the current run, not the whole sweep. Results stream to CSV incrementally
- **Bayesian hearability calibration** — per-bin Gaussian calibration maps logits to calibrated probabilities
- **Attack-run evaluation** — real-world detection metrics at calibrated precision thresholds with OOM recovery and incremental CSV saving
- **Interactive dashboard** — Streamlit app for model exploration, bin analysis, and attack-run diagnosis
- **Edge deployment** — TFLite export + Docker-based Raspberry Pi service with GPIO alarm, web UI, and ring buffer storage
- **Schmitt-trigger hysteresis** — stable detection state with configurable on/off ratios for deployment

---

## Quick Start

```bash
# Install dependencies
uv sync

# Train a single model
uv run python scripts/train.py \
    --noise-path data/my_background \
    --drone-path data/my_drone \
    --arch resnet18 \
    --lr 1e-4 \
    --mixup-alpha 0.2 \
    --epochs 15 \
    --patience 0 \
    --output-dir checkpoints/my_run

# Run an architecture sweep
uv run python sweeps/sweep.py sweeps/configs/arch.yaml

# Postprocess + calibrate a sweep
uv run python scripts/evaluate.py postprocess checkpoints/<sweep_dir>
uv run python scripts/evaluate.py calibrate checkpoints/<sweep_dir>/<run_name>

# Run attack evaluation on all checkpoints
uv run python scripts/evaluate.py --noise-path data/my_background --drone-path data/my_drone attack-runs

# Launch the eval dashboard
uv run streamlit run scripts/eval_app.py --server.port 8501

# Run tests
uv run pytest tests/ -v
```

---

## Requirements

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- CUDA-capable GPU recommended (8+ GB VRAM for most models; 12+ GB for ConvNeXt-Base and long clips)
- Audio data: drone recordings + background noise (see [Data Pipeline](#data-pipeline))

```bash
uv sync                    # core deps
uv sync --group dev        # + pytest, mypy, ruff
uv sync --group eval       # + streamlit, plotly
```

---

## Project Structure

```
src/audi/
  __init__.py              # Package metadata
  config.py                # Immutable dataclasses (ModelConfig, MelConfig, OptimizerConfig)
  augment.py               # Audio augmentation transforms
  checkpoint.py            # Checkpoint loading utilities
  cli_utils.py             # CLI argument helpers
  hearability_estimator.py # ERB-band SNR scaling and estimation
  hysteresis.py            # Schmitt-trigger hysteresis for deployment
  model/
    __init__.py            # build_model() factory + arch registry
    _base.py               # AudioBackbone ABC
    panns.py               # PANNs CNN8/10/14
    vision.py              # ResNet + ConvNeXt + EfficientNet backbones
  training/
    dataset.py             # MixedDataset + binned SNR sampling
    detector.py            # DroneDetector LightningModule
    hearability.py         # ERB-band SNR scaling
    validation.py          # ROC, precision, threshold computation
    validation_plots.py    # TensorBoard visualization

scripts/
  train.py                 # Training CLI (detect/classify subcommands)
  evaluate.py              # Postprocess, calibrate, attack-runs, thresholds, ensemble
  inference.py             # Inference on audio files
  build_data.py            # Dataset building utilities
  export_tflite.py         # TFLite model export for edge deployment
  eval_ensemble.py         # Ensemble prediction combination

sweeps/
  sweep.py                 # YAML-driven sweep runner
  configs/                 # Sweep configuration YAML files (arch, regularization, etc.)

tests/                     # pytest test suite

audi-app/                  # Edge deployment (Raspberry Pi Docker service)
```

---

## Running Experiments

### Single Model Training

Train a detection model with `train.py detect` (the default subcommand):

```bash
uv run python scripts/train.py \
    --noise-path data/HF_dataset_v2_background \
    --drone-path data/HF_dataset_v2_drone \
    --arch convnext_small \
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
base_flags: --arch convnext_small --mixup-alpha 0.2 --epochs 15 --patience 0
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
4. Runs `evaluate.py postprocess` + `evaluate.py calibrate` on completion
5. Creates a timestamped directory under `checkpoints/`

The sweep runner also supports `--no-postprocess` and `--no-calibrate` flags to skip post-sweep evaluation.

#### Available Sweep Configs

| Config | What it tests |
|--------|--------------|
| `arch.yaml` | All architectures (cnn8/10/14, resnet18/34/50, convnext tiny/small/base, efficientnet b1/b3/b5, edgenet variants, mobilenet, mobilevit) |
| `regularization.yaml` | Weight decay, dropout, cosine schedule variants |
| `multinoise.yaml` | Multi-noise background (secondary noise dataset) across architectures |
| `multires.yaml` | Clip lengths: 1.28s, 2.56s, 5.12s |
| `finetune.yaml` | Finetuning from pretrained checkpoints |
| `classify.yaml` | DADS classification pretraining sweeps |
| `production.yaml` | Production sweep combining best hyperparams |
| `bce_wd_sweep.yaml` | BCE loss with varying weight decay |
| `bce_wd_warmup.yaml` | Weight decay × warmup combinations |
| `bce_push.yaml` | BCE training with extended epochs |
| `convnext_bfr.yaml` | ConvNeXt with focal loss, specaug, long clips |
| `convnext_lr.yaml` | ConvNeXt learning rate sweep |
| `convnext_reg.yaml` | ConvNeXt regularization (focal, mixup, dropout, weight decay) |
| `prod_focal.yaml` | Production focal loss sweep |
| `prod_focal_long.yaml` | Production focal loss with 30 epochs |
| `prod_escape.yaml` | Production escape (longer training, no early stopping) |

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
| `--arch` | `cnn14` | Backbone: cnn8/10/14, resnet18/34/50, convnext_tiny/small/base, efficientnet_b1/b3/b5, edgenet_*, mobilenetv4_*, mobilevitv2_* |
| `--no-pretrained` | `False` | Train from scratch (no ImageNet pretrained weights) |
| `--no-compile` | `False` | Disable `torch.compile` |
| `--dropout` | `0.0` | Dropout rate (0.2 recommended for calibration) |
| `--bn-momentum` | `0.1` | Batch norm momentum |
| `--mel-preset` | `default` | Mel spectrogram preset: `default` (128 mels) or `vit_224` (224×224) |

**Optimizer:**
| Flag | Default | Description |
|------|---------|-------------|
| `--lr` | `1e-3` | Learning rate (use 1e-3 for CNNs, 1e-4 for ResNets/ConvNeXts) |
| `--weight-decay` | `0.01` | AdamW weight decay (0.03 helps small datasets) |
| `--lr-schedule` | `constant` | LR schedule: `constant`, `cosine`, or `linear` |
| `--warmup-epochs` | `0` | LR warmup epochs (3–8 recommended with cosine/linear) |

**Training loop:**
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `30` | Maximum training epochs |
| `--batch-size` | `32` | Per-GPU batch size |
| `--steps-per-epoch` | `100` | Training steps per epoch (limits dataset passes) |
| `--val-steps-per-epoch` | `40` | Validation steps per epoch |
| `--patience` | `5` | Early stopping patience (0 = disable) |
| `--seed` | `42` | Random seed |
| `--output-dir` | `checkpoints` | Output directory |
| `--save-top-k` | `1` | Keep N best checkpoints |
| `--accumulate-grad-batches` | `1` | Gradient accumulation steps |
| `--num-workers` | (auto) | Data loader worker processes |

**Regularization:**
| Flag | Default | Description |
|------|---------|-------------|
| `--loss` | `bce` | Loss: `bce` or `focal` |
| `--label-smoothing` | `0.0` | Label smoothing factor (0.1 recommended) |
| `--per-bin-weights` | `False` | Weight loss by SNR bin difficulty |
| `--spec-augment-prob` | `0.0` | SpecAugment probability (0.3 recommended) |
| `--mixup-alpha` | `0.0` | MixUp α (0.1–0.2 recommended) |
| `--cutmix-alpha` | `0.0` | CutMix α |
| `--augment` | `False` | Enable gain jitter + background swap |

**Finetuning:**
| Flag | Default | Description |
|------|---------|-------------|
| `--finetune-from` | `None` | Path to checkpoint for full finetuning |
| `--pretrained-checkpoint` | `None` | Path to pretrained backbone weights |

### Classification Pretraining

Pretrain a backbone on raw drone-vs-non-drone audio (no background mixing):

```bash
uv run python scripts/train.py classify \
    --drone-path data/my_drone_classify \
    --model-arch resnet18 \
    --lr 1e-4 \
    --epochs 10 \
    --output-dir checkpoints/classify
```

Useful for downstream finetuning on the detection task — classification pretraining teaches the backbone to recognize drone spectral patterns before adding background noise.

### Best-Practice Configs

**Quick baseline (15 epochs, good calibration):**
```bash
--arch resnet18 --lr 1e-4 --mixup-alpha 0.2 --epochs 15 --patience 0
```

**Extended training (50 epochs, best attack-run coverage):**
```bash
--arch resnet18 --lr 1e-4 --mixup-alpha 0.2 --epochs 50 --patience 0 --save-top-k 1
```

**Best calibration (dropout 0.2):**
```bash
--arch resnet18 --lr 1e-4 --dropout 0.2 --epochs 15 --patience 0
```

**Cosine schedule with warmup:**
```bash
--arch resnet18 --lr 1e-4 --lr-schedule cosine --warmup-epochs 3 --epochs 15 --patience 0
```

**Production ConvNeXt with long clips:**
```bash
--arch convnext_small --clip-seconds 5.12 --lr 1e-4 --lr-schedule linear --warmup-epochs 8 \
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
uv run python scripts/evaluate.py \
    --noise-path data/HF_dataset_v2_background \
    --drone-path data/HF_dataset_v2_drone \
    attack-runs

# Skip auto-postprocess/calibrate (already done)
uv run python scripts/evaluate.py \
    --noise-path data/HF_dataset_v2_background \
    --drone-path data/HF_dataset_v2_drone \
    attack-runs --skip-postprocess --skip-calibrate

# Force re-evaluation of everything
uv run python scripts/evaluate.py \
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

### Other Evaluation Commands

```bash
# Per-bin thresholds at 10% FPR
uv run python scripts/evaluate.py fpr-thresholds

# Thresholds at multiple FPR targets (1%–50%)
uv run python scripts/evaluate.py fpr-multi

# Operational metrics: detections vs FPs at deployment thresholds
uv run python scripts/evaluate.py operational

# Ensemble two models by averaging logits
uv run python scripts/evaluate.py ensemble
```

---

## Data Pipeline

The `build_data.py` script handles all data preprocessing. Run subcommands to build datasets from scratch:

### Dataset Building Subcommands


#### `urban-esc`

Build an expanded background noise dataset from 8 public audio sources (ESC-50, UrbanSound8K, TUT, MUSAN, DEMAND, etc.):

```bash
uv run python scripts/build_data.py urban-esc \
    --output-path data/hf_background_urban \
    --chunk-min 5 --chunk-max 30 \
    --max-clips-per-category 0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output-path` | (required) | Output dataset directory |
| `--chunk-min` | 5 | Minimum chunk duration (seconds) |
| `--chunk-max` | 30 | Maximum chunk duration (seconds) |
| `--datasets` | all | Specific datasets: `esc50`, `urbansound`, `ambient`, `musan`, `demand`, `tut`, `sunbird` |
| `--max-clips-per-category` | 0 | Clip limit per category (0 = auto-weighted: wind 1000, cars 900, etc.) |

#### `filtered-hf`

Convert raw audio directories into train/val/test HF dataset splits:

```bash
# Build filtered dataset
uv run python scripts/build_data.py filtered-hf \
    --input-dir data/dataset_v2 \
    --output-path data/HF_dataset_v2 \
    --chunk-sec 30
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | (required) | Directory with audio subfolders |
| `--output-path` | (required) | Output HF dataset path |
| `--target-sr` | 16000 | Target sample rate |
| `--label` | (auto) | Label for classification (`drone` / `noise`) |
| `--split-background` | `True` | Split BG subfolder into `_background` dataset |
| `--chunk-sec` | 30 | Chunk audio into fixed-length segments |
| `--val-ratio` | 0.1 | Validation split ratio |
| `--test-ratio` | 0.1 | Test split ratio |

#### `chunk-spectro`

Chunk audio files and render mel spectrograms for visualization:

```bash
uv run python scripts/build_data.py chunk-spectro \
    --dataset-v2-dir data/dataset_v2 \
    --output-dir artifacts/chunked_15s \
    --chunk-sec 15 --n-mels 128
```

#### `hearability-templates`

Build ERB-band hearability templates from background noise. These templates characterize the noise floor in each frequency band and are used for SNR-based training bin assignment.

```bash
uv run python scripts/build_data.py hearability-templates \
    --dataset-path data/HF_dataset_v2_background \
    --output-path artifacts/hearability_templates \
    --erb-bands 28 --template-percentile 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-path` | (required) | HF dataset directory |
| `--output-path` | (auto) | Output directory for templates |
| `--erb-bands` | 28 | Number of ERB frequency bands |
| `--template-percentile` | 50 | Percentile for noise floor template |

#### `pretrain-drones`

Download and preprocess the geronimobasso drone audio detection dataset from HuggingFace for classification pretraining:

```bash
uv run python scripts/build_data.py pretrain-drones
```

#### `dads-classify`

Preprocess the DADS (Drone Audio Detection Signals) dataset for classification pretraining:

```bash
uv run python scripts/build_data.py dads-classify
```

#### `analyze-snr`

Analyze per-band signal-to-noise ratio between drone and noise recordings:

```bash
uv run python scripts/build_data.py analyze-snr \
    --drone-path data/drone \
    --noise-path data/noise \
    --output-dir artifacts/snr_analysis
```

| Flag | Default | Description |
|------|---------|-------------|
| `--drone-path` | (required) | Drone audio directory |
| `--noise-path` | (required) | Noise audio directory |
| `--output-dir` | (auto) | Output directory for plots |
| `--drone-gain-db` | (auto) | Drone gain levels in dB (e.g. `0 -10 -20`) |

#### `mel-stats`

Compute mel-spectrogram mean and standard deviation for model normalization:

```bash
uv run python scripts/build_data.py mel-stats
```

### Data Directory Layout

Expected dataset structure under `data/`:

```
data/
  dataset_v2/                  # Raw dataset v2 for chunking
  attack_runs/                 # Real drone flyover recordings (*.wav)
  hf_dataset_sharded_expanded/ # Secondary noise for multi-noise training (urban_esc50.py)
  HF_dataset_v2_background/    # Background noise (train/val/test splits)
  HF_dataset_v2_drone/         # Drone audio (train/val/test splits)
  hf_dataset_sharded_expanded_merged/ # Secondary noise for multi-noise training + audioset noises (audioset_fp.py)
```

All `data/` and `checkpoints/` directories are git-ignored.

---

## Evaluation Workflow

After training, the standard evaluation pipeline:

### 1. Postprocess

Generates predictions and ROC curves for every checkpoint in a sweep:

```bash
uv run python scripts/evaluate.py postprocess checkpoints/<sweep_dir>
# Or for a specific run:
uv run python scripts/evaluate.py postprocess checkpoints/<sweep_dir> <run_name>
```

Saves to `eval_data/` inside each run directory:
- `predictions_best.pt` — validation logits, labels, bin indices
- `curves_best.npz` — per-bin ROC curves, thresholds, AUC

### 2. Calibrate

Fits a Bayesian SNR-bin estimator on positive-sample logits:

```bash
uv run python scripts/evaluate.py calibrate checkpoints/<sweep_dir>/<run_name>
```

Saves `eval_data/hearability_calib.npz` — per-bin Gaussian means, stds, priors, and decision boundaries.

### 3. Eval Dashboard

```bash
uv run streamlit run scripts/eval_app.py --server.port 8501
```

Interactive web UI for exploring model predictions, per-bin ROC curves, and attack-run diagnosis.

---

## Edge Deployment (Raspberry Pi)

The `audi-app/` directory contains a complete Docker-based deployment:

- Real-time audio capture via ALSA (`arecord`)
- TFLite int8 inference at 320ms intervals
- Schmitt-trigger hysteresis for stable YES/BLUE/NO detection
- GPIO alarm outputs (buzzer, strobe LED)
- Physical buttons (reset, record toggle, pause)
- Touch-friendly web UI on port 8080
- 32 GB ring buffer with automatic FLAC compression and eviction
- systemd service for auto-start on boot

See `audi-app/README.md` for full setup instructions.

```bash
# Export a trained model to TFLite for edge deployment
uv run python scripts/export_tflite.py \
    --ckpt checkpoints/my_run/best.ckpt \
    --noise-path data/my_background \
    --drone-path data/my_drone
```

---

## License

MIT
