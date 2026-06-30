# AUDI — Type A

**Continuous audio recording + drone detection for Raspberry Pi.** Records audio in 5-minute segments to a 32GB ring buffer on disk, keeps the last 120 seconds of PCM in memory for ML inference, MUSIC direction-of-arrival estimates, and alarm snapshots, and drives GPIO outputs for configured alert levels. DOA is event-driven: it runs only after the detector reaches a positive drone state. The default deployment alerts on RED detections and records BLUE/UNKNOWN detections without firing the alarm. Touch-friendly web UI on port 8080.

Open http://raspberry-pi-ip:8080 from any browser on your local network.

## Quick Start

```bash
# On your Raspberry Pi (arm64):
make build    # Build the Docker image
make install-service && sudo systemctl start audi           # Auto-start + run now
```

Or for a fully automated install:

```bash
sudo bash scripts/install-pi.sh
```

Install-time channel selection:

```bash
# 4-channel capture, infer on all captured channels
sudo bash scripts/install-pi.sh --audio-channels 4 --detector-channels all

# 16-channel capture, infer only on channels 1, 3, 7, and 8
sudo bash scripts/install-pi.sh --audio-channels 16 --detector-channels 1,3,7,8
```

To deploy from your workstation to a Pi over SSH:

```bash
scripts/deploy-pi.sh <pi-ip> <username> '<password>'
scripts/deploy-pi.sh --host <pi-ip> --user pi --audio-channels 16 --detector-channels 1,3,7,8
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│             │     │              │     │               │
│  ALSA Mic   │────▶│  Recorder    │────▶│  Storage Mgr  │
│  (arecord)  │     │  (5m segs)   │     │  (FLAC + 32GB)│
│             │     │              │     │               │
└─────────────┘     └──────┬───────┘     └───────┬───────┘
                           │                     │
                    ┌──────▼───────┐     ┌───────▼───────┐
                    │  Ring Buffer  │     │  Alerts Store │
                    │  (120s in RAM)│     │  (pre+post    │
                    └──────┬───────┘     │   snapshots)  │
                           │             └───────────────┘
                    ┌──────▼───────┐     ┌───────────────┐
                    │  Detector    │────▶│  GPIO Alert   │
                    │  YES / NO +  │     │  + LEDs +     │
                    │  RED / BLUE  │     │  Buttons      │
                    └──────┬───────┘     └───────────────┘
                           │YES
                    ┌──────▼───────┐
                    │  MUSIC DOA   │
                    │  (on demand) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Web UI      │
                    │  (port 8080) │
                    └──────────────┘
```

### Subsystems

| Module | File | What it does |
|--------|------|-------------|
| **Recorder** | `src/recorder.py` | Captures audio via `arecord` (ALSA) in 5-minute WAV segments. Maintains an **in-memory ring buffer** of the last 120 seconds of float32 PCM samples (60s pre-alarm + 60s post-alarm). Supports pause/resume and stop/start toggling. |
| **Storage** | `src/storage.py` | Compresses old WAVs to FLAC (~6:1 ratio). Enforces a **32GB storage cap** — oldest files are evicted first when over budget or disk space runs low. |
| **Detector** | `src/detector.py` | FP32 TFLite classifier with Schmitt-trigger YES/NO hysteresis (5 of 8 full-window votes), RED/BLUE typing, alert snapshot saving, and temporal confidence tracking. |
| **DOA Estimator** | `src/doa_estimator.py` | Event-driven MUSIC azimuth estimator. Uses configured mic channels, HPS/CFAR peak picking, and harmonic STFT bins, and runs only after model-positive drone detections. |
| **GPIO** | `src/gpio_alarm.py` | Drives Pi GPIO pins: ALERT (buzzer/relay), STROBE (blinking LED), RESET (physical button to silence), PAUSE_BTN (pause 5 min), and green/yellow/red field-tag buttons for detection review. Graceful mock fallback when not on Pi hardware. |
| **Web UI** | `src/webui_server.py` | Flask server serving a **touch-optimized HTML UI**. Big Start/Stop buttons (green/red, mutual exclusive), Silence and Pause controls, live VU meter, YES/NO state card with RED/BLUE typing, alert history, system info panel. |
| **Main** | `src/main.py` | Orchestrator — starts everything, wires callbacks, handles graceful shutdown on SIGTERM/SIGINT. |

## Config

Edit `config.yaml`:

```yaml
audio:
  device: "auto"                     # Auto-discovers a USB mic, or use an ALSA device from arecord -l
  sample_rate: 16000
  channels: 4                        # Use --audio-channels 16 at install time for 16-channel capture
  segment_duration: 300              # 5-minute segments
  ring_buffer_seconds: 120           # 60s pre + 60s post alarm
  device_retry_min: 2                # Min seconds between retries
  device_retry_max: 60               # Max (doubles per failure)

storage:
  max_size_gb: 32
  compress: true                     # FLAC up to 8 channels; WavPack for 9+ channels
  data_dir: /data/recordings
  alerts_dir: /data/alerts
  max_alerts_gb: 2

detection:
  model_path: /app/models/model_combined_mn10_mined_hardneg_blue_red.tflite
  inference_interval: 0.320
  active_threshold_profile: mn10_p90
  threshold_profiles_file: ./threshold_profiles.yaml
  enabled_channels: null             # null/all = every captured channel; e.g. [1, 3, 7, 8]
  mel_mean: 10.430418
  mel_std: 5.288271
  confidence_threshold_high: 0.6550  # YES detector state
  red_color_threshold: 0.60           # RED if red confidence is at least this
  blue_color_threshold: 0.60          # BLUE if blue confidence is at least this
  alert_on_red: true                 # RED detections fire GPIO by default
  alert_on_blue: false
  alert_on_unknown: false

doa:
  enabled: true                       # Computed only when the detector reaches YES
  active_profile: triangle_3          # Runtime-switchable from the web UI
  disabled_channels: []               # Runtime bad-mic exclusions for DOA only
  mic_indices: [0, 7, 14]             # Fallback when no profiles are configured
  n_fft: 2048
  hop_length: 256
  hps:
    harmonics: 3
    fmin_hz: 100
    peak_search:
      fmin_hz: 100
      fmax_hz: 600
    cfar:
      guard_bins: 4
      ref_bins: 20
  music:
    window_s: 1.0
    azimuth_step_deg: 1.0
    half_bins: 1
    n_sources: 1
    elevation_deg: 0.0
    smoothing_predictions: 5          # Circular smoothing across recent estimates
    confidence_jump_deg: 45.0         # Larger jumps lower DOA confidence
  profiles:
    triangle_3:
      mic_indices: [0, 7, 14]
    corners_4:
      mic_indices: [1, 7, 8, 14]
    perimeter_8:
      mic_indices: [1, 3, 7, 8, 10, 12, 14, 15]
    all_16:
      mic_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

gpio:
  enabled: true
  alert_pin: 2             # Buzzer/relay
  strobe_pin: 24           # Visual indicator
  reset_pin: 23            # Reset button
  record_led_pin: null     # Disabled by default; GPIO 27 is field-tag yellow
  record_button_pin: null  # Disabled by default; GPIO 17 is field-tag red
  pause_button_pin: null   # Disabled by default; pause remains available in the UI
  field_tag_green_pin: 22  # Active-low button to GND; correct detection and classification
  field_tag_yellow_pin: 27 # Active-low button to GND; correct detection, incorrect classification
  field_tag_red_pin: 17    # Active-low button to GND; incorrect detection

web:
  host: 0.0.0.0
  port: 8080
```

Edit `threshold_profiles.yaml` to tune deployment thresholds without changing
application code. The active profile overrides the detector threshold and the
two color typing thresholds at startup. Color is assigned only after a positive
detection: RED, BLUE, or UNKNOWN.

## Deployment on Pi

### 1. Prerequisites

```bash
# Install Docker on Raspberry Pi OS
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Clone this repo
git clone <repo-url> AUDI
cd AUDI/audi-app

# Install alsa-utils for testing
sudo apt install alsa-utils flac
arecord -l  # Check your mic is detected
```

### 2. Build & Run

```bash
make build
make start
```

### 3. Auto-start on Boot

```bash
make install-service
sudo systemctl start audi          # Start immediately
```

### 4. Verify

```bash
make status
curl http://localhost:8080/api/status
```

## Docker

The compose stack uses `network_mode: host` for GPIO + ALSA access. Runtime
recordings and alert snapshots are bind-mounted to local `./data/` instead of a
Docker-managed volume.

- `docker/docker-compose.yml` — Base config
- `docker/docker-compose.pi.yml` — Pi-specific overrides (privileged mode, GPIO group)
- `docker/Dockerfile` — arm64 multi-stage build
- `docker/audi.service` — systemd unit for auto-start

### Local Data

Audio recordings persist in `audi-app/data/recordings`, and alert snapshots in
`audi-app/data/alerts`. To reset storage:

```bash
rm -rf data/recordings data/alerts
```

## Development (without Pi)

```bash
# Install deps locally
make dev-install

# Check your system
make dev-check

# Run (GPIO mock mode, no audio capture without mic)
make dev-run
```

The web UI will be at http://localhost:8080.

## Adding a Real FP32 TFLite Model

1. Export the combined detector + blue/red classifier:
   ```bash
   uv run --extra export audi-export-blue-red-tflite \
     --ckpt checkpoints/<blue-red-run>/checkpoints/best.ckpt \
     --output audi-app/models/model_combined_mn10_mined_hardneg_blue_red.tflite
   ```
2. Keep `config.yaml` pointed at:
   ```bash
   audi-app/models/model_combined_mn10_mined_hardneg_blue_red.tflite
   ```
3. If you mount a model into Docker directly, mount it at the same path:
   ```yaml
   # docker/docker-compose.yml override
   volumes:
     - /path/to/model_combined.tflite:/app/models/model_combined_mn10_mined_hardneg_blue_red.tflite
   ```
4. Update `config.yaml` and `threshold_profiles.yaml` thresholds to match the exported checkpoint.

## GPIO Wiring

| GPIO (BCM) | Physical Pin | Purpose |
|------------|-------------|---------|
| 2          | 3           | Alert output (buzzer/relay) |
| 24         | 18          | Strobe/LED output |
| 23         | 16          | Reset button input (pull-up) |
| 22         | 15          | Green field tag: correct detection and classification |
| 27         | 13          | Yellow field tag: correct detection, incorrect classification |
| 17         | 11          | Red field tag: incorrect detection |
| GND        | 6, 9, 14, 20, 25, 30, 34, 39 | Ground |

Field-tag buttons are active-low inputs. Wire each button between its GPIO pin
and ground; the Pi internal pull-up holds the button input high until pressed.

## License

Part of the AUDI drone detection project.
