# AUDI — Type A

**Continuous audio recording + detection for Raspberry Pi.** Records audio in 5-minute segments to a 32GB ring buffer on disk, keeps the last 120 seconds of PCM in memory for ML inference and alarm snapshots, and triggers GPIO outputs on detection (YES/BLUE/NO). Touch-friendly web UI on port 8080.

Open http://raspberry-pi-ip:8080 from any browser on your local network.

## Quick Start

```bash
# On your Raspberry Pi (arm64):
make build    # Build the Docker image
make install-service && sudo systemctl start audio-guard    # Auto-start + run now
```

Or for a fully automated install:

```bash
sudo bash scripts/install-pi.sh
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
                    │  Detector    │────▶│  GPIO Alarm   │
                    │  YES / BLUE  │     │  + LEDs +     │
                    │  / NO        │     │  Buttons      │
                    └──────┬───────┘     └───────────────┘
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
| **Detector** | `src/detector.py` | TFLite int8 classifier with Schmitt-trigger hysteresis (3 of 5 windows). Binary YES/NO state, alarm snapshot saving, temporal confidence tracking. |
| **GPIO** | `src/gpio_alarm.py` | Drives Pi GPIO pins: ALERT (buzzer/relay), STROBE (blinking LED), RESET (physical button to silence), REC_LED (recording indicator), REC_BTN (record toggle), PAUSE_BTN (pause 5 min). Graceful mock fallback when not on Pi hardware. |
| **Web UI** | `src/webui_server.py` | Flask server serving a **touch-optimized HTML UI**. Big Start/Stop buttons (green/red, mutual exclusive), Silence and Pause controls, live VU meter, YES/BLUE/NO state card, alert history, system info panel. |
| **Main** | `src/main.py` | Orchestrator — starts everything, wires callbacks, handles graceful shutdown on SIGTERM/SIGINT. |

## Config

Edit `config.yaml`:

```yaml
audio:
  device: "default"                  # ALSA device (arecord -l)
  sample_rate: 48000
  segment_duration: 300              # 5-minute segments
  ring_buffer_seconds: 120           # 60s pre + 60s post alarm
  device_retry_min: 2                # Min seconds between retries
  device_retry_max: 60               # Max (doubles per failure)

storage:
  max_size_gb: 32
  compress: true
  data_dir: /data/recordings
  alerts_dir: /data/alerts
  max_alerts_gb: 2

detection:
  model_path: /app/models/model.tflite
  inference_interval: 0.320
  active_threshold_profile: balanced
  threshold_profiles_file: ./threshold_profiles.yaml
  confidence_threshold_high: 0.70    # YES → GPIO alarm

gpio:
  enabled: true
  alert_pin: 17            # Buzzer/relay
  strobe_pin: 27           # Visual indicator
  reset_pin: 22            # Reset button
  record_led_pin: 23       # Recording indicator
  record_button_pin: 24    # Record toggle
  pause_button_pin: 25     # Pause 5 min

web:
  host: 0.0.0.0
  port: 8080
```

Edit `threshold_profiles.yaml` to tune deployment thresholds without changing
application code. The active profile overrides detector and color thresholds at
startup.

## Deployment on Pi

### 1. Prerequisites

```bash
# Install Docker on Raspberry Pi OS
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Clone this repo
git clone <repo-url> pi-audio-guard
cd pi-audio-guard

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
sudo systemctl start audio-guard   # Start immediately
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
- `docker/audio-guard.service` — systemd unit for auto-start

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

## Adding a Real TFLite Model

1. Export your model using `audi-export-tflite`:
   ```bash
   uv run python scripts/export_tflite.py --ckpt ... --noise-path ... --drone-path ...
   ```
2. Copy the `.tflite` file to the Pi and mount it:
   ```yaml
   # docker/docker-compose.yml override
   volumes:
     - /path/to/model.tflite:/app/models/model.tflite
   ```
3. Update `config.yaml` thresholds to match

## GPIO Wiring

| GPIO (BCM) | Physical Pin | Purpose |
|------------|-------------|---------|
| 17         | 11          | Alert output (buzzer/relay) |
| 27         | 13          | Strobe/LED output |
| 22         | 15          | Reset button input (pull-up) |
| 23         | 16          | Recording indicator LED |
| 24         | 18          | Record toggle button (pull-up) |
| 25         | 22          | Pause 5 min button (pull-up) |
| GND        | 6, 9, 14, 20, 25, 30, 34, 39 | Ground |

## License

Part of the AUDI drone detection project.
