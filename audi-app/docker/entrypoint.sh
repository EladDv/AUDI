#!/bin/bash
# =============================================================================
# AUDI - Docker Entrypoint
# =============================================================================
# Handles device initialization, auto-discovery, then launches the app.
set -euo pipefail

echo "=== AUDI Entrypoint ==="

# Ensure data directories exist
mkdir -p /app/data/recordings /app/data/alerts /data/recordings

# Auto-discover USB audio device unless a device was provided by compose/env.
AUDIO_DEVICE="${AUDIO_DEVICE:-default}"
if arecord -l 2>/dev/null | grep -q "card"; then
    echo "Audio device found:"
    arecord -l
    if [ "$AUDIO_DEVICE" = "default" ] || [ "$AUDIO_DEVICE" = "auto" ]; then
        # Extract first USB device: "card N: ... [USB ...], device M: ..."
        # Use plughw so ALSA delivers the configured app rate instead of the
        # hardware's native rate when the interface cannot do 16 kHz directly.
        USB_DEV=$(arecord -l 2>/dev/null | grep "USB" | head -1 | sed -n 's/.*card \([0-9]*\):.*device \([0-9]*\):.*/plughw:\1,\2/p' || true)
        if [ -n "$USB_DEV" ]; then
            AUDIO_DEVICE="$USB_DEV"
            echo "Auto-discovered USB device: $AUDIO_DEVICE"
        fi
    else
        echo "Using configured audio device: $AUDIO_DEVICE"
    fi
else
    echo "WARNING: No audio capture device detected."
    echo "Plug in a USB mic or configure I2S and restart."
fi

# Override device in config.yaml
if [ "$AUDIO_DEVICE" != "default" ]; then
    echo "Setting audio device to $AUDIO_DEVICE in config.yaml"
    python3 - "$AUDIO_DEVICE" <<'PY'
import sys
import yaml
from pathlib import Path

path = Path("/app/config.yaml")
cfg = yaml.safe_load(path.read_text()) or {}
cfg.setdefault("audio", {})["device"] = sys.argv[1]
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
fi

# Decrypt a device-bound model when a secure payload is present.
SECURE_APP_MANIFEST="${AUDI_SECURE_APP_MANIFEST:-/app/secure/app.tar.gz.enc.json}"
SECURE_APP_CIPHERTEXT="${AUDI_SECURE_APP_CIPHERTEXT:-/app/secure/app.tar.gz.enc}"
SECURE_APP_ARCHIVE="${AUDI_SECURE_APP_ARCHIVE:-/dev/shm/audi-secure/app.tar.gz}"
SECURE_APP_ROOT="${AUDI_SECURE_APP_ROOT:-/dev/shm/audi-secure/app}"
APP_MAIN="/app/src/main.py"
if [ -f "$SECURE_APP_MANIFEST" ] && [ -f "$SECURE_APP_CIPHERTEXT" ]; then
    echo "Secure app payload found; decrypting for this Pi"
    rm -rf "$SECURE_APP_ROOT"
    mkdir -p "$SECURE_APP_ROOT"
    python3 -m secure_payload decrypt-file \
        --manifest "$SECURE_APP_MANIFEST" \
        --ciphertext "$SECURE_APP_CIPHERTEXT" \
        --output "$SECURE_APP_ARCHIVE"
    tar -xzf "$SECURE_APP_ARCHIVE" -C "$SECURE_APP_ROOT"
    rm -f "$SECURE_APP_ARCHIVE"
    APP_MAIN="$SECURE_APP_ROOT/src/main.py"
    export PYTHONPATH="$SECURE_APP_ROOT/src:/app/src:/usr/lib/python3/dist-packages"
fi

SECURE_MANIFEST="${AUDI_SECURE_MODEL_MANIFEST:-/app/secure/model.tflite.enc.json}"
SECURE_CIPHERTEXT="${AUDI_SECURE_MODEL_CIPHERTEXT:-/app/secure/model.tflite.enc}"
SECURE_MODEL_PATH="${AUDI_SECURE_MODEL_PATH:-/dev/shm/audi-secure/model.tflite}"
if [ -f "$SECURE_MANIFEST" ] && [ -f "$SECURE_CIPHERTEXT" ]; then
    echo "Secure model payload found; decrypting for this Pi"
    python3 -m secure_payload decrypt-model \
        --manifest "$SECURE_MANIFEST" \
        --ciphertext "$SECURE_CIPHERTEXT" \
        --output "$SECURE_MODEL_PATH"
    python3 - "$SECURE_MODEL_PATH" <<'PY'
import sys
import yaml
from pathlib import Path

path = Path("/app/config.yaml")
cfg = yaml.safe_load(path.read_text()) or {}
cfg.setdefault("detection", {})["model_path"] = sys.argv[1]
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
fi

# Check GPIO access. Pi deployments require this so alarms and LEDs work.
REQUIRE_GPIO="${REQUIRE_GPIO:-false}"
if compgen -G "/dev/gpiochip*" > /dev/null; then
    echo "GPIO access: OK ($(ls /dev/gpiochip* 2>/dev/null | tr '\n' ' '))"
    if [ -z "${RPI_LGPIO_CHIP:-}" ]; then
        for chip_path in /sys/bus/gpio/devices/gpiochip*; do
            [ -r "$chip_path/of_node/compatible" ] || continue
            if tr '\0' '\n' < "$chip_path/of_node/compatible" | grep -Eq 'raspberrypi,(rp1|bcm2835|bcm2711)-gpio'; then
                export RPI_LGPIO_CHIP="${chip_path##*gpiochip}"
                echo "GPIO chip: using /dev/gpiochip${RPI_LGPIO_CHIP}"
                break
            fi
        done
    else
        echo "GPIO chip: using /dev/gpiochip${RPI_LGPIO_CHIP} from RPI_LGPIO_CHIP"
    fi
elif [ -c /dev/gpiomem ]; then
    echo "GPIO access: OK (/dev/gpiomem present; legacy GPIO device)"
elif [ "$REQUIRE_GPIO" = "true" ] || [ "$REQUIRE_GPIO" = "1" ]; then
    echo "ERROR: GPIO is required but no /dev/gpiochip* device is available." >&2
    echo "Start with the Pi override: docker compose -f docker/docker-compose.yml -f docker/docker-compose.pi.yml up -d" >&2
    exit 1
else
    echo "GPIO: no /dev/gpiochip* device found — running without GPIO hardware"
fi

echo "Starting AUDI..."
echo ""

# Launch the main app
exec python3 -u "$APP_MAIN" "$@"
