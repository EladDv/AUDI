#!/bin/bash
# =============================================================================
# AUDI - Docker Entrypoint
# =============================================================================
# Handles device initialization, auto-discovery, then launches the app.
set -euo pipefail

echo "=== AUDI Entrypoint ==="

# Ensure data directory exists
mkdir -p /data/recordings

# Auto-discover USB audio device
AUDIO_DEVICE="default"
if arecord -l 2>/dev/null | grep -q "card"; then
    echo "Audio device found:"
    arecord -l
    # Extract first USB device: "card N: ... [USB ...], device M: ..."
    USB_DEV=$(arecord -l 2>/dev/null | grep "USB" | head -1 | sed -n 's/.*card \([0-9]*\):.*device \([0-9]*\):.*/hw:\1,\2/p')
    if [ -n "$USB_DEV" ]; then
        AUDIO_DEVICE="$USB_DEV"
        echo "Auto-discovered USB device: $AUDIO_DEVICE"
    fi
else
    echo "WARNING: No audio capture device detected."
    echo "Plug in a USB mic or configure I2S and restart."
fi

# Override device in config.yaml
if [ "$AUDIO_DEVICE" != "default" ]; then
    echo "Setting audio device to $AUDIO_DEVICE in config.yaml"
    sed -i "s/device:.*/device: \"$AUDIO_DEVICE\"/" /app/config.yaml
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
exec python3 -u /app/src/main.py "$@"
