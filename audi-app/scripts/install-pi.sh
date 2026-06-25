#!/bin/bash
# =============================================================================
# AUDI - Type A Auto-Install Script for Raspberry Pi
# =============================================================================
#
# Usage:
#   curl -sL https://your-host/install.sh | bash
#   # Or locally:
#   bash scripts/install-pi.sh
#
# What it does:
#   1. Detects Pi model and OS
#   2. Installs system deps (Docker, alsa-utils, flac, git)
#   3. Clones or copies the app to /opt/AUDI
#   4. Adds user to docker, audio, gpio groups
#   5. Configures ALSA for the default mic
#   6. Builds the Docker image
#   7. Installs systemd service for auto-start on boot
#   8. Creates data directories for recordings and alerts
#   9. Starts the service
#
# Idempotent — safe to run multiple times.
# =============================================================================

set -euo pipefail

# ---- Color output ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---- Config ----
INSTALL_DIR="/opt/AUDI"
DATA_DIR="/data"
RECORDINGS_DIR="${DATA_DIR}/recordings"
ALERTS_DIR="${DATA_DIR}/alerts"
SERVICE_NAME="audi"
DOCKER_COMPOSE="docker compose"  # Pi OS uses v2 plugin
DISABLE_RADIOS=false  # pass --keep-radios to keep WiFi/BT enabled
HOTSPOT_SSID="audi-hotspot"
HOTSPOT_PASSWORD="${AUDI_HOTSPOT_PASSWORD:-12345678}"
HOTSPOT_IP="10.42.0.1/24"
HOTSPOT_URL="http://10.42.0.1:8080"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-radios) DISABLE_RADIOS=false; shift ;;
        *) shift ;;
    esac
done

# ---- Preflight ----
echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              AUDI — Type A                    ║${NC}"
echo -e "${BLUE}║           Raspberry Pi Auto Install           ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# Check we're on a Pi (or at least Linux ARM)
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "armv7l" ] && [ "$ARCH" != "armv6l" ]; then
    warn "Architecture is $ARCH (not ARM). This script targets Raspberry Pi."
    warn "The app may still work in Docker emulation mode."
fi

# Check root/sudo
if [ "$EUID" -ne 0 ]; then
    info "Some steps need root — re-running with sudo..."
    exec sudo bash "$0" "$@"
fi

# ---- 1. System Dependencies ----
info "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    alsa-utils \
    build-essential \
    ffmpeg \
    cmake \
    libsndfile1 \
    flac \
    git \
    jq \
    network-manager \
    rsync \
    > /dev/null 2>&1
ok "System deps installed"

set_config_key() {
    local file="$1"
    local key="$2"
    local value="$3"

    if grep -qE "^[[:space:]]*#?[[:space:]]*${key}=" "$file"; then
        sed -i -E "s|^[[:space:]]*#?[[:space:]]*${key}=.*|${key}=${value}|" "$file"
    else
        echo "${key}=${value}" >> "$file"
    fi
}

install_hotspot_fallback() {
    if ! command -v nmcli > /dev/null 2>&1; then
        warn "NetworkManager/nmcli is unavailable; skipping fallback hotspot setup"
        return
    fi

    info "Installing fallback hotspot (${HOTSPOT_SSID})..."
    cat > /usr/local/sbin/audi-hotspot-fallback << EOF
#!/usr/bin/env bash
set -euo pipefail

CONN="audi-hotspot"
SSID="${HOTSPOT_SSID}"
IFACE="wlan0"
AP_ADDR="${HOTSPOT_IP}"
PASSWORD="\${AUDI_HOTSPOT_PASSWORD:-${HOTSPOT_PASSWORD}}"

log() {
  echo "\$*"
}

ensure_profile() {
  if ! nmcli -t -f NAME connection show | grep -Fxq "\$CONN"; then
    log "Creating NetworkManager hotspot profile '\$CONN'"
    nmcli connection add \\
      type wifi \\
      ifname "\$IFACE" \\
      con-name "\$CONN" \\
      autoconnect no \\
      ssid "\$SSID"
  fi

  nmcli connection modify "\$CONN" \\
    connection.autoconnect no \\
    connection.interface-name "\$IFACE" \\
    802-11-wireless.mode ap \\
    802-11-wireless.ssid "\$SSID" \\
    802-11-wireless.band bg \\
    802-11-wireless.channel 6 \\
    802-11-wireless.powersave 2 \\
    wifi-sec.key-mgmt wpa-psk \\
    wifi-sec.psk "\$PASSWORD" \\
    ipv4.method shared \\
    ipv4.addresses "\$AP_ADDR" \\
    ipv6.method ignore
}

internet_ok() {
  if [[ "\${AUDI_HOTSPOT_FORCE_OFFLINE:-0}" == "1" ]]; then
    return 1
  fi

  for target in 1.1.1.1 8.8.8.8; do
    if ping -c 1 -W 2 "\$target" >/dev/null 2>&1; then
      return 0
    fi
  done

  local connectivity
  connectivity="\$(nmcli -t networking connectivity check 2>/dev/null || true)"
  [[ "\$connectivity" == "full" ]]
}

hotspot_active() {
  nmcli -t -f NAME,DEVICE connection show --active | grep -Fxq "\$CONN:\$IFACE"
}

main() {
  ensure_profile

  if internet_ok; then
    if hotspot_active; then
      log "Internet is available; disabling fallback hotspot"
      nmcli connection down "\$CONN"
    fi
    exit 0
  fi

  if hotspot_active; then
    exit 0
  fi

  log "No internet detected; starting fallback hotspot '\$SSID' at ${HOTSPOT_URL}"
  nmcli radio wifi on
  nmcli connection up "\$CONN"
}

main "\$@"
EOF
    chmod 0755 /usr/local/sbin/audi-hotspot-fallback

    cat > /etc/systemd/system/audi-hotspot-fallback.service << 'EOF'
[Unit]
Description=AUDI fallback Wi-Fi hotspot when no internet is available
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/audi-hotspot-fallback
EOF

    cat > /etc/systemd/system/audi-hotspot-fallback.timer << 'EOF'
[Unit]
Description=Periodically check whether AUDI fallback hotspot is needed

[Timer]
OnBootSec=45s
OnUnitActiveSec=30s
AccuracySec=5s
Unit=audi-hotspot-fallback.service

[Install]
WantedBy=timers.target
EOF

    systemctl daemon-reload
    systemctl enable --now audi-hotspot-fallback.timer > /dev/null
    /usr/local/sbin/audi-hotspot-fallback || true
    ok "Fallback hotspot installed (${HOTSPOT_SSID}; UI at ${HOTSPOT_URL})"
}

configure_bootloader_power() {
    if ! command -v rpi-eeprom-config > /dev/null 2>&1; then
        warn "rpi-eeprom-config unavailable; skipping bootloader power-button config"
        return
    fi

    local tmp
    tmp="$(mktemp)"
    if ! rpi-eeprom-config > "$tmp" 2>/dev/null; then
        rm -f "$tmp"
        warn "Could not read EEPROM config; skipping bootloader power-button config"
        return
    fi

    set_config_key "$tmp" "POWER_OFF_ON_HALT" "0"
    set_config_key "$tmp" "WAIT_FOR_POWER_BUTTON" "0"

    if cmp -s "$tmp" <(rpi-eeprom-config 2>/dev/null); then
        ok "Bootloader power-button config already set"
        rm -f "$tmp"
        return
    fi

    if rpi-eeprom-config --apply "$tmp" > /dev/null 2>&1; then
        ok "Bootloader configured to boot when power is applied (reboot applies EEPROM update)"
    else
        warn "Could not apply EEPROM config; leaving bootloader unchanged"
    fi
    rm -f "$tmp"
}

# ---- 2. Docker Installation ----
if ! command -v docker &> /dev/null; then
    info "Docker not found — installing..."
    curl -fsSL https://get.docker.com | bash
    systemctl enable docker
    systemctl start docker
    ok "Docker installed"
else
    ok "Docker already installed ($(docker --version))"
fi

# ---- 3. User Groups ----
info "Adding user to required groups..."
for group in docker audio gpio; do
    if getent group "$group" > /dev/null 2>&1; then
        usermod -aG "$group" "${SUDO_USER:-$USER}" 2>/dev/null || true
    fi
done
ok "User ${SUDO_USER:-$USER} added to docker, audio, gpio groups"

# ---- 4. Clone / Copy App ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd 2>/dev/null || echo "")"

if [ -f "$PARENT_DIR/config.yaml" ] && [ -d "$PARENT_DIR/src" ] && [ -d "$PARENT_DIR/docker" ]; then
    # Running from the repo — copy to install dir
    info "Found app source at $PARENT_DIR — copying to $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR"
    rsync -a --delete \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='*.pyc' \
        "$PARENT_DIR/" "$INSTALL_DIR/"
    ok "App copied to $INSTALL_DIR"
else
    # Not in repo — check if already installed
    if [ -d "$INSTALL_DIR" ] && [ -f "$INSTALL_DIR/config.yaml" ]; then
        ok "App already installed at $INSTALL_DIR"
    else
        err "Cannot find app source. Run this script from the AUDI/audi-app directory:"
        err "  cd AUDI/audi-app && bash scripts/install-pi.sh"
        exit 1
    fi
fi

# ---- 5. Create Data Directories ----
info "Creating data directories..."
mkdir -p "$RECORDINGS_DIR" "$ALERTS_DIR"
chown -R "${SUDO_USER:-1000}:${SUDO_USER:-1000}" "$DATA_DIR"
chmod 755 "$DATA_DIR"
ok "Data dirs: $RECORDINGS_DIR, $ALERTS_DIR"

# ---- 6. ALSA Configuration ----
info "Checking ALSA audio devices..."
if command -v arecord &> /dev/null; then
    if arecord -l 2>/dev/null | grep -q "card"; then
        ok "Audio capture device detected:"
        arecord -l 2>/dev/null | head -5
    else
        warn "No capture device found. Plug in a USB mic or configure I2S."
        warn "Run 'arecord -l' after connecting a mic."
    fi
else
    warn "arecord not found (should be installed above)"
fi

# Create default .asoundrc if none exists
ASOUND_FILE="${DATA_DIR}/asound.conf"
if [ ! -f "$ASOUND_FILE" ]; then
    cat > "$ASOUND_FILE" << 'EOF'
# Default ALSA configuration for AUDI Type A
# Uses the first available capture device
pcm.!default {
    type asym
    capture.pcm "mic"
}
pcm.mic {
    type plug
    slave {
        pcm "hw:0,0"
        rate 48000
    }
}
EOF
    ok "Default ALSA config created at $ASOUND_FILE"
fi

# ---- 7. Build Docker Image ----
info "Building Docker image (this may take a few minutes)..."
cd "$INSTALL_DIR"
docker compose -f docker/docker-compose.yml build 2>&1 | tail -5
ok "Docker image built"

# ---- 8. Install Systemd Service ----
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

info "Installing systemd service..."
cp docker/audi.service "$SERVICE_FILE"
sed -i "s|/home/pi/audi-app|${INSTALL_DIR}|g" "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"
ok "Systemd service installed and enabled"

# ---- 8b. Install Kiosk Autostart + Desktop Launcher ----
info "Installing kiosk autostart and desktop launcher..."
bash scripts/install-kiosk.sh --user "${SUDO_USER:-$USER}" --url "http://localhost:8080"
ok "Kiosk launcher installed"

# ---- 9. Configure Boot Options ----
info "Checking boot config..."
BOOT_CONFIG="/boot/firmware/config.txt"
if [ ! -f "$BOOT_CONFIG" ]; then
    BOOT_CONFIG="/boot/config.txt"  # Older Pi OS
fi

if [ -f "$BOOT_CONFIG" ]; then
    if [ ! -f "${BOOT_CONFIG}.audi-install-bak" ]; then
        cp "$BOOT_CONFIG" "${BOOT_CONFIG}.audi-install-bak" || true
    fi

    # Restore the deployed full-performance display/CPU defaults and keep the
    # Pi 5 USB current override needed for wired/non-PD field power.
    sed -i \
        -e '/^# AUDI field power stability profile for weak\/non-PD 5V feeds\.$/,/^usb_max_current_enable=1$/d' \
        "$BOOT_CONFIG"
    set_config_key "$BOOT_CONFIG" "max_framebuffers" "2"
    set_config_key "$BOOT_CONFIG" "arm_boost" "1"
    set_config_key "$BOOT_CONFIG" "usb_max_current_enable" "1"
    ok "Boot config set: max_framebuffers=2, arm_boost=1, usb_max_current_enable=1"

    # if [ "$DISABLE_RADIOS" = true ]; then
    #     # Permanently disable WiFi and Bluetooth at the kernel level
    #     info "Disabling WiFi and Bluetooth in device tree..."
    #     for overlay in disable-wifi disable-bt; do
    #         if ! grep -q "dtoverlay=$overlay" "$BOOT_CONFIG"; then
    #             echo "dtoverlay=$overlay" >> "$BOOT_CONFIG"
    #             ok "  Added dtoverlay=$overlay to $BOOT_CONFIG"
    #         else
    #             ok "  dtoverlay=$overlay already present"
    #         fi
    #     done

    #     # Mask systemd services as belt-and-suspenders
    #     for svc in wpa_supplicant bluetooth hciuart; do
    #         if systemctl is-enabled "$svc" 2>/dev/null | grep -qv masked; then
    #             systemctl mask "$svc" 2>/dev/null || true
    #             ok "  Masked $svc service"
    #         fi
    #     done
    # else
    #     ok "Radios kept enabled (--keep-radios flag set)"
    # fi

    # Enable I2S if the user wants it (comment only — let user decide)
    if grep -q "^#dtparam=i2s=on" "$BOOT_CONFIG"; then
        warn "I2S is commented out in $BOOT_CONFIG"
        warn "If using an I2S mic, uncomment: dtparam=i2s=on"
    fi
fi

configure_bootloader_power

# ---- 9b. Fallback Hotspot ----
install_hotspot_fallback

# ---- 10. Firewall (if ufw) ----
if command -v ufw &> /dev/null; then
    info "Configuring firewall..."
    ufw allow 8080/tcp comment "AUDI Web UI" 2>/dev/null || true
    ok "Port 8080 opened in firewall"
fi

# ---- 11. Start Service ----
info "Starting service..."
systemctl start "${SERVICE_NAME}.service"
sleep 3

# Check if container is running
if docker ps --filter "name=${SERVICE_NAME}" --format "{{.Names}}" | grep -q "^${SERVICE_NAME}$"; then
    ok "Container is running!"
else
    warn "Container not running yet — checking logs..."
    journalctl -u "${SERVICE_NAME}.service" --no-pager -n 20 2>/dev/null || true
fi

# ---- 12. Summary ----
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Installation Complete!              ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  App:        ${INSTALL_DIR}"
echo -e "  Recordings: ${RECORDINGS_DIR}"
echo -e "  Alerts:     ${ALERTS_DIR}"
echo -e "  Web UI:     http://$(hostname -I | awk '{print $1}'):8080"
echo -e "  Service:    sudo systemctl {status,start,stop,restart} ${SERVICE_NAME}"
echo -e "  Logs:       sudo journalctl -u ${SERVICE_NAME} -f"
echo -e "  Docker:     cd ${INSTALL_DIR} && make {logs,status,restart}"
echo ""
echo -e "  ${YELLOW}⚠  You may need to log out and back in for group changes${NC}"
echo -e "  ${YELLOW}   (docker, audio, gpio) to take effect.${NC}"
echo ""
echo -e "  ${BLUE}Post-install checklist:${NC}"
echo -e "   1. Plug in a USB microphone"
echo -e "   2. Run 'arecord -l' to verify it's detected"
echo -e "   3. Open http://<pi-ip>:8080 in a browser"
echo -e "   4. Check the VU meter shows audio levels"
echo ""

# ---- Quick test ----
echo -e "${BLUE}Running quick health check...${NC}"
sleep 2
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/api/status 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    ok "Web UI responding (HTTP $HTTP_CODE)"
else
    warn "Web UI not yet responding (HTTP $HTTP_CODE) — check logs: sudo journalctl -u ${SERVICE_NAME} -f"
fi
