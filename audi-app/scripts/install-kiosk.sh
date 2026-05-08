#!/bin/bash
# =============================================================================
# AUDI - Kiosk Mode Installer
# =============================================================================
# Installs GUI autostart and desktop launchers for the AUDI web UI.
#
# Usage:
#   sudo bash scripts/install-kiosk.sh
#   sudo bash scripts/install-kiosk.sh --user pi --url http://localhost:8080
# =============================================================================

set -euo pipefail

APP_NAME="AUDI Type A"
DEFAULT_URL="http://localhost:8080"
URL="$DEFAULT_URL"
TARGET_USER="${SUDO_USER:-${USER:-pi}}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)
            TARGET_USER="${2:?missing value for --user}"
            shift 2
            ;;
        --url)
            URL="${2:?missing value for --url}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: sudo bash scripts/install-kiosk.sh [--user USER] [--url URL]" >&2
            exit 2
            ;;
    esac
done

if [[ "$EUID" -ne 0 ]]; then
    exec sudo bash "$0" --user "$TARGET_USER" --url "$URL"
fi

if ! id "$TARGET_USER" >/dev/null 2>&1; then
    echo "ERROR: user '$TARGET_USER' does not exist" >&2
    exit 1
fi

TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_KIOSK="$APP_DIR/scripts/kiosk.sh"
BIN_DIR="/usr/local/bin"
KIOSK_BIN="$BIN_DIR/audi-kiosk"
AUTOSTART_DIR="$TARGET_HOME/.config/autostart"
APPLICATIONS_DIR="$TARGET_HOME/.local/share/applications"
DESKTOP_DIR="$TARGET_HOME/Desktop"
AUTOSTART_FILE="$AUTOSTART_DIR/audi-kiosk.desktop"
APP_FILE="$APPLICATIONS_DIR/audi-kiosk.desktop"
DESKTOP_FILE="$DESKTOP_DIR/AUDI.desktop"

echo "Installing kiosk dependencies..."
apt-get update -qq
apt-get install -y -qq unclutter curl xdg-utils >/dev/null
if ! command -v chromium-browser >/dev/null 2>&1 && ! command -v chromium >/dev/null 2>&1; then
    apt-get install -y -qq chromium-browser >/dev/null || apt-get install -y -qq chromium >/dev/null
fi

echo "Installing kiosk launcher to $KIOSK_BIN ..."
install -m 0755 "$SOURCE_KIOSK" "$KIOSK_BIN"

mkdir -p "$AUTOSTART_DIR" "$APPLICATIONS_DIR" "$DESKTOP_DIR"

create_desktop_file() {
    local path="$1"
    local autostart="$2"

    {
        echo "[Desktop Entry]"
        echo "Type=Application"
        echo "Name=$APP_NAME"
        echo "Comment=Open AUDI in kiosk mode"
        echo "Exec=$KIOSK_BIN $URL"
        echo "Icon=utilities-terminal"
        echo "Terminal=false"
        echo "Categories=Utility;"
        if [[ "$autostart" == "true" ]]; then
            echo "X-GNOME-Autostart-enabled=true"
            echo "Hidden=false"
            echo "NoDisplay=false"
        fi
    } > "$path"
}

create_desktop_file "$AUTOSTART_FILE" true
create_desktop_file "$APP_FILE" false
create_desktop_file "$DESKTOP_FILE" false

chmod 0644 "$AUTOSTART_FILE" "$APP_FILE"
chmod 0755 "$DESKTOP_FILE"
chown -R "$TARGET_USER:$TARGET_USER" "$AUTOSTART_DIR" "$APPLICATIONS_DIR" "$DESKTOP_DIR"

if command -v gio >/dev/null 2>&1; then
    sudo -u "$TARGET_USER" gio set "$DESKTOP_FILE" metadata::trusted true 2>/dev/null || true
fi

echo "Kiosk autostart installed for user: $TARGET_USER"
echo "Autostart file: $AUTOSTART_FILE"
echo "Desktop launcher: $DESKTOP_FILE"
echo "App menu launcher: $APP_FILE"
echo "URL: $URL"
echo ""
echo "To launch now, run:"
echo "  sudo -u $TARGET_USER DISPLAY=\${DISPLAY:-:0} $KIOSK_BIN $URL"
