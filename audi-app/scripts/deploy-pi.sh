#!/usr/bin/env bash
# Deploy AUDI to a Raspberry Pi over SSH, then run the Pi-side installer.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/deploy-pi.sh <ip-address> <username> <password>

Options:
  --host IP              Raspberry Pi IP/hostname
  --user NAME            SSH username (default: pi)
  --password PASSWORD    SSH/sudo password
  --staging-dir PATH     Remote staging directory (default: /home/<user>/audi-app-deploy)
  --keep-radios          Pass through to scripts/install-pi.sh
  --audio-channels N     Set capture channel count in config.yaml (default: installer default)
  --detector-channels L  Set inference channels: all, null, or comma-separated indexes
  --no-verify            Skip post-install HTTP/container checks
  --no-reboot            Do not reboot the Pi after deployment
  -h, --help             Show this help

Examples:
  scripts/deploy-pi.sh 10.100.102.108 pi 'your-password'
  AUDI_PI_PASSWORD='your-password' scripts/deploy-pi.sh --host 10.100.102.108 --user pi
EOF
}

info() { printf '\033[0;34m[INFO]\033[0m  %s\n' "$*"; }
ok()   { printf '\033[0;32m[OK]\033[0m    %s\n' "$*"; }
err()  { printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "Missing required local command: $1"
        err "Install it first, then rerun this script."
        exit 1
    fi
}

shell_quote() {
    printf '%q' "$1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PI_HOST=""
PI_USER="pi"
PI_PASSWORD="${AUDI_PI_PASSWORD:-}"
REMOTE_STAGING=""
KEEP_RADIOS=false
INSTALL_AUDIO_CHANNELS="${AUDI_AUDIO_CHANNELS:-}"
INSTALL_DETECTOR_CHANNELS="${AUDI_DETECTOR_CHANNELS:-}"
VERIFY=true
REBOOT=true
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            PI_HOST="${2:-}"
            shift 2
            ;;
        --user|--name)
            PI_USER="${2:-}"
            shift 2
            ;;
        --password)
            PI_PASSWORD="${2:-}"
            shift 2
            ;;
        --staging-dir)
            REMOTE_STAGING="${2:-}"
            shift 2
            ;;
        --keep-radios)
            KEEP_RADIOS=true
            shift
            ;;
        --audio-channels|--channels)
            INSTALL_AUDIO_CHANNELS="${2:-}"
            shift 2
            ;;
        --detector-channels|--inference-channels)
            INSTALL_DETECTOR_CHANNELS="${2:-}"
            shift 2
            ;;
        --no-verify)
            VERIFY=false
            shift
            ;;
        --no-reboot)
            REBOOT=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                POSITIONAL+=("$1")
                shift
            done
            ;;
        -*)
            err "Unknown option: $1"
            usage
            exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL[@]} -gt 0 && -z "$PI_HOST" ]]; then
    PI_HOST="${POSITIONAL[0]}"
fi
if [[ ${#POSITIONAL[@]} -gt 1 ]]; then
    PI_USER="${POSITIONAL[1]}"
fi
if [[ ${#POSITIONAL[@]} -gt 2 && -z "$PI_PASSWORD" ]]; then
    PI_PASSWORD="${POSITIONAL[2]}"
fi

if [[ -z "$PI_HOST" || -z "$PI_USER" || -z "$PI_PASSWORD" ]]; then
    usage
    exit 2
fi

if [[ -z "$REMOTE_STAGING" ]]; then
    REMOTE_STAGING="/home/${PI_USER}/audi-app-deploy"
fi

if [[ ! -f "$APP_DIR/config.yaml" || ! -d "$APP_DIR/src" || ! -d "$APP_DIR/docker" ]]; then
    err "Could not find AUDI app root from $APP_DIR"
    exit 1
fi

need_cmd ssh
need_cmd sshpass
need_cmd rsync

SSH_OPTS=(
    -o PreferredAuthentications=password
    -o PubkeyAuthentication=no
    -o StrictHostKeyChecking=no
    -o ConnectTimeout=10
)

remote_bash() {
    local script="$1"
    SSHPASS="$PI_PASSWORD" sshpass -e ssh "${SSH_OPTS[@]}" "${PI_USER}@${PI_HOST}" \
        "bash -lc $(shell_quote "$script")"
}

remote_sudo_bash() {
    local script="$1"
    printf '%s\n' "$PI_PASSWORD" | SSHPASS="$PI_PASSWORD" sshpass -e ssh "${SSH_OPTS[@]}" "${PI_USER}@${PI_HOST}" \
        "sudo -S -p '' bash -lc $(shell_quote "$script")"
}

INSTALL_ARGS=()
if [[ "$KEEP_RADIOS" == "true" ]]; then
    INSTALL_ARGS+=(--keep-radios)
fi
if [[ -n "$INSTALL_AUDIO_CHANNELS" ]]; then
    INSTALL_ARGS+=(--audio-channels "$(shell_quote "$INSTALL_AUDIO_CHANNELS")")
fi
if [[ -n "$INSTALL_DETECTOR_CHANNELS" ]]; then
    INSTALL_ARGS+=(--detector-channels "$(shell_quote "$INSTALL_DETECTOR_CHANNELS")")
fi

echo ""
info "Deploying AUDI to ${PI_USER}@${PI_HOST}"
info "Local app: $APP_DIR"
info "Remote staging: $REMOTE_STAGING"

info "Checking SSH access..."
remote_bash "hostname && uname -m" >/tmp/audi-deploy-ssh-check.txt
cat /tmp/audi-deploy-ssh-check.txt
ok "SSH access verified"

info "Preparing remote staging directory..."
remote_bash "mkdir -p $(shell_quote "$REMOTE_STAGING")"

info "Copying app source and model files..."
SSHPASS="$PI_PASSWORD" rsync -az --delete \
    -e "sshpass -e ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='.pytest_cache' \
    --exclude='data' \
    "$APP_DIR/" "${PI_USER}@${PI_HOST}:${REMOTE_STAGING}/"
ok "Source copied"

info "Running Pi installer with sudo..."
install_cmd="cd $(shell_quote "$REMOTE_STAGING") && bash scripts/install-pi.sh ${INSTALL_ARGS[*]}"
remote_sudo_bash "$install_cmd"
ok "Installer finished"

if [[ "$VERIFY" == "true" ]]; then
    info "Verifying service, container, and web API..."
    verify_cmd='
set -euo pipefail
status_file="$(mktemp /tmp/audi_status_check.XXXXXX.json)"
trap '"'"'rm -f "$status_file"'"'"' EXIT
for _ in $(seq 1 45); do
    if curl -fsS -m 3 -o "$status_file" http://localhost:8080/api/status; then
        break
    fi
    sleep 2
done

curl -fsS -m 5 -o "$status_file" http://localhost:8080/api/status
export AUDI_STATUS_FILE="$status_file"
python3 - <<'"'"'PY'"'"'
import json
import os
with open(os.environ["AUDI_STATUS_FILE"]) as f:
    data = json.load(f)
det = data.get("detector", {})
print("api_status=ok")
print("model=%s" % det.get("model_path"))
print("combined_loaded=%s" % det.get("blue_red_model_loaded"))
print("profile=%s" % det.get("threshold_profile"))
print("alert_level=%s" % det.get("alert_level"))
PY

systemctl status audi --no-pager -n 12
docker ps --filter name=audi --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
docker inspect -f "health={{.State.Health.Status}}" audi
'
    remote_sudo_bash "$verify_cmd"
    ok "Verification finished"
fi

if [[ "$REBOOT" == "true" ]]; then
    info "Rebooting Pi to finish deployment..."
    remote_sudo_bash "nohup sh -c 'sleep 2; /sbin/reboot' >/dev/null 2>&1 &"
    ok "Reboot requested"
fi

echo ""
ok "AUDI is deployed"
echo "Web UI: http://${PI_HOST}:8080"
echo "Logs:   ssh ${PI_USER}@${PI_HOST} 'sudo journalctl -u audi -f'"
if [[ "$REBOOT" == "true" ]]; then
    echo "Reboot: wait about 60 seconds, then reopen the Web UI."
fi
