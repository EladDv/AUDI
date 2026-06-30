#!/usr/bin/env bash
# Deploy AUDI with app/code and model encrypted for one Raspberry Pi fingerprint.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/deploy-secure-pi.sh --host IP --user pi --password PASSWORD [options]

Options:
  --host IP              Raspberry Pi IP/hostname
  --user NAME            SSH username (default: pi)
  --password PASSWORD    SSH/sudo password
  --model PATH           Plaintext model to encrypt (default: config.yaml model_path)
  --audio-channels N     Passed through to deploy-pi.sh
  --detector-channels L  Passed through to deploy-pi.sh
  --staging-dir PATH     Passed through to deploy-pi.sh
  --keep-radios          Passed through to deploy-pi.sh
  --no-verify            Passed through to deploy-pi.sh
  --no-reboot            Passed through to deploy-pi.sh
  --keep-temp            Keep temporary secure deploy tree for inspection
  -h, --help             Show this help

The temporary deploy tree excludes plaintext models/*.tflite and plaintext app
source, adds encrypted app/model payloads under secure/, and then runs
scripts/deploy-pi.sh from that tree.
EOF
}

info() { printf '\033[0;34m[INFO]\033[0m  %s\n' "$*"; }
ok()   { printf '\033[0;32m[OK]\033[0m    %s\n' "$*"; }
err()  { printf '\033[0;31m[ERROR]\033[0m %s\n' "$*" >&2; }

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "Missing required command: $1"
        exit 1
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PI_HOST=""
PI_USER="pi"
PI_PASSWORD="${AUDI_PI_PASSWORD:-}"
MODEL_PATH=""
KEEP_TEMP=false
PASSTHROUGH=()

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
        --model)
            MODEL_PATH="${2:-}"
            shift 2
            ;;
        --audio-channels|--channels|--detector-channels|--inference-channels|--staging-dir)
            PASSTHROUGH+=("$1" "${2:-}")
            shift 2
            ;;
        --keep-radios|--no-verify|--no-reboot)
            PASSTHROUGH+=("$1")
            shift
            ;;
        --keep-temp)
            KEEP_TEMP=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown option: $1"
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$PI_HOST" || -z "$PI_USER" || -z "$PI_PASSWORD" ]]; then
    usage
    exit 2
fi

need_cmd python3
need_cmd rsync
need_cmd ssh
need_cmd sshpass

if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$(
        python3 - "$APP_DIR/config.yaml" "$APP_DIR" <<'PY'
import sys
from pathlib import Path
import yaml

config = Path(sys.argv[1])
app_dir = Path(sys.argv[2])
cfg = yaml.safe_load(config.read_text()) or {}
model = Path(cfg.get("detection", {}).get("model_path", ""))
if not model.is_absolute():
    model = app_dir / model
print(model)
PY
    )"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    err "Model not found: $MODEL_PATH"
    exit 1
fi

SSH_OPTS=(
    -o PreferredAuthentications=password
    -o PubkeyAuthentication=no
    -o StrictHostKeyChecking=no
    -o ConnectTimeout=10
)

info "Reading Pi hardware fingerprint from ${PI_USER}@${PI_HOST}"
FINGERPRINT_JSON="$(
    SSHPASS="$PI_PASSWORD" sshpass -e ssh "${SSH_OPTS[@]}" "${PI_USER}@${PI_HOST}" \
        "python3 - fingerprint" < "$APP_DIR/src/secure_payload.py"
)"
printf '%s\n' "$FINGERPRINT_JSON" | python3 -m json.tool >/dev/null
ok "Fingerprint captured"

TMP_DIR="$(mktemp -d /tmp/audi-secure-deploy.XXXXXX)"
cleanup() {
    if [[ "$KEEP_TEMP" == "true" ]]; then
        info "Keeping temp deploy tree: $TMP_DIR"
    else
        rm -rf "$TMP_DIR"
    fi
}
trap cleanup EXIT

info "Creating secure deploy tree"
rsync -a --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='data' \
    --exclude='models/*.tflite' \
    "$APP_DIR/" "$TMP_DIR/"

mkdir -p "$TMP_DIR/secure"

APP_ARCHIVE="$TMP_DIR/secure/app.tar.gz.plain"
tar \
    --exclude='src/__pycache__' \
    --exclude='webui/__pycache__' \
    -C "$APP_DIR" \
    -czf "$APP_ARCHIVE" \
    src webui
python3 "$APP_DIR/src/secure_payload.py" encrypt-file \
    --input "$APP_ARCHIVE" \
    --out-dir "$TMP_DIR/secure" \
    --source-name app.tar.gz \
    --kind audi-device-bound-app \
    --fingerprint-json "$FINGERPRINT_JSON" >/tmp/audi-secure-app-encrypt.txt
cat /tmp/audi-secure-app-encrypt.txt
rm -f "$APP_ARCHIVE"

python3 "$APP_DIR/src/secure_payload.py" encrypt-model \
    --model "$MODEL_PATH" \
    --out-dir "$TMP_DIR/secure" \
    --fingerprint-json "$FINGERPRINT_JSON" >/tmp/audi-secure-encrypt.txt
cat /tmp/audi-secure-encrypt.txt

model_name="$(basename "$MODEL_PATH")"
mv "$TMP_DIR/secure/${model_name}.enc" "$TMP_DIR/secure/model.tflite.enc"
mv "$TMP_DIR/secure/${model_name}.enc.json" "$TMP_DIR/secure/model.tflite.enc.json"

find "$TMP_DIR/src" -type f -name '*.py' ! -name 'secure_payload.py' -delete
rm -rf "$TMP_DIR/webui"
mkdir -p "$TMP_DIR/webui"
ok "Plaintext app source and models excluded; encrypted payloads prepared"

info "Running secure deploy"
AUDI_PI_PASSWORD="$PI_PASSWORD" "$TMP_DIR/scripts/deploy-pi.sh" \
    --host "$PI_HOST" \
    --user "$PI_USER" \
    "${PASSTHROUGH[@]}"
