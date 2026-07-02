#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <mvdr|mean|channel0> <output-dir> [num-workers]" >&2
  exit 2
fi

beamformer="$1"
output_dir="$2"
num_workers="${3:-4}"

cache_dir="data/pyroom_mvdr_cache_app_2048_512_deglitch_v2"
common_args=(
  --noise-path data/20260603_uma16channel_lebanon_false_hunt
  --drone-path data/HF_dataset_v2_drone
  --bg-noise-path data/HF_dataset_v7_background
  --clip-seconds 5.12
  --sample-rate 16000
  --highpass-hz 125.0
  --positive-probability 0.5
  --validation-fraction 0.15
  --min-azimuth-deg -180.0
  --max-azimuth-deg 180.0
  --min-elevation-deg 5.0
  --max-elevation-deg 70.0
  --min-distance-m 20.0
  --max-distance-m 450.0
  --drone-reference-distance-m 10.0
  --stft-n-fft 2048
  --hop-length 512
  --diagonal-loading 0.0001
  --sensor-noise-db -45.0
  --temperature-c 20.0
  --humidity-percent 50.0
  --deglitch-threshold 0.001
  --deglitch-loudness-ratio 8.0
  --deglitch-diff-ratio 12.0
  --deglitch-window-samples 64
  --bg-noise-probability 0.25
  --bg-noise-multi-probability 0.5
  --bg-noise-count 3
  --bg-noise-max-attenuation-db -40.0
  --snr-bin easy:0:5:0.20
  --snr-bin medium:-5:0:0.25
  --snr-bin hard:-10:-5:0.25
  --snr-bin very_hard:-15:-10:0.20
  --snr-bin extreme:-20:-15:0.10
)

case "$beamformer" in
  mvdr|mean|channel0) ;;
  *)
    echo "bad beamformer: $beamformer" >&2
    exit 2
    ;;
esac

rm -rf "$output_dir" "${output_dir}".tmp "${output_dir}".cache
rm -rf "$(dirname "$output_dir")/.${output_dir##*/}.tmp"
rm -rf "$(dirname "$output_dir")/.${output_dir##*/}.train.shards"
rm -rf "$(dirname "$output_dir")/.${output_dir##*/}.validation.shards"
rm -rf "$(dirname "$output_dir")/.${output_dir##*/}.test.shards"

if [[ "$beamformer" == "mvdr" ]]; then
  uv run audi-data pyroom-mvdr-cache \
    --noise-path data/20260603_uma16channel_lebanon_false_hunt \
    --cache-dir "$cache_dir" \
    --sample-rate 16000 \
    --highpass-hz 125.0 \
    --validation-fraction 0.15 \
    --split all \
    --stft-n-fft 2048 \
    --hop-length 512 \
    --diagonal-loading 0.0001 \
    --mvdr-cache-seconds 30.0 \
    --deglitch-threshold 0.001 \
    --deglitch-loudness-ratio 8.0 \
    --deglitch-diff-ratio 12.0 \
    --deglitch-window-samples 64
fi

for split in train validation test; do
  case "$split" in
    train) rows=150000 ;;
    validation|test) rows=25000 ;;
  esac

  args=(
    "${common_args[@]}"
    --beamformer "$beamformer"
    --split "$split"
    --num-examples "$rows"
    --num-workers "$num_workers"
    --output-dir "$output_dir"
    --seed 42
  )
  if [[ "$beamformer" == "mvdr" ]]; then
    args+=(--mvdr-cache-dir "$cache_dir" --mvdr-cache-seconds 30.0)
  fi

  uv run audi-data pyroom-dataset "${args[@]}"
done

uv run python - "$output_dir" <<'PY'
import sys
from pathlib import Path
from datasets import load_from_disk

path = Path(sys.argv[1])
dd = load_from_disk(str(path))
print("dataset", path)
for split in dd:
    print(split, len(dd[split]), dd[split].features)
PY
