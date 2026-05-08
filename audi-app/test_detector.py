"""
Test detector on attack-run WAVs.
Directly feeds WAV audio through the TFLite model pipeline.
"""

import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path

# Add audi-app/src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detector import (
    TFLiteClassifier,
    HysteresisState,
    resample_audio,
    compute_mel_spectrogram,
)
from recorder import AudioRingBuffer


def load_config():
    import yaml

    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    det_cfg = config["detection"]
    audio_cfg = config["audio"]

    capture_sr = audio_cfg.get("sample_rate", 16000)

    # Load TFLite classifier (same as DetectionEngine does)
    clf = TFLiteClassifier(
        model_path=det_cfg["model_path"],
        num_threads=det_cfg.get("num_threads", 2),
        n_mels=det_cfg["n_mels"],
        n_fft=det_cfg["n_fft"],
        hop_length=det_cfg["hop_length"],
        model_sample_rate=det_cfg["model_sample_rate"],
        window_samples=det_cfg["window_samples"],
        mel_mean=det_cfg.get("mel_mean"),
        mel_std=det_cfg.get("mel_std"),
    )

    # Hysteresis (same config)
    hyst = HysteresisState(
        threshold=det_cfg["confidence_threshold_high"],
        window=det_cfg.get("hysteresis_window", 5),
        ratio=det_cfg.get("hysteresis_ratio", 0.6),
        margin=det_cfg.get("hysteresis_margin", 0.05),
    )

    window_samples = det_cfg["window_samples"]
    model_sr = det_cfg["model_sample_rate"]
    interval = det_cfg.get("inference_interval", 0.320)
    window_sec = window_samples / model_sr

    # Load attack WAV
    wav_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "../data/attack_runs/200m_attackrun.wav"
    )
    audio, file_sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Resample if needed
    if file_sr != capture_sr:
        audio = resample_audio(audio, file_sr, capture_sr)

    print(f"File: {wav_path}")
    print(f"Duration: {len(audio) / capture_sr:.1f}s, SR: {capture_sr}Hz")
    print(f"Window: {window_sec:.2f}s ({window_samples} samples)")
    print(
        f"Total windows: ~{int(len(audio) / (window_samples * det_cfg.get('stride', 0.0625)))}"
    )
    print(f"Threshold: {det_cfg['confidence_threshold_high']}")
    print(f"Model loaded: {clf.is_loaded}")
    print()

    # Sliding window inference (matching attack-run eval stride)
    stride = det_cfg.get("stride", 0.0625)
    step_samples = int(window_samples * stride)
    if step_samples < 1:
        step_samples = int(capture_sr * interval)

    scores = []
    yes_count = 0
    t0 = time.perf_counter()
    total_windows = 0

    for start in range(0, len(audio) - window_samples, step_samples):
        chunk = audio[start : start + window_samples]
        spec = clf.preprocess(chunk, capture_sr)
        logit = clf.predict(spec)
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
        raw = float(prob)
        alarm = hyst.add(raw)
        state = "YES" if alarm else "NO"
        scores.append(raw)
        total_windows += 1
        if state == "YES":
            yes_count += 1

    elapsed = time.perf_counter() - t0

    scores_arr = np.array(scores)
    print(
        f"Windows processed: {total_windows} in {elapsed:.1f}s ({total_windows / elapsed:.0f} windows/s)"
    )
    print()
    print(
        f"Score stats: min={scores_arr.min():.4f} max={scores_arr.max():.4f} "
        f"mean={scores_arr.mean():.4f} median={np.median(scores_arr):.4f}"
    )
    print(
        f"Scores > 0.30: {(scores_arr > 0.30).sum()} ({(scores_arr > 0.30).mean() * 100:.1f}%)"
    )
    print(
        f"Scores > 0.50: {(scores_arr > 0.50).sum()} ({(scores_arr > 0.50).mean() * 100:.1f}%)"
    )
    print(f"YES detections (hysteresis): {yes_count}")
    print(f"Max score: {scores_arr.max():.4f}")

    # Show top 10 scores
    top_idx = np.argsort(scores_arr)[-10:][::-1]
    print("\nTop 10 scores:")
    for i in top_idx:
        t = i * step_samples / capture_sr
        print(f"  t={t:.1f}s  score={scores_arr[i]:.4f}")


if __name__ == "__main__":
    main()
