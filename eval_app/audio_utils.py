"""Audio loading and processing utilities for the eval dashboard."""

from __future__ import annotations

from pathlib import Path as _Path

import numpy as np
import torch
import torchaudio

from audi.training.detector import DroneDetector

_SR = 16000
# Eval dashboard window length.
_CLIP_S = 5.12
_CLIP_SAMPLES = int(_SR * _CLIP_S)

_ATTACK_DIR = _Path(__file__).resolve().parent.parent / "data" / "attack_runs"


def load_audio(filepath: str, target_sr: int = _SR) -> tuple[np.ndarray, int]:
    """Load wav, resample if needed, return float32 mono array."""
    try:
        wav, sr = torchaudio.load(filepath)
    except Exception:
        import soundfile as sf
        wav_np, sr = sf.read(filepath)
        wav = torch.as_tensor(wav_np.T, dtype=torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
    return wav.squeeze(0).numpy().astype(np.float32), target_sr


def sliding_windows(audio: np.ndarray, window_samples: int, stride: float) -> np.ndarray:
    """Yield windows of shape (n_windows, window_samples)."""
    step = int(window_samples * stride)
    if step < 1:
        step = 1
    n_windows = max(1, (len(audio) - window_samples) // step + 1)
    windows = np.zeros((n_windows, window_samples), dtype=np.float32)
    for i in range(n_windows):
        start = i * step
        windows[i] = audio[start: start + window_samples]
    return windows


def compute_mel_image(wav: np.ndarray, model: DroneDetector) -> np.ndarray:
    """Compute mel spectrogram as numpy image for display."""
    t = torch.as_tensor(wav, dtype=torch.float32).unsqueeze(0).to(model.device)
    mel = model._to_db(model._mel_transform(t)).cpu()
    return mel.squeeze(0).numpy()


@torch.no_grad()
def predict_windows(
    model: DroneDetector, audio: np.ndarray, device: str, stride: float,
    batch_size: int = 32,
) -> np.ndarray:
    """Run sliding-window inference, return logits for each window."""
    windows = sliding_windows(audio, _CLIP_SAMPLES, stride)
    all_logits = []
    for i in range(0, len(windows), batch_size):
        batch = torch.as_tensor(windows[i: i + batch_size], dtype=torch.float32).to(device)
        logits = model(batch).cpu().numpy()
        all_logits.append(logits)
    return np.concatenate(all_logits) if all_logits else np.array([])


def window_time_axis(audio_samples: int, window_samples: int, stride: float) -> np.ndarray:
    """Return time (seconds) of the center of each window."""
    step = int(window_samples * stride)
    n = max(1, (audio_samples - window_samples) // step + 1)
    return np.array([(i * step + window_samples / 2) / _SR for i in range(n)])
