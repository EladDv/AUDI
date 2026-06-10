"""Audio feature extraction and TFLite inference for the Pi detector."""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger("audio_guard.detector")


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Build a mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    if f_max is None:
        f_max = sample_rate / 2.0

    n_freqs = n_fft // 2 + 1
    mel_min = _hz_to_mel(np.array(f_min))
    mel_max = _hz_to_mel(np.array(f_max))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        f_left = bins[m]
        f_center = bins[m + 1]
        f_right = bins[m + 2]
        for k in range(f_left, f_center):
            filters[m, k] = (k - f_left) / max(1, f_center - f_left)
        for k in range(f_center, f_right):
            filters[m, k] = (f_right - k) / max(1, f_right - f_center)
    return filters


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 1024,
    win_length: int | None = None,
    hop_length: int = 160,
    f_min: float = 0.0,
    f_max: float | None = None,
    power: float = 2.0,
) -> np.ndarray:
    """Compute mel-dB spectrogram matching the training torchaudio frontend.

    The detector checkpoint was trained with:
    ``torchaudio.transforms.MelSpectrogram(..., norm=None, mel_scale="htk",
    center=True, pad_mode="reflect")`` followed by ``AmplitudeToDB()``.
    Librosa's high-level ``melspectrogram`` defaults do not match those
    settings, so this function implements the STFT path explicitly.
    """

    if f_max is None:
        f_max = sample_rate / 2.0
    if power != 2.0:
        raise ValueError("Only power=2.0 matches the trained detector frontend")
    if win_length is None:
        win_length = n_fft
    if win_length <= 0 or win_length > n_fft:
        raise ValueError(
            f"win_length must be in [1, n_fft], got {win_length} for n_fft={n_fft}"
        )

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    # torchaudio Spectrogram(center=True, pad_mode="reflect") pads by n_fft // 2
    # before framing. torch.hann_window defaults to periodic=True.
    pad = n_fft // 2
    padded = np.pad(audio, (pad, pad), mode="reflect")
    if len(padded) < n_fft:
        padded = np.pad(padded, (0, n_fft - len(padded)), mode="constant")

    from numpy.lib.stride_tricks import sliding_window_view

    frames = sliding_window_view(padded, n_fft)[::hop_length]
    window = _stft_window(n_fft, win_length)
    spec = np.fft.rfft(frames * window[np.newaxis, :], n=n_fft, axis=1)
    power_spec = (spec.real * spec.real + spec.imag * spec.imag).T

    mel_fb = _torchaudio_mel_filterbank(
        n_mels=n_mels,
        n_fft=n_fft,
        sample_rate=sample_rate,
        f_min=float(f_min),
        f_max=float(f_max),
    )
    mel = mel_fb @ power_spec
    mel = 10.0 * np.log10(np.maximum(mel, 1e-10))

    return mel.astype(np.float32)


@lru_cache(maxsize=16)
def _periodic_hann_window(n_fft: int) -> np.ndarray:
    return np.hanning(n_fft + 1)[:-1].astype(np.float32)


@lru_cache(maxsize=16)
def _stft_window(n_fft: int, win_length: int) -> np.ndarray:
    window = _periodic_hann_window(win_length)
    if win_length == n_fft:
        return window
    left = (n_fft - win_length) // 2
    right = n_fft - win_length - left
    return np.pad(window, (left, right), mode="constant").astype(np.float32)


@lru_cache(maxsize=16)
def _torchaudio_mel_filterbank(
    *,
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Return torchaudio-compatible HTK, unnormalized mel filters."""
    import librosa

    return librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        htk=True,
        norm=None,
    ).astype(np.float32)


def resample_audio(
    audio: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample audio using scipy. Returns float32 array."""
    if orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio.astype(np.float64), up, down).astype(np.float32)


class TFLiteClassifier:
    """Load and run a TFLite int8 model for drone detection."""

    def __init__(
        self,
        model_path: str,
        num_threads: int = 2,
        n_mels: int = 128,
        n_fft: int = 1024,
        win_length: int | None = None,
        hop_length: int = 160,
        model_sample_rate: int = 16000,
        window_samples: int = 40960,
        mel_mean: float | None = None,
        mel_std: float | None = None,
    ):
        self.model_path = model_path
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = n_fft if win_length is None else win_length
        self.hop_length = hop_length
        self.model_sample_rate = model_sample_rate
        self.window_samples = window_samples
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._loaded = False
        self.output_size = 1
        self.expected_frames: int | None = None

        try:
            self._load_model(num_threads)
        except Exception as e:
            logger.warning("TFLite model not loaded: %s - using mock", e)

    def _load_model(self, num_threads: int):
        from ai_edge_litert.interpreter import Interpreter

        self._interpreter = Interpreter(
            model_path=self.model_path,
            num_threads=num_threads,
        )
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._loaded = True

        inp_shape = self._input_details[0]["shape"]
        out_shape = self._output_details[0]["shape"]
        if len(inp_shape) == 4:
            if int(inp_shape[1]) != 3:
                raise ValueError(f"Expected 3 input channels, got {inp_shape}")
            if int(inp_shape[2]) != self.n_mels:
                raise ValueError(
                    f"Configured n_mels={self.n_mels} but model input is {inp_shape}"
                )
            self.expected_frames = int(inp_shape[3])
        if len(out_shape) == 0:
            self.output_size = 1
        elif len(out_shape) == 1:
            self.output_size = int(out_shape[0])
        else:
            self.output_size = int(np.prod(out_shape[1:]))
        logger.info(
            "TFLite model loaded: %s, input=%s, output=%s, threads=%d",
            Path(self.model_path).name,
            inp_shape,
            out_shape,
            num_threads,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def preprocess(self, audio: np.ndarray, capture_sr: int) -> np.ndarray:
        """Convert raw audio to model-ready spectrogram.

        Args:
            audio: Float32 audio array at capture_sr.
            capture_sr: Sample rate of the input audio.

        Returns:
            Spectrogram of shape (1, 3, n_mels, n_frames).
        """
        if capture_sr != self.model_sample_rate:
            audio = resample_audio(audio, capture_sr, self.model_sample_rate)

        if len(audio) < self.window_samples:
            audio = np.pad(audio, (0, self.window_samples - len(audio)))
        else:
            audio = audio[: self.window_samples]

        audio_rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        if audio_rms > 1e-8:
            audio = audio / audio_rms
        peak = float(np.max(np.abs(audio)))
        if peak > 0.98:
            audio = audio * (0.98 / peak)

        mel = compute_mel_spectrogram(
            audio,
            self.model_sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )
        target_frames = self.expected_frames or (self.window_samples // self.hop_length)
        if mel.shape[-1] < target_frames:
            raise ValueError(
                f"Mel spectrogram has {mel.shape[-1]} frames, "
                f"but model expects {target_frames}; check window_samples"
            )
        mel = mel[..., :target_frames]

        if self.mel_mean is not None and self.mel_std is not None:
            mel = (mel - self.mel_mean) / max(self.mel_std, 1e-8)
        else:
            rms = np.sqrt(np.mean(mel**2)) + 1e-8
            mel = mel / rms

        mel = np.stack([mel] * 3, axis=0)
        mel = mel[np.newaxis, ...]

        return mel.astype(np.float32)

    def predict_logits(self, spec: np.ndarray) -> np.ndarray:
        """Run inference and return all raw logits as a flat float32 array."""
        if not self._loaded:
            return np.zeros(1, dtype=np.float32)

        input_shape = self._input_details[0].get("shape")
        if input_shape is not None and len(input_shape) >= 1:
            expected_batch = int(input_shape[0])
            if expected_batch > 1 and spec.shape[0] == 1:
                spec = np.repeat(spec, expected_batch, axis=0)
        self._interpreter.set_tensor(self._input_details[0]["index"], spec)
        self._interpreter.invoke()
        logits = self._interpreter.get_tensor(self._output_details[0]["index"])
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim > 1:
            logits = logits[0]
        return logits.flatten()

    def predict(self, spec: np.ndarray) -> float:
        """Run inference on a preprocessed spectrogram.

        Returns:
            First raw logit (before sigmoid), for binary detector models.
        """
        return float(self.predict_logits(spec)[0])
