"""Immutable configuration dataclasses for the audi training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

LINEAR_STFT_MEAN_DB = -7.860577
LINEAR_STFT_STD_DB = 21.280184


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and backbone configuration.

    Attributes:
        arch: EfficientAT model architecture name (e.g. "mn10_as", "dymn10_as").
        num_classes: Number of output logits (1 for binary).
        pretrained: Whether to load AudioSet backbone weights.
        compile: Whether to apply torch.compile to the model.
        detector_head_hidden_dims: Optional hidden dims for EfficientAT detector heads.
        detector_head_dropout: Dropout between detector head hidden layers.
    """

    arch: str = "mn10_as"
    num_classes: int = 1
    pretrained: bool = True
    compile: bool = True
    detector_head_hidden_dims: tuple[int, ...] = ()
    detector_head_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.num_classes < 1:
            raise ValueError(
                f"num_classes must be >= 1, got {self.num_classes}"
            )
        if any(dim <= 0 for dim in self.detector_head_hidden_dims):
            raise ValueError("detector_head_hidden_dims must be positive")
        if self.detector_head_dropout < 0:
            raise ValueError("detector_head_dropout must be non-negative")


@dataclass(frozen=True)
class MelConfig:
    """Mel spectrogram parameters.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        n_mels: Number of mel filterbank bins.
        n_fft: FFT window size.
        win_length: Window length in samples. Defaults to ``n_fft``.
        hop_length: Hop length in samples.
        mean_db: Pre-computed scalar mean for normalization (None to skip).
        std_db: Pre-computed scalar std for normalization (None to skip).
    """

    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    win_length: int | None = None
    hop_length: int = 160
    mean_db: float | None = 10.430418
    std_db: float | None = 5.288271
    # PCEN (disabled by default — when True, replaces dB conversion + scalar norm)
    use_pcen: bool = False
    pcen_s: float = 0.025
    pcen_alpha: float = 0.98
    pcen_delta: float = 2.0
    pcen_r: float = 0.5
    pcen_eps: float = 1e-6
    # Multi-frontend: "mel" (default), "stft", "stft_bands", or
    # comma-separated combinations.
    frontend_type: str = "mel"
    stft_bands_hz: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        win_length = self.n_fft if self.win_length is None else self.win_length
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be > 0, got {self.n_mels}")
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be > 0, got {self.n_fft}")
        if win_length <= 0:
            raise ValueError(f"win_length must be > 0, got {win_length}")
        if win_length > self.n_fft:
            raise ValueError(
                f"win_length must be <= n_fft, got {win_length} > {self.n_fft}"
            )
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be > 0, got {self.hop_length}")
        object.__setattr__(self, "win_length", win_length)

@dataclass(frozen=True)
class AugmentationConfig:
    """Audio augmentation pipeline configuration.

    All probabilities are in [0, 1]. Set ``enable=False`` to disable entirely.

    Attributes:
        enable: Master switch — when False, no augmentations are applied.
        gain_jitter_db: ±dB random gain on drone and background.
        pitch_prob: Probability of pitch shift.
        pitch_semitones: ±semitones random pitch shift.
        stretch_prob: Probability of time stretch.
        time_stretch_range: (min, max) speed factor for drone.
        reverb_prob: Probability of applying reverb to mix.
        reverb_decay: (min, max) RT60 in seconds.
        eq_prob: Probability of random 2-band parametric EQ.
        eq_gain_db: ±dB per EQ band.
        noise_inject_prob: Probability of Gaussian noise injection.
        noise_inject_db: Target dBFS of noise relative to signal RMS.
        time_mask_prob: Probability of time-domain zeroing.
        time_mask_count: Number of masks to apply.
        time_mask_max_ratio: Max fraction of signal to zero per mask.
        lowpass_prob: Probability of low-pass filtering.
        lowpass_cutoff_range: (min, max) cutoff frequency in Hz.
    """

    enable: bool = False
    gain_jitter_db: float = 3.0
    pitch_prob: float = 0.25
    pitch_semitones: float = 1.0
    stretch_prob: float = 0.25
    time_stretch_range: tuple[float, float] = (0.9, 1.1)
    reverb_prob: float = 0.25
    reverb_decay: tuple[float, float] = (0.1, 0.5)
    eq_prob: float = 0.25
    eq_gain_db: float = 6.0
    noise_inject_prob: float = 0.25
    noise_inject_db: float = -40.0
    time_mask_prob: float = 0.25
    time_mask_count: int = 2
    time_mask_max_ratio: float = 0.1
    lowpass_prob: float = 0.25
    lowpass_cutoff_range: tuple[float, float] = (2000, 8000)
    shift_prob: float = 0.25
    shift_max_ratio: float = 0.5
    atmospheric_prob: float = 0.25
    atmospheric_snr_min: float = -30.0
    atmospheric_snr_max: float = 0.0
    atmospheric_cutoff_min: float = 500.0
    atmospheric_cutoff_max: float = 8000.0
    doppler_prob: float = 0.25
    doppler_max_speed_mps: float = 30.0


@dataclass(frozen=True)
class SNRBin:
    """A labelled SNR range with sampling probability.

    Attributes:
        name: Human-readable bin name (e.g. "easy", "hard").
        low_db: Lower bound in dB (inclusive).
        high_db: Upper bound in dB (inclusive).
        probability: Sampling weight for this bin.
    """

    name: str
    low_db: float
    high_db: float
    probability: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SNRBin name must not be empty")
        if self.probability <= 0:
            raise ValueError(
                f"SNRBin probability must be > 0, got {self.probability}"
            )


@dataclass(frozen=True)
class MixConfig:
    """Dataset mixing configuration.

    Attributes:
        noise_path: Path to background noise HF dataset.
        drone_path: Path to drone audio HF dataset.
        hard_noise_path: Optional hard-negative background dataset.
        hard_noise_prob: Probability of drawing the base background from
            ``hard_noise_path`` instead of ``noise_path``.
        noise2_path: Optional secondary background dataset for multi-noise mixing.
        noise2_count: Maximum number of extra noise layers (1-5).
        noise2_max_attenuation_db: Minimum level of extra noise layers relative to base.
        snr_bins: SNR bins for drone mixing.
        target_length_samples: Fixed segment length in samples.
        positive_probability: Fraction of samples that contain a drone.
        highpass_hz: Highpass cutoff applied to all audio sources.
        sample_rate: Audio sample rate in Hz.
        dataset_length: Override for virtual dataset size (None = auto).
        aug: Augmentation configuration (None = disabled).
    """

    noise_path: Path
    drone_path: Path
    snr_bins: list[SNRBin] = field(default_factory=list)
    target_length_samples: int = 20480
    positive_probability: float = 0.5
    highpass_hz: float = 125.0
    sample_rate: int = 16000
    hard_noise_path: Path | None = None
    hard_noise_prob: float = 0.0
    noise2_path: Path | None = None
    noise2_prob: float = 0.25
    noise2_multi_noise_prob: float = 0.5
    noise2_count: int = 2
    noise2_max_attenuation_db: float = -50.0
    dataset_length: int | None = None
    aug: AugmentationConfig | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.positive_probability <= 1.0:
            raise ValueError(
                f"positive_probability must be in [0, 1], got {self.positive_probability}"
            )
        if not 0.0 <= self.hard_noise_prob <= 1.0:
            raise ValueError(
                f"hard_noise_prob must be in [0, 1], got {self.hard_noise_prob}"
            )


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer and learning rate schedule configuration.

    Attributes:
        lr: Peak learning rate.
        weight_decay: AdamW decoupled weight decay.
        schedule: Learning rate schedule type.
        warmup_epochs: Number of linear warmup epochs (0 = none).
        max_epochs: Total epochs (for cosine schedule T_max).
    """

    lr: float = 1e-3
    weight_decay: float = 0.01
    schedule: Literal["constant", "cosine", "linear"] = "constant"
    warmup_epochs: int = 0
    max_epochs: int = 30

def parse_snr_bins(specs: list[str]) -> list[SNRBin]:
    """Parse SNR bin specifications from CLI strings.

    Format: ``"name:low_db:high_db:prob"``

    Args:
        specs: List of colon-delimited bin specifications.

    Returns:
        List of parsed SNRBin instances.

    Raises:
        SystemExit: If any spec has invalid format or if list is empty.
    """
    bins: list[SNRBin] = []
    for spec in specs:
        parts = [p.strip() for p in str(spec).split(":")]
        if len(parts) != 4:
            raise SystemExit(
                f"Bad --snr-bin {spec!r}. Expected name:low_db:high_db:prob"
            )
        name, lo, hi, pr = parts
        bins.append(
            SNRBin(
                name=name,
                low_db=float(lo),
                high_db=float(hi),
                probability=float(pr),
            )
        )
    if not bins:
        raise SystemExit("Provide at least one --snr-bin")
    return bins
