"""Immutable configuration dataclasses for the audi training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and backbone configuration.

    Attributes:
        arch: Model architecture name (e.g. "cnn14", "resnet18").
        num_classes: Number of output logits (1 for binary).
        pretrained: Whether to load ImageNet/AudioSet backbone weights.
        compile: Whether to apply torch.compile to the model.
    """

    arch: str = "cnn14"
    num_classes: int = 1
    pretrained: bool = True
    compile: bool = True

    def __post_init__(self) -> None:
        if self.num_classes < 1:
            raise ValueError(
                f"num_classes must be >= 1, got {self.num_classes}"
            )


@dataclass(frozen=True)
class MelConfig:
    """Mel spectrogram parameters.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        n_mels: Number of mel filterbank bins.
        n_fft: FFT window size.
        hop_length: Hop length in samples.
        mean_db: Pre-computed scalar mean for normalization (None to skip).
        std_db: Pre-computed scalar std for normalization (None to skip).
    """

    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
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

    @classmethod
    def vit_224(cls) -> MelConfig:
        """ViT-compatible config producing 224 mel bins (row dimension).

        Pair with target_length_samples=36704 (2.294 s) to get exactly
        224 time frames at default hop_length=160 — yields [B, 3, 224, 224].
        Both dimensions are divisible by 7 (required by FasterViT).
        """
        return cls(n_mels=224)


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
        background_swap_prob: Probability of mixing in secondary background.
        background_swap_db: dB level of secondary background relative to primary.
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
    background_swap_prob: float = 0.25
    background_swap_db: float = -10.0
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
    noise2_path: Path | None = None
    noise2_count: int = 2
    noise2_max_attenuation_db: float = -50.0
    dataset_length: int | None = None
    aug: AugmentationConfig | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.positive_probability <= 1.0:
            raise ValueError(
                f"positive_probability must be in [0, 1], got {self.positive_probability}"
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


@dataclass(frozen=True)
class TrainConfig:
    """Complete training run configuration.

    Aggregates all sub-configs into one immutable object.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    mix: MixConfig | None = None  # Set at runtime, no default paths
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    batch_size: int = 32
    steps_per_epoch: int = 100
    val_steps_per_epoch: int = 40
    num_workers: int = 4
    accumulate_grad_batches: int = 1
    seed: int = 42
    patience: int = 5
    save_top_k: int = 3
    output_dir: Path = Path("checkpoints")

    # Regularization
    dropout: float = 0.0
    bn_momentum: float = 0.1
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    spec_augment_prob: bool = 0.0
    per_bin_weights: bool = False
    label_smoothing: float = 0.0

    # Loss
    loss_type: Literal["bce", "focal"] = "bce"

    # Finetuning
    finetune_from: Path | None = None
    pretrained_checkpoint: Path | None = None


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
