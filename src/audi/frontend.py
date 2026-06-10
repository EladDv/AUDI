"""Multi-frontend feature extraction: mel, linear STFT, CQT, CWT.

CQT  (librosa): log-spaced frequency bins, better for harmonic drone signals.
CWT  (ssqueezepy): multi-scale wavelet transform, better for transient detection.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T

from audi.config import LINEAR_STFT_MEAN_DB, LINEAR_STFT_STD_DB

DEFAULT_BANDED_STFT_BANDS_HZ = (
    (100.0, 500.0),
    (800.0, 1000.0),
    (1200.0, 1600.0),
    (1800.0, 2000.0),
)


def parse_frequency_bands_hz(
    bands: str | None,
) -> tuple[tuple[float, float], ...] | None:
    """Parse comma-separated frequency ranges, e.g. ``100-500,800-1000``."""
    if bands is None:
        return None
    parsed = []
    for raw_band in bands.split(","):
        band = raw_band.strip()
        if not band:
            continue
        bounds = band.split("-")
        if len(bounds) != 2:
            raise ValueError(
                f"Invalid frequency band '{band}'. Expected '<low>-<high>'."
            )
        low_hz = float(bounds[0])
        high_hz = float(bounds[1])
        if low_hz < 0 or high_hz < 0:
            raise ValueError(f"Frequency band '{band}' must be non-negative.")
        if low_hz == high_hz:
            raise ValueError(f"Frequency band '{band}' must have non-zero width.")
        parsed.append((min(low_hz, high_hz), max(low_hz, high_hz)))
    if not parsed:
        raise ValueError("At least one frequency band is required.")
    return tuple(parsed)


class LinearSTFT(nn.Module):
    """Linear-frequency log-power STFT frontend.

    Produces ``n_bins`` frequency rows from the one-sided STFT frequency axis.
    Optional frequency bands are selected before resizing. This is intentionally
    not a mel projection.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 160,
        n_bins: int = 128,
        n_fft: int = 1024,
        win_length: int | None = None,
        mean_db: float = LINEAR_STFT_MEAN_DB,
        std_db: float = LINEAR_STFT_STD_DB,
        frequency_bands_hz: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.n_fft = n_fft
        self.win_length = n_fft if win_length is None else win_length
        if self.win_length <= 0:
            raise ValueError(f"win_length must be > 0, got {self.win_length}")
        if self.win_length > self.n_fft:
            raise ValueError(
                f"win_length must be <= n_fft, got {self.win_length} > {self.n_fft}"
            )
        self.mean_db = float(mean_db)
        self.std_db = max(float(std_db), 1e-8)
        self.frequency_bands_hz = frequency_bands_hz
        self.register_buffer(
            "window", torch.hann_window(self.win_length), persistent=False
        )
        if frequency_bands_hz is not None:
            freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
            mask = torch.zeros(freqs.shape, dtype=torch.bool)
            for low_hz, high_hz in frequency_bands_hz:
                lo = min(float(low_hz), float(high_hz))
                hi = max(float(low_hz), float(high_hz))
                mask |= (freqs >= lo) & (freqs <= hi)
            self.register_buffer("frequency_mask", mask, persistent=False)
        else:
            self.frequency_mask = None

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=wav.device, dtype=wav.dtype),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        magnitude = torch.nan_to_num(
            spec.abs(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(max=1e10)
        power = magnitude.pow(2).clamp(min=1e-10, max=1e20)
        spec_db = 10.0 * torch.log10(power)
        if self.frequency_mask is not None:
            mask = self.frequency_mask.to(device=spec_db.device)
            spec_db = spec_db[:, mask, :]
        if spec_db.shape[1] > self.n_bins:
            spec_db = torch.nn.functional.interpolate(
                spec_db.unsqueeze(1),
                size=(self.n_bins, spec_db.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        elif spec_db.shape[1] == self.n_bins:
            pass
        else:
            pad = self.n_bins - spec_db.shape[1]
            spec_db = torch.nn.functional.pad(spec_db, (0, 0, 0, pad))
        spec_db = (spec_db - self.mean_db) / self.std_db
        return torch.nan_to_num(spec_db, nan=0.0, posinf=0.0, neginf=0.0)


class CQT(nn.Module):
    """Constant-Q Transform using librosa.

    Produces log-spaced frequency bins. Unlike mel, CQT has constant Q factor
    so frequency resolution scales with frequency — better for harmonic analysis
    of drone rotor tones where fundamental and harmonics are logarithmically spaced.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 160,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        f_min: float | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.f_min = f_min

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        import librosa

        # wav: (B, T) -> cqt: (B, n_bins, T_frames)
        wav_np = wav.cpu().numpy()
        batch_out = []
        for i in range(wav_np.shape[0]):
            cqt = np.abs(librosa.cqt(
                wav_np[i],
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
                fmin=self.f_min,
            ))
            cqt = np.log(cqt + 1e-6)
            batch_out.append(torch.from_numpy(cqt).to(wav.device))
        return torch.stack(batch_out)  # (B, n_bins, T)


class CWT(nn.Module):
    """Continuous Wavelet Transform using ssqueezepy.

    Multi-scale wavelet decomposition — better for detecting transient/impulsive
    events and rotor modulation patterns.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 160,
        n_scales: int = 64,
        nv: int = 32,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_scales = n_scales
        self.nv = nv

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        import ssqueezepy

        wav_np = wav.cpu().numpy()
        batch_out = []
        for i in range(wav_np.shape[0]):
            Tx, *_ = ssqueezepy.cwt(
                wav_np[i],
                wavelet="morlet",
                fs=self.sample_rate,
                nv=self.nv,
                scales="log",
            )
            # Tx: (n_scales, n_times) — decimate to hop_length stride
            Tx = Tx[:, ::self.hop_length]
            # Truncate or pad to expected length
            n_bins = self.n_scales
            if Tx.shape[0] > n_bins:
                # Sub-sample scales
                idx = np.linspace(0, Tx.shape[0] - 1, n_bins, dtype=int)
                Tx = Tx[idx]
            elif Tx.shape[0] < n_bins:
                pad = np.zeros((n_bins - Tx.shape[0], Tx.shape[1]), dtype=Tx.dtype)
                Tx = np.concatenate([Tx, pad], axis=0)
            Tx = np.log(np.abs(Tx) + 1e-6)
            batch_out.append(torch.from_numpy(Tx).to(wav.device))
        return torch.stack(batch_out)  # (B, n_scales, T)


class MultiFrontend(nn.Module):
    """Run multiple frontends on the same audio, concatenate along channel dim."""

    def __init__(self, frontends: list[nn.Module], hop_length: int):
        super().__init__()
        self.frontends = nn.ModuleList(frontends)
        self.hop_length = hop_length

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        outputs = []
        for m in self.frontends:
            if isinstance(m, T.MelSpectrogram):
                out = m(wav)
            elif isinstance(m, (CQT, CWT)):
                out = m(wav)
            else:
                out = m(wav)
            # Ensure all outputs have same time dimension
            outputs.append(out)
        # Align time dimension to the shortest
        min_t = min(o.shape[-1] for o in outputs)
        aligned = [o[..., :min_t] for o in outputs]
        return torch.cat(aligned, dim=1)


class StackedFrontend(nn.Module):
    """Run frontends as image channels instead of concatenating frequency rows."""

    def __init__(self, frontends: list[nn.Module], hop_length: int):
        super().__init__()
        self.frontends = nn.ModuleList(frontends)
        self.hop_length = hop_length

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        outputs = [m(wav) for m in self.frontends]
        min_t = min(o.shape[-1] for o in outputs)
        aligned = [o[..., :min_t] for o in outputs]
        return torch.stack(aligned, dim=1)


def build_frontend(
    frontend_type: str,
    sample_rate: int = 16000,
    hop_length: int = 160,
    n_mels: int = 128,
    n_fft: int = 1024,
    win_length: int | None = None,
    use_pcen: bool = False,
    pcen_s: float = 0.025,
    pcen_alpha: float = 0.98,
    pcen_delta: float = 2.0,
    pcen_r: float = 0.5,
    pcen_eps: float = 1e-6,
    cqt_bins: int = 84,
    cqt_bpo: int = 12,
    cwt_scales: int = 64,
    stft_bands_hz: tuple[tuple[float, float], ...] | None = None,
) -> tuple[nn.Module, int, nn.Module | None]:
    """Build frontend feature extractors.

    Args:
        frontend_type: Comma-separated, e.g. "mel", "stft", "cqt", "mel,cqt".
        sample_rate: Audio sample rate.
        hop_length: Hop length in samples.
        n_mels: Mel bins.
        n_fft: FFT size.
        win_length: Window length in samples.
        use_pcen: Use PCEN instead of log-mel.
        pcen_*: PCEN parameters.
        cqt_bins: Number of CQT bins.
        cqt_bpo: CQT bins per octave.
        cwt_scales: Number of CWT scales.
        stft_bands_hz: Frequency buckets used by ``stft_bands``.

    Returns:
        (frontend_module, total_channels, pcen_module_or_none)
    """
    types = [t.strip() for t in frontend_type.split(",")]
    fe_modules: list[nn.Module] = []
    channels = 0
    pcen_mod = None

    for ft in types:
        if ft == "mel":
            mel = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            fe_modules.append(mel)
            channels += n_mels
            if use_pcen:
                from audi.training.detector import PCEN
                pcen_mod = PCEN(
                    s=pcen_s, alpha=pcen_alpha, delta=pcen_delta,
                    r=pcen_r, eps=pcen_eps,
                )
        elif ft == "cqt":
            cqt = CQT(
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_bins=cqt_bins,
                bins_per_octave=cqt_bpo,
            )
            fe_modules.append(cqt)
            channels += cqt_bins
        elif ft == "stft":
            stft = LinearSTFT(
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_bins=n_mels,
                n_fft=n_fft,
                win_length=win_length,
            )
            fe_modules.append(stft)
            channels += n_mels
        elif ft == "stft_bands":
            stft = LinearSTFT(
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_bins=n_mels,
                n_fft=n_fft,
                win_length=win_length,
                frequency_bands_hz=(
                    stft_bands_hz or DEFAULT_BANDED_STFT_BANDS_HZ
                ),
            )
            fe_modules.append(stft)
            channels += n_mels
        elif ft == "cwt":
            cwt = CWT(
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_scales=cwt_scales,
            )
            fe_modules.append(cwt)
            channels += cwt_scales

    if not fe_modules:
        raise ValueError(f"No valid frontend types in '{frontend_type}'")

    if types in (
        ["stft", "stft", "stft"],
        ["stft_bands", "stft_bands", "stft_bands"],
    ):
        return StackedFrontend(fe_modules, hop_length), 3, None

    frontend = MultiFrontend(fe_modules, hop_length)
    return frontend, channels, pcen_mod
