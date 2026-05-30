"""Multi-frontend feature extraction: mel, CQT, CWT.

CQT  (librosa): log-spaced frequency bins, better for harmonic drone signals.
CWT  (ssqueezepy): multi-scale wavelet transform, better for transient detection.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T


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


def build_frontend(
    frontend_type: str,
    sample_rate: int = 16000,
    hop_length: int = 160,
    n_mels: int = 128,
    n_fft: int = 1024,
    use_pcen: bool = False,
    pcen_s: float = 0.025,
    pcen_alpha: float = 0.98,
    pcen_delta: float = 2.0,
    pcen_r: float = 0.5,
    pcen_eps: float = 1e-6,
    cqt_bins: int = 84,
    cqt_bpo: int = 12,
    cwt_scales: int = 64,
) -> tuple[MultiFrontend, int, nn.Module | None]:
    """Build frontend feature extractors.

    Args:
        frontend_type: Comma-separated, e.g. "mel", "cqt", "cwt", "mel,cqt".
        sample_rate: Audio sample rate.
        hop_length: Hop length in samples.
        n_mels: Mel bins.
        n_fft: FFT size.
        use_pcen: Use PCEN instead of log-mel.
        pcen_*: PCEN parameters.
        cqt_bins: Number of CQT bins.
        cqt_bpo: CQT bins per octave.
        cwt_scales: Number of CWT scales.

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

    frontend = MultiFrontend(fe_modules, hop_length)
    return frontend, channels, pcen_mod
