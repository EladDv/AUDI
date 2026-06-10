"""cmd_mel_stats data-building subcommand."""
from __future__ import annotations


def run() -> None:
    from pathlib import Path

    import torch
    import torchaudio.transforms
    from torch.utils.data import DataLoader

    from audi.cli_utils import NUM_WORKERS
    from audi.config import MixConfig, parse_snr_bins
    from audi.training.dataset import make_dataset
    
    SR = 16000
    N_MELS = 128
    N_FFT = 1024
    WIN_LENGTH = 1024
    HOP = 160
    HIGH_PASS = 125.0
    CLIP_SECONDS = 2.56
    NUM_SAMPLES = 5000
    BATCH_SIZE = 64
    
    _PROJECT = Path(__file__).resolve().parents[2]  # project root
    PROJECT = _PROJECT
    snr_bins = parse_snr_bins(
        [
            "easy:-5:0:0.20",
            "medium:-10:-5:0.20",
            "hard:-15:-10:0.20",
            "very_hard:-20:-15:0.20",
            "extreme:-25:-20:0.15",
            "far_field:-30:-25:0.10",
        ]
    )
    clip_samples = int(SR * CLIP_SECONDS)
    
    val_mix_cfg = MixConfig(
        noise_path=PROJECT / "data/HF_filtered_background",
        drone_path=PROJECT / "data/HF_filtered_drone",
        snr_bins=snr_bins,
        target_length_samples=clip_samples,
        positive_probability=0.5,
        highpass_hz=HIGH_PASS,
        dataset_length=NUM_SAMPLES,
    )
    ds = make_dataset(
        cfg=val_mix_cfg,
        split="train",
        return_bin=False,
        return_components=False,
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP,
        n_mels=N_MELS,
    )
    to_db = torchaudio.transforms.AmplitudeToDB()
    
    count = 0
    mean = None
    m2 = None
    
    print(f"Processing {NUM_SAMPLES} samples in batches of {BATCH_SIZE}...")
    for i, (wav, _) in enumerate(dl):
        with torch.no_grad():
            mel_db = to_db(mel(wav))
        B, M, T = mel_db.shape
        frames = mel_db.permute(0, 2, 1).reshape(-1, M)
        for j in range(frames.shape[0]):
            x = frames[j]
            count += 1
            delta = x - mean if mean is not None else x
            if mean is None:
                mean = x.clone()
                m2 = torch.zeros_like(x)
            else:
                mean = mean + delta / count
                delta2 = x - mean
                m2 = m2 + delta * delta2
        if (i + 1) % 10 == 0:
            print(f"  {count} frames processed...")
    
    print(f"\nTotal frames: {count}")
    total_var = m2 / count
    total_std = torch.sqrt(total_var)
    scalar_mean = float(mean.mean())
    scalar_std = float(total_std.mean())
    
    print(f"\n=== Per-mel-bin stats (n_mels={N_MELS}) ===")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape:  {total_std.shape}")
    print(f"Mean[:5]:   {mean[:5].tolist()}")
    print(f"Std[:5]:    {total_std[:5].tolist()}")
    print(f"Mean[-5:]:  {mean[-5:].tolist()}")
    print(f"Std[-5:]:   {total_std[-5:].tolist()}")
    print("\n=== Scalar (averaged across mel bins) ===")
    print(f"Scalar mean: {scalar_mean:.6f}")
    print(f"Scalar std:  {scalar_std:.6f}")
    print("\n=== Python literals for copy-paste ===\n")
    print(f"# Per-bin (n_mels={N_MELS})")
    for start in range(0, N_MELS, 8):
        chunk_mean = mean[start : start + 8].tolist()
        label = "MEL_MEAN_PER_BIN" if start == 0 else " " * 17
        print(f"{label} = {[round(v, 6) for v in chunk_mean]},")
    print()
    for start in range(0, N_MELS, 8):
        chunk_std = total_std[start : start + 8].tolist()
        label = "MEL_STD_PER_BIN" if start == 0 else " " * 16
        print(f"{label}  = {[round(v, 6) for v in chunk_std]},")
    print("\n# Scalar (averaged across mel bins)")
    print(f"MEL_MEAN_SCALAR = {scalar_mean:.6f}")
    print(f"MEL_STD_SCALAR  = {scalar_std:.6f}")
    
    
    # ====================================================================
    # Entry point
    # ====================================================================
    
    
