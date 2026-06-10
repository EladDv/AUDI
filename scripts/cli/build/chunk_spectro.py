"""cmd_chunk_spectro data-building subcommand."""
from __future__ import annotations

from pathlib import Path


def run() -> None:
    import argparse
    
    import librosa
    import matplotlib
    
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from librosa import display as librosa_display
    
    _AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")
    _DEFAULT_CHUNK_SEC = 15.0
    _DEFAULT_TARGET_SR = 16000
    _DEFAULT_N_MELS = 128
    _DEFAULT_N_FFT = 1024
    _DEFAULT_FMAX = 8000
    
    def iter_audio_files(dir_path: Path) -> list[Path]:
        return sorted(
            [
                p
                for p in dir_path.rglob("*")
                if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
            ]
        )
    
    def load_mono_resampled(path: Path, target_sr: int) -> np.ndarray:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return y
    
    ap = argparse.ArgumentParser(
        description="Chunk audio files into segments and draw spectrograms."
    )
    ap.add_argument("--dataset-v2-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--chunk-sec", type=float, default=_DEFAULT_CHUNK_SEC)
    ap.add_argument("--target-sr", type=int, default=_DEFAULT_TARGET_SR)
    ap.add_argument("--n-mels", type=int, default=_DEFAULT_N_MELS)
    ap.add_argument("--n-fft", type=int, default=_DEFAULT_N_FFT)
    ap.add_argument("--win-length", type=int, default=None)
    ap.add_argument("--hop-length", type=int, default=None)
    ap.add_argument("--fmax", type=int, default=_DEFAULT_FMAX)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    win_length = args.n_fft if args.win_length is None else args.win_length
    hop_length = args.n_fft // 4 if args.hop_length is None else args.hop_length
    if win_length <= 0 or win_length > args.n_fft:
        raise SystemExit(
            f"--win-length must be in [1, --n-fft], got {win_length}"
        )
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = iter_audio_files(args.dataset_v2_dir)
    print(f"Found {len(audio_files)} audio files in {args.dataset_v2_dir}")
    for file_path in audio_files:
        print(f"\nProcessing: {file_path.name}")
        y = load_mono_resampled(file_path, args.target_sr)
        total_samples = len(y)
        chunk_samples = int(args.chunk_sec * args.target_sr)
        n_chunks = total_samples // chunk_samples
        if total_samples % chunk_samples > 0:
            y = np.pad(y, (0, chunk_samples - total_samples % chunk_samples))
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_samples
            chunk = y[start : start + chunk_samples]
            chunk_tag = f"{file_path.stem}_{chunk_idx:03d}"
            out_wav = args.output_dir / chunk_tag / f"{chunk_tag}.wav"
            out_png = args.output_dir / chunk_tag / f"{chunk_tag}_spec.png"
            if out_wav.exists() and out_png.exists() and not args.overwrite:
                continue
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            import soundfile as sf
    
            sf.write(str(out_wav), chunk, args.target_sr)
            mel_spec = librosa.feature.melspectrogram(
                y=chunk,
                sr=args.target_sr,
                n_fft=args.n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=args.n_mels,
                fmax=args.fmax,
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            fig, ax = plt.subplots(figsize=(12, 4))
            librosa_display.specshow(
                mel_db,
                sr=args.target_sr,
                x_axis="time",
                y_axis="mel",
                ax=ax,
                fmax=args.fmax,
            )
            ax.set_title(f"{chunk_tag} — Mel Spectrogram")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(
                f"  Chunk {chunk_idx:03d}: {out_wav.name} ({len(chunk) / args.target_sr:.1f}s)"
            )
    print(f"\nDone! Output: {args.output_dir}")
    
    
    # ====================================================================
    # pretrain-drones — Build HF drone pretrain dataset from geronimobasso
    # ====================================================================
    
    
