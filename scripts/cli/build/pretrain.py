"""cmd_pretrain_drones data-building subcommand."""
from __future__ import annotations

from pathlib import Path


def run() -> None:
    import numpy as np
    from datasets import (
        Audio,
        Dataset,
        DatasetDict,
        Features,
        Value,
        concatenate_datasets,
        load_dataset,
    )
    
    _SR = 16000
    _CLIP_SAMPLES = int(_SR * 2.56)
    CHUNK_SIZE = 5000
    OUT_DIR = Path("data/hf_pretrain_drone")
    
    def loop_to_length(audio, target_len):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) >= target_len:
            return audio[:target_len]
        repeats = int(np.ceil(target_len / len(audio)))
        return np.tile(audio, repeats)[:target_len].astype(np.float32)
    
    print("Loading geronimobasso/drone-audio-detection-samples ...")
    ds = load_dataset("geronimobasso/drone-audio-detection-samples")
    print(f"Splits: {list(ds.keys())}")
    
    features = Features(
        {
            "audio": Audio(sampling_rate=_SR),
            "label": Value("string"),
            "is_drone": Value("int8"),
            "source_tag": Value("string"),
            "source_file": Value("string"),
        }
    )
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_splits = {}
    
    for split in ["train", "test"] if "test" in ds else ["train"]:
        if split not in ds:
            continue
        drone_ds = ds[split].filter(lambda x: x["label"] == 1)
        out_split = split if split != "test" else "validation"
        print(f"\nProcessing {out_split}: {len(drone_ds)} drone samples")
        tmp_dir = OUT_DIR / f"_tmp_{out_split}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        chunk_dirs = []
        n = len(drone_ds)
        for chunk_start in range(0, n, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n)
            records = []
            for i in range(chunk_start, chunk_end):
                sample = drone_ds[i]
                arr = np.asarray(sample["audio"]["array"], dtype=np.float32)
                sr = sample["audio"]["sampling_rate"]
                if sr != _SR:
                    from scipy.signal import resample
    
                    arr = resample(arr, int(len(arr) * _SR / sr)).astype(
                        np.float32
                    )
                arr = loop_to_length(arr, _CLIP_SAMPLES)
                records.append(
                    {
                        "audio": {"array": arr, "sampling_rate": _SR},
                        "label": "drone",
                        "is_drone": 1,
                        "source_tag": "geronimobasso",
                        "source_file": f"pretrain_{i:06d}",
                    }
                )
            chunk_ds = Dataset.from_list(records, features=features)
            chunk_dir = tmp_dir / f"chunk_{chunk_start:06d}"
            chunk_ds.save_to_disk(str(chunk_dir))
            chunk_dirs.append(chunk_dir)
            print(f"  chunk {chunk_start}-{chunk_end - 1} → {chunk_dir}")
    
        print(f"  combining {len(chunk_dirs)} chunks ...")
        all_ds = concatenate_datasets(
            [Dataset.load_from_disk(str(cd)) for cd in chunk_dirs]
        )
        import shutil
    
        for cd in chunk_dirs:
            shutil.rmtree(cd, ignore_errors=True)
        out_splits[out_split] = all_ds
    
    if "validation" not in out_splits and "train" in out_splits:
        train_ds = out_splits["train"]
        n_val = min(5000, len(train_ds) // 20)
        out_splits["validation"] = train_ds.select(range(n_val))
        out_splits["train"] = train_ds.select(range(n_val, len(train_ds)))
    
    out = DatasetDict(out_splits)
    out.save_to_disk(str(OUT_DIR))
    print(f"\nSaved to {OUT_DIR}")
    for s, d in out_splits.items():
        labels = [x["label"] for x in d]
        n_drone = sum(1 for lb in labels if lb == "drone")
        print(f"  {s}: {len(d)} samples ({n_drone} drone)")
    
    import shutil
    
    for p in OUT_DIR.glob("_tmp_*"):
        shutil.rmtree(p)
    
    
    # ====================================================================
    # dads-classify — Build HF DADS classification dataset
    # ====================================================================
    
    
