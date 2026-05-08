"""cmd_dads_classify data-building subcommand."""
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
    OUT_DIR = Path("data/hf_pretrain_classify")
    
    def loop_to_length(audio, target):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) >= target:
            return audio[:target]
        return np.tile(audio, int(np.ceil(target / len(audio))))[
            :target
        ].astype(np.float32)
    
    print("Loading geronimobasso/drone-audio-detection-samples ...")
    ds = load_dataset("geronimobasso/drone-audio-detection-samples")
    
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
    
    for split in ["train", "test"]:
        if split not in ds:
            continue
        out_name = "validation" if split == "test" else split
        print(f"\nProcessing {out_name}: {len(ds[split])} samples")
        tmp_dir = OUT_DIR / f"_tmp_{out_name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        chunk_dirs = []
        n = len(ds[split])
        for cs in range(0, n, CHUNK_SIZE):
            ce = min(cs + CHUNK_SIZE, n)
            cd = tmp_dir / f"chunk_{cs:06d}"
            cd.mkdir(parents=True, exist_ok=True)
            records = []
            for i in range(cs, ce):
                sample = ds[split][i]
                arr = np.asarray(sample["audio"]["array"], dtype=np.float32)
                sr = sample["audio"]["sampling_rate"]
                if sr != _SR:
                    from scipy.signal import resample
    
                    arr = resample(arr, int(len(arr) * _SR / sr)).astype(
                        np.float32
                    )
                arr = loop_to_length(arr, _CLIP_SAMPLES)
                label_val = int(sample["label"])
                records.append(
                    {
                        "audio": {"array": arr, "sampling_rate": _SR},
                        "label": "drone" if label_val == 1 else "non-drone",
                        "is_drone": label_val,
                        "source_tag": "dads",
                        "source_file": f"dads_{i:06d}",
                    }
                )
            chunk_ds = Dataset.from_list(records, features=features)
            chunk_ds.save_to_disk(str(cd))
            chunk_dirs.append(cd)
            print(f"  chunk {cs}-{ce - 1}")
    
        print(f"  combining {len(chunk_dirs)} chunks ...")
        all_ds = concatenate_datasets(
            [Dataset.load_from_disk(str(cd)) for cd in chunk_dirs]
        )
        import shutil
    
        for cd in chunk_dirs:
            shutil.rmtree(cd, ignore_errors=True)
        out_splits[out_name] = all_ds
    
    if "validation" not in out_splits:
        train = out_splits["train"]
        nv = min(5000, len(train) // 20)
        out_splits["validation"] = train.select(range(nv))
        out_splits["train"] = train.select(range(nv, len(train)))
    
    DatasetDict(out_splits).save_to_disk(str(OUT_DIR))
    for s, d in out_splits.items():
        drones = sum(1 for x in d if x["is_drone"] == 1)
        print(
            f"  {s}: {len(d)} samples ({drones} drone, {len(d) - drones} non-drone)"
        )
    
    for p in OUT_DIR.glob("_tmp_*"):
        import shutil
    
        shutil.rmtree(p)
    print(f"\nSaved: {OUT_DIR}")
    
    
    # ====================================================================
    # analyze-snr — V2 SNR analysis with per-band metrics
    # ====================================================================
    
    
