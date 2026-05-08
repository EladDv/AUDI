"""Build expanded noise dataset from HF sources — ESC-50 + UrbanSound8K + 6 new datasets.

Combines ESC-50 + UrbanSound8K with ambient_sounds, MUSAN-noise, DEMAND,
TUT-urban, and Sunbird (urban Uganda). Classifies
all clips into 8 categories, chunks into 30-60s segments, and saves as
per-category HF DatasetDicts (each with train/val/test splits).

Categories: people, environment, wind, cars, urban, mechanical, indoor, bioacoustic

Usage:
  uv run python scripts/cli/build/urban_esc.py
  uv run python scripts/cli/build/urban_esc.py --datasets esc50 urbansound ambient
  HF_TOKEN=... uv run python scripts/cli/build/urban_esc.py
"""

from __future__ import annotations


def run() -> None:
    import argparse
    import io
    import os
    import random
    import shutil
    import tempfile
    import traceback
    import warnings
    from pathlib import Path
    from typing import Any

    import numpy as np
    import requests

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    import librosa
    from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset

    # -----------------------------------------------------------------------
    # Constants
    # -----------------------------------------------------------------------
    CATEGORIES = [
        "people",
        "environment",
        "wind",
        "cars",
        "urban",
        "mechanical",
        "indoor",
        "bioacoustic",
    ]
    _SR = 16000

    # -----------------------------------------------------------------------
    # Classification maps
    # -----------------------------------------------------------------------
    ESC50_MAP: dict[str, str] = {
        "clapping": "people",
        "breathing": "people",
        "coughing": "people",
        "footsteps": "people",
        "laughing": "people",
        "snoring": "people",
        "drinking_sipping": "people",
        "door_wood_knock": "people",
        "wind": "wind",
        "rain": "wind",
        "sea_waves": "wind",
        "crackling_fire": "wind",
        "thunderstorm": "wind",
        "car_horn": "cars",
        "engine": "cars",
        "train": "cars",
        "helicopter": "cars",
        "airplane": "cars",
        "siren": "cars",
        "dog": "environment",
        "rooster": "environment",
        "pig": "environment",
        "cow": "environment",
        "frog": "environment",
        "cat": "environment",
        "hen": "environment",
        "insects": "environment",
        "sheep": "environment",
        "crow": "environment",
        "crickets": "environment",
        "chirping_birds": "environment",
        "water_drops": "environment",
        "pouring_water": "environment",
        "toilet_flush": "indoor",
        "door_wood_creaks": "indoor",
        "can_opening": "indoor",
        "washing_machine": "mechanical",
        "vacuum_cleaner": "mechanical",
        "clock_alarm": "mechanical",
        "clock_tick": "mechanical",
        "glass_breaking": "environment",
        "chainsaw": "mechanical",
        "church_bells": "urban",
        "firecrackers": "urban",
        "hand_saw": "mechanical",
    }

    URBANSOUND_MAP: dict[str, str] = {
        "air_conditioner": "mechanical",
        "car_horn": "cars",
        "dog_bark": "environment",
        "drilling": "mechanical",
        "engine_idling": "cars",
        "gun_shot": "urban",
        "jackhammer": "mechanical",
        "siren": "cars",
        "street_music": "people",
    }

    AMBIENT_MAP: dict[str, str] = {
        "airplane overhead": "cars",
        "Birds_chirping": "environment",
        "Birds chirping": "environment",
        "birds chirping": "environment",
        "birds_chirping": "environment",
        "Bus": "cars",
        "bus": "cars",
        "Car_engine": "cars",
        "Car engine": "cars",
        "car_engine": "cars",
        "Car_reverse": "cars",
        "car_reverse": "cars",
        "Coffee_machine": "mechanical",
        "coffee_machine": "mechanical",
        "Coughing": "people",
        "coughing": "people",
        "Cutlery": "indoor",
        "cutlery": "indoor",
        "Door_closing": "indoor",
        "door_closing": "indoor",
        "Doorbell": "indoor",
        "doorbell": "indoor",
        "Escalator": "mechanical",
        "escalator": "mechanical",
        "Footsteps": "people",
        "Footsteps_on_concrete": "people",
        "footsteps_on_concrete": "people",
        "Fridge_humming": "mechanical",
        "fridge_humming": "mechanical",
        "Hair_dryer": "mechanical",
        "hair_dryer": "mechanical",
        "Honking": "cars",
        "honking": "cars",
        "Knock": "people",
        "knock": "people",
        "Leafes": "environment",
        "leafes": "environment",
        "leaves": "environment",
        "Microwave": "mechanical",
        "microwave": "mechanical",
        "Motorcycle_engine": "cars",
        "motorcycle_engine": "cars",
        "music_background": "people",
        "background_music": "people",
        "River": "wind",
        "river": "wind",
        "Shower": "indoor",
        "shower": "indoor",
        "Bike_bell": "cars",
        "bike_bell": "cars",
    }

    DEMAND_SCENE_MAP: dict[str, str] = {
        "DKITCHEN": "indoor",
        "DWASHING": "mechanical",
        "DLIVING": "indoor",
        "OFFICE": "indoor",
        "MEETING": "people",
        "RESTAURANT": "indoor",
        "OHALLWAY": "indoor",
        "OFIELD": "environment",
        "NPARK": "environment",
        "NRIVER": "wind",
        "SCAFE": "indoor",
        "SPSQUARE": "urban",
        "SSTREET": "urban",
        "TBUS": "cars",
        "TMETRO": "cars",
        "BUS": "cars",
        "CAFE": "indoor",
        "PRESCHOOL": "people",
        "STRAFFIC": "cars",
        "SSTRAFFIC": "cars",
    }

    TUT_SCENE_MAP: dict[str, str] = {
        "airport": "urban",
        "bus": "cars",
        "metro": "cars",
        "metro_station": "urban",
        "park": "environment",
        "public_square": "urban",
        "shopping_mall": "indoor",
        "street_pedestrian": "urban",
        "street_traffic": "cars",
        "tram": "cars",
    }

    # -----------------------------------------------------------------------
    # Audio helpers
    # -----------------------------------------------------------------------
    def _load_mono_resampled(arr: np.ndarray, orig_sr: int, target_sr: int = _SR) -> np.ndarray:
        y = np.asarray(arr, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=0)
        if orig_sr != target_sr and y.size > 0:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        return np.asarray(y, dtype=np.float32)

    def _decode_audio(sample: dict, audio_key: str = "audio") -> np.ndarray | None:
        aud = sample.get(audio_key)
        if aud is None:
            return None
        if isinstance(aud, dict):
            if "array" in aud:
                arr = aud.get("array")
                sr = aud.get("sampling_rate", _SR)
                if arr is None:
                    return None
                return _load_mono_resampled(np.asarray(arr, dtype=np.float32), int(sr))
            if "bytes" in aud:
                raw = aud["bytes"]
                if raw is None:
                    return None
                try:
                    import soundfile as sf
                    data, sr = sf.read(io.BytesIO(raw))
                    return _load_mono_resampled(data, int(sr))
                except Exception:
                    return None
        if hasattr(aud, "numpy"):
            return aud.numpy().astype(np.float32)
        if hasattr(aud, "__array__"):
            return np.asarray(aud, dtype=np.float32)
        if hasattr(aud, "get_all_samples"):
            try:
                samples = aud.get_all_samples()
                arr = np.array(samples.data, dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                sr = samples.sample_rate
                if sr != _SR and arr.size > 0:
                    arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=_SR)
                return arr.astype(np.float32)
            except Exception:
                return None
        return None

    def _tile_to_length(y: np.ndarray, length: int) -> np.ndarray:
        arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return np.zeros(length, dtype=np.float32)
        if arr.size >= length:
            return arr[:length]
        reps = int(np.ceil(float(length) / float(arr.size)))
        return np.tile(arr, reps)[:length].astype(np.float32)

    def _build_chunks(
        clips: list[np.ndarray],
        *,
        min_dur: float = 30.0,
        max_dur: float = 60.0,
        target_sr: int = _SR,
        rng: random.Random,
    ) -> list[np.ndarray]:
        if not clips:
            return []
        rng.shuffle(clips)
        chunks: list[np.ndarray] = []
        buf: list[np.ndarray] = []
        buf_samples = 0
        min_samples = int(min_dur * target_sr)
        max_samples = int(max_dur * target_sr)
        for arr in clips:
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            if buf_samples + len(arr) > max_samples and buf_samples >= min_samples:
                chunks.append(np.concatenate(buf))
                buf = []
                buf_samples = 0
            buf.append(arr)
            buf_samples += len(arr)
        if buf:
            chunks.append(np.concatenate(buf))
        return chunks

    def _features() -> Features:
        return Features(
            {
                "audio": Audio(sampling_rate=_SR),
                "category": Value("string"),
                "source": Value("string"),
            }
        )

    # -----------------------------------------------------------------------
    # Dataset collectors
    # -----------------------------------------------------------------------
    def _print_buckets(name: str, buckets: dict[str, list[np.ndarray]]) -> None:
        print(f"  {name}:")
        for cat in CATEGORIES:
            clips = buckets[cat]
            total_s = sum(len(a) / _SR for a in clips)
            print(f"    {cat}: {len(clips)} clips, {total_s:.0f}s")
        total_clips = sum(len(v) for v in buckets.values())
        total_s = sum(sum(len(a) / _SR for a in v) for v in buckets.values())
        print(f"    TOTAL: {total_clips} clips, {total_s / 3600:.1f}h")

    def _collect_esc50() -> dict[str, list[np.ndarray]]:
        print("Loading ESC-50...")
        ds = load_dataset("ashraq/esc50", split="train")
        print(f"  {len(ds)} clips")
        buckets: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES}
        for i in range(len(ds)):
            row = ds[i]
            cat = ESC50_MAP.get((row.get("category") or "").strip(), "environment")
            aud = row["audio"]
            arr = _load_mono_resampled(aud["array"], aud["sampling_rate"])
            if arr.size > 0:
                buckets[cat].append(arr)
        _print_buckets("ESC-50", buckets)
        return buckets

    def _collect_urbansound() -> dict[str, list[np.ndarray]]:
        print("Loading UrbanSound8K...")
        ds = load_dataset("danavery/urbansound8K", split="train")
        print(f"  {len(ds)} clips")
        buckets: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES}
        for i in range(len(ds)):
            row = ds[i]
            cat = URBANSOUND_MAP.get((row.get("class") or "").strip(), "environment")
            aud = row["audio"]
            arr = _load_mono_resampled(aud["array"], aud["sampling_rate"])
            if arr.size > 0:
                buckets[cat].append(arr)
        _print_buckets("UrbanSound8K", buckets)
        return buckets

    def _collect_ambient_sounds() -> dict[str, list[np.ndarray]]:
        print("Loading ambient_sounds...")
        ds = load_dataset("maxF6YsK/ambient_sounds", split="train")
        print(f"  {len(ds)} clips")
        buckets: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES}
        token = os.environ.get("HF_TOKEN", "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        api_url = "https://huggingface.co/api/datasets/maxF6YsK/ambient_sounds"
        try:
            r = requests.get(api_url, headers=headers, timeout=30)
            siblings = r.json().get("siblings", []) if r.status_code == 200 else []
        except Exception:
            siblings = []
        file_map = {}
        for sib in siblings:
            rf = sib.get("rfilename", "")
            if rf.endswith((".m4a", ".wav", ".mp3", ".flac", ".ogg")):
                file_map[os.path.basename(rf)] = rf
        for i in range(len(ds)):
            row = ds[i]
            ps_key = str(row.get("primarySound", "")).strip().lower().replace(" ", "_")
            cat = AMBIENT_MAP.get(ps_key, AMBIENT_MAP.get(row.get("primarySound", "").strip(), "environment"))
            audio_file = row.get("audioFile", "")
            if audio_file in file_map:
                dl_url = f"https://huggingface.co/datasets/maxF6YsK/ambient_sounds/resolve/main/{file_map[audio_file]}"
                try:
                    r2 = requests.get(dl_url, headers=headers, timeout=30)
                    if r2.status_code == 200:
                        ext = os.path.splitext(audio_file)[1] or ".m4a"
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                            tmp.write(r2.content)
                            tmp_path = tmp.name
                        try:
                            y, sr = librosa.load(tmp_path, sr=_SR, mono=True)
                            arr = np.asarray(y, dtype=np.float32)
                            if arr.size > 0:
                                buckets[cat].append(arr)
                        finally:
                            os.unlink(tmp_path)
                except Exception as e:
                    print(f"  Warning: failed to load {audio_file}: {e}")
        _print_buckets("ambient_sounds", buckets)
        return buckets

    def _collect_musan() -> dict[str, list[np.ndarray]]:
        print("Loading MUSAN noise...")
        ds = load_dataset("bilguun/musan-noise", split="train")
        print(f"  {len(ds)} clips")
        buckets: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES}
        count = 0
        for i in range(len(ds)):
            row = ds[i]
            arr = _decode_audio(row)
            if arr is None or arr.size < _SR // 2:
                continue
            buckets["environment"].append(arr)
            count += 1
        print(f"  collected {count} noise samples")
        _print_buckets("MUSAN", buckets)
        return buckets

    def _collect_demand() -> dict[str, list[np.ndarray]]:
        print("Loading DEMAND...")
        ds = load_dataset("voice-biomarkers/DEMAND-acoustic-noise", split="train")
        print(f"  {len(ds)} clips")
        buckets: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES}
        count = 0
        for i in range(len(ds)):
            row = ds[i]
            fn = str(row.get("file_name", ""))
            scene = "indoor"
            for prefix, cat in DEMAND_SCENE_MAP.items():
                if fn.upper().startswith(prefix.upper()):
                    scene = cat
                    break
            arr = _decode_audio(row)
            if arr is None or arr.size == 0:
                continue
            buckets[scene].append(arr)
            count += 1
            if count % 1000 == 0:
                print(f"  ... processed {count} samples")
        _print_buckets("DEMAND", buckets)
        return buckets

    def _collect_tut() -> dict[str, list[np.ndarray]]:
        print("Loading TUT-urban...")
        ds = load_dataset("wetdog/TUT-urban-acoustic-scenes-2018-development", split="train")
        total = len(ds)
        max_samples = min(total, 8000)
        print(f"  {total} clips, sampling {max_samples}")
        buckets: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES}
        count = 0
        indices = np.random.default_rng(42).choice(total, size=max_samples, replace=False)
        for idx in indices:
            row = ds[int(idx)]
            scene = str(row.get("scene_label", "")).strip()
            cat = TUT_SCENE_MAP.get(scene, "urban")
            arr = _decode_audio(row)
            if arr is None or arr.size == 0:
                continue
            buckets[cat].append(arr)
            count += 1
            if count % 1000 == 0:
                print(f"  ... processed {count} samples")
        _print_buckets("TUT-urban", buckets)
        return buckets

    def _collect_sunbird() -> dict[str, list[np.ndarray]]:
        print("Loading Sunbird urban-noise-uganda-61k...")
        ds = load_dataset("Sunbird/urban-noise-uganda-61k", "large", split="train")
        ds = ds.cast_column("audio", Audio(sampling_rate=_SR, decode=True))
        total = len(ds)
        max_samples = min(total, 500)
        print(f"  {total} clips, sampling {max_samples}")
        buckets: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES}
        count = 0
        indices = np.random.default_rng(42).choice(total, size=max_samples, replace=False)
        for idx in indices:
            row = ds[int(idx)]
            cls_name = str(row.get("class", "")).strip().lower()
            if "traffic" in cls_name or "vehicle" in cls_name or "car" in cls_name:
                cat = "cars"
            elif "speech" in cls_name or "voice" in cls_name or "talking" in cls_name:
                cat = "people"
            elif "construction" in cls_name or "machinery" in cls_name:
                cat = "mechanical"
            elif "nature" in cls_name or "animal" in cls_name or "bird" in cls_name:
                cat = "environment"
            elif "wind" in cls_name or "rain" in cls_name:
                cat = "wind"
            else:
                cat = "urban"
            arr = _decode_audio(row)
            if arr is None or arr.size == 0:
                continue
            buckets[cat].append(arr)
            count += 1
            if count % 100 == 0:
                print(f"  ... processed {count} samples")
        _print_buckets("Sunbird", buckets)
        return buckets

    def _merge_buckets(*bucket_dicts: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
        merged: dict[str, list[np.ndarray]] = {c: [] for c in CATEGORIES}
        for bd in bucket_dicts:
            for cat in CATEGORIES:
                merged[cat].extend(bd.get(cat, []))
        return merged

    # -----------------------------------------------------------------------
    # CLI
    # -----------------------------------------------------------------------
    ap = argparse.ArgumentParser(
        description="Build expanded HF noise dataset from 8 public sources"
    )
    ap.add_argument(
        "--output-path", type=Path, default=Path("data/hf_background_urban")
    )
    ap.add_argument("--chunk-min", type=float, default=30.0)
    ap.add_argument("--chunk-max", type=float, default=60.0)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-clips-per-category", type=int, default=0,
        help="Max clips per category after merging. 0 = per-category weighted limits (wind/environment 1000, cars/urban 900, mechanical 800, people/indoor 700).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--datasets", type=str, nargs="*", default=None,
        help="Specific datasets. Options: esc50, urbansound, ambient, musan, demand, tut, sunbird"
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    target_sr = _SR

    include = (
        set(args.datasets)
        if args.datasets
        else {"esc50", "urbansound", "ambient", "musan", "demand", "tut", "sunbird"}
    )

    # -----------------------------------------------------------------------
    # Collect
    # -----------------------------------------------------------------------
    all_buckets: list[dict[str, list[np.ndarray]]] = []

    if "esc50" in include:
        try:
            all_buckets.append(_collect_esc50())
        except Exception as e:
            print(f"ERROR collecting ESC-50: {e}")
    if "urbansound" in include:
        try:
            all_buckets.append(_collect_urbansound())
        except Exception as e:
            print(f"ERROR collecting UrbanSound8K: {e}")
    if "ambient" in include:
        try:
            all_buckets.append(_collect_ambient_sounds())
        except Exception as e:
            print(f"ERROR collecting ambient_sounds: {e}")
    if "musan" in include:
        try:
            all_buckets.append(_collect_musan())
        except Exception as e:
            print(f"ERROR collecting MUSAN: {e}")
    if "demand" in include:
        try:
            all_buckets.append(_collect_demand())
        except Exception as e:
            print(f"ERROR collecting DEMAND: {e}")
    if "tut" in include:
        try:
            all_buckets.append(_collect_tut())
        except Exception as e:
            print(f"ERROR collecting TUT: {e}")
    if "sunbird" in include:
        try:
            all_buckets.append(_collect_sunbird())
        except Exception as e:
            print(f"ERROR collecting Sunbird: {e}")

    merged = _merge_buckets(*all_buckets)
    print("\n" + "=" * 50)
    print("MERGED (all sources):")
    _print_buckets("ALL", merged)

    # -----------------------------------------------------------------------
    # Balance
    # -----------------------------------------------------------------------
    max_clips = args.max_clips_per_category
    if max_clips == 0:
        category_limits: dict[str, int] = {
            "wind": 1000,
            "environment": 1000,
            "cars": 900,
            "urban": 900,
            "mechanical": 800,
            "people": 700,
            "indoor": 700,
            "bioacoustic": 600,
        }
    else:
        category_limits = {c: max_clips for c in CATEGORIES}

    print("\nBalancing categories...")
    for cat in CATEGORIES:
        limit = category_limits.get(cat, 0)
        clips = merged[cat]
        if limit > 0 and len(clips) > limit:
            rng.shuffle(clips)
            merged[cat] = clips[:limit]
    print("After balancing:")
    _print_buckets("BALANCED", merged)

    # -----------------------------------------------------------------------
    # Chunk and build per-category DatasetDicts
    # -----------------------------------------------------------------------
    print(f"\nBuilding chunks ({args.chunk_min}-{args.chunk_max}s per chunk)...")

    args.output_path.mkdir(parents=True, exist_ok=True)

    for cat in CATEGORIES:
        clips = merged[cat]
        if not clips:
            print(f"  {cat}: 0 clips — SKIPPED")
            continue

        chunks = _build_chunks(
            clips,
            min_dur=float(args.chunk_min),
            max_dur=float(args.chunk_max),
            target_sr=target_sr,
            rng=rng,
        )
        print(f"  {cat}: {len(chunks)} chunks from {len(clips)} clips")

        if not chunks:
            print(f"  {cat}: 0 chunks — SKIPPED")
            continue

        records = [
            {
                "audio": {"array": arr.astype(np.float32), "sampling_rate": target_sr},
                "category": cat,
                "source": "expanded_noise_v1",
            }
            for arr in chunks
            if arr.size > 0
        ]

        ds = Dataset.from_list(records, features=_features())
        total = len(ds)
        test_n = max(1, int(total * args.test_ratio))
        val_n = max(1, int(total * args.val_ratio))
        train_n = total - test_n - val_n
        if train_n <= 0:
            train_n = 1
            test_n = max(1, total - train_n)
            val_n = total - train_n - test_n

        shuffled = ds.shuffle(seed=args.seed)
        splits = DatasetDict(
            {
                "train": shuffled.select(range(train_n)),
                "validation": shuffled.select(range(train_n, train_n + val_n)),
                "test": shuffled.select(range(train_n + val_n, total)),
            }
        )
        cat_path = args.output_path / cat
        if cat_path.exists():
            if args.overwrite:
                shutil.rmtree(cat_path)
            else:
                raise SystemExit(f"Output dir exists: {cat_path}. Use --overwrite to replace.")
        splits.save_to_disk(str(cat_path))
        print(f"    Saved: {cat_path} ({train_n} train, {val_n} val, {test_n} test)")

    print(f"\nDone — saved to {args.output_path}")
