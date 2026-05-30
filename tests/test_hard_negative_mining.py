import csv
import json

import numpy as np
import soundfile as sf

from audi.hard_negative_mining import (
    AlertRun,
    build_alert_exclusions,
    clip_with_padding,
    discover_field_recordings,
    extract_alert_runs,
    is_allowed,
    write_manifest,
)


def test_extract_alert_runs_keeps_only_persistent_runs():
    scores = np.array([0.1, 0.9, 0.95, 0.2, 0.91, 0.92, 0.93, 0.1])
    times = np.arange(len(scores), dtype=np.float32)

    runs = extract_alert_runs(scores, times, threshold=0.8, min_windows=3)

    assert runs == [
        AlertRun(start_s=4.0, end_s=6.0, max_score=0.93, mean_score=0.92, n_windows=3)
    ]


def test_is_allowed_rejects_windows_that_overlap_exclusions():
    exclusions = [(10.0, 20.0)]

    assert is_allowed(0.0, 5.0, exclusions)
    assert not is_allowed(5.0, 12.0, exclusions)
    assert not is_allowed(18.0, 24.0, exclusions)
    assert is_allowed(21.0, 25.0, exclusions)


def test_clip_with_padding_returns_fixed_length():
    audio = np.arange(5, dtype=np.float32)

    clip = clip_with_padding(audio, start_sample=3, length_samples=5)

    np.testing.assert_array_equal(clip, np.array([3, 4, 0, 0, 0], dtype=np.float32))


def test_build_alert_exclusions_uses_segments_and_conservative_unlabeled_full_window(tmp_path):
    field_dir = tmp_path
    alerts = field_dir / "alerts"
    alerts.mkdir()
    labels = field_dir / "labels.csv"
    labels.write_text(
        "alert_dir,label\n"
        "yes_1000,drone\n"
        "yes_2000,nodrone\n"
        "yes_3000,drone\n"
    )
    (field_dir / "segments.json").write_text(json.dumps({"yes_1000": [[50.0, 70.0]]}))
    for alert_id, timestamp in [
        ("yes_1000", 1000),
        ("yes_2000", 2000),
        ("yes_3000", 3000),
        ("yes_4000", 4000),
    ]:
        d = alerts / alert_id
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({"timestamp": timestamp}))

    exclusions = build_alert_exclusions(field_dir, buffer_s=5.0)

    assert (985.0, 1015.0) in exclusions
    assert (2940.0, 3060.0) in exclusions
    assert (3940.0, 4060.0) in exclusions
    assert all(not (1940.0 <= start <= 2060.0) for start, _end in exclusions)


def test_discover_field_recordings_only_uses_field_recording_dir(tmp_path):
    field_dir = tmp_path / "field"
    rec_dir = field_dir / "recordings"
    rec_dir.mkdir(parents=True)
    sf.write(rec_dir / "seg_123_000001.flac", np.zeros(1600, dtype=np.float32), 16000)
    (tmp_path / "551").mkdir()
    sf.write(tmp_path / "551" / "other.wav", np.zeros(1600, dtype=np.float32), 16000)

    recordings = discover_field_recordings(field_dir)

    assert len(recordings) == 1
    assert recordings[0].path.name == "seg_123_000001.flac"
    assert recordings[0].start_epoch == 123.0


def test_discover_field_recordings_skips_unreadable_audio(tmp_path):
    field_dir = tmp_path / "field"
    rec_dir = field_dir / "recordings"
    rec_dir.mkdir(parents=True)
    sf.write(rec_dir / "seg_123_000001.flac", np.zeros(1600, dtype=np.float32), 16000)
    (rec_dir / "seg_456_000002.flac").write_text("not audio")

    recordings = discover_field_recordings(field_dir)

    assert [r.path.name for r in recordings] == ["seg_123_000001.flac"]


def test_write_manifest_outputs_csv(tmp_path):
    rows = [
        {
            "clip_path": "clips/a.wav",
            "source_path": "recordings/seg_1.flac",
            "start_s": 1.0,
            "end_s": 6.12,
            "max_score": 0.9,
        }
    ]

    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest)

    with manifest.open() as f:
        loaded = list(csv.DictReader(f))
    assert loaded[0]["clip_path"] == "clips/a.wav"
    assert loaded[0]["max_score"] == "0.9"
