"""Tests for audi.config."""

import pytest

from audi.config import (
    MelConfig,
    MixConfig,
    ModelConfig,
    SNRBin,
    parse_snr_bins,
)


class TestSNRBin:
    def test_valid_bin(self):
        b = SNRBin(name="easy", low_db=-5, high_db=0, probability=0.2)
        assert b.name == "easy"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SNRBin(name="", low_db=-5, high_db=0, probability=0.2)

    def test_zero_probability_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            SNRBin(name="easy", low_db=-5, high_db=0, probability=0.0)


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.arch == "mn10_as"
        assert cfg.num_classes == 1
        assert cfg.pretrained is True

    def test_zero_num_classes_raises(self):
        with pytest.raises(ValueError, match="num_classes"):
            ModelConfig(num_classes=0)


class TestMelConfig:
    def test_default_win_length_matches_n_fft(self):
        cfg = MelConfig()

        assert cfg.win_length == cfg.n_fft

    def test_custom_win_length_is_exposed(self):
        cfg = MelConfig(n_fft=1024, win_length=512)

        assert cfg.win_length == 512

    def test_win_length_must_not_exceed_n_fft(self):
        with pytest.raises(ValueError, match="win_length"):
            MelConfig(n_fft=512, win_length=1024)


class TestMixConfig:
    def test_positive_probability_bounds(self):
        with pytest.raises(ValueError, match="positive_probability"):
            MixConfig(
                noise_path="/tmp/a",
                drone_path="/tmp/b",
                positive_probability=1.5,
            )

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError, match="positive_probability"):
            MixConfig(
                noise_path="/tmp/a",
                drone_path="/tmp/b",
                positive_probability=-0.1,
            )


class TestParseSNRBins:
    def test_valid(self):
        bins = parse_snr_bins(["easy:-5:0:0.20", "hard:-15:-10:0.20"])
        assert len(bins) == 2
        assert bins[0].name == "easy"
        assert bins[1].low_db == -15.0

    def test_empty_raises(self):
        with pytest.raises(SystemExit):
            parse_snr_bins([])

    def test_bad_format_raises(self):
        with pytest.raises(SystemExit):
            parse_snr_bins(["easy:-5:0"])  # missing probability
