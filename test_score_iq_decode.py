from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import advanced_link_skdsp_v4_robust as link
import generate_tx_iq_dataset as genmod
import score_iq_decode as scorer


TEST_MESSAGE = (
    "This is a message-mode payload used for testing score_iq_decode with the v4 robust link. "
    "It should decode correctly in clean conditions."
)


def test_load_iq_file_npy(tmp_path: Path):
    iq = (np.arange(16) + 1j * np.arange(16)).astype(np.complex64)
    path = tmp_path / "x.npy"
    np.save(path, iq)

    loaded = scorer.load_iq_file(path)
    assert loaded.dtype == np.complex64
    np.testing.assert_array_equal(loaded, iq)


def test_load_iq_file_raw_complex64(tmp_path: Path):
    iq = (np.arange(16) + 1j * np.arange(16)).astype(np.complex64)
    path = tmp_path / "x.iq"
    iq.tofile(path)

    loaded = scorer.load_iq_file(path)
    assert loaded.dtype == np.complex64
    np.testing.assert_array_equal(loaded, iq)


def test_reconstruct_expected_random_payload():
    n_bits = 13
    seed = 7

    expected = link.build_payload_bytes_from_random_bits(n_bits, seed)[:-4]
    rebuilt = scorer.reconstruct_expected_random_payload(n_bits, seed)

    assert rebuilt == expected


def test_bytes_to_bitstring():
    data = bytes([0b10101010, 0b11110000])
    assert scorer.bytes_to_bitstring(data) == "1010101011110000"
    assert scorer.bytes_to_bitstring(data, n_bits=12) == "101010101111"


def test_levenshtein_distance():
    assert scorer.levenshtein_distance("abc", "abc") == 0
    assert scorer.levenshtein_distance("abc", "ab") == 1
    assert scorer.levenshtein_distance("abc", "axc") == 1
    assert scorer.levenshtein_distance("", "abc") == 3


def test_score_from_strings():
    assert scorer.score_from_strings("abc", "abc") == 1.0
    assert scorer.score_from_strings("abc", None) == 0.0
    assert scorer.score_from_strings("abc", "abx") == 1.0
    assert scorer.score_from_strings("abcd", "wxyz") == pytest.approx(0.25)


def test_score_decode_message_success():
    rx_result = {
        "message": "hello world",
        "payload_len": 11,
    }
    metadata = {
        "payload_source": "message",
        "message": "hello world",
    }

    assert scorer.score_decode(rx_result, metadata) == 1.0


def test_score_decode_message_partial():
    rx_result = {
        "message": "hedlo wolxd",
        "payload_len": 11,
    }
    metadata = {
        "payload_source": "message",
        "message": "hello world",
    }

    score = scorer.score_decode(rx_result, metadata)
    assert 0.0 < score < 1.0


def test_score_decode_message_failure_none():
    metadata = {
        "payload_source": "message",
        "message": "hello world",
    }
    assert scorer.score_decode(None, metadata) == 0.0


def test_score_decode_random_bits_success():
    n_bits = 511
    seed = 5
    expected_payload = scorer.reconstruct_expected_random_payload(n_bits, seed)

    rx_result = {
        "message": None,
        "payload_bytes": expected_payload,
        "payload_len": len(expected_payload),
    }
    metadata = {
        "payload_source": f"random_bits:{n_bits}",
        "random_bits": n_bits,
        "random_seed": seed,
    }

    assert scorer.score_decode(rx_result, metadata) == 1.0


def test_score_decode_random_bits_partial():
    n_bits = 16
    seed = 5
    expected_payload = scorer.reconstruct_expected_random_payload(n_bits, seed)
    wrong = bytearray(expected_payload)
    wrong[0] ^= 0b00000011

    rx_result = {
        "message": None,
        "payload_bytes": bytes(wrong),
        "payload_len": len(wrong),
    }
    metadata = {
        "payload_source": f"random_bits:{n_bits}",
        "random_bits": n_bits,
        "random_seed": seed,
    }

    score = scorer.score_decode(rx_result, metadata)
    assert 0.0 < score < 1.0


def test_write_temp_raw_iq():
    iq = (np.arange(32) + 1j * np.arange(32)).astype(np.complex64)
    temp_path = scorer.write_temp_raw_iq(iq)

    try:
        assert temp_path.exists()
        loaded = np.fromfile(temp_path, dtype=np.complex64)
        np.testing.assert_array_equal(loaded, iq)
    finally:
        temp_path.unlink(missing_ok=True)


def test_end_to_end_main_message_mode(tmp_path: Path):
    iq_path = tmp_path / "whole_iq.npy"
    meta_path = tmp_path / "whole_meta.json"

    result = link.build_tx_iq_object(
        message=TEST_MESSAGE,
        random_bits=None,
        target_num_samples=None,
        fec="conv",
        interleave=True,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
        snr_db=100.0,
        noise_color="white",
        fading_mode="none",
        seed=1,
    )

    np.save(iq_path, result.iq)

    meta = dict(result.metadata)
    meta["message"] = TEST_MESSAGE
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    score = scorer.main(["--iq", str(iq_path), "--metadata", str(meta_path)])
    assert score == 1.0


def test_end_to_end_main_random_bits_mode(tmp_path: Path):
    iq_path = tmp_path / "whole_iq.npy"
    meta_path = tmp_path / "whole_meta.json"

    n_bits = 777
    seed = 11

    result = link.build_tx_iq_object(
        message=None,
        random_bits=n_bits,
        random_seed=seed,
        target_num_samples=None,
        fec="conv",
        interleave=True,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=433_920_000.0,
        carrier_hz=0.0,
        snr_db=100.0,
        noise_color="white",
        fading_mode="none",
        seed=1,
    )

    np.save(iq_path, result.iq)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result.metadata, f, indent=2)

    score = scorer.main(["--iq", str(iq_path), "--metadata", str(meta_path)])
    assert score == 1.0


def test_end_to_end_generated_dataset_message(tmp_path: Path):
    out_root = tmp_path / "dataset"
    produced = genmod.generate_dataset(
        output_root=out_root,
        num_outputs=1,
        min_total_samples=8192,
        max_total_samples=8192,
        section_len=1024,
        num_sections=3,
        seed=123,
        random_payload_probability=0.0,
    )

    sample_dir = produced[0]
    iq_path = sample_dir / "whole_iq.npy"
    meta_path = sample_dir / "whole_meta.json"

    score = scorer.main(["--iq", str(iq_path), "--metadata", str(meta_path)])
    assert 0.0 <= score <= 1.0


def test_end_to_end_generated_dataset_random(tmp_path: Path):
    out_root = tmp_path / "dataset"
    produced = genmod.generate_dataset(
        output_root=out_root,
        num_outputs=1,
        min_total_samples=8192,
        max_total_samples=8192,
        section_len=1024,
        num_sections=3,
        seed=456,
        random_payload_probability=1.0,
    )

    sample_dir = produced[0]
    iq_path = sample_dir / "whole_iq.npy"
    meta_path = sample_dir / "whole_meta.json"

    score = scorer.main(["--iq", str(iq_path), "--metadata", str(meta_path)])
    assert 0.0 <= score <= 1.0