# test_generate_tx_iq_dataset.py
#
# Pytest tests for the decodable-only dataset generator:
#   generate_tx_iq_dataset.py
#
# This version matches the corrected behavior:
# - requested target sample count is a MINIMUM, not an exact final length
# - the generator may increase the final IQ length to fit at least one full decodable packet
# - each saved whole IQ sample must decode correctly with advanced_link_skdsp_v4_robust
#
# Usage:
#   pytest -q test_generate_tx_iq_dataset.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import advanced_link_skdsp_v4_robust as link
import generate_tx_iq_dataset as genmod


def test_random_phrase():
    import random
    rng = random.Random(1)
    s = genmod.random_phrase(100, rng)
    assert isinstance(s, str)
    assert len(s) > 0
    assert len(s) <= 100


def test_realistic_params():
    import random
    rng = random.Random(1)
    p = genmod.realistic_params(rng)

    assert "sample_rate_hz" in p
    assert "rf_center_hz" in p
    assert "carrier_hz" in p
    assert "fec" in p
    assert "interleave" in p
    assert "snr_db" in p
    assert "noise_color" in p
    assert "fading_mode" in p
    assert "freq_offset" in p
    assert "timing_offset" in p
    assert p["fec"] in {"rep3", "conv", "none"}
    assert p["noise_color"] in {"white", "pink"}
    assert p["fading_mode"] in {"none", "rician_block", "multipath_static"}

    # corrected generator keeps carrier at zero for reliable acceptance
    assert p["carrier_hz"] == 0.0


def test_conservative_fallback_params():
    import random
    rng = random.Random(2)
    p = genmod.conservative_fallback_params(rng)

    assert p["carrier_hz"] == 0.0
    assert p["fec"] == "conv"
    assert p["interleave"] is True
    assert p["noise_color"] == "white"
    assert p["burst_probability"] == 0.0
    assert p["fading_mode"] == "none"


def test_cut_random_sections():
    iq = (np.arange(10000) + 1j * np.arange(10000)).astype(np.complex64)
    cuts = genmod.cut_random_sections(iq=iq, num_sections=3, section_len=1024, seed=1)

    assert cuts["sections"].shape == (3, 1024)
    assert cuts["sections"].dtype == np.complex64
    assert len(cuts["starts"]) == 3

    for sec, start in zip(cuts["sections"], cuts["starts"]):
        np.testing.assert_array_equal(sec, iq[start:start + 1024])


def test_build_payload_description_message_mode():
    import random
    rng = random.Random(10)

    message, random_bits, random_seed, payload_desc = genmod._build_payload_description(
        rng=rng,
        idx=0,
        random_payload_probability=0.0,
    )

    assert isinstance(message, str)
    assert random_bits is None
    assert payload_desc["mode"] == "message"
    assert payload_desc["message"] == message
    assert payload_desc["message_length"] == len(message)
    assert payload_desc["message_preview"] == message[:120]


def test_build_payload_description_random_mode():
    import random
    rng = random.Random(11)

    message, random_bits, random_seed, payload_desc = genmod._build_payload_description(
        rng=rng,
        idx=3,
        random_payload_probability=1.0,
    )

    assert message is None
    assert isinstance(random_bits, int)
    assert isinstance(random_seed, int)
    assert payload_desc["mode"] == "random_bits"
    assert payload_desc["random_bits"] == random_bits
    assert payload_desc["random_seed"] == random_seed


def test_candidate_matches_payload_message():
    payload_desc = {
        "mode": "message",
        "message": "hello world",
    }
    assert genmod._candidate_matches_payload({"message": "hello world"}, payload_desc) is True
    assert genmod._candidate_matches_payload({"message": "hello worxd"}, payload_desc) is False
    assert genmod._candidate_matches_payload(None, payload_desc) is False


def test_candidate_matches_payload_random_bits():
    n_bits = 17
    seed = 5
    expected_payload = link.build_payload_bytes_from_random_bits(n_bits, seed)[:-4]

    payload_desc = {
        "mode": "random_bits",
        "random_bits": n_bits,
        "random_seed": seed,
    }

    assert genmod._candidate_matches_payload({"payload_bytes": expected_payload}, payload_desc) is True
    assert genmod._candidate_matches_payload({"payload_bytes": b"\x00" * len(expected_payload)}, payload_desc) is False
    assert genmod._candidate_matches_payload(None, payload_desc) is False


def test_build_candidate_iq_with_full_packet_message():
    import random
    rng = random.Random(20)
    params = genmod.conservative_fallback_params(rng)

    message = "A" * 600
    requested = 12000

    tx_result = genmod._build_candidate_iq_with_full_packet(
        message=message,
        random_bits=None,
        random_seed=1,
        requested_target_num_samples=requested,
        params=params,
    )

    assert isinstance(tx_result.iq, np.ndarray)
    assert tx_result.iq.dtype == np.complex64
    assert len(tx_result.iq) >= requested
    assert tx_result.metadata["actual_num_samples"] == len(tx_result.iq)


def test_build_candidate_iq_with_full_packet_random_bits():
    import random
    rng = random.Random(21)
    params = genmod.conservative_fallback_params(rng)

    requested = 12000
    tx_result = genmod._build_candidate_iq_with_full_packet(
        message=None,
        random_bits=700 * 8,
        random_seed=7,
        requested_target_num_samples=requested,
        params=params,
    )

    assert len(tx_result.iq) >= requested
    assert tx_result.metadata["actual_num_samples"] == len(tx_result.iq)


def test_decode_candidate_with_v4_message():
    import random
    rng = random.Random(30)
    params = genmod.conservative_fallback_params(rng)

    message = (
        "This is a clean test message for decode verification using the v4 robust "
        "link implementation. " * 8
    )

    tx_result = genmod._build_candidate_iq_with_full_packet(
        message=message,
        random_bits=None,
        random_seed=1,
        requested_target_num_samples=12000,
        params=params,
    )

    meta = dict(tx_result.metadata)
    rx_result = genmod._decode_candidate_with_v4(tx_result.iq, meta)

    assert rx_result is not None
    assert rx_result["message"] == message


def test_decode_candidate_with_v4_random_bits():
    import random
    rng = random.Random(31)
    params = genmod.conservative_fallback_params(rng)

    n_bits = 700 * 8
    seed = 7

    tx_result = genmod._build_candidate_iq_with_full_packet(
        message=None,
        random_bits=n_bits,
        random_seed=seed,
        requested_target_num_samples=12000,
        params=params,
    )

    meta = dict(tx_result.metadata)
    rx_result = genmod._decode_candidate_with_v4(tx_result.iq, meta)

    assert rx_result is not None
    expected = link.build_payload_bytes_from_random_bits(n_bits, seed)[:-4]
    assert rx_result["payload_bytes"] == expected


def test_build_decodable_sample_message_mode():
    import random
    rng = random.Random(123)

    whole_iq, whole_meta = genmod.build_decodable_sample(
        dataset_index=0,
        target_num_samples=12000,
        rng=rng,
        random_payload_probability=0.0,
        max_attempts_per_sample=20,
    )

    assert isinstance(whole_iq, np.ndarray)
    assert whole_iq.dtype == np.complex64
    assert len(whole_iq) >= 12000
    assert whole_meta["actual_num_samples"] == len(whole_iq)
    assert whole_meta["requested_target_num_samples"] == 12000
    assert whole_meta["decode_verified"] is True
    assert whole_meta["payload_desc"]["mode"] == "message"
    assert "message" in whole_meta["payload_desc"]

    rx_result = genmod._decode_candidate_with_v4(whole_iq, whole_meta)
    assert rx_result is not None
    assert rx_result["message"] == whole_meta["payload_desc"]["message"]


def test_build_decodable_sample_random_mode():
    import random
    rng = random.Random(456)

    whole_iq, whole_meta = genmod.build_decodable_sample(
        dataset_index=1,
        target_num_samples=14000,
        rng=rng,
        random_payload_probability=1.0,
        max_attempts_per_sample=20,
    )

    assert isinstance(whole_iq, np.ndarray)
    assert whole_iq.dtype == np.complex64
    assert len(whole_iq) >= 14000
    assert whole_meta["actual_num_samples"] == len(whole_iq)
    assert whole_meta["requested_target_num_samples"] == 14000
    assert whole_meta["decode_verified"] is True
    assert whole_meta["payload_desc"]["mode"] == "random_bits"

    rx_result = genmod._decode_candidate_with_v4(whole_iq, whole_meta)
    assert rx_result is not None

    n_bits = whole_meta["payload_desc"]["random_bits"]
    seed = whole_meta["payload_desc"]["random_seed"]
    expected = link.build_payload_bytes_from_random_bits(n_bits, seed)[:-4]
    assert rx_result["payload_bytes"] == expected


def test_save_sample_bundle(tmp_path: Path):
    out_dir = tmp_path / "sample_000000"

    whole_iq = (np.arange(4096) + 1j * np.arange(4096)).astype(np.complex64)
    whole_meta = {"a": 1, "b": 2}
    sections = np.stack([whole_iq[:1024], whole_iq[100:1124], whole_iq[200:1224]], axis=0)
    sections_meta = {"starts": [0, 100, 200], "num_sections": 3, "section_len": 1024}

    genmod.save_sample_bundle(
        out_dir=out_dir,
        whole_iq=whole_iq,
        whole_meta=whole_meta,
        sections=sections,
        sections_meta=sections_meta,
    )

    assert (out_dir / "whole_iq.npy").exists()
    assert (out_dir / "whole_meta.json").exists()
    assert (out_dir / "sections.npy").exists()
    assert (out_dir / "sections_meta.json").exists()

    loaded_whole = np.load(out_dir / "whole_iq.npy")
    loaded_sections = np.load(out_dir / "sections.npy")

    with open(out_dir / "whole_meta.json", "r", encoding="utf-8") as f:
        loaded_whole_meta = json.load(f)
    with open(out_dir / "sections_meta.json", "r", encoding="utf-8") as f:
        loaded_sections_meta = json.load(f)

    np.testing.assert_array_equal(loaded_whole, whole_iq)
    np.testing.assert_array_equal(loaded_sections, sections)
    assert loaded_whole_meta == whole_meta
    assert loaded_sections_meta == sections_meta


def test_generate_dataset_message_only(tmp_path: Path):
    out_root = tmp_path / "dataset"

    produced = genmod.generate_dataset(
        output_root=out_root,
        num_outputs=2,
        min_total_samples=12000,
        max_total_samples=12000,
        section_len=1024,
        num_sections=3,
        seed=100,
        random_payload_probability=0.0,
        max_attempts_per_sample=20,
    )

    assert len(produced) == 2

    for sdir in produced:
        assert (sdir / "whole_iq.npy").exists()
        assert (sdir / "whole_meta.json").exists()
        assert (sdir / "sections.npy").exists()
        assert (sdir / "sections_meta.json").exists()

        whole_iq = np.load(sdir / "whole_iq.npy")
        sections = np.load(sdir / "sections.npy")

        with open(sdir / "whole_meta.json", "r", encoding="utf-8") as f:
            whole_meta = json.load(f)
        with open(sdir / "sections_meta.json", "r", encoding="utf-8") as f:
            sections_meta = json.load(f)

        assert whole_meta["decode_verified"] is True
        assert whole_meta["payload_desc"]["mode"] == "message"
        assert len(whole_iq) >= 12000
        assert whole_meta["actual_num_samples"] == len(whole_iq)
        assert whole_meta["requested_target_num_samples"] == 12000
        assert sections.shape == (3, 1024)

        for idx, start in enumerate(sections_meta["starts"]):
            np.testing.assert_array_equal(sections[idx], whole_iq[start:start + 1024])

        rx_result = genmod._decode_candidate_with_v4(whole_iq, whole_meta)
        assert rx_result is not None
        assert rx_result["message"] == whole_meta["payload_desc"]["message"]


def test_generate_dataset_random_only(tmp_path: Path):
    out_root = tmp_path / "dataset"

    produced = genmod.generate_dataset(
        output_root=out_root,
        num_outputs=2,
        min_total_samples=12000,
        max_total_samples=12000,
        section_len=1024,
        num_sections=3,
        seed=200,
        random_payload_probability=1.0,
        max_attempts_per_sample=20,
    )

    assert len(produced) == 2

    for sdir in produced:
        whole_iq = np.load(sdir / "whole_iq.npy")

        with open(sdir / "whole_meta.json", "r", encoding="utf-8") as f:
            whole_meta = json.load(f)

        assert whole_meta["decode_verified"] is True
        assert whole_meta["payload_desc"]["mode"] == "random_bits"
        assert len(whole_iq) >= 12000
        assert whole_meta["actual_num_samples"] == len(whole_iq)
        assert whole_meta["requested_target_num_samples"] == 12000

        rx_result = genmod._decode_candidate_with_v4(whole_iq, whole_meta)
        assert rx_result is not None

        n_bits = whole_meta["payload_desc"]["random_bits"]
        seed = whole_meta["payload_desc"]["random_seed"]
        expected = link.build_payload_bytes_from_random_bits(n_bits, seed)[:-4]
        assert rx_result["payload_bytes"] == expected


@pytest.mark.parametrize("num_outputs", [1, 3])
def test_generate_dataset_count(tmp_path: Path, num_outputs: int):
    out_root = tmp_path / "dataset"

    produced = genmod.generate_dataset(
        output_root=out_root,
        num_outputs=num_outputs,
        min_total_samples=10000,
        max_total_samples=12000,
        section_len=1024,
        num_sections=3,
        seed=77,
        random_payload_probability=0.5,
        max_attempts_per_sample=20,
    )

    assert len(produced) == num_outputs
    sample_dirs = sorted([p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    assert len(sample_dirs) == num_outputs

    for sdir in sample_dirs:
        with open(sdir / "whole_meta.json", "r", encoding="utf-8") as f:
            whole_meta = json.load(f)
        whole_iq = np.load(sdir / "whole_iq.npy")

        assert whole_meta["decode_verified"] is True
        assert whole_meta["actual_num_samples"] == len(whole_iq)
        assert len(whole_iq) >= whole_meta["requested_target_num_samples"]


def test_generate_dataset_main(tmp_path: Path):
    out_root = tmp_path / "dataset_cli"

    result = genmod.main([
        "--output-root", str(out_root),
        "--num-outputs", "1",
        "--min-total-samples", "10000",
        "--max-total-samples", "10000",
        "--section-len", "1024",
        "--num-sections", "3",
        "--seed", "123",
        "--random-payload-probability", "0.5",
        "--max-attempts-per-sample", "20",
    ])

    assert result["count"] == 1
    assert out_root.exists()

    sample_dirs = sorted([p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    assert len(sample_dirs) == 1

    with open(sample_dirs[0] / "whole_meta.json", "r", encoding="utf-8") as f:
        whole_meta = json.load(f)
    whole_iq = np.load(sample_dirs[0] / "whole_iq.npy")

    assert whole_meta["decode_verified"] is True
    assert whole_meta["actual_num_samples"] == len(whole_iq)
    assert len(whole_iq) >= 10000