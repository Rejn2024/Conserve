import json
from pathlib import Path

import numpy as np
import pytest

import advanced_link_skdsp_v3_tx_flexible as txmod
import generate_tx_iq_dataset as genmod
import load_tx_iq_data as loadmod


def test_build_tx_iq_object_message_arbitrary_length():
    target_len = 20000
    result = txmod.build_tx_iq_object(
        message="hello world " * 100,
        random_bits=None,
        target_num_samples=target_len,
        fec="conv",
        interleave=True,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
        snr_db=30.0,
        noise_color="white",
        fading_mode="none",
    )

    assert isinstance(result.iq, np.ndarray)
    assert result.iq.dtype == np.complex64
    assert len(result.iq) == target_len
    assert result.metadata["actual_num_samples"] == target_len
    assert result.metadata["payload_source"] == "message"


def test_build_tx_iq_object_random_bits_arbitrary_length():
    target_len = 12345
    result = txmod.build_tx_iq_object(
        message=None,
        random_bits=777,
        random_seed=17,
        target_num_samples=target_len,
        fec="rep3",
        interleave=True,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=433_920_000.0,
        carrier_hz=50_000.0,
        snr_db=28.0,
        noise_color="white",
        fading_mode="rician_block",
        rician_k_db=10.0,
    )

    assert len(result.iq) == target_len
    assert result.metadata["payload_source"] == "random_bits:777"
    assert result.metadata["actual_num_samples"] == target_len


def test_tx_command_rejects_both_message_and_random_bits(tmp_path: Path):
    with pytest.raises(SystemExit):
        txmod.main([
            "tx",
            "--message", "abc",
            "--random-bits", "100",
            "--output", str(tmp_path / "x.npy"),
        ])


def test_tx_command_rejects_missing_payload(tmp_path: Path):
    with pytest.raises(SystemExit):
        txmod.main([
            "tx",
            "--output", str(tmp_path / "x.npy"),
        ])


def test_random_bits_padding_to_bytes():
    n_bits = 13
    seed = 9
    payload_crc = txmod.build_payload_bytes_from_random_bits(n_bits, seed)
    payload = payload_crc[:-4]
    expected_bits = txmod.prbs_bits(n_bits, seed=seed)
    expected_bits = expected_bits + [0] * ((-n_bits) % 8)
    expected_payload = txmod.bits_to_bytes_msb(expected_bits)
    assert payload == expected_payload


def test_cut_random_sections():
    iq = (np.arange(10000) + 1j * np.arange(10000)).astype(np.complex64)
    cuts = genmod.cut_random_sections(iq=iq, num_sections=3, section_len=1024, seed=1)

    assert cuts["sections"].shape == (3, 1024)
    assert len(cuts["starts"]) == 3

    for sec, start in zip(cuts["sections"], cuts["starts"]):
        np.testing.assert_array_equal(sec, iq[start:start + 1024])


def test_generate_dataset_and_loaders(tmp_path: Path):
    out_root = tmp_path / "dataset"

    produced = genmod.generate_dataset(
        output_root=out_root,
        num_outputs=4,
        min_total_samples=8192,
        max_total_samples=12000,
        section_len=1024,
        num_sections=3,
        seed=3,
        random_payload_probability=0.5,
    )

    assert len(produced) == 4

    sample_dirs = loadmod.list_sample_dirs(out_root)
    assert len(sample_dirs) == 4

    for sdir in sample_dirs:
        assert (sdir / "whole_iq.npy").exists()
        assert (sdir / "whole_meta.json").exists()
        assert (sdir / "sections.npy").exists()
        assert (sdir / "sections_meta.json").exists()

        whole = loadmod.load_whole_iq(sdir)
        sections = loadmod.load_sections(sdir)
        bundle = loadmod.load_sample_bundle(sdir)

        iq = whole["iq"]
        whole_meta = whole["meta"]
        secs = sections["sections"]
        sections_meta = sections["meta"]

        assert iq.dtype == np.complex64
        assert secs.dtype == np.complex64
        assert secs.shape == (3, 1024)

        assert whole_meta["actual_num_samples"] == len(iq)
        assert sections_meta["whole_num_samples"] == len(iq)
        assert len(sections_meta["starts"]) == 3

        for idx, start in enumerate(sections_meta["starts"]):
            np.testing.assert_array_equal(secs[idx], iq[start:start + 1024])

        assert bundle["whole_iq"].shape == iq.shape
        assert bundle["sections"].shape == secs.shape


def test_save_tx_iq_object(tmp_path: Path):
    result = txmod.build_tx_iq_object(
        message="abc123" * 100,
        target_num_samples=9000,
        fec="none",
        interleave=False,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
        snr_db=100.0,
    )

    iq_path = tmp_path / "whole_iq.npy"
    meta_path = tmp_path / "whole_meta.json"
    out_iq, out_meta = txmod.save_tx_iq_object(result, iq_path=iq_path, metadata_path=meta_path)

    assert Path(out_iq).exists()
    assert Path(out_meta).exists()

    iq = np.load(out_iq)
    with open(out_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert iq.shape[0] == 9000
    assert meta["actual_num_samples"] == 9000


@pytest.mark.parametrize("fec", ["none", "rep3", "conv"])
def test_all_fec_modes_build(fec: str):
    if fec == "conv" and txmod.fec_conv is None:
        pytest.skip("scikit-dsp-comm not installed")

    result = txmod.build_tx_iq_object(
        message="abcd" * 200,
        target_num_samples=10000,
        fec=fec,
        interleave=(fec != "none"),
        sample_rate_hz=1_000_000.0,
        rf_center_hz=2_400_000_000.0,
        carrier_hz=25_000.0,
        snr_db=32.0,
        noise_color="pink",
        fading_mode="rician_block",
        rician_k_db=8.0,
        burst_probability=0.0,
    )
    assert len(result.iq) == 10000


@pytest.mark.parametrize("noise_color", ["white", "pink", "brown", "blue", "violet"])
def test_all_noise_colours_build(noise_color: str):
    result = txmod.build_tx_iq_object(
        message="xyz" * 300,
        target_num_samples=8192,
        fec="none",
        interleave=False,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
        snr_db=30.0,
        noise_color=noise_color,
    )
    assert len(result.iq) == 8192
    assert result.metadata["noise_color"] == noise_color


@pytest.mark.parametrize("fading_mode", ["none", "rician_block", "rayleigh_block", "multipath_static"])
def test_all_fading_modes_build(fading_mode: str):
    kwargs = {}
    if fading_mode == "multipath_static":
        kwargs["multipath_taps"] = [1 + 0j, 0.1 + 0.05j, 0.03 - 0.02j]

    result = txmod.build_tx_iq_object(
        message="hello" * 200,
        target_num_samples=8192,
        fec="none",
        interleave=False,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
        snr_db=30.0,
        fading_mode=fading_mode,
        **kwargs,
    )
    assert len(result.iq) == 8192
    assert result.metadata["fading_mode"] == fading_mode