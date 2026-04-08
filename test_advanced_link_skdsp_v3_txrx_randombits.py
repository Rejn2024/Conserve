# test_advanced_link_skdsp_v3_txrx_randombits.py
#
# Pytest regression suite for:
#   advanced_link_skdsp_v3_txrx_randombits.py
#
# Covers:
# - TX with message payloads
# - TX with random bit-string payloads
# - metadata writing and consistency
# - RF centre frequency / carrier placement
# - RX with metadata
# - RX without metadata using CLI fallback
# - FEC modes: none / rep3 / conv
# - interleaving on/off
# - noise colours
# - SNR sweeps
# - sample-rate changes
# - carrier / timing / residual freq offsets
# - fading modes
# - impulsive noise
#
# Tests that are likely to exceed the current receiver capability are marked xfail.
#
# Usage:
#   pytest -q test_advanced_link_skdsp_v3_txrx_randombits.py

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import advanced_link_skdsp_v3_txrx_randombits as link


TEST_MESSAGE = (
    "This is a test message for advanced_link_skdsp_v3_txrx_randombits. "
    "It exercises TX/RX message mode, protected headers, RF metadata, "
    "equalization, coarse acquisition, and decode."
)


def _fft_peak_hz(iq: np.ndarray, fs_hz: float) -> float:
    spec = np.fft.fftshift(np.fft.fft(iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1.0 / fs_hz))
    return float(freqs[np.argmax(np.abs(spec))])


def _bits_to_bytes_for_expected_payload(n_bits: int, seed: int) -> bytes:
    bits = link.prbs_bits(n_bits, seed=seed)
    pad = (-n_bits) % 8
    if pad:
        bits = bits + [0] * pad
    return link.bits_to_bytes_msb(bits)


def run_case(
    *,
    message: str | None = TEST_MESSAGE,
    random_bits: int | None = None,
    random_seed: int = 1,
    fec: str = "conv",
    interleave: bool | None = None,
    noise_color: str = "white",
    snr_db: float = 30.0,
    fading_mode: str = "none",
    fading_block_len: int = 256,
    rician_k_db: float = 10.0,
    multipath_taps: str | None = None,
    freq_offset: float = 0.0,
    timing_offset: float = 1.0,
    burst_probability: float = 0.0,
    burst_power_ratio_db: float = 12.0,
    burst_color: str = "white",
    sample_rate_hz: float = 1_000_000.0,
    rf_center_hz: float = 2_400_000_000.0,
    carrier_hz: float = 0.0,
    rx_sample_rate_hz: float | None = None,
    rx_rf_center_hz: float | None = None,
    coarse_freq_search_hz: float = 20_000.0,
    coarse_freq_bins: int = 81,
    sample_phase_search: int = 2,
    eq_taps: int = 7,
):
    """
    End-to-end TX -> RX helper.

    Returns:
        tx_result, rx_result, iq_path, meta_path
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        iq_path = td_path / "test_tx.iq"
        meta_path = td_path / "test_tx.iq.json"

        if interleave is None:
            interleave = (fec != "none")

        tx_args = [
            "tx",
            "--output", str(iq_path),
            "--metadata-path", str(meta_path),
            "--fec", fec,
            "--sps", "8",
            "--beta", "0.35",
            "--span", "6",
            "--sample-rate-hz", str(sample_rate_hz),
            "--rf-center-hz", str(rf_center_hz),
            "--carrier-hz", str(carrier_hz),
            "--snr-db", str(snr_db),
            "--noise-color", noise_color,
            "--fading-mode", fading_mode,
            "--fading-block-len", str(fading_block_len),
            "--rician-k-db", str(rician_k_db),
            "--freq-offset", str(freq_offset),
            "--timing-offset", str(timing_offset),
            "--burst-probability", str(burst_probability),
            "--burst-power-ratio-db", str(burst_power_ratio_db),
            "--burst-color", burst_color,
            "--burst-len-min", "16",
            "--burst-len-max", "64",
            "--seed", "1",
        ]

        if random_bits is not None:
            tx_args.extend(["--random-bits", str(random_bits), "--random-seed", str(random_seed)])
        else:
            tx_args.extend(["--message", message])

        if multipath_taps is not None:
            tx_args.extend(["--multipath-taps", multipath_taps])

        if interleave:
            tx_args.append("--interleave")

        tx_result = link.main(tx_args)

        assert iq_path.exists()
        assert meta_path.exists()

        if rx_sample_rate_hz is None:
            rx_sample_rate_hz = sample_rate_hz
        if rx_rf_center_hz is None:
            rx_rf_center_hz = rf_center_hz + carrier_hz

        rx_args = [
            "rx",
            "--input", str(iq_path),
            "--metadata-path", str(meta_path),
            "--fec", fec,
            "--sps", "8",
            "--beta", "0.35",
            "--span", "6",
            "--sample-rate-hz", str(rx_sample_rate_hz),
            "--rf-center-hz", str(rx_rf_center_hz),
            "--coarse-freq-search-hz", str(coarse_freq_search_hz),
            "--coarse-freq-bins", str(coarse_freq_bins),
            "--sample-phase-search", str(sample_phase_search),
            "--eq-taps", str(eq_taps),
        ]

        if interleave:
            rx_args.append("--interleave")

        rx_result = link.main(rx_args)
        return tx_result, rx_result, iq_path, meta_path


# -----------------------------------------------------------------------------
# Metadata and RF placement
# -----------------------------------------------------------------------------

def test_metadata_written_and_consistent_message_mode():
    with tempfile.TemporaryDirectory() as td:
        iq_path = Path(td) / "meta_message.iq"
        meta_path = Path(td) / "meta_message.iq.json"

        tx_result = link.main([
            "tx",
            "--message", TEST_MESSAGE,
            "--output", str(iq_path),
            "--metadata-path", str(meta_path),
            "--sample-rate-hz", "1000000",
            "--rf-center-hz", "2400000000",
            "--carrier-hz", "125000",
            "--fec", "none",
            "--snr-db", "100",
            "--noise-color", "white",
        ])

        assert iq_path.exists()
        assert meta_path.exists()

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["sample_rate_hz"] == 1_000_000.0
        assert meta["rf_center_hz"] == 2_400_000_000.0
        assert meta["carrier_hz"] == 125_000.0
        assert meta["absolute_rf_hz"] == 2_400_125_000.0
        assert tx_result["absolute_rf_hz"] == 2_400_125_000.0
        assert tx_result["payload_source"] == "message"


def test_metadata_written_and_consistent_random_bits_mode():
    with tempfile.TemporaryDirectory() as td:
        iq_path = Path(td) / "meta_rand.iq"
        meta_path = Path(td) / "meta_rand.iq.json"

        tx_result = link.main([
            "tx",
            "--random-bits", "511",
            "--random-seed", "7",
            "--output", str(iq_path),
            "--metadata-path", str(meta_path),
            "--sample-rate-hz", "1000000",
            "--rf-center-hz", "915000000",
            "--carrier-hz", "50000",
            "--fec", "rep3",
            "--snr-db", "100",
            "--noise-color", "white",
        ])

        assert iq_path.exists()
        assert meta_path.exists()

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["sample_rate_hz"] == 1_000_000.0
        assert meta["rf_center_hz"] == 915_000_000.0
        assert meta["carrier_hz"] == 50_000.0
        assert meta["absolute_rf_hz"] == 915_050_000.0
        assert tx_result["payload_source"] == "random_bits:511"


@pytest.mark.parametrize(
    "sample_rate_hz,carrier_hz",
    [
        (1_000_000.0, 50_000.0),
        (1_000_000.0, 200_000.0),
        (2_000_000.0, 300_000.0),
    ],
)
def test_digital_carrier_appears_in_spectrum(sample_rate_hz: float, carrier_hz: float):
    with tempfile.TemporaryDirectory() as td:
        iq_path = Path(td) / "carrier_test.iq"
        meta_path = Path(td) / "carrier_test.iq.json"

        link.main([
            "tx",
            "--message", TEST_MESSAGE,
            "--output", str(iq_path),
            "--metadata-path", str(meta_path),
            "--sample-rate-hz", str(sample_rate_hz),
            "--rf-center-hz", "2400000000",
            "--carrier-hz", str(carrier_hz),
            "--fec", "none",
            "--snr-db", "100",
            "--noise-color", "white",
        ])

        iq = np.fromfile(iq_path, dtype=np.complex64)
        peak_hz = _fft_peak_hz(iq, sample_rate_hz)

        # Wide tolerance: burst packet is not a single tone
        assert abs(abs(peak_hz) - abs(carrier_hz)) < 150_000.0


# -----------------------------------------------------------------------------
# Message-mode decode tests
# -----------------------------------------------------------------------------

def test_noiseless_uncoded_message_roundtrip():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="none",
        snr_db=100.0,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


def test_noiseless_rep3_message_roundtrip():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="rep3",
        interleave=True,
        snr_db=100.0,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


def test_noiseless_conv_message_roundtrip():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=100.0,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


def test_exact_rf_tuning_message_roundtrip():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=100.0,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=2_400_000_000.0,
        carrier_hz=0.0,
        rx_rf_center_hz=2_400_000_000.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


def test_rx_tuned_to_tx_rf_center_not_absolute_rf_message():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=100.0,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=2_400_000_000.0,
        carrier_hz=100_000.0,
        rx_rf_center_hz=2_400_000_000.0,
    )
    assert rx_result["message"] == TEST_MESSAGE
    assert abs(rx_result["baseband_offset_hz"] - 100_000.0) < 1e-6


# -----------------------------------------------------------------------------
# Random-bit mode decode tests
# -----------------------------------------------------------------------------

def test_noiseless_uncoded_random_bits_roundtrip():
    n_bits = 511
    seed = 7
    _, rx_result, _, _ = run_case(
        random_bits=n_bits,
        random_seed=seed,
        fec="none",
        snr_db=100.0,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=433_920_000.0,
        carrier_hz=0.0,
    )
    expected = _bits_to_bytes_for_expected_payload(n_bits, seed)
    assert rx_result["message"] is None
    assert rx_result["payload_bytes"] == expected


def test_noiseless_conv_random_bits_roundtrip():
    n_bits = 1024
    seed = 11
    _, rx_result, _, _ = run_case(
        random_bits=n_bits,
        random_seed=seed,
        fec="conv",
        interleave=True,
        snr_db=100.0,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=433_920_000.0,
        carrier_hz=0.0,
    )
    expected = _bits_to_bytes_for_expected_payload(n_bits, seed)
    assert rx_result["message"] is None
    assert rx_result["payload_bytes"] == expected


# -----------------------------------------------------------------------------
# TX functionality validation
# -----------------------------------------------------------------------------

def test_tx_rejects_message_and_random_bits_together():
    with pytest.raises(SystemExit):
        link.main([
            "tx",
            "--message", TEST_MESSAGE,
            "--random-bits", "128",
            "--output", "dummy.iq",
        ])


def test_tx_rejects_no_payload_source():
    with pytest.raises(SystemExit):
        link.main([
            "tx",
            "--output", "dummy.iq",
        ])


def test_tx_random_bits_non_byte_multiple_supported():
    n_bits = 13
    seed = 5
    payload_crc = link.build_payload_bytes_from_random_bits(n_bits, seed)
    payload = payload_crc[:-4]
    expected = _bits_to_bytes_for_expected_payload(n_bits, seed)
    assert payload == expected


# -----------------------------------------------------------------------------
# Sample-rate and fallback-metadata tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sample_rate_hz,carrier_hz",
    [
        (500_000.0, 50_000.0),
        (1_000_000.0, 100_000.0),
        (2_000_000.0, 250_000.0),
    ],
)
def test_matching_tx_rx_sample_rates(sample_rate_hz: float, carrier_hz: float):
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=100.0,
        sample_rate_hz=sample_rate_hz,
        rf_center_hz=915_000_000.0,
        carrier_hz=carrier_hz,
        rx_rf_center_hz=915_000_000.0 + carrier_hz,
    )
    assert rx_result["message"] == TEST_MESSAGE


def test_rx_without_metadata_uses_cli_fallback():
    with tempfile.TemporaryDirectory() as td:
        iq_path = Path(td) / "no_meta.iq"
        meta_path = Path(td) / "no_meta.iq.json"

        link.main([
            "tx",
            "--message", TEST_MESSAGE,
            "--output", str(iq_path),
            "--metadata-path", str(meta_path),
            "--sample-rate-hz", "1000000",
            "--rf-center-hz", "915000000",
            "--carrier-hz", "0",
            "--fec", "conv",
            "--interleave",
            "--snr-db", "100",
            "--noise-color", "white",
        ])

        meta_path.unlink()

        rx_result = link.main([
            "rx",
            "--input", str(iq_path),
            "--tx-sample-rate-hz", "1000000",
            "--tx-rf-center-hz", "915000000",
            "--tx-carrier-hz", "0",
            "--sample-rate-hz", "1000000",
            "--rf-center-hz", "915000000",
            "--fec", "conv",
            "--interleave",
            "--sps", "8",
            "--beta", "0.35",
            "--span", "6",
            "--coarse-freq-search-hz", "20000",
            "--coarse-freq-bins", "81",
            "--sample-phase-search", "2",
            "--eq-taps", "7",
        ])

        assert rx_result["message"] == TEST_MESSAGE


# -----------------------------------------------------------------------------
# Mild impairment tests expected to pass
# -----------------------------------------------------------------------------

def test_mild_uncoded_white_noise():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="none",
        snr_db=32.0,
        noise_color="white",
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


def test_mild_conv_white_noise():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=28.0,
        noise_color="white",
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


def test_mild_conv_rician():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=34.0,
        noise_color="white",
        fading_mode="rician_block",
        rician_k_db=12.0,
        fading_block_len=4096,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=2_400_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


def test_mild_conv_static_multipath():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=34.0,
        noise_color="white",
        fading_mode="multipath_static",
        multipath_taps="1+0j,0.15+0.05j,0.05-0.03j",
        sample_rate_hz=1_000_000.0,
        rf_center_hz=2_400_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


def test_mild_conv_impulsive():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=34.0,
        noise_color="white",
        burst_probability=0.00005,
        burst_power_ratio_db=8.0,
        burst_color="white",
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


# -----------------------------------------------------------------------------
# More comprehensive functionality tests that should usually pass
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("fec", ["none", "rep3", "conv"])
def test_basic_fec_modes_message(fec: str):
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec=fec,
        interleave=(fec != "none"),
        snr_db=100.0,
        noise_color="white",
        sample_rate_hz=1_000_000.0,
        rf_center_hz=433_920_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


@pytest.mark.parametrize(
    "noise_color,snr_db",
    [
        ("white", 30.0),
        ("pink", 36.0),
    ],
)
def test_supported_noise_colours_mild(noise_color: str, snr_db: float):
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        noise_color=noise_color,
        snr_db=snr_db,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


# -----------------------------------------------------------------------------
# Known limitations / expected-fail tests
# -----------------------------------------------------------------------------

@pytest.mark.xfail(reason="Current receiver is not robust enough for broad colored-noise decode coverage.")
@pytest.mark.parametrize(
    "noise_color,snr_db",
    [
        ("brown", 38.0),
        ("blue", 32.0),
        ("violet", 38.0),
    ],
)
def test_harsh_noise_colours_expected_fail(noise_color: str, snr_db: float):
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        noise_color=noise_color,
        snr_db=snr_db,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=2_400_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


@pytest.mark.xfail(reason="Current receiver remains brittle for combined carrier placement plus residual offsets.")
def test_small_carrier_plus_freq_timing_offsets_expected_fail():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=40.0,
        noise_color="white",
        freq_offset=0.00005,
        timing_offset=1.00002,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=2_400_000_000.0,
        carrier_hz=25_000.0,
        rx_rf_center_hz=2_400_025_000.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


@pytest.mark.xfail(reason="Current receiver has limited tolerance to RX sample-rate mismatch after resampling.")
@pytest.mark.parametrize(
    "tx_fs,rx_fs,carrier_hz",
    [
        (1_000_000.0, 800_000.0, 100_000.0),
        (1_000_000.0, 1_500_000.0, 100_000.0),
        (2_000_000.0, 1_000_000.0, 250_000.0),
    ],
)
def test_rx_resampling_expected_fail(tx_fs: float, rx_fs: float, carrier_hz: float):
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=100.0,
        sample_rate_hz=tx_fs,
        rf_center_hz=2_400_000_000.0,
        carrier_hz=carrier_hz,
        rx_sample_rate_hz=rx_fs,
        rx_rf_center_hz=2_400_000_000.0 + carrier_hz,
    )
    assert rx_result["message"] == TEST_MESSAGE


@pytest.mark.xfail(reason="Current receiver has no robust channel tracker for severe Rayleigh fading.")
def test_rayleigh_expected_fail():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        noise_color="white",
        snr_db=36.0,
        fading_mode="rayleigh_block",
        fading_block_len=4096,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=2_400_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE


@pytest.mark.xfail(reason="Heavy impulsive coloured interference exceeds likely receiver capability.")
def test_heavy_impulsive_brown_expected_fail():
    _, rx_result, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        noise_color="brown",
        snr_db=34.0,
        burst_probability=0.0003,
        burst_power_ratio_db=18.0,
        burst_color="brown",
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
    )
    assert rx_result["message"] == TEST_MESSAGE