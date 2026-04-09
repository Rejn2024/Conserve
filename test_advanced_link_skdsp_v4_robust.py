from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import advanced_link_skdsp_v4_robust as link


TEST_MESSAGE = (
    "This is a more robust packet-radio regression test message for the v4 link. "
    "It exercises coherent BPSK, pilots, protected headers, scrambling, equalization, "
    "and acquisition logic."
)


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
    rf_center_hz: float = 915_000_000.0,
    carrier_hz: float = 0.0,
    rx_sample_rate_hz: float | None = None,
    rx_rf_center_hz: float | None = None,
):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        iq_path = td / "test.iq"
        meta_path = td / "test.iq.json"

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

        if message is not None:
            tx_args.extend(["--message", message])
        else:
            tx_args.extend(["--random-bits", str(random_bits), "--random-seed", str(random_seed)])

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
            "--coarse-freq-search-hz", "25000",
            "--coarse-freq-bins", "101",
            "--sample-phase-search", "3",
            "--eq-taps", "7",
        ]
        if interleave:
            rx_args.append("--interleave")

        rx_result = link.main(rx_args)
        return tx_result, rx_result, iq_path, meta_path


def test_build_tx_iq_object_message():
    result = link.build_tx_iq_object(
        message=TEST_MESSAGE,
        random_bits=None,
        target_num_samples=12000,
        fec="conv",
        interleave=True,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=50_000.0,
        snr_db=30.0,
        noise_color="white",
    )
    assert isinstance(result.iq, np.ndarray)
    assert result.iq.dtype == np.complex64
    assert len(result.iq) == 12000
    assert result.metadata["payload_source"] == "message"
    assert result.metadata["actual_num_samples"] == 12000


def test_build_tx_iq_object_random_bits():
    result = link.build_tx_iq_object(
        message=None,
        random_bits=777,
        random_seed=17,
        target_num_samples=12345,
        fec="rep3",
        interleave=True,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=433_920_000.0,
        carrier_hz=25_000.0,
        snr_db=28.0,
        noise_color="pink",
    )
    assert len(result.iq) == 12345
    assert result.metadata["payload_source"] == "random_bits:777"
    assert result.metadata["actual_num_samples"] == 12345
    assert result.metadata["random_seed"] == 17


def test_build_tone_pulse_iq_object_basic_offsets():
    fs = 1_000_000.0
    pulse_len = 4096
    offsets = [-125_000.0, 75_000.0]
    result = link.build_tone_pulse_iq_object(
        tone_offsets_hz=offsets,
        pulse_num_samples=pulse_len,
        gap_num_samples=0,
        target_num_samples=pulse_len * len(offsets),
        sample_rate_hz=fs,
        snr_db=None,
    )
    assert len(result.iq) == pulse_len * len(offsets)
    assert result.metadata["waveform_type"] == "tone_pulse"
    assert result.metadata["tone_offsets_hz"] == offsets

    first = result.iq[:pulse_len]
    second = result.iq[pulse_len:2 * pulse_len]

    def estimate_peak_hz(x: np.ndarray, fs_hz: float) -> float:
        spec = np.fft.fftshift(np.fft.fft(x))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0 / fs_hz))
        return float(freqs[int(np.argmax(np.abs(spec)))])

    peak1 = estimate_peak_hz(first, fs)
    peak2 = estimate_peak_hz(second, fs)
    assert abs(peak1 - offsets[0]) < 500.0
    assert abs(peak2 - offsets[1]) < 500.0


def test_build_tone_pulse_iq_object_peak_limit():
    result = link.build_tone_pulse_iq_object(
        tone_offsets_hz=[10_000.0],
        pulse_num_samples=4000,
        tone_amplitude=2.0,
        max_peak_power=0.5,
        snr_db=None,
    )
    assert result.metadata["peak_limit_applied"] is True
    assert result.metadata["peak_power"] <= 0.5 + 1e-6


def test_save_tx_iq_object(tmp_path: Path):
    result = link.build_tx_iq_object(
        message="abc123" * 100,
        target_num_samples=9000,
        fec="none",
        interleave=False,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
        snr_db=100.0,
    )

    iq_path = tmp_path / "whole.iq"
    meta_path = tmp_path / "whole.json"
    out_iq, out_meta = link.save_tx_iq_object(result, iq_path=iq_path, metadata_path=meta_path)

    assert Path(out_iq).exists()
    assert Path(out_meta).exists()

    iq = np.fromfile(out_iq, dtype=np.complex64)
    with open(out_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert len(iq) == 9000
    assert meta["actual_num_samples"] == 9000


def test_build_tone_pulse_iq_object_multi_tone():
    result = link.build_tone_pulse_iq_object(
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
        target_num_samples=16000,
        num_tones=3,
        tone_frequencies_hz=[-120_000.0, 15_000.0, 220_000.0],
        tone_amplitudes=[1.0, 0.5, 0.25],
        pulse_on_samples=2000,
        pulse_off_samples=500,
        pulse_count=5,
        snr_db=30.0,
        noise_color="pink",
        peak_power=0.8,
        seed=42,
    )

    assert len(result.iq) == 16000
    assert result.metadata["payload_source"] == "tone_pulse"
    assert result.metadata["num_tones"] == 3
    assert result.metadata["tone_frequencies_hz"] == [-120000.0, 15000.0, 220000.0]
    assert result.metadata["pulse_on_samples"] == 2000
    assert result.metadata["pulse_off_samples"] == 500
    assert result.metadata["pulse_count"] == 5
    assert result.metadata["noise_color"] == "pink"
    assert result.metadata["measured_peak_power"] <= 0.8 * 1.5


def test_build_tone_pulse_iq_object_peak_power_no_noise():
    result = link.build_tone_pulse_iq_object(
        sample_rate_hz=1_000_000.0,
        target_num_samples=4096,
        num_tones=2,
        tone_frequencies_hz=[20_000.0, -35_000.0],
        tone_amplitudes=[1.0, 0.8],
        pulse_on_samples=1024,
        pulse_off_samples=1024,
        pulse_count=2,
        snr_db=None,
        peak_power=0.25,
        seed=7,
    )

    peak_power = float(np.max(np.abs(result.iq) ** 2))
    assert np.isclose(peak_power, 0.25, atol=1e-3)


def test_noiseless_message_uncoded():
    _, rx, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="none",
        snr_db=100.0,
        carrier_hz=0.0,
    )
    assert rx["message"] == TEST_MESSAGE


def test_noiseless_message_rep3():
    _, rx, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="rep3",
        interleave=True,
        snr_db=100.0,
        carrier_hz=0.0,
    )
    assert rx["message"] == TEST_MESSAGE


def test_noiseless_message_conv():
    _, rx, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=100.0,
        carrier_hz=0.0,
    )
    assert rx["message"] == TEST_MESSAGE


def test_noiseless_random_bits_conv():
    n_bits = 1023
    seed = 7
    _, rx, _, _ = run_case(
        message=None,
        random_bits=n_bits,
        random_seed=seed,
        fec="conv",
        interleave=True,
        snr_db=100.0,
        carrier_hz=0.0,
    )
    expected = link.build_payload_bytes_from_random_bits(n_bits, seed)[:-4]
    assert rx["message"] is None
    assert rx["payload_bytes"] == expected


@pytest.mark.parametrize("snr_db", [24.0, 28.0, 32.0])
def test_white_noise_conv_sweep(snr_db: float):
    _, rx, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=snr_db,
        noise_color="white",
        carrier_hz=0.0,
    )
    assert rx["message"] == TEST_MESSAGE


@pytest.mark.parametrize("noise_color,snr_db", [("pink", 34.0), ("brown", 40.0)])
def test_colored_noise_conv(noise_color: str, snr_db: float):
    _, rx, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=snr_db,
        noise_color=noise_color,
        carrier_hz=0.0,
    )
    assert rx["message"] == TEST_MESSAGE


def test_rician_fading_conv():
    _, rx, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=30.0,
        fading_mode="rician_block",
        rician_k_db=12.0,
        fading_block_len=1024,
        carrier_hz=0.0,
    )
    assert rx["message"] == TEST_MESSAGE


def test_static_multipath_conv():
    _, rx, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=30.0,
        fading_mode="multipath_static",
        multipath_taps="1+0j,0.15+0.04j,0.05-0.03j",
        carrier_hz=0.0,
    )
    assert rx["message"] == TEST_MESSAGE


def test_impulsive_noise_conv():
    _, rx, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=32.0,
        burst_probability=5e-5,
        burst_power_ratio_db=10.0,
        burst_color="white",
        carrier_hz=0.0,
    )
    assert rx["message"] == TEST_MESSAGE


def test_mild_freq_and_timing_offset():
    _, rx, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=32.0,
        freq_offset=5e-5,
        timing_offset=1.00002,
        carrier_hz=0.0,
    )
    assert rx["message"] == TEST_MESSAGE


def test_zero_carrier_matched_tuning():
    _, rx, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=100.0,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=2_400_000_000.0,
        carrier_hz=0.0,
        rx_rf_center_hz=2_400_000_000.0,
    )
    assert rx["message"] == TEST_MESSAGE


@pytest.mark.xfail(reason="Current v4 receiver is still brittle for nonzero digital carrier placement, even with matched RF tuning.")
def test_nonzero_carrier_matched_tuning():
    _, rx, _, _ = run_case(
        message=TEST_MESSAGE,
        fec="conv",
        interleave=True,
        snr_db=100.0,
        sample_rate_hz=1_000_000.0,
        rf_center_hz=2_400_000_000.0,
        carrier_hz=100_000.0,
        rx_rf_center_hz=2_400_100_000.0,
    )
    assert rx["message"] == TEST_MESSAGE

def test_build_tone_pulse_iq_object_multi_tone():
    result = link.build_tone_pulse_iq_object(
        sample_rate_hz=1_000_000.0,
        rf_center_hz=915_000_000.0,
        carrier_hz=0.0,
        target_num_samples=16000,
        num_tones=3,
        tone_frequencies_hz=[-120_000.0, 15_000.0, 220_000.0],
        tone_amplitudes=[1.0, 0.5, 0.25],
        pulse_on_samples=2000,
        pulse_off_samples=500,
        pulse_count=5,
        snr_db=30.0,
        noise_color="pink",
        peak_power=0.8,
        seed=42,
    )

    assert len(result.iq) == 16000
    assert result.metadata["payload_source"] == "tone_pulse"
    assert result.metadata["num_tones"] == 3
    assert result.metadata["tone_frequencies_hz"] == [-120000.0, 15000.0, 220000.0]
    assert result.metadata["pulse_on_samples"] == 2000
    assert result.metadata["pulse_off_samples"] == 500
    assert result.metadata["pulse_count"] == 5
    assert result.metadata["noise_color"] == "pink"
    assert result.metadata["measured_peak_power"] <= 0.8 * 1.5


def test_build_tone_pulse_iq_object_peak_power_no_noise():
    result = link.build_tone_pulse_iq_object(
        sample_rate_hz=1_000_000.0,
        target_num_samples=4096,
        num_tones=2,
        tone_frequencies_hz=[20_000.0, -35_000.0],
        tone_amplitudes=[1.0, 0.8],
        pulse_on_samples=1024,
        pulse_off_samples=1024,
        pulse_count=2,
        snr_db=None,
        peak_power=0.25,
        seed=7,
    )

    peak_power = float(np.max(np.abs(result.iq) ** 2))
    assert np.isclose(peak_power, 0.25, atol=1e-3)
    
def test_metadata_fallback():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        iq_path = td / "test.iq"
        meta_path = td / "test.iq.json"

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

        rx = link.main([
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
            "--coarse-freq-search-hz", "25000",
            "--coarse-freq-bins", "101",
            "--sample-phase-search", "3",
            "--eq-taps", "7",
        ])

        assert rx["message"] == TEST_MESSAGE
