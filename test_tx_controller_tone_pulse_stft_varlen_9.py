from __future__ import annotations

import math

import torch

from tx_controller_tone_pulse_stft_varlen_9 import (
    FIRST_PASS_SCALAR_FEATURE_NAMES,
    N_FIRST_PASS_SCALAR_FEATURES,
    TonePulseTXControlNetVarLen,
    build_first_pass_scalar_side_from_iq_sections,
    compute_first_pass_scalar_features_for_iq_batch,
    decode_tone_pulse_config,
)


def test_decode_tone_pulse_config_sanitizes_nan_model_outputs():
    max_tones = 8
    max_pulses = 33
    model_out = {
        "noise_color_logits": torch.full((1, 5), float("nan"), dtype=torch.float32),
        "fading_mode_logits": torch.full((1, 4), float("nan"), dtype=torch.float32),
        "burst_color_logits": torch.full((1, 5), float("nan"), dtype=torch.float32),
        "sample_rate_scale": torch.tensor([[float("nan")]], dtype=torch.float32),
        "rf_center_delta_hz": torch.tensor([[float("nan")]], dtype=torch.float32),
        "carrier_hz_norm": torch.tensor([[float("nan")]], dtype=torch.float32),
        "num_tones_cont": torch.tensor([[float("nan")]], dtype=torch.float32),
        "tone_freq_mean_norms": torch.full((1, max_tones), float("nan"), dtype=torch.float32),
        "tone_freq_std_norms": torch.full((1, max_tones), float("nan"), dtype=torch.float32),
        "tone_amp_raw": torch.full((1, max_tones), float("nan"), dtype=torch.float32),
        "tone_power_logits": torch.full((1, max_tones), float("nan"), dtype=torch.float32),
        "tone_phase_rel_rad": torch.full((1, max_tones), float("nan"), dtype=torch.float32),
        "tone_phase_offset_rad": torch.tensor([[float("nan")]], dtype=torch.float32),
        "pulse_phase_rel_rad": torch.full((1, max_pulses), float("nan"), dtype=torch.float32),
        "pulse_phase_offset_rad": torch.tensor([[float("nan")]], dtype=torch.float32),
        "pulse_on_samples": torch.tensor([[float("nan")]], dtype=torch.float32),
        "pulse_off_samples": torch.tensor([[float("nan")]], dtype=torch.float32),
        "pulse_count": torch.tensor([[float("nan")]], dtype=torch.float32),
        "start_offset": torch.tensor([[float("nan")]], dtype=torch.float32),
        "snr_db": torch.tensor([[float("nan")]], dtype=torch.float32),
        "freq_offset": torch.tensor([[float("nan")]], dtype=torch.float32),
        "timing_offset": torch.tensor([[float("nan")]], dtype=torch.float32),
        "fading_block_len_norm": torch.tensor([[float("nan")]], dtype=torch.float32),
        "rician_k_db": torch.tensor([[float("nan")]], dtype=torch.float32),
        "burst_probability": torch.tensor([[float("nan")]], dtype=torch.float32),
        "burst_power_ratio_db": torch.tensor([[float("nan")]], dtype=torch.float32),
    }

    cfg = decode_tone_pulse_config(
        model_out=model_out,
        intake_sample_rate_hz=2_000_000.0,
        rf_center_est_hz=123_000_000.0,
        desired_output_iq_len=2048,
        user_peak_power_fraction=40.0,
        rx_input_power=0.5,
        max_tones=max_tones,
        max_pulses=max_pulses,
        seed=123,
    )

    assert cfg.num_tones == 1
    assert cfg.pulse_on_samples >= 1
    assert cfg.pulse_count >= 1
    assert cfg.start_offset_samples >= 0

    finite_scalars = [
        cfg.sample_rate_hz,
        cfg.rf_center_hz,
        cfg.carrier_hz,
        cfg.snr_db,
        cfg.freq_offset,
        cfg.timing_offset,
        cfg.rician_k_db,
        cfg.burst_probability,
        cfg.burst_power_ratio_db,
    ]
    assert all(math.isfinite(x) for x in finite_scalars)
    assert all(math.isfinite(x) for x in cfg.tone_frequencies_hz)
    assert all(math.isfinite(x) for x in cfg.tone_frequency_std_hz)
    assert all(math.isfinite(x) for x in cfg.tone_amplitudes)
    assert all(math.isfinite(x) for x in cfg.pulse_phase_rotations_rad)


def test_first_pass_scalar_features_are_finite_and_follow_schema():
    iq = torch.zeros(2, 128, dtype=torch.complex64)
    iq[0, 32:96] = 1.0 + 0.0j
    iq[1, 8:24] = 0.5 + 0.25j

    result = compute_first_pass_scalar_features_for_iq_batch(iq, sample_rate_hz=1_000_000.0)
    scalar = result["scalar_side"]

    assert result["feature_names"] == FIRST_PASS_SCALAR_FEATURE_NAMES
    assert scalar.shape == (2, N_FIRST_PASS_SCALAR_FEATURES)
    assert torch.isfinite(scalar).all()

    names = {name: idx for idx, name in enumerate(FIRST_PASS_SCALAR_FEATURE_NAMES)}
    assert scalar[0, names["packet_start_frac"]] > 0.20
    assert scalar[0, names["packet_end_frac"]] < 0.80
    assert scalar[0, names["packet_duration_frac"]] > 0.40
    assert scalar[0, names["packet_geometry_valid"]] == 1.0
    assert scalar[0, names["spectral_geometry_valid"]] == 1.0


def test_first_pass_scalar_side_from_sections_feeds_default_network():
    batch = 2
    iq1 = torch.complex(torch.randn(batch, 96), torch.randn(batch, 96))
    iq2 = torch.complex(torch.randn(batch, 96), torch.randn(batch, 96))
    iq3 = torch.complex(torch.randn(batch, 96), torch.randn(batch, 96))
    scalar = build_first_pass_scalar_side_from_iq_sections([iq1, iq2, iq3], sample_rate_hz=2_000_000.0)

    model = TonePulseTXControlNetVarLen(in_ch=14, base_ch=4, max_tones=2, max_pulses=3)
    stft = [torch.randn(batch, 14, 16, 8) for _ in range(3)]
    out = model(stft, scalar)

    assert model.scalar_proj[0].in_features == N_FIRST_PASS_SCALAR_FEATURES
    assert out["tone_freq_mean_norms"].shape == (batch, 2)
    assert out["pulse_phase_rel_rad"].shape == (batch, 3)
