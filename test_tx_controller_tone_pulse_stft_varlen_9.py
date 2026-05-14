from __future__ import annotations

import math

import torch

from tx_controller_tone_pulse_stft_varlen_9 import decode_tone_pulse_config


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
