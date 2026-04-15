from __future__ import annotations

import math

import torch

from tx_controller_tone_pulse_stft_varlen_2 import ActorCritic, decode_tone_pulse_config


def test_decode_tone_pulse_config_handles_nan_scalars():
    model_out = {
        "noise_color_logits": torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "fading_mode_logits": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "burst_color_logits": torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        "sample_rate_scale": torch.tensor([[float("nan")]], dtype=torch.float32),
        "rf_center_delta_hz": torch.tensor([[float("nan")]], dtype=torch.float32),
        "carrier_hz_norm": torch.tensor([[float("nan")]], dtype=torch.float32),
        "num_tones_cont": torch.tensor([[float("nan")]], dtype=torch.float32),
        "base_tone_norm": torch.tensor([[float("nan")]], dtype=torch.float32),
        "tone_spacing_norm": torch.tensor([[float("nan")]], dtype=torch.float32),
        "tone_amp_raw": torch.zeros((1, 8), dtype=torch.float32),
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
        max_tones=8,
        seed=123,
    )

    assert cfg.num_tones == 1
    assert cfg.pulse_on_samples >= 1
    assert cfg.pulse_count >= 1
    assert cfg.start_offset_samples >= 0
    assert math.isfinite(cfg.sample_rate_hz)
    assert math.isfinite(cfg.rf_center_hz)


def test_actor_critic_reuses_varlen_backbone_and_returns_rl_shapes():
    model = ActorCritic(action_dim=7, in_ch=14, base_ch=16, n_scalar_features=6, max_tones=8)
    model.eval()

    batch_size = 3
    stft_feature_list = [
        torch.randn(batch_size, 14, 64, 17),
        torch.randn(batch_size, 14, 64, 21),
        torch.randn(batch_size, 14, 64, 13),
    ]
    scalar_side = torch.randn(batch_size, 6)

    logits, values = model(stft_feature_list=stft_feature_list, scalar_side=scalar_side)
    assert logits.shape == (batch_size, 7)
    assert values.shape == (batch_size,)

    action, log_prob, act_values = model.act(stft_feature_list=stft_feature_list, scalar_side=scalar_side)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert act_values.shape == (batch_size,)

    eval_log_prob, entropy, eval_values = model.evaluate_actions(
        stft_feature_list=stft_feature_list,
        scalar_side=scalar_side,
        actions=action,
    )
    assert eval_log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert eval_values.shape == (batch_size,)
