from __future__ import annotations

import math

import torch

from tx_controller_tone_pulse_stft_varlen_9 import (
    FIRST_PASS_SCALAR_FEATURE_NAMES,
    N_FIRST_PASS_SCALAR_FEATURES,
    FREQUENCY_STFT_FEATURE_CHANNELS,
    TARGET_INPUT_SAMPLES,
    TARGET_MAX_DETECTABLE_SAMPLES,
    TIMING_STFT_FEATURE_CHANNELS,
    TIMING_STFT_HOP_SAMPLES,
    TIMING_STFT_WINDOW_SAMPLES,
    TonePulseTXControlNetVarLen,
    preprocess_batched_iq_to_stft_feature,
    preprocess_iq_to_stft_feature,
    tone_pulse_action_dim,
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
    assert cfg.pulse_on_samples >= 5
    assert cfg.pulse_off_samples == 0
    assert cfg.pulse_count >= 1
    assert all(5 <= length <= 10_000 for length in cfg.pulse_lengths_samples)
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
    iq = torch.complex(torch.randn(batch, 288), torch.randn(batch, 288))
    scalar = build_first_pass_scalar_side_from_iq_sections([iq], sample_rate_hz=2_000_000.0)

    model = TonePulseTXControlNetVarLen(in_ch=23, base_ch=4, max_tones=2, max_pulses=3)
    stft = [torch.randn(batch, 23, 16, 8)]
    out = model(stft, scalar)

    assert model.scalar_proj[0].in_features == N_FIRST_PASS_SCALAR_FEATURES
    assert out["tone_freq_mean_norms"].shape == (batch, 2)
    assert out["pulse_phase_rel_rad"].shape == (batch, 3)
    assert out["pulse_length_log"].shape == (batch, 3)
    assert out["pulse_power_logit"].shape == (batch, 3)
    assert out["pulse_phase_ar_control"].shape == (batch, 1)
    assert out["pulse_length_ar_control"].shape == (batch, 1)
    assert out["pulse_power_ar_control"].shape == (batch, 1)
    assert torch.all(out["pulse_length_samples_cont"] >= 5.0)
    assert torch.all(out["pulse_length_samples_cont"] <= 10_000.0)


def test_tone_pulse_action_dim_uses_recurrent_pulse_state():
    assert tone_pulse_action_dim(max_tones=2, max_pulses=3) == 12 + 4 * 2
    assert tone_pulse_action_dim(max_tones=2, max_pulses=99) == 12 + 4 * 2
    model = TonePulseTXControlNetVarLen(in_ch=23, base_ch=4, max_tones=2, max_pulses=3)
    from tx_controller_tone_pulse_stft_varlen_9 import ActorCritic

    actor_critic = ActorCritic(in_ch=23, base_ch=4, max_tones=2, max_pulses=3)
    assert actor_critic.action_dim == tone_pulse_action_dim(model.max_tones, model.max_pulses)


def test_preprocessing_builds_independent_frequency_and_timing_stfts():
    iq = torch.complex(torch.randn(256), torch.randn(256))
    proc = preprocess_iq_to_stft_feature(iq, sample_rate_hz=1_000_000.0, enforce_iq_len=0)

    assert proc["frequency_feature"].shape[0] == FREQUENCY_STFT_FEATURE_CHANNELS
    assert proc["timing_feature"].shape[0] == TIMING_STFT_FEATURE_CHANNELS
    assert proc["feature"].shape[0] == FREQUENCY_STFT_FEATURE_CHANNELS + TIMING_STFT_FEATURE_CHANNELS
    assert proc["timing_feature"].shape[-1] > proc["frequency_feature"].shape[-1]
    assert TIMING_STFT_WINDOW_SAMPLES == 5
    assert TIMING_STFT_HOP_SAMPLES == 5


def test_network_uses_parallel_resunets_with_requested_temporal_scale():
    model = TonePulseTXControlNetVarLen(in_ch=23, base_ch=4, max_tones=2, max_pulses=3)

    assert model.frequency_encoder is not model.timing_encoder
    assert model.frequency_encoder.enc1.conv1.in_channels == FREQUENCY_STFT_FEATURE_CHANNELS
    assert model.timing_encoder.enc1.conv1.in_channels == TIMING_STFT_FEATURE_CHANNELS
    assert model.timing_encoder.temporal_receptive_field_samples >= TARGET_MAX_DETECTABLE_SAMPLES
    assert TIMING_STFT_WINDOW_SAMPLES <= 5
    assert TARGET_INPUT_SAMPLES is None
    assert TARGET_MAX_DETECTABLE_SAMPLES == 100_000

    native = [
        {
            "frequency_feature": torch.randn(1, FREQUENCY_STFT_FEATURE_CHANNELS, 16, 8),
            "timing_feature": torch.randn(1, TIMING_STFT_FEATURE_CHANNELS, 8, 31),
        }
        for _ in range(1)
    ]
    out = model(native, torch.randn(1, N_FIRST_PASS_SCALAR_FEATURES))
    assert out["pulse_length_log"].shape == (1, 3)


def test_pulse_length_and_power_use_independent_lstm_states():
    model = TonePulseTXControlNetVarLen(
        in_ch=23,
        base_ch=4,
        max_tones=2,
        max_pulses=3,
        pulse_length_ar_hidden=11,
        pulse_power_ar_hidden=13,
    )
    z = torch.randn(2, 96)
    length_teacher_a = torch.full((2, 3), math.log(5.0))
    length_teacher_b = torch.full((2, 3), math.log(10_000.0))
    power_teacher_a = torch.full((2, 3), -10.0)
    power_teacher_b = torch.full((2, 3), 10.0)

    assert model.pulse_length_ar_step.input_size == 3
    assert model.pulse_power_ar_step.input_size == 3
    assert model.pulse_length_ar_step.hidden_size == 11
    assert model.pulse_power_ar_step.hidden_size == 13
    assert model.pulse_length_ar_step is not model.pulse_power_ar_step

    length_a = model.pulse_length_autoregressive(z, teacher_length_logs=length_teacher_a)
    length_b = model.pulse_length_autoregressive(z, teacher_length_logs=length_teacher_b)
    power_a = model.pulse_power_autoregressive(z, teacher_power_logits=power_teacher_a)
    power_b = model.pulse_power_autoregressive(z, teacher_power_logits=power_teacher_b)

    assert "pulse_power_logit" not in length_a
    assert "pulse_length_log" not in power_a
    assert not torch.allclose(length_a["pulse_length_log_mean"], length_b["pulse_length_log_mean"])
    assert not torch.allclose(power_a["pulse_power_logit_mean"], power_b["pulse_power_logit_mean"])


def test_actor_critic_logp_entropy_include_autoregressive_pulse_terms(monkeypatch):
    from tx_controller_tone_pulse_stft_varlen_9 import ActorCritic

    batch = 2
    actor_critic = ActorCritic(in_ch=23, base_ch=4, max_tones=2, max_pulses=3).eval()
    stft = [torch.randn(batch, 23, 16, 8)]
    scalar = torch.randn(batch, N_FIRST_PASS_SCALAR_FEATURES)

    action_mean, _, log_std, _, _ = actor_critic._policy_tensors(
        stft_feature_list=stft,
        scalar_side=scalar,
    )
    dist = actor_critic._action_distribution(action_mean=action_mean, log_std=log_std)
    flat_mask = torch.ones_like(action_mean)
    flat_mask[..., actor_critic._recurrent_pulse_control_action_slice()] = 0.0
    flat_log_prob = (dist.log_prob(action_mean) * flat_mask).sum(dim=-1)
    flat_entropy = (dist.entropy() * flat_mask).sum(dim=-1)

    def fake_phase_log_prob(z, phases):
        assert phases.shape == (batch, actor_critic.max_pulses)
        return torch.full((z.shape[0],), 1.25, device=z.device), {}

    def fake_length_log_prob(z, length_logs):
        assert length_logs.shape == (batch, actor_critic.max_pulses)
        return torch.full((z.shape[0],), 1.25, device=z.device), {}

    def fake_power_log_prob(z, power_logits):
        assert power_logits.shape == (batch, actor_critic.max_pulses)
        return torch.full((z.shape[0],), 1.5, device=z.device), {}

    def fake_phase_entropy(z):
        return torch.full((z.shape[0],), 0.5, device=z.device)

    def fake_length_entropy(z):
        return torch.full((z.shape[0],), 0.75, device=z.device)

    def fake_power_entropy(z):
        return torch.full((z.shape[0],), 0.75, device=z.device)

    monkeypatch.setattr(actor_critic.backbone, "pulse_phase_autoregressive_log_prob", fake_phase_log_prob)
    monkeypatch.setattr(actor_critic.backbone, "pulse_length_autoregressive_log_prob", fake_length_log_prob)
    monkeypatch.setattr(actor_critic.backbone, "pulse_power_autoregressive_log_prob", fake_power_log_prob)
    monkeypatch.setattr(actor_critic.backbone, "pulse_phase_autoregressive_entropy", fake_phase_entropy)
    monkeypatch.setattr(actor_critic.backbone, "pulse_length_autoregressive_entropy", fake_length_entropy)
    monkeypatch.setattr(actor_critic.backbone, "pulse_power_autoregressive_entropy", fake_power_entropy)

    log_prob, entropy, _ = actor_critic.evaluate_actions(
        stft_feature_list=stft,
        scalar_side=scalar,
        actions=action_mean,
    )

    assert torch.allclose(log_prob, flat_log_prob + 4.0)
    assert torch.allclose(entropy, flat_entropy + 2.0)

    _, _, provided_action_log_prob = actor_critic.get_action_value_logp(
        {"stft_feature_list": stft, "scalar_side": scalar},
        action=action_mean,
    )
    assert torch.allclose(provided_action_log_prob, flat_log_prob + 4.0)


def test_preprocessing_uses_native_iq_length_by_default():
    short = torch.complex(torch.randn(257), torch.randn(257))
    long = torch.complex(torch.randn(401), torch.randn(401))

    short_proc = preprocess_iq_to_stft_feature(short, sample_rate_hz=1_000_000.0)
    batch_proc = preprocess_batched_iq_to_stft_feature([short, long], sample_rate_hz=1_000_000.0)

    assert short_proc["length_samples"].item() == 257
    assert batch_proc["lengths"].tolist() == [257.0, 401.0]
    assert batch_proc["frequency_feature"].shape[0] == 2
    assert batch_proc["timing_feature"].shape[0] == 2
