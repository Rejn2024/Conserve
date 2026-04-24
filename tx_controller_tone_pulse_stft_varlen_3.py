#!/usr/bin/env python3
"""
PyTorch TX controller for advanced_link_skdsp_v4_robust.build_tone_pulse_iq_object
with variable-length IQ inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import advanced_link_skdsp_v6_robust as txflex


NOISE_COLORS = ["white", "pink", "brown", "blue", "violet"]
FADING_MODES = ["none", "rician_block", "rayleigh_block", "multipath_static"]


def _as_complex_tensor(iq: Union[torch.Tensor, List[complex]]) -> torch.Tensor:
    if isinstance(iq, torch.Tensor):
        return iq.to(dtype=torch.complex64)
    return torch.as_tensor(iq, dtype=torch.complex64)


def measure_iq_power(iq: Union[torch.Tensor, List[complex]]) -> torch.Tensor:
    x = _as_complex_tensor(iq)
    if x.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.mean(torch.abs(x) ** 2).to(dtype=torch.float32)


def robust_mad_scale(x: Union[torch.Tensor, List[complex]], eps: float = 1e-12) -> torch.Tensor:
    xt = _as_complex_tensor(x)
    xr = torch.cat([xt.real, xt.imag], dim=0)
    med = torch.median(xr)
    mad = torch.median(torch.abs(xr - med))
    sigma = 1.4826 * mad + eps
    return (xt / sigma).to(dtype=torch.complex64)


def _torch_unwrap(x: torch.Tensor, dim: int = -1, period: float = 2 * torch.pi) -> torch.Tensor:
    """Compatibility shim for torch versions without torch.unwrap."""
    if hasattr(torch, "unwrap"):
        return torch.unwrap(x, dim=dim)  # type: ignore[attr-defined]

    if x.numel() == 0:
        return x

    dd = torch.diff(x, dim=dim)
    half_period = period / 2
    ddmod = torch.remainder(dd + half_period, period) - half_period
    boundary = (ddmod == -half_period) & (dd > 0)
    ddmod = torch.where(boundary, torch.full_like(ddmod, half_period), ddmod)
    ph_correct = ddmod - dd
    ph_correct = torch.where(torch.abs(dd) < half_period, torch.zeros_like(ph_correct), ph_correct)

    csum = torch.cumsum(ph_correct, dim=dim)
    head = torch.index_select(
        x,
        dim=dim,
        index=torch.tensor([0], device=x.device),
    )
    return torch.cat([head, x.narrow(dim, 1, x.shape[dim] - 1) + csum], dim=dim)




def _as_batch_complex_tensor(iq_batch: Union[torch.Tensor, Sequence[Union[torch.Tensor, List[complex]]]]) -> torch.Tensor:
    if isinstance(iq_batch, torch.Tensor):
        return iq_batch.to(dtype=torch.complex64)

    if len(iq_batch) == 0:
        raise ValueError("iq_batch must contain at least one IQ array")

    rows = [_as_complex_tensor(x) for x in iq_batch]
    lengths = [int(r.numel()) for r in rows]
    max_len = max(lengths)
    if max_len <= 0:
        raise ValueError("iq_batch entries must contain at least one sample")

    padded = []
    for row in rows:
        if row.numel() < max_len:
            row = F.pad(row, (0, max_len - int(row.numel())))
        padded.append(row)
    return torch.stack(padded, dim=0).to(dtype=torch.complex64)


def preprocess_batched_iq_to_stft_feature(
    iq_batch: Union[torch.Tensor, Sequence[Union[torch.Tensor, List[complex]]]],
    sample_rate_hz: float,
    nperseg: int = 128,
    noverlap: int = 96,
    nfft: int = 128,
) -> Dict[str, torch.Tensor]:
    x_batch = _as_batch_complex_tensor(iq_batch)
    if x_batch.ndim != 2:
        raise ValueError(f"iq_batch must have shape [B, T], got {tuple(x_batch.shape)}")

    feats = []
    rx_powers = []
    peaks = []
    lengths = []
    for i in range(x_batch.shape[0]):
        x = x_batch[i]
        lengths.append(int(x.numel()))
        proc = preprocess_iq_to_stft_feature(
            iq=x,
            sample_rate_hz=sample_rate_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
        )
        feats.append(proc["feature"])
        rx_powers.append(proc["rx_power"].to(dtype=torch.float32))
        peaks.append(proc["peak_hz"].to(dtype=torch.float32))

    return {
        "feature": torch.stack(feats, dim=0).to(dtype=torch.float32),
        "rx_power": torch.stack(rx_powers, dim=0).to(dtype=torch.float32),
        "peak_hz": torch.stack(peaks, dim=0).to(dtype=torch.float32),
        "lengths": torch.as_tensor(lengths, dtype=torch.float32),
    }
def preprocess_iq_to_stft_feature(
    iq: Union[torch.Tensor, List[complex]],
    sample_rate_hz: float,
    nperseg: int = 128,
    noverlap: int = 96,
    nfft: int = 128,
) -> Dict[str, torch.Tensor]:
    x_t = _as_complex_tensor(iq)
    if x_t.numel() < 8:
        x_t = F.pad(x_t, (0, 8 - int(x_t.numel())))

    x_t = (x_t - torch.mean(x_t)).to(dtype=torch.complex64)
    x_t = robust_mad_scale(x_t)

    nperseg_eff = int(min(nperseg, max(8, x_t.numel())))
    noverlap_eff = int(min(noverlap, max(0, nperseg_eff - 1)))
    hop_length = max(1, nperseg_eff - noverlap_eff)
    window = torch.hann_window(nperseg_eff, dtype=torch.float32, device=x_t.device)
    Z = torch.stft(
        x_t,
        n_fft=nfft,
        hop_length=hop_length,
        win_length=nperseg_eff,
        window=window,
        center=False,
        return_complex=True,
    )

    Z = torch.fft.fftshift(Z, dim=0)
    f = torch.fft.fftshift(torch.fft.fftfreq(nfft, d=1.0 / max(sample_rate_hz, 1.0)).to(x_t.device))

    mag = torch.log1p(torch.abs(Z)).to(dtype=torch.float32)
    phase = torch.angle(Z).to(dtype=torch.float32)
    real = Z.real.to(dtype=torch.float32)
    imag = Z.imag.to(dtype=torch.float32)
    power = (torch.abs(Z) ** 2).to(dtype=torch.float32)
    power_log = torch.log1p(power)
    denom = torch.sum(power, dim=0, keepdim=True) + 1e-12
    centroid = torch.sum(f[:, None] * power, dim=0, keepdim=True) / denom
    centroid_plane = centroid.repeat(mag.shape[0], 1).to(dtype=torch.float32)
    spread = torch.sqrt(torch.sum(((f[:, None] - centroid) ** 2) * power, dim=0, keepdim=True) / denom + 1e-12)
    spread_plane = (spread / max(sample_rate_hz, 1.0)).repeat(mag.shape[0], 1).to(dtype=torch.float32)

    # Spectral flatness per frame: geometric mean / arithmetic mean over frequency.
    power_pos = torch.clamp(power, min=1e-12)
    log_geom = torch.mean(torch.log(power_pos), dim=0, keepdim=True)
    geom = torch.exp(log_geom)
    arith = torch.mean(power_pos, dim=0, keepdim=True) + 1e-12
    flatness = (geom / arith).to(dtype=torch.float32)
    flatness_plane = flatness.repeat(mag.shape[0], 1)

    # Frame power normalized by average frame power to capture temporal power dynamics.
    frame_power = torch.mean(power, dim=0, keepdim=True)
    frame_power_norm = (frame_power / (torch.mean(frame_power) + 1e-12)).to(dtype=torch.float32)
    frame_power_plane = frame_power_norm.repeat(mag.shape[0], 1)

    # Delta features (temporal and frequency derivatives).
    delta_t_mag = torch.diff(mag, dim=1, prepend=mag[:, :1])
    delta_f_mag = torch.diff(mag, dim=0, prepend=mag[:1, :])
    delta_t_power = torch.diff(power_log, dim=1, prepend=power_log[:, :1])
    phase_unwrapped_t = _torch_unwrap(phase, dim=1)
    delta_t_phase = torch.diff(phase_unwrapped_t, dim=1, prepend=phase_unwrapped_t[:, :1])

    feat = torch.stack(
        [
            mag,
            phase,
            centroid_plane / max(sample_rate_hz, 1.0),
            torch.full_like(mag, torch.log10(torch.tensor(max(sample_rate_hz, 1.0), device=mag.device)) / 10.0),
            real,
            imag,
            power_log,
            delta_t_mag,
            delta_f_mag,
            delta_t_phase,
            delta_t_power,
            spread_plane,
            flatness_plane,
            frame_power_plane,
        ],
        dim=0,
    )

    return {
        "feature": feat.to(dtype=torch.float32),
        "rx_power": measure_iq_power(iq),
        "peak_hz": f[torch.argmax(torch.mean(power, dim=1))].to(dtype=torch.float32) if power.numel() else torch.tensor(0.0, dtype=torch.float32, device=x_t.device),
    }


@dataclass
class TonePulseControlConfig:
    sample_rate_hz: float
    rf_center_hz: float
    carrier_hz: float
    num_tones: int
    tone_frequencies_hz: List[float]
    tone_amplitudes: List[float]
    pulse_on_samples: int
    pulse_off_samples: int
    pulse_count: int
    start_offset_samples: int
    snr_db: Optional[float]
    noise_color: str
    freq_offset: float
    timing_offset: float
    fading_mode: str
    fading_block_len: int
    rician_k_db: float
    burst_probability: float
    burst_len_min: int
    burst_len_max: int
    burst_power_ratio_db: float
    burst_color: str
    peak_power: Optional[float]
    seed: int = 1


class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class VarLenSTFTEncoder(nn.Module):
    """Encodes one variable-size [B,C,F,T] STFT map into [B,D]."""

    def __init__(self, in_ch: int = 14, base_ch: int = 24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock2D(base_ch, base_ch)
        self.layer2 = ResidualBlock2D(base_ch, base_ch * 2, stride=2)
        self.layer3 = ResidualBlock2D(base_ch * 2, base_ch * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = base_ch * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        return self.pool(z).flatten(1)


class TonePulseTXControlNetVarLen(nn.Module):
    """Shared STFT encoder over a variable-length list of IQ windows."""

    def __init__(self, in_ch: int = 14, base_ch: int = 24, max_tones: int = 8):
        """
        Args:
            in_ch: Number of channels per STFT input map [B, C, F, T].
            base_ch: Base convolution width used by the STFT encoder.
            max_tones: Upper bound on synthesized tone count (also output width for tone amplitudes).
        """
        super().__init__()
        self.max_tones = max_tones

        # Four dedicated STFT-ResNet branches, one per output category.
        self.env_encoder = VarLenSTFTEncoder(in_ch=in_ch, base_ch=base_ch)
        self.tone_encoder = VarLenSTFTEncoder(in_ch=in_ch, base_ch=base_ch)
        self.pulse_encoder = VarLenSTFTEncoder(in_ch=in_ch, base_ch=base_ch)
        self.impairment_encoder = VarLenSTFTEncoder(in_ch=in_ch, base_ch=base_ch)

        def _make_fusion(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, 160),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(160, 96),
                nn.ReLU(inplace=True),
            )

        self.env_fusion = _make_fusion(self.env_encoder.feature_dim)
        self.tone_fusion = _make_fusion(self.tone_encoder.feature_dim)
        self.pulse_fusion = _make_fusion(self.pulse_encoder.feature_dim)
        self.impairment_fusion = _make_fusion(self.impairment_encoder.feature_dim)
        self.shared_fusion = nn.Sequential(
            nn.Linear(96 * 4, 96),
            nn.ReLU(inplace=True),
        )

        # Categorical/channel environment heads.
        self.noise_color_head = nn.Linear(96, len(NOISE_COLORS))
        self.fading_mode_head = nn.Linear(96, len(FADING_MODES))
        self.burst_color_head = nn.Linear(96, len(NOISE_COLORS))

        # Frequency/rate/tone structure heads.
        self.sample_rate_scale_head = nn.Linear(96, 1)
        self.rf_center_delta_head = nn.Linear(96, 1)
        self.carrier_norm_head = nn.Linear(96, 1)
        self.num_tones_head = nn.Linear(96, 1)
        self.base_tone_norm_head = nn.Linear(96, 1)
        self.tone_spacing_norm_head = nn.Linear(96, 1)
        self.tone_amp_raw_head = nn.Linear(96, max_tones)

        # Pulse timing/envelope heads.
        self.pulse_on_head = nn.Linear(96, 1)
        self.pulse_off_head = nn.Linear(96, 1)
        self.pulse_count_head = nn.Linear(96, 1)
        self.start_offset_head = nn.Linear(96, 1)

        # Impairments/interference heads.
        self.snr_db_head = nn.Linear(96, 1)
        self.freq_offset_head = nn.Linear(96, 1)
        self.timing_offset_head = nn.Linear(96, 1)
        self.fading_block_head = nn.Linear(96, 1)
        self.rician_k_db_head = nn.Linear(96, 1)
        self.burst_prob_head = nn.Linear(96, 1)
        self.burst_power_ratio_head = nn.Linear(96, 1)

    def _encode_branch_features(
        self,
        encoder: VarLenSTFTEncoder,
        fusion: nn.Sequential,
        stft_feature_list: List[torch.Tensor],
    ) -> torch.Tensor:
        emb = [encoder(x) for x in stft_feature_list]
        z_stft = torch.stack(emb, dim=0).mean(dim=0)
        return fusion(z_stft)

    def encode_fused_features(self, stft_feature_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode variable-length STFT observations into a shared latent.

        This method is used by supervised control heads in this class and by the RL ActorCritic
        wrapper so both training modes share the same representation network.
        """
        z_env, z_tone, z_pulse, z_impair = self.encode_category_features(stft_feature_list=stft_feature_list)
        return self.shared_fusion(torch.cat([z_env, z_tone, z_pulse, z_impair], dim=-1))

    def encode_category_features(self, stft_feature_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_env = self._encode_branch_features(
            encoder=self.env_encoder,
            fusion=self.env_fusion,
            stft_feature_list=stft_feature_list,
        )
        z_tone = self._encode_branch_features(
            encoder=self.tone_encoder,
            fusion=self.tone_fusion,
            stft_feature_list=stft_feature_list,
        )
        z_pulse = self._encode_branch_features(
            encoder=self.pulse_encoder,
            fusion=self.pulse_fusion,
            stft_feature_list=stft_feature_list,
        )
        z_impair = self._encode_branch_features(
            encoder=self.impairment_encoder,
            fusion=self.impairment_fusion,
            stft_feature_list=stft_feature_list,
        )
        return z_env, z_tone, z_pulse, z_impair

    def forward(self, stft_feature_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        z_env, z_tone, z_pulse, z_impair = self.encode_category_features(stft_feature_list=stft_feature_list)
        r = {
            "noise_color_logits": self.noise_color_head(z_env),
            "fading_mode_logits": self.fading_mode_head(z_env),
            "burst_color_logits": self.burst_color_head(z_env),
            "sample_rate_scale": 0.5 + 2.0 * torch.sigmoid(self.sample_rate_scale_head(z_tone)),
            "rf_center_delta_hz": 500_000.0 * torch.tanh(self.rf_center_delta_head(z_tone)),
            "carrier_hz_norm": 0.45 * torch.tanh(self.carrier_norm_head(z_tone)),
            "num_tones_cont": 1.0 + (self.max_tones - 1.0) * torch.sigmoid(self.num_tones_head(z_tone)),
            "base_tone_norm": 0.45 * torch.tanh(self.base_tone_norm_head(z_tone)),
            "tone_spacing_norm": 0.20 * torch.sigmoid(self.tone_spacing_norm_head(z_tone)),
            "tone_amp_raw": self.tone_amp_raw_head(z_tone),
            "pulse_on_samples": 128.0 + 8192.0 * torch.sigmoid(self.pulse_on_head(z_pulse)),
            "pulse_off_samples": 0.0 + 8192.0 * torch.sigmoid(self.pulse_off_head(z_pulse)),
            "pulse_count": 1.0 + 32.0 * torch.sigmoid(self.pulse_count_head(z_pulse)),
            "start_offset": 0.0 + 4096.0 * torch.sigmoid(self.start_offset_head(z_pulse)),
            "snr_db": 50.0 * torch.sigmoid(self.snr_db_head(z_impair)),
            "freq_offset": 0.005 * torch.tanh(self.freq_offset_head(z_impair)),
            "timing_offset": 1.0 + 0.002 * torch.tanh(self.timing_offset_head(z_impair)),
            "fading_block_len_norm": torch.sigmoid(self.fading_block_head(z_impair)),
            "rician_k_db": 20.0 * torch.sigmoid(self.rician_k_db_head(z_impair)),
            "burst_probability": 1e-2 * torch.sigmoid(self.burst_prob_head(z_impair)),
            "burst_power_ratio_db": 30.0 * torch.sigmoid(self.burst_power_ratio_head(z_impair)),
        }

        return r

class ActorCritic(nn.Module):
    """
    RL-friendly Actor-Critic model that reuses the TonePulseTXControlNetVarLen shared network.

    The backbone is exactly the variable-length STFT encoder + scalar fusion stack used by
    TonePulseTXControlNetVarLen. The actor/critic heads expose the typical interfaces used by
    policy-gradient / PPO style loops in RL notebooks.
    """

    def __init__(
        self,
        action_dim: Optional[int] = None,
        in_ch: int = 14,
        base_ch: int = 24,
        max_tones: int = 8,
        action_std_init: float = 0.25,
    ):
        super().__init__()
        self.backbone = TonePulseTXControlNetVarLen(
            in_ch=in_ch,
            base_ch=base_ch,
            max_tones=max_tones,
        )
        self.max_tones = max_tones
        self.action_dim = int(action_dim) if action_dim is not None else (12 + int(max_tones))
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self._action_std_floor = 1e-4
        self._init_log_std = math.log(max(action_std_init, self._action_std_floor))
        self.log_std_head = nn.Linear(96, self.action_dim)
        self.value_head = nn.Linear(96, 1)

    def _encode(self, stft_feature_list: List[torch.Tensor]) -> torch.Tensor:
        return self.backbone.encode_fused_features(
            stft_feature_list=stft_feature_list,
        )

    def _continuous_action_mean(
        self,
        stft_feature_list: List[torch.Tensor],
    ) -> torch.Tensor:
        y = self.backbone(stft_feature_list)

        def _expected_index(logits: torch.Tensor) -> torch.Tensor:
            probs = torch.softmax(logits, dim=-1)
            idx = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
            return (probs * idx).sum(dim=-1, keepdim=True)

        pieces = [
            _expected_index(y["noise_color_logits"]),
            _expected_index(y["fading_mode_logits"]),
            _expected_index(y["burst_color_logits"]),
            y["rf_center_delta_hz"],
            y["carrier_hz_norm"],
            y["num_tones_cont"],
            y["base_tone_norm"],
            y["tone_spacing_norm"],
            y["tone_amp_raw"],
            y["pulse_on_samples"],
            y["pulse_off_samples"],
            y["pulse_count"],
            y["start_offset"],
        ]
        action_mean = torch.cat(pieces, dim=-1)
        if action_mean.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action mean width {action_mean.shape[-1]} does not match action_dim {self.action_dim}. "
                "Set action_dim to 12 + max_tones (default) or a matching custom width."
            )
        return action_mean

    def _log_std_from_features(self, fused_features: torch.Tensor) -> torch.Tensor:
        # Keep per-action std positive and numerically stable while allowing
        # input-conditioned exploration from STFT features.
        raw = self.log_std_head(fused_features)
        return self._init_log_std + torch.tanh(raw)

    def forward(self, stft_feature_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self._continuous_action_mean(stft_feature_list=stft_feature_list)
        fused = self._encode(stft_feature_list=stft_feature_list)
        value = self.value_head(fused).squeeze(-1)
        log_std = self._log_std_from_features(fused)
        return action_mean, value, log_std

    def forward_observation(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper for notebook-style observations.

        Expected keys:
            observation["stft_feature_list"] -> List[Tensor[B, C, F, T]]
        """
        return self.forward(
            stft_feature_list=observation["stft_feature_list"],
        )

    def _action_distribution(self, action_mean: torch.Tensor, log_std: torch.Tensor) -> torch.distributions.Normal:
        std = torch.exp(log_std).clamp_min(self._action_std_floor)
        return torch.distributions.Normal(loc=action_mean, scale=std)

    def act(
        self,
        stft_feature_list: List[torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value, log_std = self.forward(stft_feature_list=stft_feature_list)
        dist = self._action_distribution(action_mean=action_mean, log_std=log_std)
        action = action_mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self,
        stft_feature_list: List[torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value, log_std = self.forward(stft_feature_list=stft_feature_list)
        dist = self._action_distribution(action_mean=action_mean, log_std=log_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value

    def get_action_value_logp(
        self,
        observation: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO-friendly helper used by notebook training loops.

        Args:
            observation: Dict containing:
                - "stft_feature_list": List[Tensor[B, C, F, T]]
            action: Optional Tensor[B, action_dim] of preselected continuous actions.
                If omitted, an action is sampled (or mean action if deterministic=True).
            deterministic: If True and `action` is None, choose mean action.

        Returns:
            Tuple of (action, value, log_prob) with action shaped [B, action_dim].
        """
        action_mean, value, log_std = self.forward_observation(observation)
        dist = self._action_distribution(action_mean=action_mean, log_std=log_std)
        if action is None:
            action = action_mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, value, log_prob


def _nearest_block_len(x_norm: float) -> int:
    choices = torch.tensor([128, 256, 512, 1024, 2048, 4096], dtype=torch.int64)
    idx = int(round(x_norm * (len(choices) - 1)))
    idx = max(0, min(idx, len(choices) - 1))
    return int(choices[idx].item())


def decode_tone_pulse_config(
    model_out: Dict[str, torch.Tensor],
    intake_sample_rate_hz: float,
    rf_center_est_hz: float,
    desired_output_iq_len: Optional[int],
    user_peak_power_fraction: Optional[float],
    rx_input_power: float,
    max_tones: int,
    seed: int,
    action_overrides: Optional[Dict[str, object]] = None,
) -> TonePulseControlConfig:
    def _safe_scalar(name: str, default: float) -> float:
        value = float(model_out[name].item())
        return value if math.isfinite(value) else float(default)

    noise_color = NOISE_COLORS[int(torch.argmax(model_out["noise_color_logits"], dim=-1).item())]
    fading_mode = FADING_MODES[int(torch.argmax(model_out["fading_mode_logits"], dim=-1).item())]
    burst_color = NOISE_COLORS[int(torch.argmax(model_out["burst_color_logits"], dim=-1).item())]

    sample_rate_hz = float(intake_sample_rate_hz * _safe_scalar("sample_rate_scale", 1.0))
    rf_center_hz = float(rf_center_est_hz + _safe_scalar("rf_center_delta_hz", 0.0))
    carrier_hz = float(_safe_scalar("carrier_hz_norm", 0.0)) * sample_rate_hz

    num_tones = int(round(_safe_scalar("num_tones_cont", 1.0)))
    num_tones = max(1, min(num_tones, max_tones))

    base_f = float(_safe_scalar("base_tone_norm", 0.0)) * sample_rate_hz
    spacing = float(_safe_scalar("tone_spacing_norm", 0.0)) * sample_rate_hz

    amp_raw = model_out["tone_amp_raw"].detach().reshape(-1)[:num_tones]

    if action_overrides:
        def _override_scalar(name: str, default: float) -> float:
            v = action_overrides.get(name, default)
            try:
                out = float(v)
            except Exception:
                out = float(default)
            return out if math.isfinite(out) else float(default)

        noise_idx = int(round(_override_scalar("noise_color", NOISE_COLORS.index(noise_color))))
        noise_idx = max(0, min(noise_idx, len(NOISE_COLORS) - 1))
        noise_color = NOISE_COLORS[noise_idx]

        fading_idx = int(round(_override_scalar("fading_mode", FADING_MODES.index(fading_mode))))
        fading_idx = max(0, min(fading_idx, len(FADING_MODES) - 1))
        fading_mode = FADING_MODES[fading_idx]

        burst_idx = int(round(_override_scalar("burst_color", NOISE_COLORS.index(burst_color))))
        burst_idx = max(0, min(burst_idx, len(NOISE_COLORS) - 1))
        burst_color = NOISE_COLORS[burst_idx]

        rf_center_hz = _override_scalar("rf_center_hz", rf_center_hz)
        carrier_hz = _override_scalar("carrier_hz", carrier_hz)
        num_tones = int(round(_override_scalar("num_tones", num_tones)))
        num_tones = max(1, min(num_tones, max_tones))
        base_f = _override_scalar("base_f", base_f)
        spacing = abs(_override_scalar("spacing", spacing))

        amp_raw_override = action_overrides.get("amp_raw", None)
        if amp_raw_override is not None:
            try:
                amp_raw_t = torch.as_tensor(amp_raw_override, dtype=torch.float32).reshape(-1)
                if amp_raw_t.numel() > 0:
                    amp_raw = amp_raw_t[:num_tones]
            except Exception:
                pass

    offsets = (torch.arange(num_tones, dtype=torch.float64) - 0.5 * (num_tones - 1)) * spacing
    tone_frequencies_hz = [
        float(
            torch.clamp(
                base_f + df,
                min=-0.49 * sample_rate_hz,
                max=0.49 * sample_rate_hz,
            ).item()
        )
        for df in offsets
    ]

    tone_amplitudes = (0.05 + 0.95 * torch.sigmoid(amp_raw)).to(dtype=torch.float64).tolist()

    peak_power = None if user_peak_power_fraction is None else float(user_peak_power_fraction) * float(rx_input_power)

    pulse_on_samples = int(round(_safe_scalar("pulse_on_samples", 128.0)))
    pulse_off_samples = int(round(_safe_scalar("pulse_off_samples", 0.0)))
    pulse_count = int(round(_safe_scalar("pulse_count", 1.0)))
    start_offset_samples = int(round(_safe_scalar("start_offset", 0.0)))
    if action_overrides:
        def _to_int(name: str, default: int) -> int:
            try:
                return int(round(float(action_overrides.get(name, default))))
            except Exception:
                return int(default)

        pulse_on_samples = _to_int("pulse_on_samples", pulse_on_samples)
        pulse_off_samples = _to_int("pulse_off_samples", pulse_off_samples)
        pulse_count = _to_int("pulse_count", pulse_count)
        start_offset_samples = _to_int("start_offset_samples", start_offset_samples)

    if desired_output_iq_len is not None and desired_output_iq_len > 0:
        pulse_on_samples = min(pulse_on_samples, desired_output_iq_len)
        start_offset_samples = min(start_offset_samples, max(0, desired_output_iq_len - 1))

    return TonePulseControlConfig(
        sample_rate_hz=sample_rate_hz,
        rf_center_hz=rf_center_hz,
        carrier_hz=carrier_hz,
        num_tones=num_tones,
        tone_frequencies_hz=tone_frequencies_hz,
        tone_amplitudes=tone_amplitudes,
        pulse_on_samples=max(1, pulse_on_samples),
        pulse_off_samples=max(0, pulse_off_samples),
        pulse_count=max(1, pulse_count),
        start_offset_samples=max(0, start_offset_samples),
        snr_db=_safe_scalar("snr_db", 0.0),
        noise_color=noise_color,
        freq_offset=_safe_scalar("freq_offset", 0.0),
        timing_offset=_safe_scalar("timing_offset", 1.0),
        fading_mode=fading_mode,
        fading_block_len=_nearest_block_len(_safe_scalar("fading_block_len_norm", 0.0)),
        rician_k_db=_safe_scalar("rician_k_db", 0.0),
        burst_probability=_safe_scalar("burst_probability", 0.0),
        burst_len_min=16,
        burst_len_max=64,
        burst_power_ratio_db=_safe_scalar("burst_power_ratio_db", 0.0),
        burst_color=burst_color,
        peak_power=peak_power,
        seed=seed,
    )


def build_controlled_tone_pulse_from_variable_inputs(
    model: TonePulseTXControlNetVarLen,
    rx_iq_windows: List[Union[torch.Tensor, List[complex]]],
    intake_sample_rate_hz: float,
    desired_output_iq_len: Optional[int] = None,
    user_peak_power_fraction: Optional[float] = None,
    seed: int = 1,
    device: str = "cpu",
) -> Dict[str, object]:
    """Build one controlled tone-pulse transmission from three RX IQ windows."""
    batched = [_as_batch_complex_tensor([x]) for x in rx_iq_windows]
    out = build_controlled_tone_pulse_batch_from_iq_batches(
        model=model,
        rx_iq_batches=batched,
        intake_sample_rate_hz=intake_sample_rate_hz,
        desired_output_iq_len=desired_output_iq_len,
        user_peak_power_fraction=user_peak_power_fraction,
        seed=seed,
        device=device,
    )
    return out[0]


def build_controlled_tone_pulse_batch_from_iq_batches(
    model: TonePulseTXControlNetVarLen,
    rx_iq_batches: Sequence[Union[torch.Tensor, Sequence[Union[torch.Tensor, List[complex]]]]],
    intake_sample_rate_hz: float,
    desired_output_iq_len: Optional[int] = None,
    user_peak_power_fraction: Optional[float] = None,
    action_overrides: Optional[Sequence[Optional[Dict[str, object]]]] = None,
    seed: int = 1,
    device: str = "cpu",
) -> List[Dict[str, object]]:
    if len(rx_iq_batches) != 3:
        raise ValueError(f"rx_iq_batches must contain exactly 3 IQ batch inputs, got {len(rx_iq_batches)}")

    proc = [preprocess_batched_iq_to_stft_feature(x, intake_sample_rate_hz) for x in rx_iq_batches]
    batch_size = int(proc[0]["feature"].shape[0])
    if any(int(p["feature"].shape[0]) != batch_size for p in proc):
        raise ValueError("All three IQ inputs must use the same batch size")

    stft_tensors = [p["feature"].to(device) for p in proc]
    rx_power_stack = torch.stack([p["rx_power"].to(device) for p in proc], dim=0)
    peak_stack = torch.stack([p["peak_hz"].to(device) for p in proc], dim=0)
    rx_input_power_t = rx_power_stack.mean(dim=0)
    rf_center_est_t = peak_stack.mean(dim=0)

    lengths = torch.stack([p["lengths"].to(device) for p in proc], dim=0).transpose(0, 1)
    model = model.to(device)
    y = model(stft_tensors)
    if isinstance(y, tuple):
        if hasattr(model, "backbone"):
            y = model.backbone(stft_tensors)
        else:
            raise TypeError("model(stft_tensors) returned tuple; expected dict of control heads")

    if action_overrides is not None and len(action_overrides) != batch_size:
        raise ValueError("action_overrides must have one entry per batch sample")

    out = []
    for i in range(batch_size):
        model_out_i = {k: v[i : i + 1] for k, v in y.items()}
        action_override_i = None if action_overrides is None else action_overrides[i]
        cfg = decode_tone_pulse_config(
            model_out=model_out_i,
            intake_sample_rate_hz=float(intake_sample_rate_hz),
            rf_center_est_hz=float(rf_center_est_t[i].item()),
            desired_output_iq_len=desired_output_iq_len,
            user_peak_power_fraction=user_peak_power_fraction,
            rx_input_power=float(rx_input_power_t[i].item()),
            max_tones=model.max_tones,
            seed=seed + i,
            action_overrides=action_override_i,
        )

        tx_result = txflex.build_tone_pulse_iq_object(
            sample_rate_hz=cfg.sample_rate_hz,
            rf_center_hz=cfg.rf_center_hz,
            carrier_hz=cfg.carrier_hz,
            target_num_samples=desired_output_iq_len,
            num_tones=cfg.num_tones,
            tone_frequencies_hz=cfg.tone_frequencies_hz,
            tone_amplitudes=cfg.tone_amplitudes,
            pulse_on_samples=cfg.pulse_on_samples,
            pulse_off_samples=cfg.pulse_off_samples,
            pulse_count=cfg.pulse_count,
            start_offset_samples=cfg.start_offset_samples,
            snr_db=cfg.snr_db,
            noise_color=cfg.noise_color,
            freq_offset=cfg.freq_offset,
            timing_offset=cfg.timing_offset,
            fading_mode=cfg.fading_mode,
            fading_block_len=cfg.fading_block_len,
            rician_k_db=cfg.rician_k_db,
            multipath_taps=None,
            burst_probability=cfg.burst_probability,
            burst_len_min=cfg.burst_len_min,
            burst_len_max=cfg.burst_len_max,
            burst_power_ratio_db=cfg.burst_power_ratio_db,
            burst_color=cfg.burst_color,
            peak_power=cfg.peak_power,
            seed=cfg.seed,
        )

        tx_metadata = dict(tx_result.metadata)
        tx_metadata["controller_input_lengths"] = [int(v) for v in lengths[i].tolist()]
        tx_metadata["controller_rx_input_power"] = float(rx_input_power_t[i].item())

        out.append(
            {
                "tx_config": cfg,
                "model_outputs": model_out_i,
                "tx_iq": _as_complex_tensor(tx_result.iq),
                "tx_metadata": tx_metadata,
                "rx_input_power": float(rx_input_power_t[i].item()),
                "rf_center_est_hz": float(rf_center_est_t[i].item()),
            }
        )

    return out


if __name__ == "__main__":
    iq_a = torch.complex(torch.randn(900), torch.randn(900)).to(dtype=torch.complex64)
    iq_b = torch.complex(torch.randn(1700), torch.randn(1700)).to(dtype=torch.complex64)
    iq_c = torch.complex(torch.randn(3200), torch.randn(3200)).to(dtype=torch.complex64)

    model = TonePulseTXControlNetVarLen(in_ch=14, base_ch=16, max_tones=8)
    out = build_controlled_tone_pulse_from_variable_inputs(
        model=model,
        rx_iq_windows=[iq_a, iq_b, iq_c],
        intake_sample_rate_hz=1_000_000.0,
        desired_output_iq_len=24_000,
        user_peak_power_fraction=0.1,
        seed=11,
        device="cpu",
    )

    print("Generated IQ samples:", len(out["tx_iq"]))
    print("Controlled num_tones:", out["tx_config"].num_tones)
    print("Tone frequencies (Hz):", out["tx_config"].tone_frequencies_hz)
