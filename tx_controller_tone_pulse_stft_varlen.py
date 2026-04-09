#!/usr/bin/env python3
"""
PyTorch TX controller for advanced_link_skdsp_v4_robust.build_tone_pulse_iq_object
with variable-length IQ inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

import advanced_link_skdsp_v4_robust as txflex


NOISE_COLORS = ["white", "pink", "brown", "blue", "violet"]
FADING_MODES = ["none", "rician_block", "rayleigh_block", "multipath_static"]


def _as_complex_tensor(iq: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(iq, torch.Tensor):
        return iq.to(dtype=torch.complex64)
    return torch.as_tensor(np.asarray(iq), dtype=torch.complex64)


def measure_iq_power(iq: Union[np.ndarray, torch.Tensor]) -> float:
    x = _as_complex_tensor(iq)
    if x.numel() == 0:
        return 0.0
    return float(torch.mean(torch.abs(x) ** 2).item())


def robust_mad_scale(x: Union[np.ndarray, torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    xt = _as_complex_tensor(x)
    xr = torch.cat([xt.real, xt.imag], dim=0)
    med = torch.median(xr)
    mad = torch.median(torch.abs(xr - med))
    sigma = 1.4826 * mad + eps
    return (xt / sigma).to(dtype=torch.complex64)


def preprocess_iq_to_stft_feature(
    iq: Union[np.ndarray, torch.Tensor],
    sample_rate_hz: float,
    nperseg: int = 128,
    noverlap: int = 96,
    nfft: int = 128,
) -> Dict[str, Union[np.ndarray, np.float32]]:
    x_t = _as_complex_tensor(iq)
    if x_t.numel() < 8:
        x_t = F.pad(x_t, (0, 8 - int(x_t.numel())))

    x_t = (x_t - torch.mean(x_t)).to(dtype=torch.complex64)
    x_t = robust_mad_scale(x_t)
    x = x_t.detach().cpu().numpy()

    f, _, Z = signal.stft(
        x,
        fs=sample_rate_hz,
        window="hann",
        nperseg=min(nperseg, len(x)),
        noverlap=min(noverlap, max(0, len(x) - 1)),
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    Z = np.fft.fftshift(Z, axes=0)
    f = np.fft.fftshift(f)

    mag = np.log1p(np.abs(Z)).astype(np.float32)
    phase = np.angle(Z).astype(np.float32)
    power = np.abs(Z) ** 2
    denom = np.sum(power, axis=0, keepdims=True) + 1e-12
    centroid = (np.sum(f[:, None] * power, axis=0, keepdims=True) / denom).astype(np.float32)
    centroid_plane = np.repeat(centroid, repeats=mag.shape[0], axis=0)

    feat = np.stack(
        [
            mag,
            phase,
            centroid_plane / max(sample_rate_hz, 1.0),
            np.full_like(mag, np.log10(max(sample_rate_hz, 1.0)) / 10.0, dtype=np.float32),
        ],
        axis=0,
    )

    return {
        "feature": feat.astype(np.float32),
        "rx_power": np.float32(measure_iq_power(iq)),
        "peak_hz": np.float32(f[np.argmax(np.mean(power, axis=1))]) if power.size else np.float32(0.0),
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

    def __init__(self, in_ch: int = 4, base_ch: int = 24):
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

    def __init__(self, in_ch: int = 4, base_ch: int = 24, n_scalar_features: int = 6, max_tones: int = 8):
        super().__init__()
        self.encoder = VarLenSTFTEncoder(in_ch=in_ch, base_ch=base_ch)
        self.max_tones = max_tones

        self.scalar_proj = nn.Sequential(nn.Linear(n_scalar_features, 48), nn.ReLU(inplace=True))
        fused_dim = self.encoder.feature_dim + 48
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(160, 96),
            nn.ReLU(inplace=True),
        )

        self.noise_color_head = nn.Linear(96, len(NOISE_COLORS))
        self.fading_mode_head = nn.Linear(96, len(FADING_MODES))
        self.burst_color_head = nn.Linear(96, len(NOISE_COLORS))

        self.sample_rate_scale_head = nn.Linear(96, 1)
        self.rf_center_delta_head = nn.Linear(96, 1)
        self.carrier_norm_head = nn.Linear(96, 1)
        self.num_tones_head = nn.Linear(96, 1)
        self.base_tone_norm_head = nn.Linear(96, 1)
        self.tone_spacing_norm_head = nn.Linear(96, 1)
        self.tone_amp_raw_head = nn.Linear(96, max_tones)
        self.pulse_on_head = nn.Linear(96, 1)
        self.pulse_off_head = nn.Linear(96, 1)
        self.pulse_count_head = nn.Linear(96, 1)
        self.start_offset_head = nn.Linear(96, 1)
        self.snr_db_head = nn.Linear(96, 1)
        self.freq_offset_head = nn.Linear(96, 1)
        self.timing_offset_head = nn.Linear(96, 1)
        self.fading_block_head = nn.Linear(96, 1)
        self.rician_k_db_head = nn.Linear(96, 1)
        self.burst_prob_head = nn.Linear(96, 1)
        self.burst_power_ratio_head = nn.Linear(96, 1)

    def forward(self, stft_feature_list: List[torch.Tensor], scalar_side: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = [self.encoder(x) for x in stft_feature_list]
        z_stft = torch.stack(emb, dim=0).mean(dim=0)
        z = torch.cat([z_stft, self.scalar_proj(scalar_side)], dim=-1)
        z = self.fusion(z)

        return {
            "noise_color_logits": self.noise_color_head(z),
            "fading_mode_logits": self.fading_mode_head(z),
            "burst_color_logits": self.burst_color_head(z),
            "sample_rate_scale": 0.5 + 2.0 * torch.sigmoid(self.sample_rate_scale_head(z)),
            "rf_center_delta_hz": 500_000.0 * torch.tanh(self.rf_center_delta_head(z)),
            "carrier_hz_norm": 0.45 * torch.tanh(self.carrier_norm_head(z)),
            "num_tones_cont": 1.0 + (self.max_tones - 1.0) * torch.sigmoid(self.num_tones_head(z)),
            "base_tone_norm": 0.45 * torch.tanh(self.base_tone_norm_head(z)),
            "tone_spacing_norm": 0.20 * torch.sigmoid(self.tone_spacing_norm_head(z)),
            "tone_amp_raw": self.tone_amp_raw_head(z),
            "pulse_on_samples": 128.0 + 8192.0 * torch.sigmoid(self.pulse_on_head(z)),
            "pulse_off_samples": 0.0 + 8192.0 * torch.sigmoid(self.pulse_off_head(z)),
            "pulse_count": 1.0 + 32.0 * torch.sigmoid(self.pulse_count_head(z)),
            "start_offset": 0.0 + 4096.0 * torch.sigmoid(self.start_offset_head(z)),
            "snr_db": 50.0 * torch.sigmoid(self.snr_db_head(z)),
            "freq_offset": 0.005 * torch.tanh(self.freq_offset_head(z)),
            "timing_offset": 1.0 + 0.002 * torch.tanh(self.timing_offset_head(z)),
            "fading_block_len_norm": torch.sigmoid(self.fading_block_head(z)),
            "rician_k_db": 20.0 * torch.sigmoid(self.rician_k_db_head(z)),
            "burst_probability": 1e-2 * torch.sigmoid(self.burst_prob_head(z)),
            "burst_power_ratio_db": 30.0 * torch.sigmoid(self.burst_power_ratio_head(z)),
        }


def _nearest_block_len(x_norm: float) -> int:
    choices = np.array([128, 256, 512, 1024, 2048, 4096], dtype=np.int64)
    idx = int(round(x_norm * (len(choices) - 1)))
    idx = max(0, min(idx, len(choices) - 1))
    return int(choices[idx])


def decode_tone_pulse_config(
    model_out: Dict[str, torch.Tensor],
    intake_sample_rate_hz: float,
    rf_center_est_hz: float,
    desired_output_iq_len: Optional[int],
    user_peak_power_fraction: Optional[float],
    rx_input_power: float,
    max_tones: int,
    seed: int,
) -> TonePulseControlConfig:
    noise_color = NOISE_COLORS[int(torch.argmax(model_out["noise_color_logits"], dim=-1).item())]
    fading_mode = FADING_MODES[int(torch.argmax(model_out["fading_mode_logits"], dim=-1).item())]
    burst_color = NOISE_COLORS[int(torch.argmax(model_out["burst_color_logits"], dim=-1).item())]

    sample_rate_hz = float(intake_sample_rate_hz * model_out["sample_rate_scale"].item())
    rf_center_hz = float(rf_center_est_hz + model_out["rf_center_delta_hz"].item())
    carrier_hz = float(model_out["carrier_hz_norm"].item()) * sample_rate_hz

    num_tones = int(round(float(model_out["num_tones_cont"].item())))
    num_tones = max(1, min(num_tones, max_tones))

    base_f = float(model_out["base_tone_norm"].item()) * sample_rate_hz
    spacing = float(model_out["tone_spacing_norm"].item()) * sample_rate_hz
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

    amp_raw = model_out["tone_amp_raw"].detach().reshape(-1)[:num_tones]
    tone_amplitudes = (0.05 + 0.95 * torch.sigmoid(amp_raw)).to(dtype=torch.float64).tolist()

    peak_power = None if user_peak_power_fraction is None else float(user_peak_power_fraction) * float(rx_input_power)

    pulse_on_samples = int(round(float(model_out["pulse_on_samples"].item())))
    pulse_off_samples = int(round(float(model_out["pulse_off_samples"].item())))
    pulse_count = int(round(float(model_out["pulse_count"].item())))
    start_offset_samples = int(round(float(model_out["start_offset"].item())))

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
        snr_db=float(model_out["snr_db"].item()),
        noise_color=noise_color,
        freq_offset=float(model_out["freq_offset"].item()),
        timing_offset=float(model_out["timing_offset"].item()),
        fading_mode=fading_mode,
        fading_block_len=_nearest_block_len(float(model_out["fading_block_len_norm"].item())),
        rician_k_db=float(model_out["rician_k_db"].item()),
        burst_probability=float(model_out["burst_probability"].item()),
        burst_len_min=16,
        burst_len_max=64,
        burst_power_ratio_db=float(model_out["burst_power_ratio_db"].item()),
        burst_color=burst_color,
        peak_power=peak_power,
        seed=seed,
    )


def build_controlled_tone_pulse_from_variable_inputs(
    model: TonePulseTXControlNetVarLen,
    rx_iq_windows: List[Union[np.ndarray, torch.Tensor]],
    intake_sample_rate_hz: float,
    desired_output_iq_len: Optional[int] = None,
    user_peak_power_fraction: Optional[float] = None,
    seed: int = 1,
    device: str = "cpu",
) -> Dict[str, object]:
    if not rx_iq_windows:
        raise ValueError("rx_iq_windows must contain at least one IQ array")

    proc = [preprocess_iq_to_stft_feature(x, intake_sample_rate_hz) for x in rx_iq_windows]
    stft_tensors = [torch.from_numpy(p["feature"]).unsqueeze(0).to(device) for p in proc]  # type: ignore[arg-type]

    rx_power_t = torch.tensor([float(p["rx_power"]) for p in proc], dtype=torch.float32)
    peak_t = torch.tensor([float(p["peak_hz"]) for p in proc], dtype=torch.float32)
    rx_input_power = float(rx_power_t.mean().item())
    rf_center_est_hz = float(peak_t.mean().item())

    lengths = torch.tensor([int(_as_complex_tensor(x).numel()) for x in rx_iq_windows], dtype=torch.float32)
    max_len = torch.clamp(torch.max(lengths), min=1.0)
    scalar_side = torch.tensor(
        [[
            torch.log10(torch.tensor(max(intake_sample_rate_hz, 1.0), dtype=torch.float32)).item() / 10.0,
            float((torch.mean(lengths) / max_len).item()),
            float((torch.std(lengths, unbiased=False) / max_len).item()),
            rx_input_power,
            float((peak_t.mean() / max(intake_sample_rate_hz, 1.0)).item()),
            float((peak_t.std(unbiased=False) / max(intake_sample_rate_hz, 1.0)).item()),
        ]],
        dtype=torch.float32,
    ).to(device)

    model = model.to(device)
    # model.eval()
    # with torch.no_grad():
    y = model(stft_tensors, scalar_side)

    cfg = decode_tone_pulse_config(
        model_out=y,
        intake_sample_rate_hz=float(intake_sample_rate_hz),
        rf_center_est_hz=rf_center_est_hz,
        desired_output_iq_len=desired_output_iq_len,
        user_peak_power_fraction=user_peak_power_fraction,
        rx_input_power=rx_input_power,
        max_tones=model.max_tones,
        seed=seed,
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
    tx_metadata["controller_input_lengths"] = [int(v) for v in lengths.tolist()]
    tx_metadata["controller_rx_input_power"] = rx_input_power

    return {
        "tx_config": cfg,
        "model_outputs": y,
        "tx_iq": tx_result.iq.astype(np.complex64),
        "tx_metadata": tx_metadata,
        "rx_input_power": rx_input_power,
        "rf_center_est_hz": rf_center_est_hz,
    }


if __name__ == "__main__":
    iq_a = (np.random.randn(900) + 1j * np.random.randn(900)).astype(np.complex64)
    iq_b = (np.random.randn(1700) + 1j * np.random.randn(1700)).astype(np.complex64)
    iq_c = (np.random.randn(3200) + 1j * np.random.randn(3200)).astype(np.complex64)

    model = TonePulseTXControlNetVarLen(in_ch=4, base_ch=16, n_scalar_features=6, max_tones=8)
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
