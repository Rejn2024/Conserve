#!/usr/bin/env python3
"""
tx_controller_net_stft_3input.py

PyTorch-based TX controller for advanced_link_skdsp_v3_tx_flexible.py

This version:
- takes THREE IQ captures/windows as input
- estimates RF centre frequency from the intake IQ
- preprocesses each intake IQ by:
    * DC subtraction
    * robust MAD scaling
    * STFT
- uses a shared-weight 3-branch STFT ResNet
- predicts TX control settings, including multipath taps
- calls advanced_link_skdsp_v3_tx_flexible.build_tx_iq_object(...)
- supports:
    * message text payloads
    * message=None => random bit payloads
    * user-specified desired output IQ length

Important behavior
------------------
- If message is not None:
    build_controlled_tx_waveform_from_three_inputs(...) transmits that message.
- If message is None:
    build_controlled_tx_waveform_from_three_inputs(...) transmits random bits.
    The number of random bits can be given explicitly via random_bits.
    If random_bits is None, a default random payload length is chosen.

- desired_output_iq_len controls target_num_samples in the flexible TX module.

Dependencies
------------
- numpy
- scipy
- torch
- advanced_link_skdsp_v3_tx_flexible.py (must be importable)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

import advanced_link_skdsp_v4_robust as txflex


FEC_NONE = "none"
FEC_REP3 = "rep3"
FEC_CONV = "conv"

NOISE_COLORS = ["white", "pink", "brown", "blue", "violet"]
FADING_MODES = ["none", "rician_block", "rayleigh_block", "multipath_static"]


# =============================================================================
# Basic DSP helpers for preprocessing
# =============================================================================

def measure_iq_power(iq: np.ndarray) -> float:
    iq = np.asarray(iq)
    if iq.size == 0:
        return 0.0
    return float(np.mean(np.abs(iq) ** 2))


def robust_mad_scale(x: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    xr = np.concatenate([x.real, x.imag])
    med = np.median(xr)
    mad = np.median(np.abs(xr - med))
    sigma = 1.4826 * mad + eps
    return (x / sigma).astype(np.complex64), float(sigma)


def subtract_dc(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)).astype(np.complex64)


# =============================================================================
# RF centre estimation from intake data
# =============================================================================

def estimate_rf_center_from_intake_iq(
    iq: np.ndarray,
    sample_rate_hz: float,
    nominal_rf_center_hz: float = 0.0,
) -> Dict[str, float]:
    """
    Estimate RF centre from intake IQ using FFT centroid and peak.
    """
    x = np.asarray(iq, dtype=np.complex64)
    x = subtract_dc(x)
    x, sigma = robust_mad_scale(x)

    spec = np.fft.fftshift(np.fft.fft(x))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0 / sample_rate_hz))
    mag2 = np.abs(spec) ** 2

    peak_idx = int(np.argmax(mag2))
    peak_hz = float(freqs[peak_idx])

    denom = float(np.sum(mag2)) + 1e-12
    centroid_hz = float(np.sum(freqs * mag2) / denom)

    est_digital_center_hz = 0.5 * peak_hz + 0.5 * centroid_hz
    rf_center_est_hz = float(nominal_rf_center_hz + est_digital_center_hz)

    return {
        "dc_i": float(np.mean(iq.real)),
        "dc_q": float(np.mean(iq.imag)),
        "mad_sigma": float(sigma),
        "peak_hz": peak_hz,
        "centroid_hz": centroid_hz,
        "digital_center_est_hz": est_digital_center_hz,
        "rf_center_est_hz": rf_center_est_hz,
    }


# =============================================================================
# STFT preprocessing
# =============================================================================

def stft_feature_map(
    iq: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 128,
    noverlap: int = 96,
    nfft: int = 128,
) -> Dict[str, np.ndarray]:
    x = np.asarray(iq, dtype=np.complex64)
    x = subtract_dc(x)
    x, sigma = robust_mad_scale(x)

    f, t, Z = signal.stft(
        x,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
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

    return {
        "stft_mag": mag,
        "stft_phase": phase,
        "stft_centroid_plane_hz": centroid_plane,
        "stft_freqs_hz": f.astype(np.float32),
        "mad_sigma": np.float32(sigma),
    }


def preprocess_single_iq(
    iq: np.ndarray,
    input_sample_rate_hz: float,
    target_len: int = 1024,
    nperseg: int = 128,
    noverlap: int = 96,
    nfft: int = 128,
) -> Tuple[np.ndarray, Dict[str, float]]:
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) != target_len:
        iq = signal.resample(iq, target_len).astype(np.complex64)

    rf_est = estimate_rf_center_from_intake_iq(
        iq=iq,
        sample_rate_hz=input_sample_rate_hz,
        nominal_rf_center_hz=0.0,
    )
    stft_feats = stft_feature_map(
        iq=iq,
        sample_rate_hz=input_sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )

    mag = stft_feats["stft_mag"]
    phase = stft_feats["stft_phase"]
    centroid_plane = stft_feats["stft_centroid_plane_hz"] / max(input_sample_rate_hz, 1.0)

    sr_norm = np.float32(np.log10(max(input_sample_rate_hz, 1.0)) / 10.0)
    sr_plane = np.full_like(mag, sr_norm, dtype=np.float32)

    rf_center_norm = np.float32(rf_est["digital_center_est_hz"] / max(input_sample_rate_hz, 1.0))
    rf_plane = np.full_like(mag, rf_center_norm, dtype=np.float32)

    feat = np.stack(
        [
            mag,
            phase,
            centroid_plane.astype(np.float32),
            sr_plane,
            rf_plane,
        ],
        axis=0,
    )  # [C,F,T]

    scalar_info = {
        "dc_i": rf_est["dc_i"],
        "dc_q": rf_est["dc_q"],
        "mad_sigma": rf_est["mad_sigma"],
        "peak_hz": rf_est["peak_hz"],
        "centroid_hz": rf_est["centroid_hz"],
        "digital_center_est_hz": rf_est["digital_center_est_hz"],
        "rf_center_est_hz": rf_est["rf_center_est_hz"],
        "rx_input_power": measure_iq_power(iq),
    }

    return feat, scalar_info


# =============================================================================
# Config
# =============================================================================

@dataclass
class TXControlConfig:
    sample_rate_hz: float
    rf_center_hz: float
    carrier_hz: float
    fec: str
    interleave: bool
    interleave_rows: int
    snr_db: float
    noise_color: str
    fading_mode: str
    fading_block_len: int
    rician_k_db: float
    freq_offset: float
    timing_offset: float
    burst_probability: float
    burst_len_min: int
    burst_len_max: int
    burst_power_ratio_db: float
    burst_color: str
    multipath_taps: List[complex]
    seed: int = 1


# =============================================================================
# 2D ResNet blocks
# =============================================================================

class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class STFTResNetBackbone(nn.Module):
    def __init__(self, in_ch: int = 5, base_ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock2D(base_ch, base_ch, stride=1),
            ResidualBlock2D(base_ch, base_ch, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock2D(base_ch, base_ch * 2, stride=2),
            ResidualBlock2D(base_ch * 2, base_ch * 2, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock2D(base_ch * 2, base_ch * 4, stride=2),
            ResidualBlock2D(base_ch * 4, base_ch * 4, stride=1),
        )
        self.layer4 = nn.Sequential(
            ResidualBlock2D(base_ch * 4, base_ch * 8, stride=2),
            ResidualBlock2D(base_ch * 8, base_ch * 8, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = base_ch * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return x


# =============================================================================
# Three-input network
# =============================================================================

class TXControlNetSTFT3Input(nn.Module):
    """
    Three-branch STFT ResNet.
    Each branch shares weights.
    """
    def __init__(self, in_ch: int = 5, base_ch: int = 32, n_scalar_features: int = 15):
        super().__init__()
        self.backbone = STFTResNetBackbone(in_ch=in_ch, base_ch=base_ch)
        feat_dim = self.backbone.feature_dim

        self.scalar_proj = nn.Sequential(
            nn.Linear(n_scalar_features, 64),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(3 * feat_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        self.fec_head = nn.Linear(128, 3)
        self.noise_color_head = nn.Linear(128, 5)
        self.fading_mode_head = nn.Linear(128, 4)
        self.burst_color_head = nn.Linear(128, 5)
        self.interleave_head = nn.Linear(128, 1)

        self.sample_rate_scale_head = nn.Linear(128, 1)
        self.rf_center_delta_head = nn.Linear(128, 1)
        self.carrier_hz_head = nn.Linear(128, 1)
        self.snr_db_head = nn.Linear(128, 1)
        self.fading_block_len_head = nn.Linear(128, 1)
        self.rician_k_db_head = nn.Linear(128, 1)
        self.freq_offset_head = nn.Linear(128, 1)
        self.timing_offset_head = nn.Linear(128, 1)
        self.burst_probability_head = nn.Linear(128, 1)
        self.burst_power_ratio_db_head = nn.Linear(128, 1)
        self.multipath_taps_head = nn.Linear(128, 6)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        scalar_side: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        z3 = self.backbone(x3)
        zs = self.scalar_proj(scalar_side)

        z = torch.cat([z1, z2, z3, zs], dim=-1)
        z = self.fusion(z)

        out = {
            "fec_logits": self.fec_head(z),
            "noise_color_logits": self.noise_color_head(z),
            "fading_mode_logits": self.fading_mode_head(z),
            "burst_color_logits": self.burst_color_head(z),
            "interleave_logit": self.interleave_head(z),

            "sample_rate_scale": 0.5 + 2.0 * torch.sigmoid(self.sample_rate_scale_head(z)),
            "rf_center_delta_hz": 250_000.0 * torch.tanh(self.rf_center_delta_head(z)),
            "carrier_hz_norm": torch.tanh(self.carrier_hz_head(z)),
            "snr_db": 40.0 * torch.sigmoid(self.snr_db_head(z)),
            "fading_block_len_norm": torch.sigmoid(self.fading_block_len_head(z)),
            "rician_k_db": 20.0 * torch.sigmoid(self.rician_k_db_head(z)),
            "freq_offset": 0.005 * torch.tanh(self.freq_offset_head(z)),
            "timing_offset": 1.0 + 0.002 * torch.tanh(self.timing_offset_head(z)),
            "burst_probability": 1e-3 * torch.sigmoid(self.burst_probability_head(z)),
            "burst_power_ratio_db": 30.0 * torch.sigmoid(self.burst_power_ratio_db_head(z)),
            "multipath_taps_raw": self.multipath_taps_head(z),
        }
        return out


# =============================================================================
# Output decoding
# =============================================================================

def _nearest_fading_block_len(x_norm: float) -> int:
    choices = np.array([128, 256, 512, 1024, 2048, 4096], dtype=np.int64)
    idx = int(round(x_norm * (len(choices) - 1)))
    idx = max(0, min(idx, len(choices) - 1))
    return int(choices[idx])


def decode_multipath_taps(raw: torch.Tensor) -> List[complex]:
    v = raw.detach().cpu().numpy().reshape(-1)
    t0 = (1.0 + 0.25 * np.tanh(v[0])) + 1j * (0.10 * np.tanh(v[1]))
    t1 = (0.40 * np.tanh(v[2])) + 1j * (0.40 * np.tanh(v[3]))
    t2 = (0.20 * np.tanh(v[4])) + 1j * (0.20 * np.tanh(v[5]))
    return [complex(t0), complex(t1), complex(t2)]


def decode_tx_control_outputs(
    model_out: Dict[str, torch.Tensor],
    intake_sample_rate_hz: float,
    rf_center_est_hz: float,
) -> TXControlConfig:
    fec_idx = int(torch.argmax(model_out["fec_logits"], dim=-1).item())
    noise_idx = int(torch.argmax(model_out["noise_color_logits"], dim=-1).item())
    fading_idx = int(torch.argmax(model_out["fading_mode_logits"], dim=-1).item())
    burst_idx = int(torch.argmax(model_out["burst_color_logits"], dim=-1).item())

    fec = [FEC_NONE, FEC_REP3, FEC_CONV][fec_idx]
    noise_color = NOISE_COLORS[noise_idx]
    fading_mode = FADING_MODES[fading_idx]
    burst_color = NOISE_COLORS[burst_idx]
    interleave = bool(torch.sigmoid(model_out["interleave_logit"]).item() > 0.5)

    sample_rate_hz = float(intake_sample_rate_hz * model_out["sample_rate_scale"].item())
    rf_center_hz = float(rf_center_est_hz + model_out["rf_center_delta_hz"].item())
    carrier_hz = float(model_out["carrier_hz_norm"].item()) * 0.45 * sample_rate_hz

    return TXControlConfig(
        sample_rate_hz=sample_rate_hz,
        rf_center_hz=rf_center_hz,
        carrier_hz=carrier_hz,
        fec=fec,
        interleave=interleave,
        interleave_rows=8,
        snr_db=float(model_out["snr_db"].item()),
        noise_color=noise_color,
        fading_mode=fading_mode,
        fading_block_len=_nearest_fading_block_len(float(model_out["fading_block_len_norm"].item())),
        rician_k_db=float(model_out["rician_k_db"].item()),
        freq_offset=float(model_out["freq_offset"].item()),
        timing_offset=float(model_out["timing_offset"].item()),
        burst_probability=float(model_out["burst_probability"].item()),
        burst_len_min=16,
        burst_len_max=64,
        burst_power_ratio_db=float(model_out["burst_power_ratio_db"].item()),
        burst_color=burst_color,
        multipath_taps=decode_multipath_taps(model_out["multipath_taps_raw"]),
        seed=1,
    )


# =============================================================================
# Helper for random-bit defaulting
# =============================================================================

def _default_random_bits_for_length(desired_output_iq_len: Optional[int]) -> int:
    """
    Choose a reasonable random payload length when message=None and random_bits is not supplied.
    """
    if desired_output_iq_len is None:
        return 4096
    # Keep this moderate and variable with requested output length.
    # Clamp to a practical range.
    est = max(1024, min(16384, desired_output_iq_len // 4))
    return int(est)


# =============================================================================
# Main three-input adaptive builder using advanced_link_skdsp_v3_tx_flexible
# =============================================================================

def build_controlled_tx_waveform_from_three_inputs(
    model: TXControlNetSTFT3Input,
    rx_iq_window_1: np.ndarray,
    rx_iq_window_2: np.ndarray,
    rx_iq_window_3: np.ndarray,
    intake_sample_rate_hz: float,
    message: Optional[str],
    desired_output_iq_len: Optional[int] = None,
    random_bits: Optional[int] = None,
    random_seed: int = 1,
    user_peak_power_fraction: Optional[float] = None,
    device: str = "cpu",
) -> Dict[str, object]:
    """
    Build a TX IQ object using three intake IQ windows.

    Parameters
    ----------
    model:
        The 3-input STFT controller network.

    rx_iq_window_1, rx_iq_window_2, rx_iq_window_3:
        Three intake IQ arrays.

    intake_sample_rate_hz:
        Sample rate of the intake IQ arrays.

    message:
        If not None, transmit this message.
        If None, use random bits.

    desired_output_iq_len:
        Desired number of TX IQ samples to generate.
        Passed through to advanced_link_skdsp_v3_tx_flexible.build_tx_iq_object(...)
        as target_num_samples.

    random_bits:
        Used only when message is None.
        If None and message is None, a default random bit length is chosen.

    random_seed:
        Seed for random-bit payload generation when message is None.

    user_peak_power_fraction:
        If not None, override the model and constrain the final TX IQ so that

            peak(|tx|^2) <= user_peak_power_fraction * rx_input_power

        where rx_input_power is the average received input power across the
        three intake IQ windows.

    Returns
    -------
    Dict containing:
        - preprocessing info
        - estimated RF centre
        - predicted tx_config
        - built TX IQ object
        - whole TX metadata
        - power limit info
    """
    feat1, info1 = preprocess_single_iq(rx_iq_window_1, intake_sample_rate_hz)
    feat2, info2 = preprocess_single_iq(rx_iq_window_2, intake_sample_rate_hz)
    feat3, info3 = preprocess_single_iq(rx_iq_window_3, intake_sample_rate_hz)

    x1 = torch.from_numpy(feat1).unsqueeze(0).to(device)
    x2 = torch.from_numpy(feat2).unsqueeze(0).to(device)
    x3 = torch.from_numpy(feat3).unsqueeze(0).to(device)

    rf_center_est_hz = float(np.mean([
        info1["rf_center_est_hz"],
        info2["rf_center_est_hz"],
        info3["rf_center_est_hz"],
    ]))

    rx_input_power = float(np.mean([
        info1["rx_input_power"],
        info2["rx_input_power"],
        info3["rx_input_power"],
    ]))

    scalar_side = torch.tensor(
        [[
            np.log10(max(intake_sample_rate_hz, 1.0)) / 10.0,
            info1["digital_center_est_hz"] / max(intake_sample_rate_hz, 1.0),
            info2["digital_center_est_hz"] / max(intake_sample_rate_hz, 1.0),
            info3["digital_center_est_hz"] / max(intake_sample_rate_hz, 1.0),
            np.mean([info1["mad_sigma"], info2["mad_sigma"], info3["mad_sigma"]]),
            np.mean([info1["dc_i"], info2["dc_i"], info3["dc_i"]]),
            np.mean([info1["dc_q"], info2["dc_q"], info3["dc_q"]]),
            rx_input_power,
            info1["peak_hz"] / max(intake_sample_rate_hz, 1.0),
            info2["peak_hz"] / max(intake_sample_rate_hz, 1.0),
            info3["peak_hz"] / max(intake_sample_rate_hz, 1.0),
            info1["centroid_hz"] / max(intake_sample_rate_hz, 1.0),
            info2["centroid_hz"] / max(intake_sample_rate_hz, 1.0),
            info3["centroid_hz"] / max(intake_sample_rate_hz, 1.0),
            rf_center_est_hz / max(intake_sample_rate_hz, 1.0),
        ]],
        dtype=torch.float32,
    ).to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        y = model(x1, x2, x3, scalar_side)

    tx_cfg = decode_tx_control_outputs(
        model_out=y,
        intake_sample_rate_hz=float(intake_sample_rate_hz),
        rf_center_est_hz=rf_center_est_hz,
    )

    use_random_bits = message is None
    if use_random_bits and random_bits is None:
        random_bits = _default_random_bits_for_length(desired_output_iq_len)

    tx_result = txflex.build_tx_iq_object(
        message=message,
        random_bits=random_bits if use_random_bits else None,
        random_seed=random_seed,
        target_num_samples=desired_output_iq_len,
        fec=tx_cfg.fec,
        interleave=tx_cfg.interleave,
        interleave_rows=tx_cfg.interleave_rows,
        sps=8,
        beta=0.35,
        span=6,
        sample_rate_hz=tx_cfg.sample_rate_hz,
        rf_center_hz=tx_cfg.rf_center_hz,
        carrier_hz=tx_cfg.carrier_hz,
        snr_db=tx_cfg.snr_db,
        noise_color=tx_cfg.noise_color,
        freq_offset=tx_cfg.freq_offset,
        timing_offset=tx_cfg.timing_offset,
        fading_mode=tx_cfg.fading_mode,
        fading_block_len=tx_cfg.fading_block_len,
        rician_k_db=tx_cfg.rician_k_db,
        multipath_taps=tx_cfg.multipath_taps,
        burst_probability=tx_cfg.burst_probability,
        burst_len_min=tx_cfg.burst_len_min,
        burst_len_max=tx_cfg.burst_len_max,
        burst_power_ratio_db=tx_cfg.burst_power_ratio_db,
        burst_color=tx_cfg.burst_color,
        # idle_gap_bits=0,
        seed=tx_cfg.seed,
    )

    tx_iq = tx_result.iq.astype(np.complex64)
    applied_peak_fraction = None
    power_limit_info = {
        "rx_input_power": rx_input_power,
        "peak_limit_applied": False,
        "user_peak_power_fraction": user_peak_power_fraction,
        "pre_limit_peak_power": float(np.max(np.abs(tx_iq) ** 2)) if len(tx_iq) else 0.0,
        "post_limit_peak_power": float(np.max(np.abs(tx_iq) ** 2)) if len(tx_iq) else 0.0,
        "applied_amplitude_scale": 1.0,
    }

    if user_peak_power_fraction is not None:
        if user_peak_power_fraction <= 0:
            raise ValueError("user_peak_power_fraction must be > 0")

        peak_cap = float(user_peak_power_fraction) * rx_input_power
        pre_peak = float(np.max(np.abs(tx_iq) ** 2)) if len(tx_iq) else 0.0

        if pre_peak > 0.0 and pre_peak > peak_cap:
            scale = math.sqrt(peak_cap / pre_peak)
            tx_iq = (tx_iq * scale).astype(np.complex64)
            post_peak = float(np.max(np.abs(tx_iq) ** 2))
            power_limit_info = {
                "rx_input_power": rx_input_power,
                "peak_limit_applied": True,
                "user_peak_power_fraction": float(user_peak_power_fraction),
                "peak_cap": peak_cap,
                "pre_limit_peak_power": pre_peak,
                "post_limit_peak_power": post_peak,
                "applied_amplitude_scale": float(scale),
            }
        else:
            power_limit_info = {
                "rx_input_power": rx_input_power,
                "peak_limit_applied": True,
                "user_peak_power_fraction": float(user_peak_power_fraction),
                "peak_cap": peak_cap,
                "pre_limit_peak_power": pre_peak,
                "post_limit_peak_power": pre_peak,
                "applied_amplitude_scale": 1.0,
            }

        applied_peak_fraction = float(user_peak_power_fraction)

    # Keep metadata aligned with the actual returned IQ
    tx_metadata = dict(tx_result.metadata)
    tx_metadata["actual_num_samples"] = int(len(tx_iq))
    tx_metadata["avg_power"] = float(np.mean(np.abs(tx_iq) ** 2)) if len(tx_iq) else 0.0
    tx_metadata["peak_power"] = float(np.max(np.abs(tx_iq) ** 2)) if len(tx_iq) else 0.0
    tx_metadata["user_peak_power_fraction"] = applied_peak_fraction

    return {
        "input_info_1": info1,
        "input_info_2": info2,
        "input_info_3": info3,
        "rf_center_est_hz": rf_center_est_hz,
        "rx_input_power": rx_input_power,
        "tx_config": tx_cfg,
        "model_outputs": y,
        "tx_iq": tx_iq,
        "tx_metadata": tx_metadata,
        "payload_mode": "random_bits" if use_random_bits else "message",
        "random_bits_used": random_bits if use_random_bits else None,
        "desired_output_iq_len": desired_output_iq_len,
        "power_limit_info": power_limit_info,
    }


# =============================================================================
# CLI export helper
# =============================================================================

def complex_taps_to_arg(taps: List[complex]) -> str:
    return ",".join(f"{t.real:+.6f}{t.imag:+.6f}j" for t in taps)


def tx_config_to_cli_args(
    cfg: TXControlConfig,
    message: Optional[str],
    desired_output_iq_len: Optional[int] = None,
    random_bits: Optional[int] = None,
    random_seed: int = 1,
    output_path: str = "tx_controlled.npy",
) -> List[str]:
    args = [
        "tx",
        "--output", output_path,
        "--sample-rate-hz", str(cfg.sample_rate_hz),
        "--rf-center-hz", str(cfg.rf_center_hz),
        "--carrier-hz", str(cfg.carrier_hz),
        "--fec", cfg.fec,
        "--snr-db", str(cfg.snr_db),
        "--noise-color", cfg.noise_color,
        "--fading-mode", cfg.fading_mode,
        "--fading-block-len", str(cfg.fading_block_len),
        "--rician-k-db", str(cfg.rician_k_db),
        "--freq-offset", str(cfg.freq_offset),
        "--timing-offset", str(cfg.timing_offset),
        "--burst-probability", str(cfg.burst_probability),
        "--burst-len-min", str(cfg.burst_len_min),
        "--burst-len-max", str(cfg.burst_len_max),
        "--burst-power-ratio-db", str(cfg.burst_power_ratio_db),
        "--burst-color", cfg.burst_color,
        "--multipath-taps", complex_taps_to_arg(cfg.multipath_taps),
    ]

    if desired_output_iq_len is not None:
        args.extend(["--target-num-samples", str(desired_output_iq_len)])

    if message is None:
        if random_bits is None:
            random_bits = _default_random_bits_for_length(desired_output_iq_len)
        args.extend(["--random-bits", str(random_bits), "--random-seed", str(random_seed)])
    else:
        args.extend(["--message", message])

    if cfg.interleave:
        args.append("--interleave")

    return args


if __name__ == "__main__":
    iq1 = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
    iq2 = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
    iq3 = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)

    model = TXControlNetSTFT3Input(in_ch=5, base_ch=16, n_scalar_features=15)

    # Example 1: message payload, explicit output length
    result_msg = build_controlled_tx_waveform_from_three_inputs(
            model=model,
            rx_iq_window_1=iq1,
            rx_iq_window_2=iq2,
            rx_iq_window_3=iq3,
            intake_sample_rate_hz=1_000_000.0,
            message="Adaptive transmitter test packet",
            desired_output_iq_len=20000,
            user_peak_power_fraction=0.10,
            device="cpu",
        )

    print("Message-mode RF estimate:", result_msg["rf_center_est_hz"])
    print("Message-mode TX config:")
    print(result_msg["tx_config"])
    print("Message-mode output samples:", len(result_msg["tx_iq"]))
    print("Message-mode payload mode:", result_msg["payload_mode"])

    # Example 2: random-bit payload, explicit output length
    result_rand = build_controlled_tx_waveform_from_three_inputs(model=model,
                                                                rx_iq_window_1=iq1,
                                                                rx_iq_window_2=iq2,
                                                                rx_iq_window_3=iq3,
                                                                intake_sample_rate_hz=1_000_000.0,
                                                                message=None,
                                                                desired_output_iq_len=25000,
                                                                random_bits=4096,
                                                                random_seed=7,
                                                                user_peak_power_fraction=0.10,
                                                                device="cpu",
                                                            )

    print("Random-bit mode RF estimate:", result_rand["rf_center_est_hz"])
    print("Random-bit mode TX config:")
    print(result_rand["tx_config"])
    print("Random-bit output samples:", len(result_rand["tx_iq"]))
    print("Random-bit payload mode:", result_rand["payload_mode"])
    print("Random bits used:", result_rand["random_bits_used"])