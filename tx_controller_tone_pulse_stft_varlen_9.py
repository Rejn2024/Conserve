#!/usr/bin/env python3
"""
PyTorch TX controller for advanced_link_skdsp_v7_robust.build_tone_pulse_iq_object
with variable-length IQ inputs and explicit tone-phase controls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import advanced_link_skdsp_v7_robust as txflex


NOISE_COLORS = ["white", "pink", "brown", "blue", "violet"]
FADING_MODES = ["none", "rician_block", "rayleigh_block", "multipath_static"]
DEFAULT_COMPLEX_DTYPE = txflex.DEFAULT_COMPLEX_DTYPE

FIRST_PASS_SCALAR_FEATURE_SCHEMA_VERSION = "first_pass_scalar_features_v1"
FIRST_PASS_TIME_PEAK_COUNT = 3
FIRST_PASS_SCALAR_FEATURE_NAMES: Tuple[str, ...] = (
    "packet_start_frac",
    "packet_end_frac",
    "packet_duration_frac",
    "active_frame_frac",
    "time_energy_centroid_frac",
    "time_energy_spread_frac",
    "time_peak_0_center_frac",
    "time_peak_0_height_norm",
    "time_peak_1_center_frac",
    "time_peak_1_height_norm",
    "time_peak_2_center_frac",
    "time_peak_2_height_norm",
    "center_freq_frac",
    "occupied_bw_frac",
    "low_edge_frac",
    "high_edge_frac",
    "spectral_energy_concentration",
    "rx_power_db",
    "noise_floor_db",
    "snr_est_db",
    "papr_db",
    "sample_rate_ratio",
    "packet_geometry_valid",
    "spectral_geometry_valid",
    "power_context_valid",
    "time_peaks_valid",
)
N_FIRST_PASS_SCALAR_FEATURES = len(FIRST_PASS_SCALAR_FEATURE_NAMES)


def _complex64_for_limited_op(x: torch.Tensor) -> torch.Tensor:
    return txflex._complex64_for_limited_op(x)


def _restore_complex_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return txflex._restore_complex_dtype(x, dtype)


def _as_complex_tensor(iq: Union[torch.Tensor, List[complex]]) -> torch.Tensor:
    if isinstance(iq, torch.Tensor):
        return iq.to(dtype=DEFAULT_COMPLEX_DTYPE)
    return torch.as_tensor(iq, dtype=DEFAULT_COMPLEX_DTYPE)


def _sanitize_complex_iq(x: torch.Tensor, clamp: float = 1.0e6) -> torch.Tensor:
    """Replace non-finite IQ samples and clamp extreme outliers before DSP ops."""
    return torch.complex(
        torch.clamp(torch.nan_to_num(x.real, nan=0.0, posinf=0.0, neginf=0.0), min=-float(clamp), max=float(clamp)),
        torch.clamp(torch.nan_to_num(x.imag, nan=0.0, posinf=0.0, neginf=0.0), min=-float(clamp), max=float(clamp)),
    ).to(dtype=x.dtype)


def _sanitize_stft_feature(feat: torch.Tensor, clamp: float = 1.0e6) -> torch.Tensor:
    """Keep STFT feature planes finite for policy/distribution construction."""
    feat = torch.nan_to_num(feat.to(dtype=torch.float32), nan=0.0, posinf=float(clamp), neginf=-float(clamp))
    return torch.clamp(feat, min=-float(clamp), max=float(clamp))


def measure_iq_power(iq: Union[torch.Tensor, List[complex]]) -> torch.Tensor:
    x = _complex64_for_limited_op(_sanitize_complex_iq(_as_complex_tensor(iq)))
    if x.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.mean(torch.abs(x) ** 2).to(dtype=torch.float32)


def robust_mad_scale(x: Union[torch.Tensor, List[complex]], eps: float = 1e-12) -> torch.Tensor:
    xt = _sanitize_complex_iq(_as_complex_tensor(x))
    work = _complex64_for_limited_op(xt)
    xr = torch.cat([work.real, work.imag], dim=0).to(dtype=torch.float32)
    med = torch.median(xr)
    mad = torch.median(torch.abs(xr - med))
    sigma = 1.4826 * mad + eps
    return _restore_complex_dtype(work / sigma, xt.dtype)


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
        return iq_batch.to(dtype=DEFAULT_COMPLEX_DTYPE)

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
    return torch.stack(padded, dim=0).to(dtype=DEFAULT_COMPLEX_DTYPE)


def compute_first_pass_scalar_features_for_iq_batch(
    iq_batch: Union[torch.Tensor, Sequence[Union[torch.Tensor, List[complex]]]],
    sample_rate_hz: float,
    *,
    reference_sample_rate_hz: float = 1.0e9,
    time_bins: int = 64,
) -> Dict[str, torch.Tensor]:
    """Build the first-pass scalar observation vector from batched IQ.

    The vector follows ``FIRST_PASS_SCALAR_FEATURE_NAMES`` and is intended to be
    fused beside STFT/CNN embeddings.  All features are computed from the IQ
    samples available to the policy: packet/time geometry, coarse spectral
    overlap, power context, sample-rate scale, and validity flags.
    """

    x = _complex64_for_limited_op(_sanitize_complex_iq(_as_batch_complex_tensor(iq_batch)))
    if x.ndim != 2:
        raise ValueError(f"iq_batch must have shape [B, T], got {tuple(x.shape)}")
    if x.shape[1] < 2:
        x = F.pad(x, (0, 2 - int(x.shape[1])))

    device = x.device
    dtype = torch.float32
    batch_size, n_samples = int(x.shape[0]), int(x.shape[1])
    eps = torch.tensor(1e-12, dtype=dtype, device=device)
    sample_rate = float(max(sample_rate_hz, 1.0))

    power = torch.nan_to_num((x.real.square() + x.imag.square()).to(dtype), nan=0.0, posinf=0.0, neginf=0.0)
    total_power = torch.sum(power, dim=1).clamp_min(eps)
    mean_power = torch.mean(power, dim=1).clamp_min(eps)
    peak_power = torch.max(power, dim=1).values.clamp_min(eps)
    noise_floor = torch.quantile(power, 0.10, dim=1).clamp_min(eps)

    # Packet geometry from a robust per-row threshold.  The threshold floats
    # above a quiet-floor estimate so low-SNR packets still produce conservative
    # active masks without peeking at labels or decode results.
    threshold = noise_floor + 0.10 * (peak_power - noise_floor).clamp_min(0.0)
    active = power > threshold[:, None]
    valid_packet = active.any(dim=1).to(dtype)
    idx = torch.arange(n_samples, dtype=dtype, device=device)
    idx_norm = idx / float(max(n_samples - 1, 1))

    active_i = active.to(torch.int64)
    start_idx = torch.argmax(active_i, dim=1).to(dtype)
    last_from_end = torch.argmax(torch.flip(active_i, dims=[1]), dim=1).to(dtype)
    end_idx = (n_samples - 1) - last_from_end
    start_frac = torch.where(valid_packet > 0, start_idx / float(max(n_samples - 1, 1)), torch.zeros_like(start_idx))
    end_frac = torch.where(valid_packet > 0, end_idx / float(max(n_samples - 1, 1)), torch.zeros_like(end_idx))
    duration_frac = torch.where(
        valid_packet > 0,
        (end_idx - start_idx + 1.0).clamp_min(0.0) / float(max(n_samples, 1)),
        torch.zeros_like(end_idx),
    )
    active_frame_frac = active.to(dtype).mean(dim=1)

    time_centroid = torch.sum(power * idx_norm[None, :], dim=1) / total_power
    time_spread = torch.sqrt(torch.sum(power * (idx_norm[None, :] - time_centroid[:, None]).square(), dim=1) / total_power + eps)

    n_bins = max(1, min(int(time_bins), n_samples))
    pooled = F.adaptive_avg_pool1d(power.unsqueeze(1), n_bins).squeeze(1)
    pooled_peak = torch.max(pooled, dim=1, keepdim=True).values.clamp_min(eps)
    peak_heights, peak_bin_idx = torch.topk(
        pooled,
        k=min(FIRST_PASS_TIME_PEAK_COUNT, n_bins),
        dim=1,
        largest=True,
        sorted=True,
    )
    if peak_heights.shape[1] < FIRST_PASS_TIME_PEAK_COUNT:
        pad_width = FIRST_PASS_TIME_PEAK_COUNT - int(peak_heights.shape[1])
        peak_heights = F.pad(peak_heights, (0, pad_width))
        peak_bin_idx = F.pad(peak_bin_idx, (0, pad_width))
    peak_centers = peak_bin_idx.to(dtype) / float(max(n_bins - 1, 1))
    peak_heights_norm = peak_heights / pooled_peak
    valid_time_peaks = (pooled_peak.squeeze(1) > eps).to(dtype)

    spectrum = torch.fft.fftshift(torch.fft.fft(x, dim=1), dim=1)
    spec_power = torch.nan_to_num((spectrum.real.square() + spectrum.imag.square()).to(dtype), nan=0.0, posinf=0.0, neginf=0.0)
    spec_total = torch.sum(spec_power, dim=1).clamp_min(eps)
    freq_frac = torch.fft.fftshift(torch.fft.fftfreq(n_samples, d=1.0 / sample_rate).to(device=device, dtype=dtype)) / sample_rate
    center_freq_frac = torch.sum(spec_power * freq_frac[None, :], dim=1) / spec_total
    cdf = torch.cumsum(spec_power, dim=1) / spec_total[:, None]
    low_idx = torch.argmax((cdf >= 0.05).to(torch.int64), dim=1)
    high_idx = torch.argmax((cdf >= 0.95).to(torch.int64), dim=1)
    low_edge_frac = freq_frac[low_idx]
    high_edge_frac = freq_frac[high_idx]
    occupied_bw_frac = (high_edge_frac - low_edge_frac).clamp_min(0.0)
    spectral_concentration = torch.max(spec_power, dim=1).values / spec_total
    spectral_valid = (spec_total > eps).to(dtype)

    rx_power_db = 10.0 * torch.log10(mean_power)
    noise_floor_db = 10.0 * torch.log10(noise_floor)
    snr_est_db = 10.0 * torch.log10((mean_power - noise_floor).clamp_min(eps) / noise_floor)
    papr_db = 10.0 * torch.log10(peak_power / mean_power)
    sample_rate_ratio = torch.full(
        (batch_size,),
        float(sample_rate / max(reference_sample_rate_hz, 1.0)),
        dtype=dtype,
        device=device,
    )
    power_valid = torch.isfinite(mean_power).to(dtype)

    pieces: List[torch.Tensor] = [
        start_frac,
        end_frac,
        duration_frac,
        active_frame_frac,
        time_centroid,
        time_spread,
    ]
    for j in range(FIRST_PASS_TIME_PEAK_COUNT):
        pieces.extend([peak_centers[:, j], peak_heights_norm[:, j]])
    pieces.extend(
        [
            center_freq_frac,
            occupied_bw_frac,
            low_edge_frac,
            high_edge_frac,
            spectral_concentration,
            rx_power_db,
            noise_floor_db,
            snr_est_db,
            papr_db,
            sample_rate_ratio,
            valid_packet,
            spectral_valid,
            power_valid,
            valid_time_peaks,
        ]
    )
    feature = torch.stack(pieces, dim=1).to(dtype=dtype)
    feature = torch.nan_to_num(feature, nan=0.0, posinf=1.0e6, neginf=-1.0e6).clamp(-1.0e6, 1.0e6)
    if feature.shape[1] != N_FIRST_PASS_SCALAR_FEATURES:
        raise RuntimeError(
            f"first-pass scalar feature width {feature.shape[1]} does not match schema width {N_FIRST_PASS_SCALAR_FEATURES}"
        )
    return {
        "scalar_side": feature,
        "feature_names": FIRST_PASS_SCALAR_FEATURE_NAMES,
        "schema_version": FIRST_PASS_SCALAR_FEATURE_SCHEMA_VERSION,
    }


def build_first_pass_scalar_side_from_iq_sections(
    iq_sections: Sequence[Union[torch.Tensor, Sequence[Union[torch.Tensor, List[complex]]]]],
    sample_rate_hz: float,
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Concatenate section IQ and return the first-pass scalar feature matrix."""

    if len(iq_sections) != 3:
        raise ValueError(f"iq_sections must contain exactly 3 IQ batches, got {len(iq_sections)}")
    batches = [_as_batch_complex_tensor(section) for section in iq_sections]
    batch_size = int(batches[0].shape[0])
    if any(int(batch.shape[0]) != batch_size for batch in batches):
        raise ValueError("All IQ sections must use the same batch size")
    max_lens = [int(batch.shape[1]) for batch in batches]
    if any(length <= 0 for length in max_lens):
        raise ValueError("IQ sections must contain at least one sample")
    combined = torch.cat(batches, dim=1)
    if device is not None:
        combined = combined.to(device=device)
    return compute_first_pass_scalar_features_for_iq_batch(combined, sample_rate_hz)["scalar_side"]


def _legacy_scalar_side_from_preprocessed(
    proc: Sequence[Dict[str, torch.Tensor]],
    intake_sample_rate_hz: float,
    device: Union[str, torch.device],
) -> torch.Tensor:
    batch_size = int(proc[0]["feature"].shape[0])
    rx_power_stack = torch.stack([p["rx_power"].to(device) for p in proc], dim=0)
    peak_stack = torch.stack([p["peak_hz"].to(device) for p in proc], dim=0)
    lengths = torch.stack([p["lengths"].to(device) for p in proc], dim=0).transpose(0, 1)
    max_len = torch.clamp(torch.max(lengths, dim=1).values, min=1.0)
    return torch.stack(
        [
            torch.log10(torch.tensor(max(intake_sample_rate_hz, 1.0), dtype=torch.float32, device=device)) / 10.0
            * torch.ones(batch_size, dtype=torch.float32, device=device),
            torch.mean(lengths, dim=1) / max_len,
            torch.std(lengths, dim=1, unbiased=False) / max_len,
            rx_power_stack.mean(dim=0).to(dtype=torch.float32),
            peak_stack.mean(dim=0) / max(intake_sample_rate_hz, 1.0),
            peak_stack.std(dim=0, unbiased=False) / max(intake_sample_rate_hz, 1.0),
        ],
        dim=1,
    ).to(dtype=torch.float32, device=device)


def _expected_model_scalar_features(model: nn.Module) -> int:
    scalar_proj = getattr(model, "scalar_proj", None)
    if scalar_proj is None and hasattr(model, "backbone"):
        scalar_proj = getattr(model.backbone, "scalar_proj", None)
    if scalar_proj is not None:
        first = scalar_proj[0] if isinstance(scalar_proj, nn.Sequential) else scalar_proj
        in_features = getattr(first, "in_features", None)
        if in_features is not None:
            return int(in_features)
    return N_FIRST_PASS_SCALAR_FEATURES


def _fit_scalar_side_width(scalar_side: torch.Tensor, expected_width: int) -> torch.Tensor:
    if scalar_side.shape[1] == expected_width:
        return scalar_side
    if scalar_side.shape[1] > expected_width:
        return scalar_side[:, :expected_width]
    pad = torch.zeros(
        scalar_side.shape[0],
        expected_width - int(scalar_side.shape[1]),
        dtype=scalar_side.dtype,
        device=scalar_side.device,
    )
    return torch.cat([scalar_side, pad], dim=1)


def build_scalar_side_for_model(
    model: nn.Module,
    iq_sections: Sequence[Union[torch.Tensor, Sequence[Union[torch.Tensor, List[complex]]]]],
    intake_sample_rate_hz: float,
    *,
    device: Union[str, torch.device] = "cpu",
    preprocessed: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
) -> torch.Tensor:
    """Return scalar side features with the width expected by ``model``.

    New models default to the first-pass schema.  Legacy six-scalar models keep
    the previous length/rate/power/peak summary for checkpoint compatibility.
    Custom widths receive the first-pass vector truncated or zero-padded.
    """

    expected = _expected_model_scalar_features(model)
    if expected == 6 and preprocessed is not None:
        return _legacy_scalar_side_from_preprocessed(preprocessed, intake_sample_rate_hz, device)
    scalar_side = build_first_pass_scalar_side_from_iq_sections(iq_sections, intake_sample_rate_hz, device=device)
    return _fit_scalar_side_width(scalar_side.to(dtype=torch.float32, device=device), expected)


def preprocess_batched_iq_to_stft_feature(
    iq_batch: Union[torch.Tensor, Sequence[Union[torch.Tensor, List[complex]]]],
    sample_rate_hz: float,
    nperseg: int = 128,
    noverlap: int = 96,
    nfft: int = 128,
) -> Dict[str, torch.Tensor]:
    x_batch = _sanitize_complex_iq(_as_batch_complex_tensor(iq_batch))
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
        "feature": _sanitize_stft_feature(torch.stack(feats, dim=0)),
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
    x_t = _sanitize_complex_iq(_as_complex_tensor(iq))
    if x_t.numel() < 8:
        x_t = F.pad(x_t, (0, 8 - int(x_t.numel())))

    # torch.stft/torch.fft and some reductions have limited complex32 support;
    # run this compatibility segment in complex64, then emit float32 features.
    x_work = _complex64_for_limited_op(x_t)
    x_work = (x_work - torch.mean(x_work)).to(dtype=torch.complex64)
    x_work = _complex64_for_limited_op(robust_mad_scale(x_work))

    nperseg_eff = int(min(nperseg, max(8, x_work.numel())))
    noverlap_eff = int(min(noverlap, max(0, nperseg_eff - 1)))
    hop_length = max(1, nperseg_eff - noverlap_eff)
    window = torch.hann_window(nperseg_eff, dtype=torch.float32, device=x_work.device)
    Z = torch.stft(
        x_work,
        n_fft=nfft,
        hop_length=hop_length,
        win_length=nperseg_eff,
        window=window,
        center=False,
        return_complex=True,
    )

    Z = torch.fft.fftshift(Z, dim=0)
    f = torch.fft.fftshift(torch.fft.fftfreq(nfft, d=1.0 / max(sample_rate_hz, 1.0)).to(x_work.device))

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
        "feature": _sanitize_stft_feature(feat),
        "rx_power": measure_iq_power(iq),
        "peak_hz": f[torch.argmax(torch.mean(power, dim=1))].to(dtype=torch.float32) if power.numel() else torch.tensor(0.0, dtype=torch.float32, device=x_work.device),
    }


@dataclass
class TonePulseControlConfig:
    sample_rate_hz: float
    rf_center_hz: float
    carrier_hz: float
    num_tones: int
    tone_frequencies_hz: List[float]
    tone_frequency_std_hz: List[float]
    tone_amplitudes: List[float]
    tone_initial_phases_rad: List[float]
    tone_phase_offset_rad: float
    pulse_on_samples: int
    pulse_off_samples: int
    pulse_count: int
    start_offset_samples: int
    pulse_phase_rotations_rad: List[float]
    tone_pulse_on_samples: List[int]
    tone_pulse_off_samples: List[int]
    tone_pulse_count: List[int]
    tone_start_offset_samples: List[int]
    tone_pulse_phase_rotations_rad: List[List[float]]
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
    """Encodes one variable-size [B,C,F,T] STFT map into [B,D] with a ResNet backbone."""

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

    def __init__(
        self,
        in_ch: int = 14,
        base_ch: int = 24,
        n_scalar_features: int = N_FIRST_PASS_SCALAR_FEATURES,
        max_tones: int = 8,
        max_pulses: int = 33,
        pulse_phase_ar_components: int = 4,
        pulse_phase_ar_hidden: int = 64,
    ):
        """
        Args:
            in_ch: Number of channels per STFT input map [B, C, F, T].
            base_ch: Base convolution width used by the STFT encoder.
            n_scalar_features: Number of non-image scalar side features fused with STFT features.
            max_tones: Upper bound on synthesized tone count (also output width for tone amplitudes).
            max_pulses: Upper bound on synthesized pulse count (output width for per-pulse phase controls).
            pulse_phase_ar_components: Number of circular mixture components per autoregressive pulse step.
            pulse_phase_ar_hidden: Hidden width for the LSTM pulse-phase mixture decoder.
        """
        super().__init__()
        self.encoder = VarLenSTFTEncoder(in_ch=in_ch, base_ch=base_ch)
        self.max_tones = max_tones
        self.max_pulses = max_pulses
        self.pulse_phase_ar_components = max(1, int(pulse_phase_ar_components))
        self.pulse_phase_ar_hidden = max(8, int(pulse_phase_ar_hidden))
        self.n_windows = 3

        self.window_fusion = nn.Sequential(
            nn.Linear(self.encoder.feature_dim * self.n_windows, self.encoder.feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder.feature_dim * 2, self.encoder.feature_dim),
            nn.ReLU(inplace=True),
        )

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
        self.tone_freq_mean_norm_heads = nn.ModuleList([nn.Linear(96, 1) for _ in range(max_tones)])
        self.tone_freq_std_norm_heads = nn.ModuleList([nn.Linear(96, 1) for _ in range(max_tones)])
        self.tone_power_raw_heads = nn.ModuleList([nn.Linear(96, 1) for _ in range(max_tones)])
        self.tone_phase_rel_heads = nn.ModuleList([nn.Linear(96, 1) for _ in range(max_tones)])
        self.tone_phase_offset_head = nn.Linear(96, 1)
        # Autoregressive circular mixture for pulse-relative phases.  The legacy
        # per-slot scalar heads are retained for checkpoint key compatibility;
        # the active path below uses recurrent LSTM mixture heads.
        self.pulse_phase_rel_heads = nn.ModuleList([nn.Linear(96, 1) for _ in range(max_pulses)])
        self.pulse_phase_ar_init = nn.Linear(96, 2 * self.pulse_phase_ar_hidden)
        self.pulse_phase_ar_step = nn.LSTMCell(4, self.pulse_phase_ar_hidden)
        self.pulse_phase_ar_logits_head = nn.Linear(self.pulse_phase_ar_hidden, self.pulse_phase_ar_components)
        self.pulse_phase_ar_loc_head = nn.Linear(self.pulse_phase_ar_hidden, self.pulse_phase_ar_components)
        self.pulse_phase_ar_concentration_head = nn.Linear(self.pulse_phase_ar_hidden, self.pulse_phase_ar_components)
        self.pulse_phase_offset_head = nn.Linear(96, 1)
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

    def _fuse_window_embeddings(self, emb: List[torch.Tensor]) -> torch.Tensor:
        if len(emb) != self.n_windows:
            raise ValueError(f"Expected exactly {self.n_windows} STFT windows, got {len(emb)}")
        z_cat = torch.cat(emb, dim=-1)
        return self.window_fusion(z_cat)

    def _pulse_index_features(self, step: int, batch_size: int, device: torch.device) -> torch.Tensor:
        denom = max(1, self.max_pulses - 1)
        frac = float(step) / float(denom)
        angle = 2.0 * math.pi * frac
        return torch.stack(
            [
                torch.full((batch_size,), math.sin(angle), dtype=torch.float32, device=device),
                torch.full((batch_size,), math.cos(angle), dtype=torch.float32, device=device),
            ],
            dim=-1,
        )

    def pulse_phase_autoregressive_mixture(
        self,
        z: torch.Tensor,
        teacher_phases: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Unroll an LSTM autoregressive circular mixture over pulse phases.

        Args:
            z: Fused controller features with shape [B, 96].
            teacher_phases: Optional [B, max_pulses] phase tensor.  When given,
                step i+1 is conditioned on the wrapped phase from step i, enabling
                correct autoregressive log-prob evaluation under teacher forcing.
            deterministic: If False and no teacher is provided, sample the next
                previous phase from the mixture before advancing the LSTM state.
                The returned ``pulse_phase_rel_rad`` is the unrolled phase
                vector used by callers.
        """

        batch_size = int(z.shape[0])
        device = z.device
        init_state = self.pulse_phase_ar_init(z)
        hidden_init, cell_init = torch.chunk(init_state, 2, dim=-1)
        hidden = torch.tanh(hidden_init)
        cell = torch.tanh(cell_init)
        prev_phase = torch.zeros(batch_size, dtype=torch.float32, device=device)
        logits_steps: List[torch.Tensor] = []
        loc_steps: List[torch.Tensor] = []
        conc_steps: List[torch.Tensor] = []
        phase_steps: List[torch.Tensor] = []

        teacher = None
        if teacher_phases is not None:
            teacher = teacher_phases.to(device=device, dtype=torch.float32)
            if teacher.ndim == 1:
                teacher = teacher.unsqueeze(0)
            if teacher.shape[0] != batch_size:
                raise ValueError("teacher_phases batch size must match z")

        for step in range(int(self.max_pulses)):
            step_feat = self._pulse_index_features(step, batch_size, device)
            ar_in = torch.cat(
                [torch.sin(prev_phase).unsqueeze(-1), torch.cos(prev_phase).unsqueeze(-1), step_feat],
                dim=-1,
            )
            hidden, cell = self.pulse_phase_ar_step(ar_in, (hidden, cell))
            logits = self.pulse_phase_ar_logits_head(hidden)
            loc = _wrap_phase_rad(torch.pi * torch.tanh(self.pulse_phase_ar_loc_head(hidden)))
            concentration = F.softplus(self.pulse_phase_ar_concentration_head(hidden)) + 1.0e-3
            probs = torch.softmax(logits, dim=-1)
            mean_sin = torch.sum(probs * torch.sin(loc), dim=-1)
            mean_cos = torch.sum(probs * torch.cos(loc), dim=-1)
            mean_phase = torch.atan2(mean_sin, mean_cos)

            if teacher is not None and step < teacher.shape[1]:
                current_phase = _wrap_phase_rad(teacher[:, step])
            elif deterministic:
                current_phase = mean_phase
            else:
                comp = torch.distributions.Categorical(logits=logits).sample()
                picked_loc = loc.gather(1, comp.unsqueeze(-1)).squeeze(-1)
                picked_conc = concentration.gather(1, comp.unsqueeze(-1)).squeeze(-1)
                std = torch.rsqrt(picked_conc.clamp_min(1.0e-3))
                current_phase = _wrap_phase_rad(picked_loc + std * torch.randn_like(picked_loc))

            logits_steps.append(logits)
            loc_steps.append(loc)
            conc_steps.append(concentration)
            phase_steps.append(current_phase)

            prev_phase = current_phase

        return {
            "pulse_phase_rel_rad": torch.stack(phase_steps, dim=1),
            "pulse_phase_rel_mix_logits": torch.stack(logits_steps, dim=1),
            "pulse_phase_rel_mix_loc_rad": torch.stack(loc_steps, dim=1),
            "pulse_phase_rel_mix_concentration": torch.stack(conc_steps, dim=1),
        }

    def pulse_phase_autoregressive_log_prob(
        self,
        z: torch.Tensor,
        phases: torch.Tensor,
        *,
        shifts: Tuple[int, ...] = (-1, 0, 1),
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Teacher-forced log probability under the autoregressive phase mixture."""

        phases = phases.to(device=z.device, dtype=torch.float32)
        if phases.ndim == 1:
            phases = phases.unsqueeze(0)
        mix = self.pulse_phase_autoregressive_mixture(z, teacher_phases=phases, deterministic=True)
        logits = mix["pulse_phase_rel_mix_logits"]
        loc = mix["pulse_phase_rel_mix_loc_rad"]
        concentration = mix["pulse_phase_rel_mix_concentration"]
        scale = torch.rsqrt(concentration.clamp_min(1.0e-3))
        value = _wrap_phase_rad(phases[:, : self.max_pulses]).unsqueeze(-1)
        shifted_terms = []
        for shift in shifts:
            shifted = value + float(shift) * 2.0 * torch.pi
            shifted_terms.append(torch.distributions.Normal(loc, scale).log_prob(shifted))
        wrapped_component_logp = torch.logsumexp(torch.stack(shifted_terms, dim=0), dim=0)
        log_mix = torch.log_softmax(logits, dim=-1)
        per_step = torch.logsumexp(log_mix + wrapped_component_logp, dim=-1)
        return per_step.sum(dim=-1), mix

    def pulse_phase_autoregressive_entropy(self, z: torch.Tensor) -> torch.Tensor:
        """Approximate entropy of the autoregressive mixture using deterministic feedback."""

        mix = self.pulse_phase_autoregressive_mixture(z, deterministic=True)
        logits = mix["pulse_phase_rel_mix_logits"]
        concentration = mix["pulse_phase_rel_mix_concentration"]
        cat = torch.distributions.Categorical(logits=logits)
        scale = torch.rsqrt(concentration.clamp_min(1.0e-3))
        component_entropy = torch.distributions.Normal(torch.zeros_like(scale), scale).entropy()
        weights = torch.softmax(logits, dim=-1)
        return (cat.entropy() + torch.sum(weights * component_entropy, dim=-1)).sum(dim=-1)

    def model_outputs_from_fused(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        pulse_phase_mix = self.pulse_phase_autoregressive_mixture(z, deterministic=True)
        return {
            "noise_color_logits": self.noise_color_head(z),
            "fading_mode_logits": self.fading_mode_head(z),
            "burst_color_logits": self.burst_color_head(z),
            "sample_rate_scale": 0.5 + 2.0 * torch.sigmoid(self.sample_rate_scale_head(z)),
            "rf_center_delta_hz": 500_000.0 * torch.tanh(self.rf_center_delta_head(z)),
            "carrier_hz_norm": 0.45 * torch.tanh(self.carrier_norm_head(z)),
            "num_tones_cont": 1.0 + (self.max_tones - 1.0) * torch.sigmoid(self.num_tones_head(z)),
            "tone_freq_mean_norms": torch.cat(
                [0.45 * torch.tanh(h(z)) for h in self.tone_freq_mean_norm_heads],
                dim=1,
            ),
            "tone_freq_std_norms": torch.cat(
                [0.33 * torch.sigmoid(h(z)) for h in self.tone_freq_std_norm_heads],
                dim=1,
            ),
            # Keep a raw per-tone amplitude control (varlen_4 style).
            # `tone_power_logits` is retained as a compatibility alias for existing RL wiring.
            "tone_amp_raw": torch.cat([h(z) for h in self.tone_power_raw_heads], dim=1),
            "tone_power_logits": torch.cat([h(z) for h in self.tone_power_raw_heads], dim=1),
            "tone_phase_rel_rad": torch.cat([torch.pi * torch.tanh(h(z)) for h in self.tone_phase_rel_heads], dim=1),
            "tone_phase_offset_rad": torch.pi * torch.tanh(self.tone_phase_offset_head(z)),
            **pulse_phase_mix,
            "pulse_phase_offset_rad": torch.pi * torch.tanh(self.pulse_phase_offset_head(z)),
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

    def forward(self, stft_feature_list: List[torch.Tensor], scalar_side: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = [self.encoder(x) for x in stft_feature_list]
        z_stft = self._fuse_window_embeddings(emb)
        z = torch.cat([z_stft, self.scalar_proj(scalar_side)], dim=-1)
        z = self.fusion(z)
        return self.model_outputs_from_fused(z)


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
        n_scalar_features: int = N_FIRST_PASS_SCALAR_FEATURES,
        max_tones: int = 8,
        max_pulses: int = 33,
        action_std_init: float = 0.25,
        pulse_phase_ar_components: int = 4,
        pulse_phase_ar_hidden: int = 64,
    ):
        super().__init__()
        self.backbone = TonePulseTXControlNetVarLen(
            in_ch=in_ch,
            base_ch=base_ch,
            n_scalar_features=n_scalar_features,
            max_tones=max_tones,
            max_pulses=max_pulses,
            pulse_phase_ar_components=pulse_phase_ar_components,
            pulse_phase_ar_hidden=pulse_phase_ar_hidden,
        )
        self.max_tones = max_tones
        self.max_pulses = max_pulses
        self.action_dim = int(action_dim) if action_dim is not None else (11 + 4 * int(max_tones) + int(max_pulses))
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self._action_std_floor = 1e-4
        self._init_log_std = math.log(max(action_std_init, self._action_std_floor))
        self.log_std_head = nn.Linear(96, self.action_dim)
        self.value_head = nn.Linear(96, 1)

    def _encode(
        self,
        stft_feature_list: List[torch.Tensor],
        scalar_side: torch.Tensor,
    ) -> torch.Tensor:
        emb = [self.backbone.encoder(x) for x in stft_feature_list]
        z_stft = self.backbone._fuse_window_embeddings(emb)
        z = torch.cat([z_stft, self.backbone.scalar_proj(scalar_side)], dim=-1)
        return self.backbone.fusion(z)

    def _action_mean_from_model_out(self, y: Dict[str, torch.Tensor]) -> torch.Tensor:
        def _expected_index(logits: torch.Tensor) -> torch.Tensor:
            probs = torch.softmax(logits, dim=-1)
            idx = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
            return (probs * idx).sum(dim=-1, keepdim=True)

        pieces = [
            _expected_index(y["noise_color_logits"]),
            _expected_index(y["fading_mode_logits"]),
            y["rf_center_delta_hz"],
            y["carrier_hz_norm"],
            y["num_tones_cont"],
            y["tone_freq_mean_norms"],
            y["tone_freq_std_norms"],
            y["tone_power_logits"],
            y["tone_phase_rel_rad"],
            y["tone_phase_offset_rad"],
            y["pulse_phase_rel_rad"],
            y["pulse_phase_offset_rad"],
            y["pulse_on_samples"],
            y["pulse_off_samples"],
            y["pulse_count"],
            y["start_offset"],
        ]
        action_mean = torch.cat(pieces, dim=-1)
        if action_mean.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action mean width {action_mean.shape[-1]} does not match action_dim {self.action_dim}. "
                "Set action_dim to 11 + 4*max_tones + max_pulses (default) or a matching custom width."
            )
        return action_mean

    def _continuous_action_mean(
        self,
        stft_feature_list: List[torch.Tensor],
        scalar_side: torch.Tensor,
    ) -> torch.Tensor:
        return self._action_mean_from_model_out(self.backbone(stft_feature_list=stft_feature_list, scalar_side=scalar_side))

    def _log_std_from_features(self, fused_features: torch.Tensor) -> torch.Tensor:
        raw = self.log_std_head(fused_features)
        return self._init_log_std + torch.tanh(raw)

    def _pulse_phase_action_slice(self) -> slice:
        start = 5 + (4 * int(self.max_tones)) + 1
        return slice(start, start + int(self.max_pulses))

    def _policy_tensors(
        self,
        stft_feature_list: List[torch.Tensor],
        scalar_side: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        fused = self._encode(stft_feature_list=stft_feature_list, scalar_side=scalar_side)
        y = self.backbone.model_outputs_from_fused(fused)
        action_mean = self._action_mean_from_model_out(y)
        value = self.value_head(fused).squeeze(-1)
        log_std = self._log_std_from_features(fused)
        return action_mean, value, log_std, fused, y

    def forward(
        self,
        stft_feature_list: List[torch.Tensor],
        scalar_side: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value, log_std, _, _ = self._policy_tensors(
            stft_feature_list=stft_feature_list,
            scalar_side=scalar_side,
        )
        return action_mean, value, log_std

    def forward_observation(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scalar_side = observation.get("scalar_side")
        if scalar_side is None:
            stft_list = observation["stft_feature_list"]
            if not stft_list:
                raise ValueError("observation['stft_feature_list'] must contain at least one tensor")
            ref = stft_list[0]
            n_scalar_features = self.backbone.scalar_proj[0].in_features
            scalar_side = torch.zeros(
                ref.shape[0],
                n_scalar_features,
                dtype=ref.dtype,
                device=ref.device,
            )
        return self.forward(
            stft_feature_list=observation["stft_feature_list"],
            scalar_side=scalar_side,
        )

    def _policy_tensors_from_observation(
        self,
        observation: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        scalar_side = observation.get("scalar_side")
        if scalar_side is None:
            stft_list = observation["stft_feature_list"]
            if not stft_list:
                raise ValueError("observation['stft_feature_list'] must contain at least one tensor")
            ref = stft_list[0]
            scalar_side = torch.zeros(
                ref.shape[0],
                self.backbone.scalar_proj[0].in_features,
                dtype=ref.dtype,
                device=ref.device,
            )
        return self._policy_tensors(stft_feature_list=observation["stft_feature_list"], scalar_side=scalar_side)

    def _action_distribution(self, action_mean: torch.Tensor, log_std: torch.Tensor) -> torch.distributions.Normal:
        std = torch.exp(log_std).clamp_min(self._action_std_floor)
        action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        std = torch.nan_to_num(std, nan=self._action_std_floor, posinf=20.0, neginf=self._action_std_floor).clamp_min(self._action_std_floor)
        return torch.distributions.Normal(loc=action_mean, scale=std)

    def _normal_log_prob_excluding_pulse_phase(
        self,
        dist: torch.distributions.Normal,
        action: torch.Tensor,
    ) -> torch.Tensor:
        per_dim = dist.log_prob(action)
        pulse_slice = self._pulse_phase_action_slice()
        mask = torch.ones(per_dim.shape[-1], dtype=torch.bool, device=per_dim.device)
        mask[pulse_slice] = False
        return per_dim[..., mask].sum(dim=-1)

    def _normal_entropy_excluding_pulse_phase(self, dist: torch.distributions.Normal) -> torch.Tensor:
        entropy = dist.entropy()
        pulse_slice = self._pulse_phase_action_slice()
        mask = torch.ones(entropy.shape[-1], dtype=torch.bool, device=entropy.device)
        mask[pulse_slice] = False
        return entropy[..., mask].sum(dim=-1)

    def _sample_hybrid_action(
        self,
        *,
        action_mean: torch.Tensor,
        log_std: torch.Tensor,
        fused: torch.Tensor,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._action_distribution(action_mean=action_mean, log_std=log_std)
        action = action_mean.clone() if deterministic else dist.rsample()
        pulse_slice = self._pulse_phase_action_slice()
        if deterministic:
            pulse_phases = action_mean[:, pulse_slice]
            pulse_log_prob, _ = self.backbone.pulse_phase_autoregressive_log_prob(fused, pulse_phases)
        else:
            mix = self.backbone.pulse_phase_autoregressive_mixture(fused, deterministic=False)
            pulse_phases = mix["pulse_phase_rel_rad"]
            pulse_log_prob, _ = self.backbone.pulse_phase_autoregressive_log_prob(fused, pulse_phases)
            action[:, pulse_slice] = pulse_phases
        log_prob = self._normal_log_prob_excluding_pulse_phase(dist, action) + pulse_log_prob
        entropy = self._normal_entropy_excluding_pulse_phase(dist) + self.backbone.pulse_phase_autoregressive_entropy(fused)
        return action, log_prob, entropy

    def act(
        self,
        stft_feature_list: List[torch.Tensor],
        scalar_side: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value, log_std, fused, _ = self._policy_tensors(stft_feature_list=stft_feature_list, scalar_side=scalar_side)
        action, log_prob, _ = self._sample_hybrid_action(
            action_mean=action_mean,
            log_std=log_std,
            fused=fused,
            deterministic=deterministic,
        )
        return action, log_prob, value

    def evaluate_actions(
        self,
        stft_feature_list: List[torch.Tensor],
        scalar_side: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value, log_std, fused, _ = self._policy_tensors(stft_feature_list=stft_feature_list, scalar_side=scalar_side)
        dist = self._action_distribution(action_mean=action_mean, log_std=log_std)
        pulse_slice = self._pulse_phase_action_slice()
        pulse_log_prob, _ = self.backbone.pulse_phase_autoregressive_log_prob(fused, actions[:, pulse_slice])
        log_prob = self._normal_log_prob_excluding_pulse_phase(dist, actions) + pulse_log_prob
        entropy = self._normal_entropy_excluding_pulse_phase(dist) + self.backbone.pulse_phase_autoregressive_entropy(fused)
        return log_prob, entropy, value

    def get_action_value_logp(
        self,
        observation: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value, log_std, fused, _ = self._policy_tensors_from_observation(observation)
        if action is None:
            action, log_prob, _ = self._sample_hybrid_action(
                action_mean=action_mean,
                log_std=log_std,
                fused=fused,
                deterministic=deterministic,
            )
        else:
            dist = self._action_distribution(action_mean=action_mean, log_std=log_std)
            pulse_slice = self._pulse_phase_action_slice()
            pulse_log_prob, _ = self.backbone.pulse_phase_autoregressive_log_prob(fused, action[:, pulse_slice])
            log_prob = self._normal_log_prob_excluding_pulse_phase(dist, action) + pulse_log_prob
        return action, value, log_prob



def _finite_scalar_value(
    value: Union[torch.Tensor, float, int],
    default: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """Return a finite Python float, falling back when RL/model output is NaN/Inf."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            out = float(default)
        else:
            out = float(value.detach().reshape(-1)[0].item())
    else:
        out = float(value)
    if not math.isfinite(out):
        out = float(default)
    if min_value is not None:
        out = max(float(min_value), out)
    if max_value is not None:
        out = min(float(max_value), out)
    return out


def _finite_model_scalar(
    model_out: Dict[str, torch.Tensor],
    key: str,
    default: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    return _finite_scalar_value(model_out[key], default=default, min_value=min_value, max_value=max_value)


def _finite_vector(
    value: torch.Tensor,
    *,
    default: float = 0.0,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sanitize vector model outputs before converting them to Python config values."""
    y = value.detach().reshape(-1).to(dtype=dtype)
    y = torch.nan_to_num(y, nan=float(default), posinf=float(default), neginf=float(default))
    if min_value is not None or max_value is not None:
        y = torch.clamp(y, min=min_value, max=max_value)
    return y


def _nearest_block_len(x_norm: float) -> int:
    choices = torch.tensor([128, 256, 512, 1024, 2048, 4096], dtype=torch.int64)
    idx = int(round(x_norm * (len(choices) - 1)))
    idx = max(0, min(idx, len(choices) - 1))
    return int(choices[idx].item())


def _wrap_phase_rad(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _apply_per_pulse_phase_rotation(
    iq: Union[torch.Tensor, List[complex]],
    *,
    pulse_on_samples: int,
    pulse_off_samples: int,
    pulse_count: int,
    start_offset_samples: int,
    pulse_phase_rotations_rad: Sequence[float],
) -> torch.Tensor:
    x = _as_complex_tensor(iq).clone()
    if x.numel() == 0 or pulse_count <= 0:
        return x

    cycle = max(1, int(pulse_on_samples) + int(pulse_off_samples))
    for k in range(int(pulse_count)):
        start = int(start_offset_samples) + (k * cycle)
        end = min(start + int(pulse_on_samples), int(x.numel()))
        if start >= int(x.numel()):
            break
        theta = float(pulse_phase_rotations_rad[min(k, len(pulse_phase_rotations_rad) - 1)])
        rot = torch.complex(
            torch.cos(torch.tensor(theta, dtype=torch.float32, device=x.device)),
            torch.sin(torch.tensor(theta, dtype=torch.float32, device=x.device)),
        ).to(dtype=DEFAULT_COMPLEX_DTYPE)
        x[start:end] = x[start:end] * rot
    return x


def _apply_per_tone_pulse_phase_rotation(
    iq: Union[torch.Tensor, List[complex]],
    *,
    sample_rate_hz: float,
    carrier_hz: float,
    tone_frequencies_hz: Sequence[float],
    tone_amplitudes: Sequence[float],
    tone_initial_phases_rad: Sequence[float],
    tone_pulse_on_samples: Sequence[int],
    tone_pulse_off_samples: Sequence[int],
    tone_pulse_count: Sequence[int],
    tone_start_offset_samples: Sequence[int],
    tone_pulse_phase_rotations_rad: Sequence[Sequence[float]],
) -> torch.Tensor:
    """Apply per-tone pulse schedules and per-tone/per-pulse phase rotations."""
    x = _as_complex_tensor(iq)
    n = int(x.numel())
    if n == 0 or len(tone_frequencies_hz) == 0:
        return torch.zeros_like(x)

    t = torch.arange(n, dtype=torch.float32, device=x.device) / max(float(sample_rate_hz), 1.0)
    y = torch.zeros(n, dtype=DEFAULT_COMPLEX_DTYPE, device=x.device)
    two_pi = float(2.0 * torch.pi)

    num_tones = len(tone_frequencies_hz)
    for k in range(num_tones):
        amp = float(tone_amplitudes[k]) if k < len(tone_amplitudes) else 0.0
        if amp == 0.0:
            continue
        phase0 = float(tone_initial_phases_rad[k]) if k < len(tone_initial_phases_rad) else 0.0
        tone_hz = float(carrier_hz) + float(tone_frequencies_hz[k])
        base = torch.complex(
            torch.cos((two_pi * tone_hz) * t + phase0),
            torch.sin((two_pi * tone_hz) * t + phase0),
        ).to(dtype=DEFAULT_COMPLEX_DTYPE) * amp

        on = max(1, int(tone_pulse_on_samples[k] if k < len(tone_pulse_on_samples) else 1))
        off = max(0, int(tone_pulse_off_samples[k] if k < len(tone_pulse_off_samples) else 0))
        count = max(0, int(tone_pulse_count[k] if k < len(tone_pulse_count) else 0))
        start0 = max(0, int(tone_start_offset_samples[k] if k < len(tone_start_offset_samples) else 0))
        phases = tone_pulse_phase_rotations_rad[k] if k < len(tone_pulse_phase_rotations_rad) else [0.0]
        if count <= 0:
            continue
        cycle = max(1, on + off)

        mask = torch.zeros(n, dtype=torch.float32, device=x.device)
        phase_vec = torch.zeros(n, dtype=torch.float32, device=x.device)
        for p in range(count):
            start = start0 + p * cycle
            if start >= n:
                break
            end = min(start + on, n)
            theta = float(phases[min(p, len(phases) - 1)]) if len(phases) else 0.0
            mask[start:end] = 1.0
            phase_vec[start:end] = theta
        rot = torch.complex(torch.cos(phase_vec), torch.sin(phase_vec)).to(dtype=DEFAULT_COMPLEX_DTYPE)
        y = y + (base * rot * mask.to(dtype=DEFAULT_COMPLEX_DTYPE))
    return y


def decode_tone_pulse_config(
    model_out: Dict[str, torch.Tensor],
    intake_sample_rate_hz: float,
    rf_center_est_hz: float,
    desired_output_iq_len: Optional[int],
    user_peak_power_fraction: Optional[float],
    rx_input_power: float,
    max_tones: int,
    max_pulses: int,
    seed: int,
) -> TonePulseControlConfig:
    noise_color = NOISE_COLORS[int(torch.argmax(model_out["noise_color_logits"], dim=-1).item())]
    fading_mode = FADING_MODES[int(torch.argmax(model_out["fading_mode_logits"], dim=-1).item())]
    burst_color = NOISE_COLORS[int(torch.argmax(model_out["burst_color_logits"], dim=-1).item())]

    sample_rate_scale = _finite_model_scalar(
        model_out, "sample_rate_scale", default=1.0, min_value=0.1, max_value=10.0
    )
    rf_center_delta_hz = _finite_model_scalar(model_out, "rf_center_delta_hz", default=0.0)
    carrier_hz_norm = _finite_model_scalar(
        model_out, "carrier_hz_norm", default=0.0, min_value=-0.49, max_value=0.49
    )
    sample_rate_hz = float(max(1.0, intake_sample_rate_hz * sample_rate_scale))
    rf_center_hz = float(rf_center_est_hz + rf_center_delta_hz)
    carrier_hz = float(carrier_hz_norm) * sample_rate_hz

    num_tones_cont = _finite_model_scalar(
        model_out, "num_tones_cont", default=1.0, min_value=1.0, max_value=float(max_tones)
    )
    num_tones = int(round(num_tones_cont))
    num_tones = max(1, min(num_tones, max_tones))

    tone_freq_mean_norms = _finite_vector(
        model_out["tone_freq_mean_norms"],
        default=0.0,
        min_value=-0.49,
        max_value=0.49,
    )[:num_tones]
    tone_freq_std_norms = _finite_vector(
        model_out["tone_freq_std_norms"],
        default=0.0,
        min_value=0.0,
        max_value=0.5,
    )[:num_tones]
    tone_freq_means_hz = tone_freq_mean_norms * sample_rate_hz
    tone_frequency_std_hz = (tone_freq_std_norms * sample_rate_hz).to(dtype=torch.float64).tolist()
    tone_frequencies_hz = [
        float(
            torch.clamp(
                mu_f,
                min=-0.49 * sample_rate_hz,
                max=0.49 * sample_rate_hz,
            ).item()
        )
        for mu_f in tone_freq_means_hz
    ]

    tone_amp_raw = _finite_vector(model_out.get("tone_amp_raw", model_out["tone_power_logits"]), default=0.0)[:num_tones]
    # Match varlen_4 mapping: independent per-tone amplitudes in ~[0.05, 1.0].
    tone_amplitudes = (0.05 + 0.95 * torch.sigmoid(tone_amp_raw)).to(dtype=torch.float64).tolist()
    tone_phase_rel = _finite_vector(model_out["tone_phase_rel_rad"], default=0.0, dtype=torch.float64)[:num_tones]
    tone_phase_offset_rad = _finite_model_scalar(model_out, "tone_phase_offset_rad", default=0.0)
    tone_initial_phases_rad = torch.zeros(num_tones, dtype=torch.float64)
    if num_tones > 1:
        tone_initial_phases_rad[1:] = torch.cumsum(tone_phase_rel[1:], dim=0)
    tone_initial_phases_rad = _wrap_phase_rad(tone_initial_phases_rad.to(dtype=torch.float32)).to(dtype=torch.float64).tolist()

    peak_power = None if user_peak_power_fraction is None else float(user_peak_power_fraction) * float(rx_input_power)

    pulse_on_samples = int(
        round(_finite_model_scalar(model_out, "pulse_on_samples", default=128.0, min_value=1.0))
    )
    pulse_off_samples = int(
        round(_finite_model_scalar(model_out, "pulse_off_samples", default=0.0, min_value=0.0))
    )
    pulse_count = int(
        round(
            _finite_model_scalar(
                model_out, "pulse_count", default=1.0, min_value=1.0, max_value=float(max_pulses)
            )
        )
    )
    start_offset_samples = int(
        round(_finite_model_scalar(model_out, "start_offset", default=0.0, min_value=0.0))
    )
    pulse_count = max(1, min(pulse_count, max_pulses))
    pulse_phase_rel = _finite_vector(model_out["pulse_phase_rel_rad"], default=0.0, dtype=torch.float64)[:pulse_count]
    pulse_phase_offset_rad = _finite_model_scalar(model_out, "pulse_phase_offset_rad", default=0.0)
    pulse_phase_rotations_rad = torch.zeros(pulse_count, dtype=torch.float64)
    pulse_phase_rotations_rad[0] = pulse_phase_offset_rad
    if pulse_count > 1:
        pulse_phase_rotations_rad[1:] = pulse_phase_offset_rad + torch.cumsum(pulse_phase_rel[1:], dim=0)
    pulse_phase_rotations_rad = _wrap_phase_rad(pulse_phase_rotations_rad.to(dtype=torch.float32)).to(dtype=torch.float64).tolist()

    tone_pulse_on_samples = []
    tone_pulse_off_samples = []
    tone_pulse_count = []
    tone_start_offset_samples = []
    tone_pulse_phase_rotations_rad = []
    for k in range(num_tones):
        on_k = max(1, pulse_on_samples + int(round((k - (num_tones - 1) / 2.0) * 0.1 * pulse_on_samples)))
        off_k = max(0, pulse_off_samples + int(round((k - (num_tones - 1) / 2.0) * 0.1 * max(1, pulse_off_samples))))
        count_k = max(1, pulse_count - k)
        start_k = max(0, start_offset_samples + (k * max(1, on_k // 8)))
        if desired_output_iq_len is not None and desired_output_iq_len > 0:
            on_k = min(on_k, desired_output_iq_len)
            start_k = min(start_k, max(0, desired_output_iq_len - 1))
        tone_pulse_on_samples.append(int(on_k))
        tone_pulse_off_samples.append(int(off_k))
        tone_pulse_count.append(int(count_k))
        tone_start_offset_samples.append(int(start_k))
        rel_k = _finite_vector(model_out["pulse_phase_rel_rad"], default=0.0, dtype=torch.float64)[:count_k]
        phase_list_k = torch.zeros(count_k, dtype=torch.float64)
        phase_list_k[0] = tone_phase_offset_rad + pulse_phase_offset_rad + float(tone_initial_phases_rad[k])
        if count_k > 1:
            phase_list_k[1:] = phase_list_k[0] + torch.cumsum(rel_k[1:], dim=0) + (k * 0.25)
        tone_pulse_phase_rotations_rad.append(
            _wrap_phase_rad(phase_list_k.to(dtype=torch.float32)).to(dtype=torch.float64).tolist()
        )

    if desired_output_iq_len is not None and desired_output_iq_len > 0:
        pulse_on_samples = min(pulse_on_samples, desired_output_iq_len)
        start_offset_samples = min(start_offset_samples, max(0, desired_output_iq_len - 1))

    return TonePulseControlConfig(
        sample_rate_hz=sample_rate_hz,
        rf_center_hz=rf_center_hz,
        carrier_hz=carrier_hz,
        num_tones=num_tones,
        tone_frequencies_hz=tone_frequencies_hz,
        tone_frequency_std_hz=tone_frequency_std_hz,
        tone_amplitudes=tone_amplitudes,
        tone_initial_phases_rad=tone_initial_phases_rad,
        tone_phase_offset_rad=tone_phase_offset_rad,
        pulse_on_samples=max(1, pulse_on_samples),
        pulse_off_samples=max(0, pulse_off_samples),
        pulse_count=pulse_count,
        start_offset_samples=max(0, start_offset_samples),
        pulse_phase_rotations_rad=pulse_phase_rotations_rad,
        tone_pulse_on_samples=tone_pulse_on_samples,
        tone_pulse_off_samples=tone_pulse_off_samples,
        tone_pulse_count=tone_pulse_count,
        tone_start_offset_samples=tone_start_offset_samples,
        tone_pulse_phase_rotations_rad=tone_pulse_phase_rotations_rad,
        snr_db=_finite_model_scalar(model_out, "snr_db", default=30.0, min_value=0.0, max_value=100.0),
        noise_color=noise_color,
        freq_offset=_finite_model_scalar(model_out, "freq_offset", default=0.0, min_value=-1.0, max_value=1.0),
        timing_offset=_finite_model_scalar(model_out, "timing_offset", default=1.0, min_value=0.0),
        fading_mode=fading_mode,
        fading_block_len=_nearest_block_len(
            _finite_model_scalar(
                model_out, "fading_block_len_norm", default=0.0, min_value=0.0, max_value=1.0
            )
        ),
        rician_k_db=_finite_model_scalar(model_out, "rician_k_db", default=0.0, min_value=0.0),
        burst_probability=_finite_model_scalar(model_out, "burst_probability", default=0.0, min_value=0.0, max_value=1.0),
        burst_len_min=16,
        burst_len_max=64,
        burst_power_ratio_db=_finite_model_scalar(model_out, "burst_power_ratio_db", default=0.0, min_value=0.0),
        burst_color=burst_color,
        peak_power=peak_power,
        seed=seed,
    )



def _finite_float(value: object, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _finite_int(value: object, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    out = int(round(_finite_float(value, float(default))))
    if min_value is not None:
        out = max(int(min_value), out)
    if max_value is not None:
        out = min(int(max_value), out)
    return out


def _choice_from_override(value: object, choices: Sequence[str], default: str) -> str:
    if isinstance(value, str):
        return value if value in choices else default
    idx = _finite_int(value, 0, min_value=0, max_value=len(choices) - 1)
    return choices[idx]


def _finite_float_list(value: object, *, default: Sequence[float], length: int) -> List[float]:
    if value is None:
        vals: List[float] = []
    elif torch.is_tensor(value):
        vals = [float(x) for x in value.detach().cpu().reshape(-1).tolist()]
    elif isinstance(value, (list, tuple)):
        vals = [float(x) for x in value]
    else:
        vals = [float(value)]

    out: List[float] = []
    default_list = list(default)
    for i in range(length):
        fallback = default_list[i] if i < len(default_list) else (default_list[-1] if default_list else 0.0)
        raw = vals[i] if i < len(vals) else fallback
        out.append(_finite_float(raw, fallback))
    return out


def _derive_tone_pulse_controls(
    *,
    num_tones: int,
    pulse_on_samples: int,
    pulse_off_samples: int,
    pulse_count: int,
    start_offset_samples: int,
    desired_output_iq_len: Optional[int],
    tone_phase_offset_rad: float,
    pulse_phase_offset_rad: float,
    tone_initial_phases_rad: Sequence[float],
    pulse_phase_rel_rad: Sequence[float],
) -> Tuple[List[int], List[int], List[int], List[int], List[List[float]]]:
    tone_pulse_on_samples: List[int] = []
    tone_pulse_off_samples: List[int] = []
    tone_pulse_count: List[int] = []
    tone_start_offset_samples: List[int] = []
    tone_pulse_phase_rotations_rad: List[List[float]] = []

    for k in range(max(1, int(num_tones))):
        on_k = max(1, int(round(pulse_on_samples + (k - (num_tones - 1) / 2.0) * 0.1 * pulse_on_samples)))
        off_k = max(0, int(round(pulse_off_samples + (k - (num_tones - 1) / 2.0) * 0.1 * max(1, pulse_off_samples))))
        count_k = max(1, int(pulse_count) - k)
        start_k = max(0, int(start_offset_samples) + (k * max(1, on_k // 8)))
        if desired_output_iq_len is not None and desired_output_iq_len > 0:
            on_k = min(on_k, int(desired_output_iq_len))
            start_k = min(start_k, max(0, int(desired_output_iq_len) - 1))

        tone_pulse_on_samples.append(int(on_k))
        tone_pulse_off_samples.append(int(off_k))
        tone_pulse_count.append(int(count_k))
        tone_start_offset_samples.append(int(start_k))

        phase0 = float(tone_phase_offset_rad) + float(pulse_phase_offset_rad)
        if k < len(tone_initial_phases_rad):
            phase0 += float(tone_initial_phases_rad[k])
        phase_list = torch.zeros(count_k, dtype=torch.float64)
        phase_list[0] = phase0
        if count_k > 1:
            rel_vals = _finite_float_list(pulse_phase_rel_rad, default=[0.0], length=count_k)
            rel_t = torch.as_tensor(rel_vals, dtype=torch.float64)
            phase_list[1:] = phase_list[0] + torch.cumsum(rel_t[1:], dim=0) + (k * 0.25)
        tone_pulse_phase_rotations_rad.append(
            _wrap_phase_rad(phase_list.to(dtype=torch.float32)).to(dtype=torch.float64).tolist()
        )

    return (
        tone_pulse_on_samples,
        tone_pulse_off_samples,
        tone_pulse_count,
        tone_start_offset_samples,
        tone_pulse_phase_rotations_rad,
    )


def apply_tone_pulse_action_overrides(
    cfg: TonePulseControlConfig,
    overrides: Optional[Dict[str, object]],
    *,
    desired_output_iq_len: Optional[int],
    max_tones: int,
    max_pulses: int,
) -> TonePulseControlConfig:
    """Return ``cfg`` with one row's RL action overrides applied.

    The overrides are deliberately constrained to the existing simulated tone-pulse
    parameter space.  This keeps batched rollouts differentiable where possible
    while ensuring each environment row can receive distinct waveform controls.
    """

    if not overrides:
        return cfg

    sample_rate_hz = _finite_float(overrides.get("sample_rate_hz", cfg.sample_rate_hz), cfg.sample_rate_hz)
    sample_rate_hz = max(1.0, sample_rate_hz)
    if "sample_rate_scale" in overrides:
        sample_rate_hz = max(1.0, cfg.sample_rate_hz * _finite_float(overrides["sample_rate_scale"], 1.0))

    rf_center_hz = _finite_float(overrides.get("rf_center_hz", cfg.rf_center_hz), cfg.rf_center_hz)
    if "rf_center_delta_hz" in overrides:
        rf_center_hz = cfg.rf_center_hz + _finite_float(overrides["rf_center_delta_hz"], 0.0)

    carrier_hz = _finite_float(overrides.get("carrier_hz", cfg.carrier_hz), cfg.carrier_hz)
    if "carrier_hz_norm" in overrides:
        carrier_hz = max(-0.49, min(0.49, _finite_float(overrides["carrier_hz_norm"], 0.0))) * sample_rate_hz

    num_tones = _finite_int(overrides.get("num_tones", cfg.num_tones), cfg.num_tones, min_value=1, max_value=max_tones)
    if "tone_frequencies_hz" in overrides:
        tone_frequencies_hz = _finite_float_list(overrides["tone_frequencies_hz"], default=cfg.tone_frequencies_hz, length=num_tones)
    elif "tone_freq_mean_norms" in overrides:
        norms = _finite_float_list(overrides["tone_freq_mean_norms"], default=[f / sample_rate_hz for f in cfg.tone_frequencies_hz], length=num_tones)
        tone_frequencies_hz = [max(-0.49, min(0.49, n)) * sample_rate_hz for n in norms]
    elif "base_f" in overrides or "spacing" in overrides:
        base_f = _finite_float(overrides.get("base_f", cfg.tone_frequencies_hz[0] if cfg.tone_frequencies_hz else 0.0), 0.0)
        spacing = _finite_float(overrides.get("spacing", 0.0), 0.0)
        center = (num_tones - 1) / 2.0
        tone_frequencies_hz = [base_f + (i - center) * spacing for i in range(num_tones)]
    else:
        tone_frequencies_hz = _finite_float_list(cfg.tone_frequencies_hz, default=[0.0], length=num_tones)
    tone_frequencies_hz = [max(-0.49 * sample_rate_hz, min(0.49 * sample_rate_hz, f)) for f in tone_frequencies_hz]

    if "tone_frequency_std_hz" in overrides:
        tone_frequency_std_hz = _finite_float_list(overrides["tone_frequency_std_hz"], default=cfg.tone_frequency_std_hz, length=num_tones)
    elif "tone_freq_std_norms" in overrides:
        tone_frequency_std_hz = [max(0.0, n) * sample_rate_hz for n in _finite_float_list(overrides["tone_freq_std_norms"], default=[0.0], length=num_tones)]
    else:
        tone_frequency_std_hz = _finite_float_list(cfg.tone_frequency_std_hz, default=[0.0], length=num_tones)

    if "tone_amplitudes" in overrides:
        tone_amplitudes = _finite_float_list(overrides["tone_amplitudes"], default=cfg.tone_amplitudes, length=num_tones)
    elif "amp_raw" in overrides:
        raw = torch.as_tensor(_finite_float_list(overrides["amp_raw"], default=[0.0], length=num_tones), dtype=torch.float32)
        tone_amplitudes = (0.05 + 0.95 * torch.sigmoid(raw)).to(dtype=torch.float64).tolist()
    elif "tone_amp_raw" in overrides:
        raw = torch.as_tensor(_finite_float_list(overrides["tone_amp_raw"], default=[0.0], length=num_tones), dtype=torch.float32)
        tone_amplitudes = (0.05 + 0.95 * torch.sigmoid(raw)).to(dtype=torch.float64).tolist()
    else:
        tone_amplitudes = _finite_float_list(cfg.tone_amplitudes, default=[1.0], length=num_tones)
    tone_amplitudes = [max(0.0, float(a)) for a in tone_amplitudes]

    tone_initial_phases_rad = _finite_float_list(
        overrides.get("tone_initial_phases_rad", cfg.tone_initial_phases_rad),
        default=cfg.tone_initial_phases_rad,
        length=num_tones,
    )
    tone_initial_phases_rad = _wrap_phase_rad(torch.as_tensor(tone_initial_phases_rad, dtype=torch.float32)).to(dtype=torch.float64).tolist()
    tone_phase_offset_rad = _finite_float(overrides.get("tone_phase_offset_rad", cfg.tone_phase_offset_rad), cfg.tone_phase_offset_rad)

    pulse_on_samples = _finite_int(overrides.get("pulse_on_samples", cfg.pulse_on_samples), cfg.pulse_on_samples, min_value=1)
    pulse_off_samples = _finite_int(overrides.get("pulse_off_samples", cfg.pulse_off_samples), cfg.pulse_off_samples, min_value=0)
    pulse_count = _finite_int(overrides.get("pulse_count", cfg.pulse_count), cfg.pulse_count, min_value=1, max_value=max_pulses)
    start_offset_samples = _finite_int(overrides.get("start_offset_samples", cfg.start_offset_samples), cfg.start_offset_samples, min_value=0)
    if desired_output_iq_len is not None and desired_output_iq_len > 0:
        start_offset_samples = min(start_offset_samples, max(0, int(desired_output_iq_len) - 1))

    pulse_phase_rotations_rad = _finite_float_list(
        overrides.get("pulse_phase_rotations_rad", cfg.pulse_phase_rotations_rad),
        default=cfg.pulse_phase_rotations_rad,
        length=pulse_count,
    )
    pulse_phase_rotations_rad = _wrap_phase_rad(torch.as_tensor(pulse_phase_rotations_rad, dtype=torch.float32)).to(dtype=torch.float64).tolist()
    pulse_phase_offset_rad = _finite_float(overrides.get("pulse_phase_offset_rad", pulse_phase_rotations_rad[0]), pulse_phase_rotations_rad[0])
    pulse_phase_rel_rad = _finite_float_list(overrides.get("pulse_phase_rel_rad", cfg.pulse_phase_rotations_rad), default=cfg.pulse_phase_rotations_rad, length=pulse_count)

    (
        tone_pulse_on_samples,
        tone_pulse_off_samples,
        tone_pulse_count,
        tone_start_offset_samples,
        tone_pulse_phase_rotations_rad,
    ) = _derive_tone_pulse_controls(
        num_tones=num_tones,
        pulse_on_samples=pulse_on_samples,
        pulse_off_samples=pulse_off_samples,
        pulse_count=pulse_count,
        start_offset_samples=start_offset_samples,
        desired_output_iq_len=desired_output_iq_len,
        tone_phase_offset_rad=tone_phase_offset_rad,
        pulse_phase_offset_rad=pulse_phase_offset_rad,
        tone_initial_phases_rad=tone_initial_phases_rad,
        pulse_phase_rel_rad=pulse_phase_rel_rad,
    )

    return replace(
        cfg,
        sample_rate_hz=sample_rate_hz,
        rf_center_hz=rf_center_hz,
        carrier_hz=carrier_hz,
        num_tones=num_tones,
        tone_frequencies_hz=tone_frequencies_hz,
        tone_frequency_std_hz=tone_frequency_std_hz,
        tone_amplitudes=tone_amplitudes,
        tone_initial_phases_rad=tone_initial_phases_rad,
        tone_phase_offset_rad=tone_phase_offset_rad,
        pulse_on_samples=pulse_on_samples,
        pulse_off_samples=pulse_off_samples,
        pulse_count=pulse_count,
        start_offset_samples=start_offset_samples,
        pulse_phase_rotations_rad=pulse_phase_rotations_rad,
        tone_pulse_on_samples=tone_pulse_on_samples,
        tone_pulse_off_samples=tone_pulse_off_samples,
        tone_pulse_count=tone_pulse_count,
        tone_start_offset_samples=tone_start_offset_samples,
        tone_pulse_phase_rotations_rad=tone_pulse_phase_rotations_rad,
        noise_color=_choice_from_override(overrides.get("noise_color", cfg.noise_color), NOISE_COLORS, cfg.noise_color),
        fading_mode=_choice_from_override(overrides.get("fading_mode", cfg.fading_mode), FADING_MODES, cfg.fading_mode),
        burst_color=_choice_from_override(overrides.get("burst_color", cfg.burst_color), NOISE_COLORS, cfg.burst_color),
        snr_db=None if overrides.get("snr_db", cfg.snr_db) is None else _finite_float(overrides.get("snr_db", cfg.snr_db), cfg.snr_db or 0.0),
        freq_offset=_finite_float(overrides.get("freq_offset", cfg.freq_offset), cfg.freq_offset),
        timing_offset=_finite_float(overrides.get("timing_offset", cfg.timing_offset), cfg.timing_offset),
        rician_k_db=_finite_float(overrides.get("rician_k_db", cfg.rician_k_db), cfg.rician_k_db),
        burst_probability=max(0.0, min(1.0, _finite_float(overrides.get("burst_probability", cfg.burst_probability), cfg.burst_probability))),
        burst_power_ratio_db=_finite_float(overrides.get("burst_power_ratio_db", cfg.burst_power_ratio_db), cfg.burst_power_ratio_db),
        peak_power=None if overrides.get("peak_power", cfg.peak_power) is None else max(0.0, _finite_float(overrides.get("peak_power", cfg.peak_power), cfg.peak_power or 0.0)),
        seed=_finite_int(overrides.get("seed", cfg.seed), cfg.seed),
    )

def build_controlled_tone_pulse_from_variable_inputs(
    model: TonePulseTXControlNetVarLen,
    rx_iq_windows: List[Union[torch.Tensor, List[complex]]],
    intake_sample_rate_hz: float,
    desired_output_iq_len: Optional[int] = None,
    user_peak_power_fraction: Optional[float] = None,
    seed: int = 1,
    device: str = "cpu",
    action_overrides: Optional[Dict[str, object]] = None,
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
        action_overrides=[action_overrides] if action_overrides else None,
    )
    return out[0]


def build_controlled_tone_pulse_batch_from_iq_batches(
    model: TonePulseTXControlNetVarLen,
    rx_iq_batches: Sequence[Union[torch.Tensor, Sequence[Union[torch.Tensor, List[complex]]]]],
    intake_sample_rate_hz: float,
    desired_output_iq_len: Optional[int] = None,
    user_peak_power_fraction: Optional[float] = None,
    seed: int = 1,
    device: str = "cpu",
    action_overrides: Optional[Sequence[Optional[Dict[str, object]]]] = None,
) -> List[Dict[str, object]]:
    if len(rx_iq_batches) != 3:
        raise ValueError(f"rx_iq_batches must contain exactly 3 IQ batch inputs, got {len(rx_iq_batches)}")

    proc = [preprocess_batched_iq_to_stft_feature(x, intake_sample_rate_hz) for x in rx_iq_batches]
    batch_size = int(proc[0]["feature"].shape[0])
    if any(int(p["feature"].shape[0]) != batch_size for p in proc):
        raise ValueError("All three IQ inputs must use the same batch size")
    if action_overrides is not None and len(action_overrides) != batch_size:
        raise ValueError(
            f"action_overrides length {len(action_overrides)} must match batch size {batch_size}"
        )

    stft_tensors = [p["feature"].to(device) for p in proc]
    rx_power_stack = torch.stack([p["rx_power"].to(device) for p in proc], dim=0)
    peak_stack = torch.stack([p["peak_hz"].to(device) for p in proc], dim=0)
    rx_input_power_t = rx_power_stack.mean(dim=0)
    rf_center_est_t = peak_stack.mean(dim=0)

    lengths = torch.stack([p["lengths"].to(device) for p in proc], dim=0).transpose(0, 1)
    scalar_side = build_scalar_side_for_model(
        model,
        rx_iq_batches,
        intake_sample_rate_hz,
        device=device,
        preprocessed=proc,
    )

    model = model.to(device)
    y = model(stft_tensors, scalar_side)

    out = []
    for i in range(batch_size):
        model_out_i = {k: v[i : i + 1] for k, v in y.items()}
        cfg = decode_tone_pulse_config(
            model_out=model_out_i,
            intake_sample_rate_hz=float(intake_sample_rate_hz),
            rf_center_est_hz=float(rf_center_est_t[i].item()),
            desired_output_iq_len=desired_output_iq_len,
            user_peak_power_fraction=user_peak_power_fraction,
            rx_input_power=float(rx_input_power_t[i].item()),
            max_tones=model.max_tones,
            max_pulses=model.max_pulses,
            seed=seed + i,
        )
        row_overrides = None if action_overrides is None else action_overrides[i]
        cfg = apply_tone_pulse_action_overrides(
            cfg,
            row_overrides,
            desired_output_iq_len=desired_output_iq_len,
            max_tones=model.max_tones,
            max_pulses=model.max_pulses,
        )

        tx_result = txflex.build_tone_pulse_iq_object(
            sample_rate_hz=cfg.sample_rate_hz,
            rf_center_hz=cfg.rf_center_hz,
            carrier_hz=cfg.carrier_hz,
            target_num_samples=desired_output_iq_len,
            num_tones=cfg.num_tones,
            tone_frequencies_hz=cfg.tone_frequencies_hz,
            tone_amplitudes=cfg.tone_amplitudes,
            tone_initial_phases_rad=cfg.tone_initial_phases_rad,
            tone_phase_offset_rad=cfg.tone_phase_offset_rad,
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

        tx_iq = _apply_per_tone_pulse_phase_rotation(
            tx_result.iq,
            sample_rate_hz=cfg.sample_rate_hz,
            carrier_hz=cfg.carrier_hz,
            tone_frequencies_hz=cfg.tone_frequencies_hz,
            tone_amplitudes=cfg.tone_amplitudes,
            tone_initial_phases_rad=cfg.tone_initial_phases_rad,
            tone_pulse_on_samples=cfg.tone_pulse_on_samples,
            tone_pulse_off_samples=cfg.tone_pulse_off_samples,
            tone_pulse_count=cfg.tone_pulse_count,
            tone_start_offset_samples=cfg.tone_start_offset_samples,
            tone_pulse_phase_rotations_rad=cfg.tone_pulse_phase_rotations_rad,
        )

        tx_metadata = dict(tx_result.metadata)
        tx_metadata["controller_input_lengths"] = [int(v) for v in lengths[i].tolist()]
        tx_metadata["controller_rx_input_power"] = float(rx_input_power_t[i].item())
        tx_metadata["controller_scalar_feature_schema"] = FIRST_PASS_SCALAR_FEATURE_SCHEMA_VERSION
        tx_metadata["controller_scalar_feature_names"] = list(FIRST_PASS_SCALAR_FEATURE_NAMES[: scalar_side.shape[1]])
        tx_metadata["controller_scalar_side"] = [float(v) for v in scalar_side[i].detach().cpu().tolist()]
        tx_metadata["controller_pulse_phase_rotations_rad"] = [float(v) for v in cfg.pulse_phase_rotations_rad]
        tx_metadata["controller_tone_pulse_on_samples"] = [int(v) for v in cfg.tone_pulse_on_samples]
        tx_metadata["controller_tone_pulse_off_samples"] = [int(v) for v in cfg.tone_pulse_off_samples]
        tx_metadata["controller_tone_pulse_count"] = [int(v) for v in cfg.tone_pulse_count]
        tx_metadata["controller_tone_start_offset_samples"] = [int(v) for v in cfg.tone_start_offset_samples]
        tx_metadata["controller_tone_pulse_phase_rotations_rad"] = [
            [float(p) for p in plist] for plist in cfg.tone_pulse_phase_rotations_rad
        ]
        tx_metadata["controller_action_overrides_applied"] = bool(row_overrides)

        out.append(
            {
                "tx_config": cfg,
                "model_outputs": model_out_i,
                "tx_iq": _as_complex_tensor(tx_iq),
                "tx_metadata": tx_metadata,
                "rx_input_power": float(rx_input_power_t[i].item()),
                "rf_center_est_hz": float(rf_center_est_t[i].item()),
            }
        )

    return out


if __name__ == "__main__":
    iq_a = torch.complex(torch.randn(900), torch.randn(900)).to(dtype=DEFAULT_COMPLEX_DTYPE)
    iq_b = torch.complex(torch.randn(1700), torch.randn(1700)).to(dtype=DEFAULT_COMPLEX_DTYPE)
    iq_c = torch.complex(torch.randn(3200), torch.randn(3200)).to(dtype=DEFAULT_COMPLEX_DTYPE)

    model = TonePulseTXControlNetVarLen(in_ch=4, base_ch=16, max_tones=8)
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
