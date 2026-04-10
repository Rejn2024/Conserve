#!/usr/bin/env python3
"""
advanced_link_skdsp_v4_robust.py
"""

from __future__ import annotations

import argparse
import binascii
import concurrent.futures
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torchaudio.functional as AF
except Exception:  # pragma: no cover - optional dependency
    AF = None

DEFAULT_TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEC_NONE = "none"
FEC_REP3 = "rep3"
FEC_CONV = "conv"

MAX_PAYLOAD_BYTES = 10_000_000
HEADER_MAGIC = 0xA55A

PREAMBLE_HALF_LEN_BITS = 64
PREAMBLE_REPS = 4
SYNC_WORD = b"\xD3\x91\xC5\x7A"
TRAINING_LEN_BITS = 128

HEADER_BYTES_LEN = 2 + 4 + 4 + 4
HEADER_BITS_LEN = HEADER_BYTES_LEN * 8
HEADER_PROT_BITS_LEN = HEADER_BITS_LEN * 3
HEADER_COPIES = 2

PILOT_INTERVAL_BITS = 128
PILOT_BLOCK_BITS = 16

POSTAMBLE_BITS = 256

# -----------------------------------------------------------------------------
# Raw IQ / metadata helpers
# -----------------------------------------------------------------------------

def save_iq(path: Union[str, Path], iq: Union[np.ndarray, torch.Tensor]) -> None:
    np.asarray(_as_numpy_complex64(iq), dtype=np.complex64).tofile(str(path))


def load_iq(path: Union[str, Path]) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.complex64)


def default_metadata_path(iq_path: Union[str, Path]) -> str:
    p = Path(iq_path)
    return str(p.with_suffix(p.suffix + ".json"))


def save_iq_metadata(
    iq_path: Union[str, Path],
    sample_rate_hz: float,
    rf_center_hz: float,
    carrier_hz: float,
    metadata_path: Optional[Union[str, Path]] = None,
) -> str:
    if metadata_path is None:
        metadata_path = default_metadata_path(iq_path)
    else:
        metadata_path = str(metadata_path)

    meta = {
        "iq_path": str(iq_path),
        "sample_rate_hz": float(sample_rate_hz),
        "rf_center_hz": float(rf_center_hz),
        "carrier_hz": float(carrier_hz),
        "absolute_rf_hz": float(rf_center_hz + carrier_hz),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return metadata_path


def load_iq_metadata(
    iq_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
) -> Optional[dict]:
    if metadata_path is None:
        metadata_path = default_metadata_path(iq_path)
    else:
        metadata_path = str(metadata_path)

    p = Path(metadata_path)
    if not p.exists():
        return None

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

        
def bytes_to_bits_msb(data: bytes) -> List[int]:
    out: List[int] = []
    for b in data:
        for i in range(7, -1, -1):
            out.append((b >> i) & 1)
    return out


def bits_to_bytes_msb(bits: List[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Bit count must be a multiple of 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        x = 0
        for bit in bits[i:i + 8]:
            x = (x << 1) | (bit & 1)
        out.append(x)
    return bytes(out)


def prbs_bits(n: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n, dtype=np.int8).tolist()


PREAMBLE_HALF_BITS = prbs_bits(PREAMBLE_HALF_LEN_BITS, seed=12345)
PREAMBLE_BITS = PREAMBLE_HALF_BITS * PREAMBLE_REPS
SYNC_BITS = bytes_to_bits_msb(SYNC_WORD)
TRAINING_BITS = prbs_bits(TRAINING_LEN_BITS, seed=67890)
PILOT_BITS = prbs_bits(PILOT_BLOCK_BITS, seed=24680)

ACCESS_BITS = PREAMBLE_BITS + SYNC_BITS + TRAINING_BITS

PREAMBLE_BITS_LEN = len(PREAMBLE_BITS)
SYNC_BITS_LEN = len(SYNC_BITS)
TRAINING_BITS_LEN = len(TRAINING_BITS)
ACCESS_BITS_LEN = len(ACCESS_BITS)


def rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    n = np.arange(-span * sps, span * sps + 1, dtype=np.float64)
    t = n / sps
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 0 and abs(abs(4 * beta * ti) - 1.0) < 1e-9:
            h[i] = (
                beta / np.sqrt(2.0)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            )
        else:
            num = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den

    h /= np.sqrt(np.sum(h ** 2))
    return h.astype(np.float64)


def _as_complex_tensor(x: Union[np.ndarray, torch.Tensor, List[complex]]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.complex64).to(DEFAULT_TORCH_DEVICE)
    return torch.as_tensor(np.asarray(x), dtype=torch.complex64, device=DEFAULT_TORCH_DEVICE)


def _as_numpy_complex64(x: Union[np.ndarray, torch.Tensor, List[complex]]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.complex64, copy=False)
    return np.asarray(x, dtype=np.complex64)


def _complex_lfilter_fir(
    x: Union[np.ndarray, torch.Tensor, List[complex]],
    taps: Union[np.ndarray, torch.Tensor, List[complex]],
) -> torch.Tensor:
    xt = _as_complex_tensor(x)
    ht = _as_complex_tensor(taps)
    if xt.numel() == 0:
        return xt

    xr = xt.real.to(dtype=torch.float32).reshape(1, 1, -1)
    xi = xt.imag.to(dtype=torch.float32).reshape(1, 1, -1)
    k_r = torch.flip(ht.real.to(dtype=torch.float32), dims=[0]).reshape(1, 1, -1).to(device=xr.device, dtype=xr.dtype)
    k_i = torch.flip(ht.imag.to(dtype=torch.float32), dims=[0]).reshape(1, 1, -1).to(device=xr.device, dtype=xr.dtype)
    pad = ht.numel() - 1

    xr_pad = F.pad(xr, (pad, 0))
    xi_pad = F.pad(xi, (pad, 0))
    yr = F.conv1d(xr_pad, k_r).reshape(-1) - F.conv1d(xi_pad, k_i).reshape(-1)
    yi = F.conv1d(xr_pad, k_i).reshape(-1) + F.conv1d(xi_pad, k_r).reshape(-1)
    return torch.complex(yr, yi).to(dtype=torch.complex64)


def _resample_complex_to_len(
    x: Union[np.ndarray, torch.Tensor, List[complex]],
    new_len: int,
) -> torch.Tensor:
    xt = _as_complex_tensor(x)
    if new_len <= 0:
        raise ValueError("new_len must be positive")
    if xt.numel() == 0:
        return xt
    if int(new_len) == int(xt.numel()):
        return xt

    old_len = int(xt.numel())
    if AF is not None:
        y_r = AF.resample(xt.real.unsqueeze(0), orig_freq=old_len, new_freq=int(new_len)).squeeze(0)
        y_i = AF.resample(xt.imag.unsqueeze(0), orig_freq=old_len, new_freq=int(new_len)).squeeze(0)
    else:
        y_r = F.interpolate(
            xt.real.reshape(1, 1, -1),
            size=int(new_len),
            mode="linear",
            align_corners=False,
        ).reshape(-1)
        y_i = F.interpolate(
            xt.imag.reshape(1, 1, -1),
            size=int(new_len),
            mode="linear",
            align_corners=False,
        ).reshape(-1)
    return torch.complex(y_r, y_i).to(dtype=torch.complex64)


def measure_power(x: Union[np.ndarray, torch.Tensor]) -> float:
    xt = _as_complex_tensor(x)
    return float(torch.mean(torch.abs(xt) ** 2).item()) if xt.numel() else 0.0

def measure_peak_power(x: Union[np.ndarray, torch.Tensor]) -> float:
    xt = _as_complex_tensor(x)
    return float(torch.max(torch.abs(xt) ** 2).item()) if xt.numel() else 0.0

def measure_peak_power(x: Union[np.ndarray, torch.Tensor]) -> float:
    xt = _as_complex_tensor(x)
    return float(torch.max(torch.abs(xt) ** 2).item()) if xt.numel() else 0.0


def bpsk_map(bits: List[int]) -> torch.Tensor:
    b = torch.tensor(bits, dtype=torch.int8, device=DEFAULT_TORCH_DEVICE)
    return torch.where(
        b > 0,
        torch.tensor(1.0, device=b.device),
        torch.tensor(-1.0, device=b.device),
    ).to(dtype=torch.complex64)


def upsample_and_shape(symbols: Union[np.ndarray, torch.Tensor], sps: int, taps: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    syms = _as_complex_tensor(symbols)
    up = torch.zeros(syms.numel() * sps, dtype=torch.complex64, device=syms.device)
    up[::sps] = syms
    return _complex_lfilter_fir(up, taps)


def tx_waveform(bits: List[int], sps: int, beta: float, span: int) -> torch.Tensor:
    syms = bpsk_map(bits)
    taps = rrc_taps(beta=beta, sps=sps, span=span)
    return upsample_and_shape(syms, sps=sps, taps=taps)


def apply_carrier_frequency(iq: Union[np.ndarray, torch.Tensor], carrier_hz: float, sample_rate_hz: float) -> torch.Tensor:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if abs(carrier_hz) >= sample_rate_hz / 2:
        raise ValueError("carrier_hz must satisfy |carrier_hz| < sample_rate_hz/2")
    x = _as_complex_tensor(iq)
    if carrier_hz == 0.0:
        return x
    n = torch.arange(x.numel(), dtype=torch.float64, device=x.device)
    phase = 2.0 * torch.pi * carrier_hz * n / sample_rate_hz
    rot = torch.complex(torch.cos(phase), torch.sin(phase)).to(dtype=torch.complex64)
    return x * rot


def apply_frequency_offset(iq: Union[np.ndarray, torch.Tensor], freq_offset: float) -> torch.Tensor:
    x = _as_complex_tensor(iq)
    if freq_offset == 0.0:
        return x
    n = torch.arange(x.numel(), dtype=torch.float64, device=x.device)
    phase = 2.0 * torch.pi * freq_offset * n
    rot = torch.complex(torch.cos(phase), torch.sin(phase)).to(dtype=torch.complex64)
    return x * rot


def apply_timing_offset_resample(iq: Union[np.ndarray, torch.Tensor], timing_offset: float) -> torch.Tensor:
    x = _as_complex_tensor(iq)
    if timing_offset == 1.0 or x.numel() == 0:
        return x
    target_len = int(x.numel())
    new_len = max(1, int(round(target_len / timing_offset)))
    y = _resample_complex_to_len(x, new_len)
    if new_len > target_len:
        y = y[:target_len]
    elif new_len < target_len:
        y = torch.cat([y, torch.zeros(target_len - new_len, dtype=y.dtype, device=y.device)])
    return y


def resample_iq(iq: Union[np.ndarray, torch.Tensor], fs_in_hz: float, fs_out_hz: float) -> torch.Tensor:
    if fs_in_hz <= 0 or fs_out_hz <= 0:
        raise ValueError("Sample rates must be positive")
    if np.isclose(fs_in_hz, fs_out_hz):
        return _as_complex_tensor(iq)
    x = _as_complex_tensor(iq)
    new_len = max(1, int(round(x.numel() * fs_out_hz / fs_in_hz)))
    return _resample_complex_to_len(x, new_len)


def _complex_colored_noise(
    n: int,
    color: str,
    power: float,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    exponents = {
        "white": 0.0,
        "pink": 1.0,
        "brown": 2.0,
        "blue": -1.0,
        "violet": -2.0,
    }
    if color not in exponents:
        raise ValueError(f"Unsupported noise color: {color}")
    alpha = exponents[color]

    if n <= 0:
        return torch.zeros(0, dtype=torch.complex64, device=device or torch.device(DEFAULT_TORCH_DEVICE))
    dev = device or torch.device(DEFAULT_TORCH_DEVICE)

    def make_real() -> torch.Tensor:
        freqs = torch.fft.rfftfreq(n, d=1.0, device=dev)
        spec = torch.complex(
            torch.randn(len(freqs), generator=generator, device=dev),
            torch.randn(len(freqs), generator=generator, device=dev),
        )
        shaping = torch.ones_like(freqs, dtype=torch.float32)
        nz = freqs > 0
        if alpha > 0:
            shaping[nz] = 1.0 / (freqs[nz] ** (alpha / 2.0))
        elif alpha < 0:
            shaping[nz] = freqs[nz] ** ((-alpha) / 2.0)
        shaping[~nz] = 0.0 if color in ("pink", "brown") else 1.0
        y = torch.fft.irfft(spec * shaping.to(spec.dtype), n=n)
        p = torch.mean(y ** 2)
        if float(p.item()) > 0:
            y = y * torch.sqrt(torch.tensor((power / 2.0), device=dev, dtype=torch.float32) / p)
        return y

    i = make_real()
    q = make_real()
    z = torch.complex(i, q).to(dtype=torch.complex64)
    p = torch.mean(torch.abs(z) ** 2)
    if float(p.item()) > 0:
        z = z * torch.sqrt(torch.tensor(power, device=dev, dtype=torch.float32) / p)
    return z


def apply_fading(
    iq: Union[np.ndarray, torch.Tensor],
    mode: str = "none",
    block_len: int = 256,
    rician_k_db: float = 6.0,
    multipath_taps: Optional[List[complex]] = None,
    seed: int = 1,
) -> torch.Tensor:
    x = _as_complex_tensor(iq)
    gen = torch.Generator(device=x.device if x.is_cuda else "cpu")
    gen.manual_seed(seed)

    if mode == "none":
        return x

    if mode == "rayleigh_block":
        out = torch.empty_like(x)
        for start in range(0, x.numel(), block_len):
            end = min(start + block_len, x.numel())
            h = torch.complex(
                torch.randn((), generator=gen, device=x.device),
                torch.randn((), generator=gen, device=x.device),
            ) / math.sqrt(2.0)
            out[start:end] = x[start:end] * h
        return out.to(dtype=torch.complex64)

    if mode == "rician_block":
        out = torch.empty_like(x)
        k_lin = 10.0 ** (rician_k_db / 10.0)
        los = np.sqrt(k_lin / (k_lin + 1.0))
        scat_scale = np.sqrt(1.0 / (k_lin + 1.0))
        for start in range(0, x.numel(), block_len):
            end = min(start + block_len, x.numel())
            scat = (
                torch.complex(
                    torch.randn((), generator=gen, device=x.device),
                    torch.randn((), generator=gen, device=x.device),
                ) / math.sqrt(2.0)
            ) * scat_scale
            h = los + scat
            out[start:end] = x[start:end] * h
        return out.to(dtype=torch.complex64)

    if mode == "multipath_static":
        taps = multipath_taps if multipath_taps is not None else [
            1.0 + 0.0j,
            0.20 + 0.08j,
            0.06 - 0.04j,
        ]
        y = _complex_lfilter_fir(x, taps)
        return y[: x.numel()].to(dtype=torch.complex64)

    raise ValueError(f"Unsupported fading mode: {mode}")


def add_impulsive_bursts(
    iq: Union[np.ndarray, torch.Tensor],
    base_noise_power: float,
    burst_probability: float = 0.0,
    burst_len_min: int = 16,
    burst_len_max: int = 64,
    burst_power_ratio_db: float = 12.0,
    burst_color: str = "white",
    seed: int = 1,
) -> torch.Tensor:
    x = _as_complex_tensor(iq)
    if burst_probability <= 0.0 or x.numel() == 0:
        return x

    gen = torch.Generator(device="cpu")
    genl = torch.Generator(device=x.device)
    gen.manual_seed(seed + 1000)
    genl.manual_seed(seed + 2000)
    out = x.clone()
    burst_power = base_noise_power * (10.0 ** (burst_power_ratio_db / 10.0))

    idx = 0
    while idx < out.numel():
        if float(torch.rand((), generator=gen).item()) < burst_probability:
            burst_len = int(torch.randint(burst_len_min, burst_len_max + 1, (1,), generator=gen).item())
            end = min(idx + burst_len, out.numel())
            out[idx:end] += _complex_colored_noise(end - idx, burst_color, burst_power, generator=genl, device=out.device)
            idx = end
        else:
            idx += 1
    return out.to(dtype=torch.complex64)


def impair_iq(
    iq: Union[np.ndarray, torch.Tensor],
    snr_db: Optional[float],
    noise_color: str,
    freq_offset: float,
    timing_offset: float,
    fading_mode: str,
    fading_block_len: int,
    rician_k_db: float,
    multipath_taps: Optional[List[complex]],
    burst_probability: float,
    burst_len_min: int,
    burst_len_max: int,
    burst_power_ratio_db: float,
    burst_color: str,
    seed: int,
) -> torch.Tensor:
    x = _as_complex_tensor(iq)
    gen = torch.Generator(device=x.device if x.is_cuda else "cpu")
    gen.manual_seed(seed)

    x = apply_fading(
        x,
        mode=fading_mode,
        block_len=fading_block_len,
        rician_k_db=rician_k_db,
        multipath_taps=multipath_taps,
        seed=seed,
    )

    x = apply_frequency_offset(x, freq_offset)
    x = apply_timing_offset_resample(x, timing_offset)

    sig_power = measure_power(x)
    if snr_db is None:
        noise_power = 0.0
        y = x
    else:
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        y = x + _complex_colored_noise(x.numel(), noise_color, noise_power, generator=gen, device=x.device)

    y = add_impulsive_bursts(
        y,
        base_noise_power=noise_power if noise_power > 0 else max(sig_power * 1e-6, 1e-12),
        burst_probability=burst_probability,
        burst_len_min=burst_len_min,
        burst_len_max=burst_len_max,
        burst_power_ratio_db=burst_power_ratio_db,
        burst_color=burst_color,
        seed=seed,
    )
    return y.to(dtype=torch.complex64)


def limit_peak_power(iq: Union[np.ndarray, torch.Tensor], max_peak_power: Optional[float]) -> Tuple[torch.Tensor, bool, float]:
    x = _as_complex_tensor(iq)
    pre_peak_power = measure_peak_power(x)
    if max_peak_power is None:
        return x, False, pre_peak_power
    if max_peak_power <= 0:
        raise ValueError("max_peak_power must be > 0 when provided")
    if pre_peak_power <= 0 or pre_peak_power <= max_peak_power:
        return x, False, pre_peak_power
    scale = math.sqrt(max_peak_power / pre_peak_power)
    return (x * scale).to(dtype=torch.complex64), True, pre_peak_power


def robust_agc_and_blanking(x: Union[np.ndarray, torch.Tensor], blanker_k: float = 6.0) -> torch.Tensor:
    xt = _as_complex_tensor(x)
    if xt.numel() == 0:
        return xt

    xt = xt - torch.mean(xt)
    mag = torch.abs(xt)
    med = torch.median(mag)
    mad = torch.median(torch.abs(mag - med)) + 1e-12
    thresh = med + blanker_k * 1.4826 * mad

    if float(thresh.item()) > 0.0:
        over = mag > thresh
        if bool(torch.any(over).item()):
            clamped = thresh * torch.exp(1j * torch.angle(xt[over]))
            xt = xt.clone()
            xt[over] = clamped

    p = torch.mean(torch.abs(xt) ** 2)
    if float(p.item()) > 0:
        xt = xt / torch.sqrt(p)
    return xt.to(dtype=torch.complex64)


def lfsr_sequence(n: int, seed: int = 0x5D) -> List[int]:
    state = seed & 0x7F
    if state == 0:
        state = 0x5D
    out = []
    for _ in range(n):
        new_bit = ((state >> 6) ^ (state >> 5)) & 1
        out.append(state & 1)
        state = ((state << 1) & 0x7F) | new_bit
    return out


def scramble_bits(bits: List[int], seed: int = 0x5D) -> List[int]:
    pn = lfsr_sequence(len(bits), seed=seed)
    return [(b ^ p) & 1 for b, p in zip(bits, pn)]


def descramble_bits(bits: List[int], seed: int = 0x5D) -> List[int]:
    return scramble_bits(bits, seed=seed)


def rep3_encode_bits(bits: List[int]) -> List[int]:
    out = []
    for b in bits:
        out.extend([b, b, b])
    return out


def rep3_decode_soft(soft: np.ndarray) -> List[int]:
    usable = (len(soft) // 3) * 3
    soft = soft[:usable]
    out = []
    for i in range(0, len(soft), 3):
        out.append(1 if np.sum(soft[i:i + 3]) >= 0 else 0)
    return out


def block_interleave_bits(bits: List[int], rows: int = 8) -> List[int]:
    if rows <= 1:
        return bits[:]
    cols = math.ceil(len(bits) / rows)
    padded = bits[:] + [0] * (rows * cols - len(bits))
    mat = np.array(padded, dtype=np.int8).reshape(rows, cols)
    out = mat.T.reshape(-1).tolist()
    return out[:len(bits)]


def block_deinterleave_soft(soft: np.ndarray, rows: int = 8) -> np.ndarray:
    if rows <= 1:
        return np.asarray(soft, dtype=np.float64)
    n = len(soft)
    cols = math.ceil(n / rows)
    padded = np.concatenate([soft, np.zeros(rows * cols - n, dtype=np.float64)])
    mat = padded.reshape(cols, rows).T
    out = mat.reshape(-1)
    return out[:n]


def conv_encode(bits: List[int]) -> List[int]:
    state = [0, 0]
    out = []
    for b in bits + [0, 0]:
        d0 = b
        d1 = state[0]
        d2 = state[1]
        o1 = (d0 ^ d1 ^ d2) & 1
        o2 = (d0 ^ d2) & 1
        out.extend([o1, o2])
        state = [d0, d1]
    return out


def conv_decode_soft(soft: np.ndarray) -> List[int]:
    soft = np.asarray(soft, dtype=np.float64)
    usable = (len(soft) // 2) * 2
    soft = soft[:usable]
    n_steps = len(soft) // 2
    if n_steps == 0:
        return []

    INF = 1e18
    pm = np.full((n_steps + 1, 4), INF, dtype=np.float64)
    prev_state = np.full((n_steps, 4), -1, dtype=np.int32)
    prev_bit = np.full((n_steps, 4), -1, dtype=np.int32)
    pm[0, 0] = 0.0

    def branch_outputs(state: int, bit: int) -> Tuple[int, int, int]:
        s0 = (state >> 1) & 1
        s1 = state & 1
        d0 = bit
        d1 = s0
        d2 = s1
        o1 = (d0 ^ d1 ^ d2) & 1
        o2 = (d0 ^ d2) & 1
        new_state = ((bit << 1) | s0) & 0b11
        return new_state, o1, o2

    for t in range(n_steps):
        y1 = soft[2 * t]
        y2 = soft[2 * t + 1]
        for s in range(4):
            if pm[t, s] >= INF / 2:
                continue
            for b in (0, 1):
                ns, o1, o2 = branch_outputs(s, b)
                e1 = 1.0 if o1 else -1.0
                e2 = 1.0 if o2 else -1.0
                metric = pm[t, s] + (y1 - e1) ** 2 + (y2 - e2) ** 2
                if metric < pm[t + 1, ns]:
                    pm[t + 1, ns] = metric
                    prev_state[t, ns] = s
                    prev_bit[t, ns] = b

    state = 0
    decoded = []
    for t in range(n_steps - 1, -1, -1):
        b = prev_bit[t, state]
        if b < 0:
            b = 0
        decoded.append(int(b))
        state = prev_state[t, state] if prev_state[t, state] >= 0 else 0
    decoded.reverse()

    if len(decoded) >= 2:
        decoded = decoded[:-2]
    return decoded


class FECCodec:
    def __init__(self, mode: str):
        self.mode = mode

    def encoded_length(self, input_bit_len: int) -> int:
        if self.mode == FEC_NONE:
            return input_bit_len
        if self.mode == FEC_REP3:
            return 3 * input_bit_len
        if self.mode == FEC_CONV:
            return 2 * (input_bit_len + 2)
        raise ValueError("Unsupported FEC mode")

    def encode_bits(self, bits: List[int]) -> List[int]:
        if self.mode == FEC_NONE:
            return bits[:]
        if self.mode == FEC_REP3:
            return rep3_encode_bits(bits)
        if self.mode == FEC_CONV:
            return conv_encode(bits)
        raise ValueError("Unsupported FEC mode")

    def decode_soft(self, soft: np.ndarray) -> List[int]:
        if self.mode == FEC_NONE:
            return (np.asarray(soft) > 0).astype(np.int8).tolist()
        if self.mode == FEC_REP3:
            return rep3_decode_soft(soft)
        if self.mode == FEC_CONV:
            return conv_decode_soft(soft)
        raise ValueError("Unsupported FEC mode")


def build_header_bytes(payload_len: int) -> bytes:
    if payload_len <= 0:
        raise ValueError("payload_len must be > 0")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError("payload_len too large")

    payload_len_inv = payload_len ^ 0xFFFFFFFF
    prefix = struct.pack(">HII", HEADER_MAGIC, payload_len, payload_len_inv)
    crc = binascii.crc32(prefix) & 0xFFFFFFFF
    return prefix + struct.pack(">I", crc)


def parse_header_bytes(header: bytes) -> Optional[int]:
    if len(header) != HEADER_BYTES_LEN:
        return None
    magic, payload_len, payload_len_inv, hdr_crc = struct.unpack(">HIII", header)
    prefix = struct.pack(">HII", magic, payload_len, payload_len_inv)
    calc_crc = binascii.crc32(prefix) & 0xFFFFFFFF
    if magic != HEADER_MAGIC:
        return None
    if payload_len == 0 or payload_len > MAX_PAYLOAD_BYTES:
        return None
    if payload_len_inv != (payload_len ^ 0xFFFFFFFF):
        return None
    if hdr_crc != calc_crc:
        return None
    return payload_len


def build_payload_bytes_from_message(message: Union[str, bytes]) -> bytes:
    payload = message.encode("utf-8") if isinstance(message, str) else bytes(message)
    if len(payload) == 0:
        raise ValueError("Empty payloads are not supported")
    payload_crc = binascii.crc32(payload) & 0xFFFFFFFF
    return payload + struct.pack(">I", payload_crc)


def build_payload_bytes_from_random_bits(n_bits: int, seed: int) -> bytes:
    if n_bits <= 0:
        raise ValueError("n_bits must be > 0")
    bits = prbs_bits(n_bits, seed=seed)
    pad = (-n_bits) % 8
    if pad:
        bits = bits + [0] * pad
    payload = bits_to_bytes_msb(bits)
    payload_crc = binascii.crc32(payload) & 0xFFFFFFFF
    return payload + struct.pack(">I", payload_crc)


def parse_payload_bytes(raw: bytes, payload_len: int) -> Optional[bytes]:
    needed = payload_len + 4
    if len(raw) < needed:
        return None
    payload = raw[:payload_len]
    rx_crc = struct.unpack(">I", raw[payload_len:payload_len + 4])[0]
    calc_crc = binascii.crc32(payload) & 0xFFFFFFFF
    if rx_crc != calc_crc:
        return None
    return payload


def insert_pilots(bits: List[int], interval: int = PILOT_INTERVAL_BITS, pilot_bits: List[int] = PILOT_BITS) -> List[int]:
    out = []
    i = 0
    while i < len(bits):
        chunk = bits[i:i + interval]
        out.extend(chunk)
        i += len(chunk)
        if i < len(bits):
            out.extend(pilot_bits)
    return out


def remove_pilots_soft(
    soft: np.ndarray,
    original_data_len: int,
    interval: int = PILOT_INTERVAL_BITS,
    pilot_bits: List[int] = PILOT_BITS,
) -> np.ndarray:
    soft = np.asarray(soft, dtype=np.float64)
    out = []
    i_data = 0
    i_soft = 0
    p_len = len(pilot_bits)

    while i_data < original_data_len and i_soft < len(soft):
        chunk = min(interval, original_data_len - i_data)
        out.extend(soft[i_soft:i_soft + chunk])
        i_soft += chunk
        i_data += chunk
        if i_data < original_data_len:
            i_soft += p_len

    return np.asarray(out[:original_data_len], dtype=np.float64)


def pilot_positions(total_data_len: int, interval: int = PILOT_INTERVAL_BITS, p_len: int = PILOT_BLOCK_BITS) -> List[Tuple[int, int, int, int]]:
    pos = []
    data_consumed = 0
    stream_idx = 0
    while data_consumed < total_data_len:
        chunk = min(interval, total_data_len - data_consumed)
        data_start = stream_idx
        data_end = stream_idx + chunk
        stream_idx += chunk
        data_consumed += chunk
        if data_consumed < total_data_len:
            pilot_start = stream_idx
            pilot_end = stream_idx + p_len
            pos.append((data_start, data_end, pilot_start, pilot_end))
            stream_idx += p_len
        else:
            pos.append((data_start, data_end, -1, -1))
    return pos


@dataclass
class TXBuildResult:
    iq: torch.Tensor
    metadata: dict


def build_tx_bitstream(
    payload_crc_bytes: bytes,
    fec_mode: str,
    interleave: bool,
    interleave_rows: int,
) -> Tuple[List[int], int, int, int, int]:
    payload_len = len(payload_crc_bytes) - 4
    header_bits = bytes_to_bits_msb(build_header_bytes(payload_len))
    header_prot = rep3_encode_bits(header_bits)
    header_total = header_prot * HEADER_COPIES

    payload_plain_bits = bytes_to_bits_msb(payload_crc_bytes)
    payload_scrambled = scramble_bits(payload_plain_bits, seed=0x5D)

    fec = FECCodec(fec_mode)
    payload_coded = fec.encode_bits(payload_scrambled)

    if interleave:
        payload_coded = block_interleave_bits(payload_coded, rows=interleave_rows)

    payload_with_pilots = insert_pilots(payload_coded, interval=PILOT_INTERVAL_BITS, pilot_bits=PILOT_BITS)

    frame_bits = ACCESS_BITS + header_total + payload_with_pilots + [0] * POSTAMBLE_BITS
    return frame_bits, payload_len, len(payload_plain_bits), len(payload_coded), len(payload_with_pilots)


def build_tx_iq_object(
    *,
    message: Optional[Union[str, bytes]] = None,
    random_bits: Optional[int] = None,
    random_seed: int = 1,
    target_num_samples: Optional[int] = None,
    fec: str = FEC_NONE,
    interleave: bool = False,
    interleave_rows: int = 8,
    sps: int = 8,
    beta: float = 0.35,
    span: int = 6,
    sample_rate_hz: float = 1_000_000.0,
    rf_center_hz: float = 0.0,
    carrier_hz: float = 0.0,
    snr_db: Optional[float] = None,
    noise_color: str = "white",
    freq_offset: float = 0.0,
    timing_offset: float = 1.0,
    fading_mode: str = "none",
    fading_block_len: int = 256,
    rician_k_db: float = 6.0,
    multipath_taps: Optional[List[complex]] = None,
    burst_probability: float = 0.0,
    burst_len_min: int = 16,
    burst_len_max: int = 64,
    burst_power_ratio_db: float = 12.0,
    burst_color: str = "white",
    seed: int = 1,
) -> TXBuildResult:
    if (message is None) == (random_bits is None):
        raise ValueError("Provide exactly one of message or random_bits")

    if random_bits is not None:
        payload_crc_bytes = build_payload_bytes_from_random_bits(random_bits, seed=random_seed)
        payload_source = f"random_bits:{random_bits}"
    else:
        payload_crc_bytes = build_payload_bytes_from_message(message)
        payload_source = "message"

    frame_bits, payload_len, payload_plain_bits, payload_coded_bits, payload_with_pilots_bits = build_tx_bitstream(
        payload_crc_bytes=payload_crc_bytes,
        fec_mode=fec,
        interleave=interleave,
        interleave_rows=interleave_rows,
    )

    one_burst_iq = tx_waveform(frame_bits, sps=sps, beta=beta, span=span)

    if target_num_samples is None:
        iq = one_burst_iq.to(dtype=torch.complex64)
        n_repeats = 1
    else:
        if target_num_samples <= 0:
            raise ValueError("target_num_samples must be > 0")
        n_repeats = int(math.ceil(target_num_samples / len(one_burst_iq)))
        iq = torch.tile(one_burst_iq, (n_repeats,))[:target_num_samples].to(dtype=torch.complex64)

    iq = apply_carrier_frequency(iq, carrier_hz=carrier_hz, sample_rate_hz=sample_rate_hz)

    iq = impair_iq(
        iq=iq,
        snr_db=snr_db,
        noise_color=noise_color,
        freq_offset=freq_offset,
        timing_offset=timing_offset,
        fading_mode=fading_mode,
        fading_block_len=fading_block_len,
        rician_k_db=rician_k_db,
        multipath_taps=multipath_taps,
        burst_probability=burst_probability,
        burst_len_min=burst_len_min,
        burst_len_max=burst_len_max,
        burst_power_ratio_db=burst_power_ratio_db,
        burst_color=burst_color,
        seed=seed,
    )

    metadata = {
        "payload_source": payload_source,
        "message_length": None if message is None else len(message),
        "message": message if isinstance(message, str) else None,
        "random_bits": random_bits,
        "random_seed": random_seed if random_bits is not None else None,
        "payload_len": payload_len,
        "payload_len_bytes": payload_len,
        "target_num_samples": target_num_samples,
        "actual_num_samples": int(iq.numel()),
        "payload_plain_bits": payload_plain_bits,
        "payload_coded_bits": payload_coded_bits,
        "payload_with_pilots_bits": payload_with_pilots_bits,
        "n_repeats": n_repeats,
        "sps": sps,
        "beta": beta,
        "span": span,
        "sample_rate_hz": sample_rate_hz,
        "rf_center_hz": rf_center_hz,
        "carrier_hz": carrier_hz,
        "absolute_rf_hz": rf_center_hz + carrier_hz,
        "fec": fec,
        "interleave": interleave,
        "interleave_rows": interleave_rows,
        "snr_db": snr_db,
        "noise_color": noise_color,
        "freq_offset": freq_offset,
        "timing_offset": timing_offset,
        "fading_mode": fading_mode,
        "fading_block_len": fading_block_len,
        "rician_k_db": rician_k_db,
        "multipath_taps": None if multipath_taps is None else [str(t) for t in multipath_taps],
        "burst_probability": burst_probability,
        "burst_len_min": burst_len_min,
        "burst_len_max": burst_len_max,
        "burst_power_ratio_db": burst_power_ratio_db,
        "burst_color": burst_color,
        "seed": seed,
        "avg_power": measure_power(iq),
    }

    return TXBuildResult(iq=iq.to(dtype=torch.complex64), metadata=metadata)


def _scale_iq_to_peak_power(iq: Union[np.ndarray, torch.Tensor], peak_power: Optional[float]) -> torch.Tensor:
    x = _as_complex_tensor(iq)
    if peak_power is None:
        return x
    if peak_power <= 0.0:
        raise ValueError("peak_power must be > 0 when provided")
    if len(x) == 0:
        return x
    current_peak_power = float(torch.max(torch.abs(x) ** 2).item())
    if current_peak_power <= 0.0:
        return x
    gain = np.sqrt(peak_power / current_peak_power)
    return (x * gain).to(dtype=torch.complex64)


def build_tone_pulse_iq_object(
    *,
    sample_rate_hz: float = 1_000_000.0,
    rf_center_hz: float = 0.0,
    carrier_hz: float = 0.0,
    target_num_samples: Optional[int] = None,
    num_tones: int = 1,
    tone_frequencies_hz: Optional[List[float]] = None,
    tone_amplitudes: Optional[List[float]] = None,
    tone_initial_phases_rad: Optional[List[float]] = None,
    pulse_on_samples: int = 4096,
    pulse_off_samples: int = 0,
    pulse_count: int = 1,
    start_offset_samples: int = 0,
    snr_db: Optional[float] = None,
    noise_color: str = "white",
    freq_offset: float = 0.0,
    timing_offset: float = 1.0,
    fading_mode: str = "none",
    fading_block_len: int = 256,
    rician_k_db: float = 6.0,
    multipath_taps: Optional[List[complex]] = None,
    burst_probability: float = 0.0,
    burst_len_min: int = 16,
    burst_len_max: int = 64,
    burst_power_ratio_db: float = 12.0,
    burst_color: str = "white",
    peak_power: Optional[float] = None,
    seed: int = 1,
) -> TXBuildResult:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if num_tones <= 0:
        raise ValueError("num_tones must be > 0")
    if pulse_on_samples <= 0:
        raise ValueError("pulse_on_samples must be > 0")
    if pulse_off_samples < 0:
        raise ValueError("pulse_off_samples must be >= 0")
    if pulse_count <= 0:
        raise ValueError("pulse_count must be > 0")
    if start_offset_samples < 0:
        raise ValueError("start_offset_samples must be >= 0")

    if tone_frequencies_hz is None:
        tone_frequencies_hz = [0.0] * num_tones
    if len(tone_frequencies_hz) != num_tones:
        raise ValueError("len(tone_frequencies_hz) must equal num_tones")
    if any(abs(f_hz) >= sample_rate_hz / 2.0 for f_hz in tone_frequencies_hz):
        raise ValueError("Each tone frequency must satisfy |f_hz| < sample_rate_hz/2")

    if tone_amplitudes is None:
        tone_amplitudes = [1.0] * num_tones
    if len(tone_amplitudes) != num_tones:
        raise ValueError("len(tone_amplitudes) must equal num_tones")
    if any(a < 0.0 for a in tone_amplitudes):
        raise ValueError("tone_amplitudes must be non-negative")

    rng = np.random.default_rng(seed)
    if tone_initial_phases_rad is None:
        tone_initial_phases_rad = rng.uniform(0.0, 2.0 * np.pi, size=num_tones).tolist()
    if len(tone_initial_phases_rad) != num_tones:
        raise ValueError("len(tone_initial_phases_rad) must equal num_tones")

    cycle_samples = pulse_on_samples + pulse_off_samples
    natural_len = start_offset_samples + (pulse_count * cycle_samples) - pulse_off_samples
    total_samples = natural_len if target_num_samples is None else int(target_num_samples)
    if total_samples <= 0:
        raise ValueError("target_num_samples must be > 0 when provided")

    gate = torch.zeros(total_samples, dtype=torch.float32, device=DEFAULT_TORCH_DEVICE)
    for k in range(pulse_count):
        start = start_offset_samples + k * cycle_samples
        end = min(start + pulse_on_samples, total_samples)
        if start >= total_samples:
            break
        gate[start:end] = 1.0

    n = torch.arange(total_samples, dtype=torch.float64, device=DEFAULT_TORCH_DEVICE)
    iq = torch.zeros(total_samples, dtype=torch.complex64, device=DEFAULT_TORCH_DEVICE)
    for a, f_hz, ph in zip(tone_amplitudes, tone_frequencies_hz, tone_initial_phases_rad):
        if a == 0.0:
            continue
        phase = 2.0 * torch.pi * f_hz * n / sample_rate_hz + ph
        iq = iq + a * torch.complex(torch.cos(phase), torch.sin(phase)).to(dtype=torch.complex64)
    iq = (iq * gate).to(dtype=torch.complex64)

    iq = _scale_iq_to_peak_power(iq, peak_power)
    iq = apply_carrier_frequency(iq, carrier_hz=carrier_hz, sample_rate_hz=sample_rate_hz)
    iq = impair_iq(
        iq=iq,
        snr_db=snr_db,
        noise_color=noise_color,
        freq_offset=freq_offset,
        timing_offset=timing_offset,
        fading_mode=fading_mode,
        fading_block_len=fading_block_len,
        rician_k_db=rician_k_db,
        multipath_taps=multipath_taps,
        burst_probability=burst_probability,
        burst_len_min=burst_len_min,
        burst_len_max=burst_len_max,
        burst_power_ratio_db=burst_power_ratio_db,
        burst_color=burst_color,
        seed=seed,
    )

    metadata = {
        "payload_source": "tone_pulse",
        "sample_rate_hz": sample_rate_hz,
        "rf_center_hz": rf_center_hz,
        "carrier_hz": carrier_hz,
        "absolute_rf_hz": rf_center_hz + carrier_hz,
        "target_num_samples": target_num_samples,
        "actual_num_samples": int(iq.numel()),
        "num_tones": num_tones,
        "tone_frequencies_hz": [float(x) for x in tone_frequencies_hz],
        "tone_amplitudes": [float(x) for x in tone_amplitudes],
        "tone_initial_phases_rad": [float(x) for x in tone_initial_phases_rad],
        "pulse_on_samples": pulse_on_samples,
        "pulse_off_samples": pulse_off_samples,
        "pulse_count": pulse_count,
        "start_offset_samples": start_offset_samples,
        "snr_db": snr_db,
        "noise_color": noise_color,
        "freq_offset": freq_offset,
        "timing_offset": timing_offset,
        "fading_mode": fading_mode,
        "fading_block_len": fading_block_len,
        "rician_k_db": rician_k_db,
        "multipath_taps": None if multipath_taps is None else [str(t) for t in multipath_taps],
        "burst_probability": burst_probability,
        "burst_len_min": burst_len_min,
        "burst_len_max": burst_len_max,
        "burst_power_ratio_db": burst_power_ratio_db,
        "burst_color": burst_color,
        "peak_power": peak_power,
        "seed": seed,
        "avg_power": measure_power(iq),
        "measured_peak_power": float(torch.max(torch.abs(iq) ** 2).item()) if iq.numel() else 0.0,
    }
    return TXBuildResult(iq=iq.to(dtype=torch.complex64), metadata=metadata)


def save_tx_iq_object(
    result: TXBuildResult,
    iq_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
) -> Tuple[str, str]:
    iq_path = str(iq_path)
    if metadata_path is None:
        metadata_path = default_metadata_path(iq_path)
    else:
        metadata_path = str(metadata_path)

    save_iq(iq_path, result.iq)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(result.metadata, f, indent=2)

    return iq_path, metadata_path


def read_message_arg(args) -> Union[str, bytes]:
    if getattr(args, "message", None) is not None:
        return args.message
    if getattr(args, "message_file", None) is not None:
        with open(args.message_file, "rb") as f:
            raw = f.read()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw
    raise ValueError("Provide --message or --message-file")


def tx_command(args):
    message = None
    if args.random_bits is None:
        message = read_message_arg(args)

    multipath_taps = None
    if args.multipath_taps:
        multipath_taps = [complex(t) for t in args.multipath_taps.split(",")]

    result = build_tx_iq_object(
        message=message,
        random_bits=args.random_bits,
        random_seed=args.random_seed,
        target_num_samples=args.target_num_samples,
        fec=args.fec,
        interleave=args.interleave,
        interleave_rows=args.interleave_rows,
        sps=args.sps,
        beta=args.beta,
        span=args.span,
        sample_rate_hz=args.sample_rate_hz,
        rf_center_hz=args.rf_center_hz,
        carrier_hz=args.carrier_hz,
        snr_db=args.snr_db,
        noise_color=args.noise_color,
        freq_offset=args.freq_offset,
        timing_offset=args.timing_offset,
        fading_mode=args.fading_mode,
        fading_block_len=args.fading_block_len,
        rician_k_db=args.rician_k_db,
        multipath_taps=multipath_taps,
        burst_probability=args.burst_probability,
        burst_len_min=args.burst_len_min,
        burst_len_max=args.burst_len_max,
        burst_power_ratio_db=args.burst_power_ratio_db,
        burst_color=args.burst_color,
        seed=args.seed,
    )

    save_iq(args.output, result.iq)
    metadata_path = save_iq_metadata(
        iq_path=args.output,
        sample_rate_hz=args.sample_rate_hz,
        rf_center_hz=args.rf_center_hz,
        carrier_hz=args.carrier_hz,
        metadata_path=args.metadata_path,
    )

    merged_meta = dict(result.metadata)
    merged_meta["output"] = args.output
    merged_meta["metadata_path"] = metadata_path
    return merged_meta


def coarse_frequency_acquire(
    iq: Union[np.ndarray, torch.Tensor],
    ref_waveform: Union[np.ndarray, torch.Tensor],
    sample_rate_hz: float,
    search_hz: float,
    n_bins: int,
) -> Tuple[int, float, float]:
    x = _as_complex_tensor(iq)
    ref = _as_complex_tensor(ref_waveform)
    if x.numel() < ref.numel():
        raise RuntimeError("IQ shorter than reference waveform")

    best_metric = -1.0
    best_start = 0
    best_cfo = 0.0

    n = torch.arange(x.numel(), dtype=torch.float64, device=x.device)
    bins = torch.linspace(-search_hz, search_hz, n_bins, dtype=torch.float64, device=x.device)
    kr = torch.flip(torch.conj(ref).real.to(dtype=torch.float32), dims=[0]).reshape(1, 1, -1)
    ki = torch.flip(torch.conj(ref).imag.to(dtype=torch.float32), dims=[0]).reshape(1, 1, -1)

    for f_hz in bins:
        phase = -2.0 * torch.pi * f_hz * n / sample_rate_hz
        rot = torch.complex(torch.cos(phase), torch.sin(phase)).to(dtype=torch.complex64)
        y = x * rot

        yr_in = y.real.to(dtype=torch.float32).reshape(1, 1, -1)
        yi_in = y.imag.to(dtype=torch.float32).reshape(1, 1, -1)
        yr = F.conv1d(yr_in, kr).reshape(-1) - F.conv1d(yi_in, ki).reshape(-1)
        yi = F.conv1d(yr_in, ki).reshape(-1) + F.conv1d(yi_in, kr).reshape(-1)
        mag = torch.abs(torch.complex(yr, yi))

        idx = int(torch.argmax(mag).item())
        metric = float(mag[idx].item())
        if metric > best_metric:
            best_metric = metric
            best_start = idx
            best_cfo = float(f_hz.item())

    print(f'from loop best_start : {best_start}, type : {type(best_start)}')
    print(f'from loop best_cfo : {best_cfo}, type : {type(best_cfo)}')
    print(f'from loop best_metric : {best_metric}, type : {type(best_metric)}')

    
    return best_start, best_cfo, best_metric


def estimate_residual_cfo_from_preamble(
    symbols: Union[np.ndarray, torch.Tensor],
    symbol_rate_hz: float,
) -> float:
    syms = _as_complex_tensor(symbols)
    half = PREAMBLE_HALF_LEN_BITS
    reps = PREAMBLE_REPS
    if syms.numel() < PREAMBLE_BITS_LEN:
        return 0.0

    phases = []
    for r in range(reps - 1):
        s1 = syms[r * half:(r + 1) * half]
        s2 = syms[(r + 1) * half:(r + 2) * half]
        if s1.numel() == half and s2.numel() == half:
            ph = torch.angle(torch.sum(s2 * torch.conj(s1)))
            phases.append(ph / (2.0 * np.pi * half))
    if not phases:
        return 0.0
    return float(torch.mean(torch.stack(phases)).item() * symbol_rate_hz)


def apply_symbol_rate_cfo(symbols: Union[np.ndarray, torch.Tensor], cfo_hz: float, symbol_rate_hz: float) -> torch.Tensor:
    syms = _as_complex_tensor(symbols)
    if cfo_hz == 0.0:
        return syms
    n = torch.arange(syms.numel(), dtype=torch.float64, device=syms.device)
    phase = -2.0 * torch.pi * cfo_hz * n / symbol_rate_hz
    rot = torch.complex(torch.cos(phase), torch.sin(phase)).to(dtype=torch.complex64)
    return (syms * rot).to(dtype=torch.complex64)


def matched_filter(iq: Union[np.ndarray, torch.Tensor], sps: int, beta: float, span: int) -> torch.Tensor:
    taps = rrc_taps(beta=beta, sps=sps, span=span)
    return _complex_lfilter_fir(iq, taps)


def extract_symbols_from_start(
    mf: Union[np.ndarray, torch.Tensor],
    start_index_samples: int,
    sps: int,
    span: int,
    sample_delta: int = 0,
) -> torch.Tensor:
    mft = _as_complex_tensor(mf)
    first = start_index_samples + 2 * span * sps + sample_delta
    if first < 0 or first >= mft.numel():
        return torch.zeros(0, dtype=torch.complex64, device=mft.device)
    return mft[first::sps].to(dtype=torch.complex64)


def design_symbol_equalizer_ls(
    rx_train: Union[np.ndarray, torch.Tensor],
    tx_train: Union[np.ndarray, torch.Tensor],
    ntaps: int = 7,
    ridge: float = 1e-3,
) -> torch.Tensor:
    rx_train_t = _as_complex_tensor(rx_train).to(dtype=torch.complex128)
    tx_train_t = _as_complex_tensor(tx_train).to(dtype=torch.complex128)

    if ntaps % 2 == 0:
        raise ValueError("ntaps must be odd")
    if rx_train_t.numel() != tx_train_t.numel():
        raise ValueError("Training sequences must have same length")
    if rx_train_t.numel() < ntaps:
        return torch.tensor([1.0 + 0.0j], dtype=torch.complex128, device=rx_train_t.device)

    half = ntaps // 2
    rpad = F.pad(rx_train_t.reshape(1, 1, -1), (half, half)).reshape(-1)
    X = torch.stack([rpad[i:i + ntaps] for i in range(rx_train_t.numel())], dim=0)

    A = X.conj().T @ X + ridge * torch.eye(ntaps, dtype=torch.complex128, device=rx_train_t.device)
    b = X.conj().T @ tx_train_t
    w = torch.linalg.solve(A, b)
    return w


def apply_symbol_equalizer(symbols: Union[np.ndarray, torch.Tensor], w: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    sym_t = symbols if isinstance(symbols, torch.Tensor) else _as_complex_tensor(symbols)
    w_t = w if isinstance(w, torch.Tensor) else _as_complex_tensor(w)
    if not torch.is_complex(sym_t):
        sym_t = sym_t.to(dtype=torch.complex64)
    if not torch.is_complex(w_t):
        w_t = w_t.to(dtype=torch.complex64)
    common_dtype = torch.promote_types(sym_t.dtype, w_t.dtype)
    sym_t = sym_t.to(dtype=common_dtype)
    w_t = w_t.to(dtype=common_dtype)
    ntaps = int(w_t.numel())
    if ntaps == 1:
        return sym_t * w_t[0]

    half = ntaps // 2
    spad = F.pad(sym_t.reshape(1, 1, -1), (half, half)).reshape(-1)
    X = torch.stack([spad[i:i + ntaps] for i in range(sym_t.numel())], dim=0)
    out = X @ w_t
    return out


def apply_pilot_phase_tracking(
    symbols: Union[np.ndarray, torch.Tensor],
    data_with_pilots_len: int,
    data_only_len: int,
    access_and_headers_len: int,
) -> torch.Tensor:
    symbols_t = _as_complex_tensor(symbols)
    if symbols_t.numel() < access_and_headers_len + data_with_pilots_len:
        return symbols_t.to(dtype=torch.complex64)

    y = symbols_t.clone()
    payload_stream = y[access_and_headers_len:access_and_headers_len + data_with_pilots_len]

    pos = pilot_positions(data_only_len, interval=PILOT_INTERVAL_BITS, p_len=PILOT_BLOCK_BITS)
    pilot_syms = bpsk_map(PILOT_BITS).to(device=symbols_t.device, dtype=symbols_t.dtype)

    phase_est = 0.0
    for data_start, data_end, pilot_start, pilot_end in pos:
        if pilot_start >= 0 and pilot_end <= payload_stream.numel():
            rx_pilot = payload_stream[pilot_start:pilot_end]
            if rx_pilot.numel() == pilot_syms.numel():
                ph = torch.angle(torch.sum(rx_pilot * torch.conj(pilot_syms)))
                phase_est = float(ph.item())
                payload_stream[data_start:pilot_start] *= torch.exp(
                    torch.tensor(-1j * phase_est, dtype=symbols_t.dtype, device=symbols_t.device)
                )
            else:
                payload_stream[data_start:data_end] *= torch.exp(
                    torch.tensor(-1j * phase_est, dtype=symbols_t.dtype, device=symbols_t.device)
                )
        else:
            payload_stream[data_start:data_end] *= torch.exp(
                torch.tensor(-1j * phase_est, dtype=symbols_t.dtype, device=symbols_t.device)
            )

    y[access_and_headers_len:access_and_headers_len + data_with_pilots_len] = payload_stream
    return y


def choose_valid_header_from_copies(header_soft_all: np.ndarray) -> Optional[int]:
    copy_len = HEADER_PROT_BITS_LEN
    vals = []
    for c in range(HEADER_COPIES):
        s = header_soft_all[c * copy_len:(c + 1) * copy_len]
        if len(s) < copy_len:
            continue
        bits = rep3_decode_soft(s)[:HEADER_BITS_LEN]
        hdr = bits_to_bytes_msb(bits)
        payload_len = parse_header_bytes(hdr)
        if payload_len is not None:
            vals.append(payload_len)
    if not vals:
        return None
    best = max(set(vals), key=vals.count)
    return int(best)


def try_decode_from_symbols(
    symbols: Union[np.ndarray, torch.Tensor],
    fec_mode: str,
    interleave: bool,
    interleave_rows: int,
    symbol_rate_hz: float,
    eq_taps: int,
) -> Optional[bytes]:
    symbols_t = _as_complex_tensor(symbols).to(dtype=torch.complex128)
    access_syms = bpsk_map(ACCESS_BITS).to(device=symbols_t.device, dtype=torch.complex128)

    if symbols_t.numel() < access_syms.numel() + HEADER_COPIES * HEADER_PROT_BITS_LEN:
        return None

    residual_cfo_hz = estimate_residual_cfo_from_preamble(symbols_t[:PREAMBLE_BITS_LEN], symbol_rate_hz)
    symbols_t = apply_symbol_rate_cfo(symbols_t, residual_cfo_hz, symbol_rate_hz)

    train_start = PREAMBLE_BITS_LEN + SYNC_BITS_LEN
    train_end = train_start + TRAINING_LEN_BITS
    if symbols_t.numel() < train_end:
        return None

    rx_train = symbols_t[train_start:train_end]
    tx_train = access_syms[train_start:train_end]

    w = design_symbol_equalizer_ls(rx_train, tx_train, ntaps=eq_taps, ridge=1e-3)
    eq_symbols = apply_symbol_equalizer(symbols_t, w)

    rx_train_eq = eq_symbols[train_start:train_end]
    ph = torch.angle(torch.sum(rx_train_eq * torch.conj(tx_train)))
    eq_symbols = eq_symbols * torch.exp(torch.tensor(-1j * float(ph.item()), dtype=eq_symbols.dtype, device=eq_symbols.device))

    soft_bits = eq_symbols.real.to(dtype=torch.float64)

    hdr_start = ACCESS_BITS_LEN
    hdr_end = hdr_start + HEADER_COPIES * HEADER_PROT_BITS_LEN
    if hdr_end > soft_bits.numel():
        return None

    payload_len = choose_valid_header_from_copies(
        soft_bits[hdr_start:hdr_end].detach().cpu().numpy().astype(np.float64, copy=False)
    )
    if payload_len is None:
        return None

    payload_plain_bits = (payload_len + 4) * 8
    fec = FECCodec(fec_mode)
    payload_coded_bits = fec.encoded_length(payload_plain_bits)

    n_pilot_blocks = max(0, math.ceil(max(payload_coded_bits - PILOT_INTERVAL_BITS, 0) / PILOT_INTERVAL_BITS))
    payload_with_pilots_bits = payload_coded_bits + n_pilot_blocks * PILOT_BLOCK_BITS

    data_start = hdr_end
    data_end = data_start + payload_with_pilots_bits
    if data_end > soft_bits.numel():
        return None

    eq_symbols = apply_pilot_phase_tracking(
        eq_symbols,
        data_with_pilots_len=payload_with_pilots_bits,
        data_only_len=payload_coded_bits,
        access_and_headers_len=data_start,
    )
    soft_bits = eq_symbols.real.to(dtype=torch.float64)

    payload_soft_with_pilots = soft_bits[data_start:data_end].detach().cpu().numpy().astype(np.float64, copy=False)
    payload_soft = remove_pilots_soft(
        payload_soft_with_pilots,
        original_data_len=payload_coded_bits,
        interval=PILOT_INTERVAL_BITS,
        pilot_bits=PILOT_BITS,
    )

    if interleave:
        payload_soft = block_deinterleave_soft(payload_soft, rows=interleave_rows)

    decoded_bits = fec.decode_soft(payload_soft)
    if len(decoded_bits) < payload_plain_bits:
        return None

    decoded_bits = decoded_bits[:payload_plain_bits]
    decoded_bits = descramble_bits(decoded_bits, seed=0x5D)
    decoded_bytes = bits_to_bytes_msb(decoded_bits)

    return parse_payload_bytes(decoded_bytes, payload_len)


def try_decode_from_symbols_numpy_legacy(
    symbols: Union[np.ndarray, torch.Tensor],
    fec_mode: str,
    interleave: bool,
    interleave_rows: int,
    symbol_rate_hz: float,
    eq_taps: int,
) -> Optional[bytes]:
    symbols = _as_numpy_complex64(symbols)
    symbols_t = symbols  # legacy compatibility alias
    access_syms = _as_numpy_complex64(bpsk_map(ACCESS_BITS))

    if len(symbols_t) < len(access_syms) + HEADER_COPIES * HEADER_PROT_BITS_LEN:
        return None

    residual_cfo_hz = estimate_residual_cfo_from_preamble(symbols[:PREAMBLE_BITS_LEN], symbol_rate_hz)
    if residual_cfo_hz != 0.0:
        n = np.arange(len(symbols), dtype=np.float64)
        symbols = (symbols * np.exp(-1j * 2.0 * np.pi * residual_cfo_hz * n / symbol_rate_hz)).astype(np.complex64)

    train_start = PREAMBLE_BITS_LEN + SYNC_BITS_LEN
    train_end = train_start + TRAINING_LEN_BITS
    if symbols_t.numel() < train_end:
        return None

    rx_train = symbols[train_start:train_end]
    tx_train = access_syms[train_start:train_end]

    w = _as_numpy_complex64(design_symbol_equalizer_ls(rx_train, tx_train, ntaps=eq_taps, ridge=1e-3))
    eq_symbols = _as_numpy_complex64(apply_symbol_equalizer(symbols, w))

    rx_train_eq = eq_symbols[train_start:train_end]
    ph = np.angle(np.sum(rx_train_eq * np.conj(tx_train)))
    eq_symbols = (eq_symbols * np.exp(-1j * ph)).astype(np.complex64)

    soft_bits = eq_symbols.real.to(dtype=torch.float64)

    hdr_start = ACCESS_BITS_LEN
    hdr_end = hdr_start + HEADER_COPIES * HEADER_PROT_BITS_LEN
    if hdr_end > soft_bits.numel():
        return None

    payload_len = choose_valid_header_from_copies(
        soft_bits[hdr_start:hdr_end].detach().cpu().numpy().astype(np.float64, copy=False)
    )
    if payload_len is None:
        return None

    payload_plain_bits = (payload_len + 4) * 8
    fec = FECCodec(fec_mode)
    payload_coded_bits = fec.encoded_length(payload_plain_bits)

    n_pilot_blocks = max(0, math.ceil(max(payload_coded_bits - PILOT_INTERVAL_BITS, 0) / PILOT_INTERVAL_BITS))
    payload_with_pilots_bits = payload_coded_bits + n_pilot_blocks * PILOT_BLOCK_BITS

    data_start = hdr_end
    data_end = data_start + payload_with_pilots_bits
    if data_end > soft_bits.numel():
        return None

    eq_symbols = _as_numpy_complex64(
        apply_pilot_phase_tracking(
            eq_symbols,
            data_with_pilots_len=payload_with_pilots_bits,
            data_only_len=payload_coded_bits,
            access_and_headers_len=data_start,
        )
    )
    soft_bits = eq_symbols.real.to(dtype=torch.float64)

    payload_soft_with_pilots = soft_bits[data_start:data_end].detach().cpu().numpy().astype(np.float64, copy=False)
    payload_soft = remove_pilots_soft(
        payload_soft_with_pilots,
        original_data_len=payload_coded_bits,
        interval=PILOT_INTERVAL_BITS,
        pilot_bits=PILOT_BITS,
    )

    if interleave:
        payload_soft = block_deinterleave_soft(payload_soft, rows=interleave_rows)

    decoded_bits = fec.decode_soft(payload_soft)
    if len(decoded_bits) < payload_plain_bits:
        return None

    decoded_bits = decoded_bits[:payload_plain_bits]
    decoded_bits = descramble_bits(decoded_bits, seed=0x5D)
    decoded_bytes = bits_to_bytes_msb(decoded_bits)
    return parse_payload_bytes(decoded_bytes, payload_len)


def try_decode_over_sample_deltas(
    mf: Union[np.ndarray, torch.Tensor],
    start_index_samples: int,
    sps: int,
    span: int,
    sample_phase_search: int,
    fec_mode: str,
    interleave: bool,
    interleave_rows: int,
    symbol_rate_hz: float,
    eq_taps: int,
) -> Tuple[Optional[bytes], Optional[int]]:
    deltas = list(range(-sample_phase_search, sample_phase_search + 1))
    mf_t = _as_complex_tensor(mf)

    def _worker(sample_delta: int) -> Tuple[int, Optional[bytes]]:
        symbols = extract_symbols_from_start(
            mf=mf_t,
            start_index_samples=start_index_samples,
            sps=sps,
            span=span,
            sample_delta=sample_delta,
        )
        payload = try_decode_from_symbols(
            symbols=symbols,
            fec_mode=fec_mode,
            interleave=interleave,
            interleave_rows=interleave_rows,
            symbol_rate_hz=symbol_rate_hz,
            eq_taps=eq_taps,
        )
        if payload is None:
            payload = try_decode_from_symbols_numpy_legacy(
                symbols=symbols,
                fec_mode=fec_mode,
                interleave=interleave,
                interleave_rows=interleave_rows,
                symbol_rate_hz=symbol_rate_hz,
                eq_taps=eq_taps,
            )
        return sample_delta, payload

    # GPU tensor decode attempts are kept sequential for determinism/stability.
    if mf_t.is_cuda:
        for d in deltas:
            sample_delta, payload = _worker(d)
            if payload is not None:
                return payload, sample_delta
        return None, None

    max_workers = max(1, min(len(deltas), 8))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_worker, deltas))

    for sample_delta, payload in results:
        if payload is not None:
            return payload, sample_delta
    return None, None


def rx_command(args):
    iq = _as_complex_tensor(load_iq(args.input))
    meta = load_iq_metadata(args.input, metadata_path=args.metadata_path)

    if meta is not None:
        tx_sample_rate_hz = float(meta["sample_rate_hz"])
        tx_rf_center_hz = float(meta["rf_center_hz"])
        tx_carrier_hz = float(meta["carrier_hz"])
        tx_absolute_rf_hz = float(meta["absolute_rf_hz"])
    else:
        if args.tx_sample_rate_hz is None:
            raise ValueError("No metadata found and --tx-sample-rate-hz was not provided")
        tx_sample_rate_hz = float(args.tx_sample_rate_hz)
        tx_rf_center_hz = float(args.tx_rf_center_hz)
        tx_carrier_hz = float(args.tx_carrier_hz)
        tx_absolute_rf_hz = tx_rf_center_hz + tx_carrier_hz

    rx_sample_rate_hz = float(args.sample_rate_hz) if args.sample_rate_hz is not None else tx_sample_rate_hz
    rx_rf_center_hz = float(args.rf_center_hz)
    sps = int(args.sps)
    symbol_rate_hz = tx_sample_rate_hz / sps
    
    # print(args)
    
    baseband_offset_hz = tx_absolute_rf_hz - rx_rf_center_hz
    # print(f'baseband_offset_hz : {baseband_offset_hz}')
    
    if abs(baseband_offset_hz) >= tx_sample_rate_hz / 2:
        raise ValueError("RX RF center causes out-of-band baseband offset")

    iq = apply_carrier_frequency(iq, carrier_hz=-baseband_offset_hz, sample_rate_hz=tx_sample_rate_hz)

    if not np.isclose(rx_sample_rate_hz, tx_sample_rate_hz):
        iq = resample_iq(iq, fs_in_hz=tx_sample_rate_hz, fs_out_hz=rx_sample_rate_hz)

    dsp_sample_rate_hz = tx_sample_rate_hz
    if not np.isclose(rx_sample_rate_hz, dsp_sample_rate_hz):
        iq = resample_iq(iq, fs_in_hz=rx_sample_rate_hz, fs_out_hz=dsp_sample_rate_hz)

    iq = robust_agc_and_blanking(iq, blanker_k=6.0)

    access_ref_waveform = tx_waveform(ACCESS_BITS, sps=sps, beta=args.beta, span=args.span)

    coarse_start, coarse_cfo_hz, coarse_metric = coarse_frequency_acquire(
        iq=iq,
        ref_waveform=access_ref_waveform,
        sample_rate_hz=dsp_sample_rate_hz,
        search_hz=args.coarse_freq_search_hz,
        n_bins=args.coarse_freq_bins,
    )

    iq = apply_carrier_frequency(iq, carrier_hz=-coarse_cfo_hz, sample_rate_hz=dsp_sample_rate_hz)
    mf = matched_filter(iq, sps=sps, beta=args.beta, span=args.span)

    payload, sample_offset_used = try_decode_over_sample_deltas(
        mf=mf,
        start_index_samples=coarse_start,
        sps=sps,
        span=args.span,
        sample_phase_search=args.sample_phase_search,
        fec_mode=args.fec,
        interleave=args.interleave,
        interleave_rows=args.interleave_rows,
        symbol_rate_hz=symbol_rate_hz,
        eq_taps=args.eq_taps,
    )

    if payload is None:
        raise RuntimeError("No valid packet found after acquisition, header decode, FEC decode, and CRC")

    try:
        message_text = payload.decode("utf-8")
    except UnicodeDecodeError:
        message_text = None

    result = {
        "payload_bytes": payload,
        "payload_len": len(payload),
        "message": message_text,
        "sample_offset_used": sample_offset_used,
        "coarse_cfo_hz": coarse_cfo_hz,
        "coarse_metric": coarse_metric,
    }

    if args.output_file:
        mode = "w" if message_text is not None else "wb"
        with open(args.output_file, mode) as f:
            if message_text is not None:
                f.write(message_text)
            else:
                f.write(payload)

    return result

def rx_command_iq(iq, meta):
    # iq = load_iq(args.input)
    # meta = load_iq_metadata(args.input, metadata_path=args.metadata_path)

    iq = _as_complex_tensor(iq)

    tx_sample_rate_hz = float(meta["sample_rate_hz"])
    tx_rf_center_hz = float(meta["rf_center_hz"])
    tx_carrier_hz = float(meta["carrier_hz"])
    tx_absolute_rf_hz = float(meta["absolute_rf_hz"])

    rx_sample_rate_hz = tx_sample_rate_hz
    rx_rf_center_hz = tx_rf_center_hz

    sps = int(meta["sps"])
    beta = float(meta["beta"])
    span = int(meta["span"])
    sample_rate_hz = float(meta["sample_rate_hz"])
    rf_center_hz = float(meta["rf_center_hz"])

    fec = meta['fec']
    # print(f'fec : {fec}')
    interleave = bool(meta['interleave'])
    interleave_rows = int(meta['interleave_rows'])
    
    coarse_freq_search_hz = 25_000.0 # float(meta["coarse_freq_search_hz"])
    coarse_freq_bins = 101 # int(meta["coarse_freq_bins"])
    sample_phase_search = 3 # float(meta["sample_phase_search"])
    eq_taps = 7 # int(meta["eq_taps"])

    symbol_rate_hz = tx_sample_rate_hz / sps

    baseband_offset_hz = tx_absolute_rf_hz - rx_rf_center_hz
    # print(f'baseband_offset_hz : {baseband_offset_hz}')
    
    if abs(baseband_offset_hz) >= tx_sample_rate_hz / 2:
        raise ValueError("RX RF center causes out-of-band baseband offset")

    iq = apply_carrier_frequency(iq, carrier_hz=-baseband_offset_hz, sample_rate_hz=tx_sample_rate_hz)

    if not np.isclose(rx_sample_rate_hz, tx_sample_rate_hz):
        iq = resample_iq(iq, fs_in_hz=tx_sample_rate_hz, fs_out_hz=rx_sample_rate_hz)

    dsp_sample_rate_hz = tx_sample_rate_hz
    if not np.isclose(rx_sample_rate_hz, dsp_sample_rate_hz):
        iq = resample_iq(iq, fs_in_hz=rx_sample_rate_hz, fs_out_hz=dsp_sample_rate_hz)

    iq = robust_agc_and_blanking(iq, blanker_k=6.0)

    access_ref_waveform = tx_waveform(ACCESS_BITS, sps=sps, beta=beta, span=span)

    # loop_coarse_start, loop_coarse_cfo_hz, loop_coarse_metric = loop_coarse_frequency_acquire(
    #     iq=iq,
    #     ref_waveform=access_ref_waveform,
    #     sample_rate_hz=dsp_sample_rate_hz,
    #     search_hz=coarse_freq_search_hz,
    #     n_bins=coarse_freq_bins,
    # )
    
    coarse_start, coarse_cfo_hz, coarse_metric = coarse_frequency_acquire(
        iq=iq,
        ref_waveform=access_ref_waveform,
        sample_rate_hz=dsp_sample_rate_hz,
        search_hz=coarse_freq_search_hz,
        n_bins=coarse_freq_bins,
    )

    

    iq = apply_carrier_frequency(iq, carrier_hz=-coarse_cfo_hz, sample_rate_hz=dsp_sample_rate_hz)
    mf = matched_filter(iq, sps=sps, beta=beta, span=span)

    payload, sample_offset_used = try_decode_over_sample_deltas(
        mf=mf,
        start_index_samples=coarse_start,
        sps=sps,
        span=span,
        sample_phase_search=sample_phase_search,
        fec_mode=fec,
        interleave=interleave,
        interleave_rows=interleave_rows,
        symbol_rate_hz=symbol_rate_hz,
        eq_taps=eq_taps,
    )

    if payload is None:
        raise RuntimeError("No valid packet found after acquisition, header decode, FEC decode, and CRC")
            
    try:
        message_text = payload.decode("utf-8")
    except UnicodeDecodeError:
        message_text = None

    result = {
        "payload_bytes": payload,
        "payload_len": len(payload),
        "message": message_text,
        "sample_offset_used": sample_offset_used,
        "coarse_cfo_hz": coarse_cfo_hz,
        "coarse_metric": coarse_metric,
    }


    return result


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    tx = sub.add_parser("tx")
    tx.add_argument("--message", type=str)
    tx.add_argument("--message-file", type=str)
    tx.add_argument("--random-bits", type=int, default=None)
    tx.add_argument("--random-seed", type=int, default=1)
    tx.add_argument("--output", type=str, default="tx_skdsp_v4.iq")
    tx.add_argument("--metadata-path", type=str, default=None)
    tx.add_argument("--target-num-samples", type=int, default=None)

    tx.add_argument("--sps", type=int, default=8)
    tx.add_argument("--beta", type=float, default=0.35)
    tx.add_argument("--span", type=int, default=6)
    tx.add_argument("--seed", type=int, default=1)

    tx.add_argument("--sample-rate-hz", type=float, default=1_000_000.0)
    tx.add_argument("--rf-center-hz", type=float, default=0.0)
    tx.add_argument("--carrier-hz", type=float, default=0.0)

    tx.add_argument("--fec", choices=[FEC_NONE, FEC_REP3, FEC_CONV], default=FEC_NONE)
    tx.add_argument("--interleave", action="store_true")
    tx.add_argument("--interleave-rows", type=int, default=8)

    tx.add_argument("--snr-db", type=float, default=None)
    tx.add_argument("--noise-color", choices=["white", "pink", "brown", "blue", "violet"], default="white")
    tx.add_argument("--freq-offset", type=float, default=0.0)
    tx.add_argument("--timing-offset", type=float, default=1.0)

    tx.add_argument("--fading-mode", choices=["none", "rayleigh_block", "rician_block", "multipath_static"], default="none")
    tx.add_argument("--fading-block-len", type=int, default=256)
    tx.add_argument("--rician-k-db", type=float, default=6.0)
    tx.add_argument("--multipath-taps", type=str, default=None)

    tx.add_argument("--burst-probability", type=float, default=0.0)
    tx.add_argument("--burst-len-min", type=int, default=16)
    tx.add_argument("--burst-len-max", type=int, default=64)
    tx.add_argument("--burst-power-ratio-db", type=float, default=12.0)
    tx.add_argument("--burst-color", choices=["white", "pink", "brown", "blue", "violet"], default="white")

    rx = sub.add_parser("rx")
    rx.add_argument("--input", type=str, default="tx_skdsp_v4.iq")
    rx.add_argument("--metadata-path", type=str, default=None)

    rx.add_argument("--sps", type=int, default=8)
    rx.add_argument("--beta", type=float, default=0.35)
    rx.add_argument("--span", type=int, default=6)

    rx.add_argument("--sample-rate-hz", type=float, default=None)
    rx.add_argument("--rf-center-hz", type=float, default=0.0)

    rx.add_argument("--tx-sample-rate-hz", type=float, default=None)
    rx.add_argument("--tx-rf-center-hz", type=float, default=0.0)
    rx.add_argument("--tx-carrier-hz", type=float, default=0.0)

    rx.add_argument("--fec", choices=[FEC_NONE, FEC_REP3, FEC_CONV], default=FEC_NONE)
    rx.add_argument("--interleave", action="store_true")
    rx.add_argument("--interleave-rows", type=int, default=8)

    rx.add_argument("--coarse-freq-search-hz", type=float, default=25_000.0)
    rx.add_argument("--coarse-freq-bins", type=int, default=101)
    rx.add_argument("--sample-phase-search", type=int, default=3)
    rx.add_argument("--eq-taps", type=int, default=7)

    rx.add_argument("--output-file", type=str)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "tx":
        if args.random_bits is None and args.message is None and args.message_file is None:
            raise SystemExit("tx requires one of --message, --message-file, or --random-bits")
        if args.random_bits is not None and (args.message is not None or args.message_file is not None):
            raise SystemExit("Use either random bits or message input, not both.")
        return tx_command(args)

    return rx_command(args)


if __name__ == "__main__":
    main()
