#!/usr/bin/env python3
"""PyTorch port of advanced_link_skdsp_v5_robust_numpy.py."""

from __future__ import annotations

import argparse
import binascii
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


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

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(x, dtype=torch.complex64, device: Optional[torch.device] = None) -> torch.Tensor:
    device = device or DEFAULT_DEVICE
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def save_iq(path: Union[str, Path], iq: torch.Tensor) -> None:
    _to_tensor(iq).detach().cpu().numpy().astype("complex64").tofile(str(path))


def load_iq(path: Union[str, Path]) -> torch.Tensor:
    import numpy as np

    x = np.fromfile(str(path), dtype=np.complex64)
    return torch.from_numpy(x).to(DEFAULT_DEVICE)


def default_metadata_path(iq_path: Union[str, Path]) -> str:
    p = Path(iq_path)
    return str(p.with_suffix(p.suffix + ".json"))


def save_iq_metadata(iq_path, sample_rate_hz, rf_center_hz, carrier_hz, metadata_path=None) -> str:
    metadata_path = default_metadata_path(iq_path) if metadata_path is None else str(metadata_path)
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


def load_iq_metadata(iq_path, metadata_path=None) -> Optional[dict]:
    metadata_path = default_metadata_path(iq_path) if metadata_path is None else str(metadata_path)
    p = Path(metadata_path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def bytes_to_bits_msb(data: bytes) -> List[int]:
    return [((b >> i) & 1) for b in data for i in range(7, -1, -1)]


def bits_to_bytes_msb(bits: List[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Bit count must be a multiple of 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        x = 0
        for bit in bits[i : i + 8]:
            x = (x << 1) | (bit & 1)
        out.append(x)
    return bytes(out)


def prbs_bits(n: int, seed: int) -> List[int]:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randint(0, 2, (n,), generator=g, dtype=torch.int64).tolist()


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


def rrc_taps(beta: float, sps: int, span: int) -> torch.Tensor:
    n = torch.arange(-span * sps, span * sps + 1, dtype=torch.float64, device=DEFAULT_DEVICE)
    t = n / sps
    h = torch.zeros_like(t)
    for i in range(t.numel()):
        ti = float(t[i])
        if abs(ti) < 1e-12:
            h[i] = 1.0 + beta * (4.0 / math.pi - 1.0)
        elif beta > 0 and abs(abs(4 * beta * ti) - 1.0) < 1e-9:
            h[i] = beta / math.sqrt(2.0) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * beta)) + (1 - 2 / math.pi) * math.cos(math.pi / (4 * beta)))
        else:
            num = math.sin(math.pi * ti * (1 - beta)) + 4 * beta * ti * math.cos(math.pi * ti * (1 + beta))
            den = math.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h = h / torch.sqrt(torch.sum(h**2))
    return h


def measure_power(x: torch.Tensor) -> float:
    x = _to_tensor(x)
    return float(torch.mean(torch.abs(x) ** 2).real.item()) if x.numel() else 0.0


def bpsk_map(bits: List[int]) -> torch.Tensor:
    b = torch.as_tensor(bits, dtype=torch.float32, device=DEFAULT_DEVICE)
    return torch.where(b > 0, torch.ones_like(b), -torch.ones_like(b)).to(torch.complex64)


def _lfilter_1d(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    xr = torch.view_as_real(x).T.unsqueeze(0)
    k = taps.flip(0).to(torch.float32).view(1, 1, -1)
    y0 = F.conv1d(xr[:, 0:1, :], k, padding=k.shape[-1] - 1)[:, :, : x.shape[0]]
    y1 = F.conv1d(xr[:, 1:2, :], k, padding=k.shape[-1] - 1)[:, :, : x.shape[0]]
    return torch.complex(y0[0, 0], y1[0, 0]).to(torch.complex64)


def upsample_and_shape(symbols: torch.Tensor, sps: int, taps: torch.Tensor) -> torch.Tensor:
    symbols = _to_tensor(symbols)
    up = torch.zeros(symbols.numel() * sps, dtype=torch.complex64, device=symbols.device)
    up[::sps] = symbols
    return _lfilter_1d(up, taps)


def tx_waveform(bits: List[int], sps: int, beta: float, span: int) -> torch.Tensor:
    return upsample_and_shape(bpsk_map(bits), sps=sps, taps=rrc_taps(beta=beta, sps=sps, span=span))


def apply_carrier_frequency(iq: torch.Tensor, carrier_hz: float, sample_rate_hz: float) -> torch.Tensor:
    iq = _to_tensor(iq)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if abs(carrier_hz) >= sample_rate_hz / 2:
        raise ValueError("carrier_hz must satisfy |carrier_hz| < sample_rate_hz/2")
    if carrier_hz == 0.0:
        return iq.to(torch.complex64)
    n = torch.arange(iq.numel(), dtype=torch.float64, device=iq.device)
    rot = torch.exp(1j * 2.0 * math.pi * carrier_hz * n / sample_rate_hz)
    return (iq * rot).to(torch.complex64)


def apply_frequency_offset(iq: torch.Tensor, freq_offset: float) -> torch.Tensor:
    iq = _to_tensor(iq)
    if freq_offset == 0.0:
        return iq.to(torch.complex64)
    n = torch.arange(iq.numel(), dtype=torch.float64, device=iq.device)
    return (iq * torch.exp(1j * 2.0 * math.pi * freq_offset * n)).to(torch.complex64)


def apply_timing_offset_resample(iq: torch.Tensor, timing_offset: float) -> torch.Tensor:
    iq = _to_tensor(iq)
    if timing_offset == 1.0 or iq.numel() == 0:
        return iq.to(torch.complex64)
    new_len = max(1, int(round(iq.numel() / timing_offset)))
    return resample_iq(iq, float(iq.numel()), float(new_len))[: iq.numel()]


def resample_iq(iq: torch.Tensor, fs_in_hz: float, fs_out_hz: float) -> torch.Tensor:
    iq = _to_tensor(iq)
    if fs_in_hz <= 0 or fs_out_hz <= 0:
        raise ValueError("Sample rates must be positive")
    if abs(fs_in_hz - fs_out_hz) < 1e-9:
        return iq.to(torch.complex64)
    n = iq.numel()
    new_len = max(1, int(round(n * fs_out_hz / fs_in_hz)))
    X = torch.fft.fft(iq)
    if new_len > n:
        pad = new_len - n
        X = torch.cat([X[: n // 2], torch.zeros(pad, dtype=X.dtype, device=X.device), X[n // 2 :]], dim=0)
    else:
        cut = n - new_len
        X = torch.cat([X[: n // 2 - cut // 2], X[n // 2 + (cut - cut // 2) :]], dim=0)
    y = torch.fft.ifft(X) * (new_len / n)
    return y.to(torch.complex64)


def _complex_colored_noise(n: int, color: str, power: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    exponents = {"white": 0.0, "pink": 1.0, "brown": 2.0, "blue": -1.0, "violet": -2.0}
    if color not in exponents:
        raise ValueError(f"Unsupported noise color: {color}")
    alpha = exponents[color]
    g = generator or torch.Generator(device="cpu")
    freqs = torch.fft.rfftfreq(n, d=1.0, device=DEFAULT_DEVICE)
    spec = torch.randn(freqs.numel(), generator=g, device=DEFAULT_DEVICE) + 1j * torch.randn(freqs.numel(), generator=g, device=DEFAULT_DEVICE)
    shaping = torch.ones_like(freqs)
    nz = freqs > 0
    if alpha > 0:
        shaping[nz] = 1.0 / (freqs[nz] ** (alpha / 2.0))
    elif alpha < 0:
        shaping[nz] = freqs[nz] ** ((-alpha) / 2.0)
    if color in ("pink", "brown"):
        shaping[~nz] = 0.0
    y = torch.fft.irfft(spec * shaping, n=n)
    p = torch.mean(y**2)
    if float(p) > 0:
        y = y * torch.sqrt(torch.tensor((power / 2.0), device=y.device) / p)
    z = torch.complex(y, torch.fft.irfft(spec * shaping, n=n))
    pz = torch.mean(torch.abs(z) ** 2)
    if float(pz) > 0:
        z = z * torch.sqrt(torch.tensor(power, device=z.device) / pz)
    return z.to(torch.complex64)


def apply_fading(iq, mode="none", block_len=256, rician_k_db=6.0, multipath_taps=None, seed=1):
    x = _to_tensor(iq)
    g = torch.Generator(device="cpu"); g.manual_seed(int(seed))
    if mode == "none":
        return x
    if mode in ("rayleigh_block", "rician_block"):
        out = x.clone()
        k_lin = 10.0 ** (rician_k_db / 10.0)
        los = math.sqrt(k_lin / (k_lin + 1.0)); scat_scale = math.sqrt(1.0 / (k_lin + 1.0))
        for start in range(0, x.numel(), block_len):
            end = min(start + block_len, x.numel())
            h = (torch.randn((), generator=g, device=x.device) + 1j * torch.randn((), generator=g, device=x.device)) / math.sqrt(2.0)
            if mode == "rician_block":
                h = los + h * scat_scale
            out[start:end] = x[start:end] * h
        return out.to(torch.complex64)
    if mode == "multipath_static":
        taps = multipath_taps if multipath_taps is not None else [1.0 + 0.0j, 0.20 + 0.08j, 0.06 - 0.04j]
        t = torch.as_tensor(taps, dtype=torch.complex64, device=x.device)
        y = F.conv1d(x.view(1, 1, -1), t.flip(0).view(1, 1, -1), padding=t.numel() - 1).view(-1)[: x.numel()]
        return y.to(torch.complex64)
    raise ValueError(f"Unsupported fading mode: {mode}")


def add_impulsive_bursts(iq, base_noise_power, burst_probability=0.0, burst_len_min=16, burst_len_max=64, burst_power_ratio_db=12.0, burst_color="white", seed=1):
    out = _to_tensor(iq).clone()
    if burst_probability <= 0.0 or out.numel() == 0:
        return out
    g = torch.Generator(device="cpu"); g.manual_seed(int(seed + 1000))
    burst_power = base_noise_power * (10.0 ** (burst_power_ratio_db / 10.0))
    idx = 0
    while idx < out.numel():
        if float(torch.rand((), generator=g)) < burst_probability:
            burst_len = int(torch.randint(burst_len_min, burst_len_max + 1, (1,), generator=g).item())
            end = min(idx + burst_len, out.numel())
            out[idx:end] = out[idx:end] + _complex_colored_noise(end - idx, burst_color, burst_power, g)
            idx = end
        else:
            idx += 1
    return out.to(torch.complex64)


def impair_iq(iq, snr_db, noise_color, freq_offset, timing_offset, fading_mode, fading_block_len, rician_k_db, multipath_taps, burst_probability, burst_len_min, burst_len_max, burst_power_ratio_db, burst_color, seed):
    x = apply_fading(iq, mode=fading_mode, block_len=fading_block_len, rician_k_db=rician_k_db, multipath_taps=multipath_taps, seed=seed)
    x = apply_frequency_offset(x, freq_offset)
    x = apply_timing_offset_resample(x, timing_offset)
    sig_power = measure_power(x)
    g = torch.Generator(device="cpu"); g.manual_seed(int(seed))
    if snr_db is None:
        noise_power = 0.0; y = x
    else:
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        y = x + _complex_colored_noise(x.numel(), noise_color, noise_power, g)
    return add_impulsive_bursts(y, base_noise_power=(noise_power if noise_power > 0 else max(sig_power * 1e-6, 1e-12)), burst_probability=burst_probability, burst_len_min=burst_len_min, burst_len_max=burst_len_max, burst_power_ratio_db=burst_power_ratio_db, burst_color=burst_color, seed=seed)


def robust_agc_and_blanking(x, blanker_k=6.0):
    x = _to_tensor(x)
    x = x - torch.mean(x)
    mag = torch.abs(x)
    med = torch.median(mag)
    mad = torch.median(torch.abs(mag - med)) + 1e-12
    thresh = med + blanker_k * 1.4826 * mad
    over = mag > thresh
    if torch.any(over):
        x = x.clone()
        x[over] = thresh * torch.exp(1j * torch.angle(x[over]))
    p = torch.mean(torch.abs(x) ** 2)
    if float(p) > 0:
        x = x / torch.sqrt(p)
    return x.to(torch.complex64)


def lfsr_sequence(n: int, seed: int = 0x5D) -> List[int]:
    state = seed & 0x7F
    if state == 0: state = 0x5D
    out = []
    for _ in range(n):
        new_bit = ((state >> 6) ^ (state >> 5)) & 1
        out.append(state & 1)
        state = ((state << 1) & 0x7F) | new_bit
    return out


def scramble_bits(bits: List[int], seed: int = 0x5D) -> List[int]:
    pn = lfsr_sequence(len(bits), seed=seed)
    return [(b ^ p) & 1 for b, p in zip(bits, pn)]

descramble_bits = scramble_bits


def rep3_encode_bits(bits: List[int]) -> List[int]:
    out = []
    for b in bits: out.extend([b, b, b])
    return out


def rep3_decode_soft(soft: torch.Tensor) -> List[int]:
    soft = _to_tensor(soft, dtype=torch.float64)
    usable = (soft.numel() // 3) * 3
    soft = soft[:usable]
    return [1 if float(torch.sum(soft[i : i + 3])) >= 0 else 0 for i in range(0, usable, 3)]


def block_interleave_bits(bits: List[int], rows: int = 8) -> List[int]:
    if rows <= 1: return bits[:]
    cols = math.ceil(len(bits) / rows)
    padded = bits[:] + [0] * (rows * cols - len(bits))
    mat = torch.tensor(padded, dtype=torch.int8).reshape(rows, cols)
    out = mat.T.reshape(-1).tolist()
    return out[: len(bits)]


def block_deinterleave_soft(soft: torch.Tensor, rows: int = 8) -> torch.Tensor:
    soft = _to_tensor(soft, dtype=torch.float64)
    if rows <= 1: return soft
    n = soft.numel(); cols = math.ceil(n / rows)
    padded = torch.cat([soft, torch.zeros(rows * cols - n, dtype=torch.float64, device=soft.device)])
    mat = padded.reshape(cols, rows).T
    return mat.reshape(-1)[:n]


def conv_encode(bits: List[int]) -> List[int]:
    state = [0, 0]; out = []
    for b in bits + [0, 0]:
        d0, d1, d2 = b, state[0], state[1]
        out.extend([(d0 ^ d1 ^ d2) & 1, (d0 ^ d2) & 1])
        state = [d0, d1]
    return out


def conv_decode_soft(soft: torch.Tensor) -> List[int]:
    soft = _to_tensor(soft, dtype=torch.float64).detach().cpu()
    usable = (soft.numel() // 2) * 2; soft = soft[:usable]; n_steps = soft.numel() // 2
    if n_steps == 0: return []
    INF = 1e18
    pm = torch.full((n_steps + 1, 4), INF, dtype=torch.float64)
    prev_state = torch.full((n_steps, 4), -1, dtype=torch.int64)
    prev_bit = torch.full((n_steps, 4), -1, dtype=torch.int64)
    pm[0, 0] = 0.0
    def bo(state, bit):
        s0 = (state >> 1) & 1; s1 = state & 1; d0 = bit
        o1 = (d0 ^ s0 ^ s1) & 1; o2 = (d0 ^ s1) & 1
        return ((bit << 1) | s0) & 0b11, o1, o2
    for t in range(n_steps):
        y1, y2 = float(soft[2 * t]), float(soft[2 * t + 1])
        for s in range(4):
            if pm[t, s] >= INF / 2: continue
            for b in (0, 1):
                ns, o1, o2 = bo(s, b); e1 = 1.0 if o1 else -1.0; e2 = 1.0 if o2 else -1.0
                metric = pm[t, s] + (y1 - e1) ** 2 + (y2 - e2) ** 2
                if metric < pm[t + 1, ns]:
                    pm[t + 1, ns] = metric; prev_state[t, ns] = s; prev_bit[t, ns] = b
    state = 0; dec = []
    for t in range(n_steps - 1, -1, -1):
        b = int(prev_bit[t, state]); dec.append(0 if b < 0 else b)
        ps = int(prev_state[t, state]); state = ps if ps >= 0 else 0
    dec.reverse();
    return dec[:-2] if len(dec) >= 2 else dec


class FECCodec:
    def __init__(self, mode: str): self.mode = mode
    def encoded_length(self, input_bit_len: int) -> int:
        return input_bit_len if self.mode == FEC_NONE else 3 * input_bit_len if self.mode == FEC_REP3 else 2 * (input_bit_len + 2)
    def encode_bits(self, bits: List[int]) -> List[int]:
        return bits[:] if self.mode == FEC_NONE else rep3_encode_bits(bits) if self.mode == FEC_REP3 else conv_encode(bits)
    def decode_soft(self, soft: torch.Tensor) -> List[int]:
        return (_to_tensor(soft, dtype=torch.float64) > 0).to(torch.int8).tolist() if self.mode == FEC_NONE else rep3_decode_soft(soft) if self.mode == FEC_REP3 else conv_decode_soft(soft)


def build_header_bytes(payload_len: int) -> bytes:
    if payload_len <= 0 or payload_len > MAX_PAYLOAD_BYTES: raise ValueError("invalid payload_len")
    payload_len_inv = payload_len ^ 0xFFFFFFFF
    prefix = struct.pack(">HII", HEADER_MAGIC, payload_len, payload_len_inv)
    crc = binascii.crc32(prefix) & 0xFFFFFFFF
    return prefix + struct.pack(">I", crc)


def parse_header_bytes(header: bytes) -> Optional[int]:
    if len(header) != HEADER_BYTES_LEN: return None
    magic, payload_len, payload_len_inv, hdr_crc = struct.unpack(">HIII", header)
    prefix = struct.pack(">HII", magic, payload_len, payload_len_inv)
    calc_crc = binascii.crc32(prefix) & 0xFFFFFFFF
    if magic != HEADER_MAGIC or payload_len == 0 or payload_len > MAX_PAYLOAD_BYTES or payload_len_inv != (payload_len ^ 0xFFFFFFFF) or hdr_crc != calc_crc:
        return None
    return payload_len


def build_payload_bytes_from_message(message):
    payload = message.encode("utf-8") if isinstance(message, str) else bytes(message)
    if len(payload) == 0: raise ValueError("Empty payloads are not supported")
    return payload + struct.pack(">I", binascii.crc32(payload) & 0xFFFFFFFF)


def build_payload_bytes_from_random_bits(n_bits: int, seed: int) -> bytes:
    if n_bits <= 0: raise ValueError("n_bits must be > 0")
    bits = prbs_bits(n_bits, seed=seed); pad = (-n_bits) % 8
    if pad: bits += [0] * pad
    payload = bits_to_bytes_msb(bits)
    return payload + struct.pack(">I", binascii.crc32(payload) & 0xFFFFFFFF)


def parse_payload_bytes(raw: bytes, payload_len: int) -> Optional[bytes]:
    needed = payload_len + 4
    if len(raw) < needed: return None
    payload = raw[:payload_len]
    rx_crc = struct.unpack(">I", raw[payload_len : payload_len + 4])[0]
    return payload if rx_crc == (binascii.crc32(payload) & 0xFFFFFFFF) else None


def insert_pilots(bits: List[int], interval: int = PILOT_INTERVAL_BITS, pilot_bits: List[int] = PILOT_BITS) -> List[int]:
    out = []; i = 0
    while i < len(bits):
        chunk = bits[i : i + interval]; out.extend(chunk); i += len(chunk)
        if i < len(bits): out.extend(pilot_bits)
    return out


def remove_pilots_soft(soft, original_data_len, interval=PILOT_INTERVAL_BITS, pilot_bits=PILOT_BITS):
    soft = _to_tensor(soft, dtype=torch.float64)
    out = []; i_data = 0; i_soft = 0; p_len = len(pilot_bits)
    while i_data < original_data_len and i_soft < soft.numel():
        chunk = min(interval, original_data_len - i_data)
        out.append(soft[i_soft : i_soft + chunk]); i_soft += chunk; i_data += chunk
        if i_data < original_data_len: i_soft += p_len
    return torch.cat(out)[:original_data_len] if out else torch.empty(0, dtype=torch.float64, device=soft.device)


def pilot_positions(total_data_len: int, interval: int = PILOT_INTERVAL_BITS, p_len: int = PILOT_BLOCK_BITS) -> List[Tuple[int, int, int, int]]:
    pos = []; data_consumed = 0; stream_idx = 0
    while data_consumed < total_data_len:
        chunk = min(interval, total_data_len - data_consumed); ds = stream_idx; de = stream_idx + chunk
        stream_idx += chunk; data_consumed += chunk
        if data_consumed < total_data_len:
            ps = stream_idx; pe = stream_idx + p_len; pos.append((ds, de, ps, pe)); stream_idx += p_len
        else: pos.append((ds, de, -1, -1))
    return pos


@dataclass
class TXBuildResult:
    iq: torch.Tensor
    metadata: dict


def build_tx_bitstream(payload_crc_bytes: bytes, fec_mode: str, interleave: bool, interleave_rows: int):
    payload_len = len(payload_crc_bytes) - 4
    header_bits = bytes_to_bits_msb(build_header_bytes(payload_len))
    header_total = rep3_encode_bits(header_bits) * HEADER_COPIES
    payload_plain_bits = bytes_to_bits_msb(payload_crc_bytes)
    payload_scrambled = scramble_bits(payload_plain_bits, seed=0x5D)
    fec = FECCodec(fec_mode)
    payload_coded = fec.encode_bits(payload_scrambled)
    if interleave: payload_coded = block_interleave_bits(payload_coded, rows=interleave_rows)
    payload_with_pilots = insert_pilots(payload_coded, interval=PILOT_INTERVAL_BITS, pilot_bits=PILOT_BITS)
    frame_bits = ACCESS_BITS + header_total + payload_with_pilots + [0] * POSTAMBLE_BITS
    return frame_bits, payload_len, len(payload_plain_bits), len(payload_coded), len(payload_with_pilots)


def build_tx_iq_object(**kwargs) -> TXBuildResult:
    message = kwargs.get("message", None); random_bits = kwargs.get("random_bits", None)
    if (message is None) == (random_bits is None): raise ValueError("Provide exactly one of message or random_bits")
    random_seed = kwargs.get("random_seed", 1); fec = kwargs.get("fec", FEC_NONE); interleave = kwargs.get("interleave", False); interleave_rows = kwargs.get("interleave_rows", 8)
    sps = kwargs.get("sps", 8); beta = kwargs.get("beta", 0.35); span = kwargs.get("span", 6)
    sample_rate_hz = kwargs.get("sample_rate_hz", 1_000_000.0); rf_center_hz = kwargs.get("rf_center_hz", 0.0); carrier_hz = kwargs.get("carrier_hz", 0.0)
    payload_crc_bytes = build_payload_bytes_from_random_bits(random_bits, seed=random_seed) if random_bits is not None else build_payload_bytes_from_message(message)
    payload_source = f"random_bits:{random_bits}" if random_bits is not None else "message"
    frame_bits, payload_len, payload_plain_bits, payload_coded_bits, payload_with_pilots_bits = build_tx_bitstream(payload_crc_bytes, fec, interleave, interleave_rows)
    iq = tx_waveform(frame_bits, sps=sps, beta=beta, span=span)
    target_num_samples = kwargs.get("target_num_samples", None)
    n_repeats = 1
    if target_num_samples is not None:
        n_repeats = int(math.ceil(target_num_samples / iq.numel())); iq = iq.repeat(n_repeats)[:target_num_samples]
    iq = apply_carrier_frequency(iq, carrier_hz=carrier_hz, sample_rate_hz=sample_rate_hz)
    iq = impair_iq(iq=iq, snr_db=kwargs.get("snr_db", None), noise_color=kwargs.get("noise_color", "white"), freq_offset=kwargs.get("freq_offset", 0.0), timing_offset=kwargs.get("timing_offset", 1.0), fading_mode=kwargs.get("fading_mode", "none"), fading_block_len=kwargs.get("fading_block_len", 256), rician_k_db=kwargs.get("rician_k_db", 6.0), multipath_taps=kwargs.get("multipath_taps", None), burst_probability=kwargs.get("burst_probability", 0.0), burst_len_min=kwargs.get("burst_len_min", 16), burst_len_max=kwargs.get("burst_len_max", 64), burst_power_ratio_db=kwargs.get("burst_power_ratio_db", 12.0), burst_color=kwargs.get("burst_color", "white"), seed=kwargs.get("seed", 1))
    meta = {"payload_source": payload_source, "message": message if isinstance(message, str) else None, "payload_len": payload_len, "payload_plain_bits": payload_plain_bits, "payload_coded_bits": payload_coded_bits, "payload_with_pilots_bits": payload_with_pilots_bits, "n_repeats": n_repeats, "sps": sps, "beta": beta, "span": span, "sample_rate_hz": sample_rate_hz, "rf_center_hz": rf_center_hz, "carrier_hz": carrier_hz, "absolute_rf_hz": rf_center_hz + carrier_hz, "fec": fec, "interleave": interleave, "interleave_rows": interleave_rows, "avg_power": measure_power(iq)}
    return TXBuildResult(iq=iq.to(torch.complex64), metadata=meta)


def save_tx_iq_object(result: TXBuildResult, iq_path, metadata_path=None):
    iq_path = str(iq_path); metadata_path = default_metadata_path(iq_path) if metadata_path is None else str(metadata_path)
    save_iq(iq_path, result.iq)
    with open(metadata_path, "w", encoding="utf-8") as f: json.dump(result.metadata, f, indent=2)
    return iq_path, metadata_path


def coarse_frequency_acquire(iq, ref_waveform, sample_rate_hz, search_hz, n_bins):
    iq = _to_tensor(iq); ref = _to_tensor(ref_waveform)
    if iq.numel() < ref.numel(): raise RuntimeError("IQ shorter than reference waveform")
    best_metric = -1.0; best_start = 0; best_cfo = 0.0
    n = torch.arange(iq.numel(), dtype=torch.float64, device=iq.device)
    for f_hz in torch.linspace(-search_hz, search_hz, n_bins, device=iq.device):
        y = iq * torch.exp(-1j * 2.0 * math.pi * f_hz * n / sample_rate_hz)
        corr = F.conv1d(y.view(1, 1, -1), torch.conj(ref.flip(0)).view(1, 1, -1)).view(-1)
        mag = torch.abs(corr); metric, idx = torch.max(mag, dim=0)
        if float(metric) > best_metric: best_metric = float(metric); best_start = int(idx); best_cfo = float(f_hz)
    return best_start, best_cfo, best_metric


def estimate_residual_cfo_from_preamble(symbols, symbol_rate_hz):
    symbols = _to_tensor(symbols)
    if symbols.numel() < PREAMBLE_BITS_LEN: return 0.0
    half = PREAMBLE_HALF_LEN_BITS; phases = []
    for r in range(PREAMBLE_REPS - 1):
        s1 = symbols[r * half : (r + 1) * half]; s2 = symbols[(r + 1) * half : (r + 2) * half]
        if s1.numel() == half and s2.numel() == half:
            phases.append(float(torch.angle(torch.sum(s2 * torch.conj(s1))) / (2.0 * math.pi * half)))
    return float(sum(phases) / len(phases) * symbol_rate_hz) if phases else 0.0


def apply_symbol_rate_cfo(symbols, cfo_hz, symbol_rate_hz):
    symbols = _to_tensor(symbols)
    if cfo_hz == 0.0: return symbols.to(torch.complex64)
    n = torch.arange(symbols.numel(), dtype=torch.float64, device=symbols.device)
    return (symbols * torch.exp(-1j * 2.0 * math.pi * cfo_hz * n / symbol_rate_hz)).to(torch.complex64)


def matched_filter(iq, sps, beta, span):
    return _lfilter_1d(_to_tensor(iq), rrc_taps(beta, sps, span))


def extract_symbols_from_start(mf, start_index_samples, sps, span, sample_delta=0):
    mf = _to_tensor(mf)
    first = start_index_samples + 2 * span * sps + sample_delta
    if first < 0 or first >= mf.numel(): return torch.empty(0, dtype=torch.complex64, device=mf.device)
    return mf[first::sps].to(torch.complex64)


def design_symbol_equalizer_ls(rx_train, tx_train, ntaps=7, ridge=1e-3):
    rx_train = _to_tensor(rx_train); tx_train = _to_tensor(tx_train)
    if ntaps % 2 == 0: raise ValueError("ntaps must be odd")
    if rx_train.numel() != tx_train.numel(): raise ValueError("Training sequences must have same length")
    if rx_train.numel() < ntaps: return torch.tensor([1.0 + 0j], dtype=torch.complex64, device=rx_train.device)
    half = ntaps // 2
    rpad = F.pad(rx_train.view(1, 1, -1), (half, half)).view(-1)
    X = torch.stack([rpad[i : i + ntaps] for i in range(rx_train.numel())], dim=0)
    A = X.conj().T @ X + ridge * torch.eye(ntaps, dtype=torch.complex64, device=rx_train.device)
    b = X.conj().T @ tx_train
    return torch.linalg.solve(A, b).to(torch.complex64)


def apply_symbol_equalizer(symbols, w):
    symbols = _to_tensor(symbols); w = _to_tensor(w)
    if w.numel() == 1: return (symbols * w[0]).to(torch.complex64)
    half = w.numel() // 2
    spad = F.pad(symbols.view(1, 1, -1), (half, half)).view(-1)
    out = torch.empty_like(symbols)
    for i in range(symbols.numel()): out[i] = torch.dot(spad[i : i + w.numel()], w)
    return out.to(torch.complex64)


def apply_pilot_phase_tracking(symbols, data_with_pilots_len, data_only_len, access_and_headers_len):
    symbols = _to_tensor(symbols)
    if symbols.numel() < access_and_headers_len + data_with_pilots_len: return symbols
    y = symbols.clone(); payload_stream = y[access_and_headers_len : access_and_headers_len + data_with_pilots_len]
    pilot_syms = bpsk_map(PILOT_BITS); phase_est = 0.0
    for ds, de, ps, pe in pilot_positions(data_only_len, PILOT_INTERVAL_BITS, PILOT_BLOCK_BITS):
        if ps >= 0 and pe <= payload_stream.numel() and payload_stream[ps:pe].numel() == pilot_syms.numel():
            phase_est = float(torch.angle(torch.sum(payload_stream[ps:pe] * torch.conj(pilot_syms))))
            payload_stream[ds:ps] *= torch.exp(torch.tensor(-1j * phase_est, device=symbols.device))
        else:
            payload_stream[ds:de] *= torch.exp(torch.tensor(-1j * phase_est, device=symbols.device))
    y[access_and_headers_len : access_and_headers_len + data_with_pilots_len] = payload_stream
    return y.to(torch.complex64)


def choose_valid_header_from_copies(header_soft_all) -> Optional[int]:
    s = _to_tensor(header_soft_all, dtype=torch.float64)
    vals = []
    for c in range(HEADER_COPIES):
        chunk = s[c * HEADER_PROT_BITS_LEN : (c + 1) * HEADER_PROT_BITS_LEN]
        if chunk.numel() < HEADER_PROT_BITS_LEN: continue
        payload_len = parse_header_bytes(bits_to_bytes_msb(rep3_decode_soft(chunk)[:HEADER_BITS_LEN]))
        if payload_len is not None: vals.append(payload_len)
    return int(max(set(vals), key=vals.count)) if vals else None


def try_decode_from_symbols(symbols, fec_mode, interleave, interleave_rows, symbol_rate_hz, eq_taps) -> Optional[bytes]:
    symbols = _to_tensor(symbols)
    if symbols.numel() < ACCESS_BITS_LEN + HEADER_COPIES * HEADER_PROT_BITS_LEN: return None
    symbols = apply_symbol_rate_cfo(symbols, estimate_residual_cfo_from_preamble(symbols[:PREAMBLE_BITS_LEN], symbol_rate_hz), symbol_rate_hz)
    train_start = PREAMBLE_BITS_LEN + SYNC_BITS_LEN; train_end = train_start + TRAINING_LEN_BITS
    if symbols.numel() < train_end: return None
    tx_train = bpsk_map(ACCESS_BITS)[train_start:train_end]
    eq_symbols = apply_symbol_equalizer(symbols, design_symbol_equalizer_ls(symbols[train_start:train_end], tx_train, ntaps=eq_taps, ridge=1e-3))
    ph = torch.angle(torch.sum(eq_symbols[train_start:train_end] * torch.conj(tx_train))); eq_symbols *= torch.exp(-1j * ph)
    soft_bits = torch.real(eq_symbols).to(torch.float64)
    hdr_start = ACCESS_BITS_LEN; hdr_end = hdr_start + HEADER_COPIES * HEADER_PROT_BITS_LEN
    payload_len = choose_valid_header_from_copies(soft_bits[hdr_start:hdr_end])
    if payload_len is None: return None
    payload_plain_bits = (payload_len + 4) * 8; fec = FECCodec(fec_mode); payload_coded_bits = fec.encoded_length(payload_plain_bits)
    n_pilot_blocks = max(0, math.ceil(max(payload_coded_bits - PILOT_INTERVAL_BITS, 0) / PILOT_INTERVAL_BITS))
    payload_with_pilots_bits = payload_coded_bits + n_pilot_blocks * PILOT_BLOCK_BITS
    data_start = hdr_end; data_end = data_start + payload_with_pilots_bits
    if data_end > soft_bits.numel(): return None
    eq_symbols = apply_pilot_phase_tracking(eq_symbols, payload_with_pilots_bits, payload_coded_bits, data_start)
    payload_soft = remove_pilots_soft(torch.real(eq_symbols).to(torch.float64)[data_start:data_end], payload_coded_bits, PILOT_INTERVAL_BITS, PILOT_BITS)
    if interleave: payload_soft = block_deinterleave_soft(payload_soft, rows=interleave_rows)
    decoded_bits = fec.decode_soft(payload_soft)
    if len(decoded_bits) < payload_plain_bits: return None
    decoded_bytes = bits_to_bytes_msb(descramble_bits(decoded_bits[:payload_plain_bits], seed=0x5D))
    return parse_payload_bytes(decoded_bytes, payload_len)


def rx_command_iq(iq, meta):
    tx_sample_rate_hz = float(meta["sample_rate_hz"]); tx_rf_center_hz = float(meta["rf_center_hz"]); tx_absolute_rf_hz = float(meta["absolute_rf_hz"])
    sps, beta, span = 8, 0.35, 6
    symbol_rate_hz = tx_sample_rate_hz / sps
    iq = apply_carrier_frequency(iq, carrier_hz=-(tx_absolute_rf_hz - tx_rf_center_hz), sample_rate_hz=tx_sample_rate_hz)
    iq = robust_agc_and_blanking(iq, blanker_k=6.0)
    access_ref_waveform = tx_waveform(ACCESS_BITS, sps=sps, beta=beta, span=span)
    coarse_start, coarse_cfo_hz, coarse_metric = coarse_frequency_acquire(iq, access_ref_waveform, tx_sample_rate_hz, 25_000.0, 101)
    mf = matched_filter(apply_carrier_frequency(iq, carrier_hz=-coarse_cfo_hz, sample_rate_hz=tx_sample_rate_hz), sps=sps, beta=beta, span=span)
    payload = try_decode_from_symbols(extract_symbols_from_start(mf, coarse_start, sps, span, sample_delta=3), FEC_NONE, True, 8, symbol_rate_hz, 7)
    if payload is None: raise RuntimeError("No valid packet found after acquisition, header decode, FEC decode, and CRC")
    try: message_text = payload.decode("utf-8")
    except UnicodeDecodeError: message_text = None
    return {"payload_bytes": payload, "payload_len": len(payload), "message": message_text, "sample_offset_used": 3, "coarse_cfo_hz": coarse_cfo_hz, "coarse_metric": coarse_metric}


# Simple CLI compatibility

def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--message", type=str, default="hello")
    args = parser.parse_args(argv)
    tx = build_tx_iq_object(message=args.message, interleave=True)
    print(json.dumps(tx.metadata, indent=2))


if __name__ == "__main__":
    main()
