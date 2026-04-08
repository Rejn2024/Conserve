#!/usr/bin/env python3
"""
advanced_link_skdsp_v3_tx_flexible.py

TX-focused module for generating arbitrary-length IQ outputs.

Features
--------
- Message payloads or random-bit payloads
- DBPSK + RRC shaping
- Protected header
- Optional payload FEC/interleaving
- RF metadata
- Colored noise / fading / impulsive noise / timing / residual freq offset
- Arbitrary-length IQ output by repeating framed bursts with configurable gaps

This is intended as a TX/data-generation module.
"""

from __future__ import annotations

import argparse
import binascii
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import signal

try:
    from sk_dsp_comm import fec_conv
except ImportError:
    fec_conv = None


PREAMBLE_HALF_LEN_BITS = 64
SYNC_WORD = b"\xD3\x91\xC5\x7A"
TRAINING_LEN_BITS = 128
HEADER_MAGIC = 0xA55A
POSTAMBLE_BITS = 256
MAX_PAYLOAD_BYTES = 10_000_000

FEC_NONE = "none"
FEC_REP3 = "rep3"
FEC_CONV = "conv"


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
PREAMBLE_BITS = PREAMBLE_HALF_BITS + PREAMBLE_HALF_BITS
SYNC_BITS = bytes_to_bits_msb(SYNC_WORD)
TRAINING_BITS = prbs_bits(TRAINING_LEN_BITS, seed=67890)
ACCESS_BITS = PREAMBLE_BITS + SYNC_BITS + TRAINING_BITS

HEADER_BYTES_LEN = 2 + 4 + 4 + 4
HEADER_BITS_LEN = HEADER_BYTES_LEN * 8
HEADER_PROT_BITS_LEN = HEADER_BITS_LEN * 3


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


def measure_power(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x) ** 2)) if len(x) else 0.0


def apply_carrier_frequency(iq: np.ndarray, carrier_hz: float, sample_rate_hz: float) -> np.ndarray:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if abs(carrier_hz) >= sample_rate_hz / 2:
        raise ValueError("carrier_hz must satisfy |carrier_hz| < sample_rate_hz/2")
    if carrier_hz == 0.0:
        return iq.astype(np.complex64)

    n = np.arange(len(iq), dtype=np.float64)
    rot = np.exp(1j * 2.0 * np.pi * carrier_hz * n / sample_rate_hz)
    return (iq * rot).astype(np.complex64)


def apply_frequency_offset(iq: np.ndarray, freq_offset: float) -> np.ndarray:
    if freq_offset == 0.0:
        return iq.astype(np.complex64)
    n = np.arange(len(iq), dtype=np.float64)
    rot = np.exp(1j * 2.0 * np.pi * freq_offset * n)
    return (iq * rot).astype(np.complex64)


def apply_timing_offset_resample(iq: np.ndarray, timing_offset: float) -> np.ndarray:
    if timing_offset == 1.0 or len(iq) == 0:
        return iq.astype(np.complex64)

    new_len = max(1, int(round(len(iq) / timing_offset)))
    y = signal.resample(iq, new_len)
    if new_len > len(iq):
        y = y[:len(iq)]
    elif new_len < len(iq):
        y = np.concatenate([y, np.zeros(len(iq) - new_len, dtype=y.dtype)])
    return y.astype(np.complex64)


def apply_fading(
    iq: np.ndarray,
    mode: str = "none",
    block_len: int = 256,
    rician_k_db: float = 6.0,
    multipath_taps: Optional[List[complex]] = None,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.asarray(iq, dtype=np.complex64)

    if mode == "none":
        return x

    if mode == "rayleigh_block":
        out = np.empty_like(x)
        for start in range(0, len(x), block_len):
            end = min(start + block_len, len(x))
            h = (rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)
            out[start:end] = x[start:end] * h
        return out.astype(np.complex64)

    if mode == "rician_block":
        out = np.empty_like(x)
        k_lin = 10.0 ** (rician_k_db / 10.0)
        los = np.sqrt(k_lin / (k_lin + 1.0))
        scat_scale = np.sqrt(1.0 / (k_lin + 1.0))
        for start in range(0, len(x), block_len):
            end = min(start + block_len, len(x))
            scat = ((rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)) * scat_scale
            h = los + scat
            out[start:end] = x[start:end] * h
        return out.astype(np.complex64)

    if mode == "multipath_static":
        taps = multipath_taps if multipath_taps is not None else [
            1.0 + 0.0j,
            0.20 + 0.08j,
            0.06 - 0.04j,
        ]
        y = np.convolve(x, np.asarray(taps, dtype=np.complex64), mode="full")[:len(x)]
        return y.astype(np.complex64)

    raise ValueError(f"Unsupported fading mode: {mode}")


def _complex_colored_noise(
    n: int,
    color: str,
    power: float,
    rng: np.random.Generator,
) -> np.ndarray:
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

    def make_real():
        freqs = np.fft.rfftfreq(n, d=1.0)
        spec = rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))
        shaping = np.ones_like(freqs, dtype=np.float64)
        nz = freqs > 0
        if alpha > 0:
            shaping[nz] = 1.0 / (freqs[nz] ** (alpha / 2.0))
        elif alpha < 0:
            shaping[nz] = freqs[nz] ** ((-alpha) / 2.0)
        shaping[~nz] = 0.0 if color in ("pink", "brown") else 1.0

        y = np.fft.irfft(spec * shaping, n=n)
        p = np.mean(y ** 2)
        if p > 0:
            y *= np.sqrt((power / 2.0) / p)
        return y

    i = make_real()
    q = make_real()
    z = i + 1j * q
    p = np.mean(np.abs(z) ** 2)
    if p > 0:
        z *= np.sqrt(power / p)
    return z.astype(np.complex64)


def add_impulsive_bursts(
    iq: np.ndarray,
    base_noise_power: float,
    burst_probability: float = 0.0,
    burst_len_min: int = 16,
    burst_len_max: int = 64,
    burst_power_ratio_db: float = 12.0,
    burst_color: str = "white",
    seed: int = 1,
) -> np.ndarray:
    if burst_probability <= 0.0 or len(iq) == 0:
        return iq.astype(np.complex64)

    rng = np.random.default_rng(seed + 1000)
    out = np.array(iq, copy=True, dtype=np.complex64)
    burst_power = base_noise_power * (10.0 ** (burst_power_ratio_db / 10.0))

    idx = 0
    while idx < len(out):
        if rng.random() < burst_probability:
            burst_len = int(rng.integers(burst_len_min, burst_len_max + 1))
            end = min(idx + burst_len, len(out))
            out[idx:end] += _complex_colored_noise(end - idx, burst_color, burst_power, rng)
            idx = end
        else:
            idx += 1

    return out.astype(np.complex64)


def impair_iq(
    iq: np.ndarray,
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
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.asarray(iq, dtype=np.complex64)

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
        y = x + _complex_colored_noise(len(x), noise_color, noise_power, rng)

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
    return y.astype(np.complex64)


def rep3_encode_bits(bits: List[int]) -> List[int]:
    out: List[int] = []
    for b in bits:
        out.extend([b, b, b])
    return out


def build_header_bytes(payload_len: int) -> bytes:
    if payload_len <= 0:
        raise ValueError("payload_len must be > 0")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError("payload_len too large")

    payload_len_inv = payload_len ^ 0xFFFFFFFF
    prefix = struct.pack(">HII", HEADER_MAGIC, payload_len, payload_len_inv)
    hdr_crc = binascii.crc32(prefix) & 0xFFFFFFFF
    return prefix + struct.pack(">I", hdr_crc)


@dataclass
class FECConfig:
    mode: str


class FECCodec:
    def __init__(self, cfg: FECConfig):
        self.cfg = cfg
        self._conv = None
        self._conv_polys = None
        self._conv_depth = None
        self._conv_k = None

        if cfg.mode == FEC_CONV:
            if fec_conv is None:
                raise ImportError("scikit-dsp-comm is required for convolutional FEC")
            self._conv_polys = ("111", "101")
            self._conv_depth = 40
            self._conv_k = len(self._conv_polys[0])
            self._conv = fec_conv.FECConv(self._conv_polys, Depth=self._conv_depth)

    def _conv_tail_bits(self) -> int:
        return self._conv_k - 1

    def _conv_flush_bits(self) -> int:
        return self._conv_depth - 1

    def encode_bits(self, bits: List[int]) -> List[int]:
        if self.cfg.mode == FEC_NONE:
            return bits[:]
        if self.cfg.mode == FEC_REP3:
            return rep3_encode_bits(bits)
        if self.cfg.mode == FEC_CONV:
            bits_arr = np.array(bits, dtype=int)
            pad_zeros = self._conv_tail_bits() + self._conv_flush_bits()
            bits_arr = np.concatenate([bits_arr, np.zeros(pad_zeros, dtype=int)])
            state = "0" * (self._conv_k - 1)
            encoded, _ = self._conv.conv_encoder(bits_arr, state)
            return [int(x) & 1 for x in np.asarray(encoded).flatten().tolist()]
        raise ValueError("Unsupported FEC mode")


def block_interleave_bits(bits: List[int], rows: int = 8) -> List[int]:
    if rows <= 1:
        return bits[:]
    cols = math.ceil(len(bits) / rows)
    padded = bits[:] + [0] * (rows * cols - len(bits))
    mat = np.array(padded, dtype=np.int8).reshape(rows, cols)
    out = mat.T.reshape(-1).tolist()
    return out[:len(bits)]


def build_payload_bytes_from_message(message: Union[str, bytes]) -> bytes:
    payload = message.encode("utf-8") if isinstance(message, str) else bytes(message)
    if len(payload) == 0:
        raise ValueError("Empty payloads are not supported")
    payload_crc = binascii.crc32(payload) & 0xFFFFFFFF
    return payload + struct.pack(">I", payload_crc)


def build_payload_bytes_from_random_bits(n_bits: int, seed: int) -> bytes:
    if n_bits <= 0:
        raise ValueError("n_bits must be > 0")
    pad = (-n_bits) % 8
    bits = prbs_bits(n_bits, seed=seed)
    if pad:
        bits = bits + [0] * pad
    payload = bits_to_bytes_msb(bits)
    payload_crc = binascii.crc32(payload) & 0xFFFFFFFF
    return payload + struct.pack(">I", payload_crc)


def dbpsk_diff_encode(bits: List[int]) -> np.ndarray:
    out = np.zeros(len(bits), dtype=np.int8)
    prev = 0
    for i, b in enumerate(bits):
        prev ^= (b & 1)
        out[i] = prev
    return out


def dbpsk_map(bits: np.ndarray) -> np.ndarray:
    return np.where(bits > 0, 1.0 + 0j, -1.0 + 0j).astype(np.complex64)


def dbpsk_symbols_from_bits(bits: List[int], prepend_ref: bool = True) -> np.ndarray:
    diff = dbpsk_diff_encode(bits)
    syms = dbpsk_map(diff)
    if prepend_ref:
        ref_sym = np.array([-1.0 + 0.0j], dtype=np.complex64)
        syms = np.concatenate([ref_sym, syms])
    return syms


def build_payload_section(args) -> Tuple[bytes, str]:
    if args.random_bits is not None:
        payload_crc = build_payload_bytes_from_random_bits(args.random_bits, seed=args.random_seed)
        return payload_crc, f"random_bits:{args.random_bits}"
    payload_crc = build_payload_bytes_from_message(read_message_arg(args))
    return payload_crc, "message"


def build_frame_bits(
    payload_crc_bytes: bytes,
    fec_cfg: FECConfig,
    interleave: bool,
    interleave_rows: int,
    idle_gap_bits: int = 0,
) -> Tuple[List[int], int, int, int]:
    payload_len = len(payload_crc_bytes) - 4
    header_bytes = build_header_bytes(payload_len)
    header_bits = bytes_to_bits_msb(header_bytes)
    header_bits_prot = rep3_encode_bits(header_bits)

    payload_bits_plain = bytes_to_bits_msb(payload_crc_bytes)
    fec = FECCodec(fec_cfg)
    payload_bits_coded = fec.encode_bits(payload_bits_plain)

    if interleave:
        payload_bits_coded = block_interleave_bits(payload_bits_coded, rows=interleave_rows)

    frame_bits = ACCESS_BITS + header_bits_prot + payload_bits_coded + [0] * POSTAMBLE_BITS
    if idle_gap_bits > 0:
        frame_bits = frame_bits + [0] * idle_gap_bits

    return frame_bits, payload_len, len(payload_bits_plain), len(payload_bits_coded)


def tx_waveform(frame_bits: List[int], sps: int, beta: float, span: int) -> np.ndarray:
    syms = dbpsk_symbols_from_bits(frame_bits, prepend_ref=True)
    taps = rrc_taps(beta=beta, sps=sps, span=span)
    up = np.zeros(len(syms) * sps, dtype=np.complex64)
    up[::sps] = syms
    y = signal.lfilter(taps, [1.0], up)
    return y.astype(np.complex64)


@dataclass
class TXBuildResult:
    iq: np.ndarray
    metadata: dict


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
    idle_gap_bits: int = 0,
    seed: int = 1,
) -> TXBuildResult:
    """
    Build a single IQ object.

    If target_num_samples is not None, the framed burst is repeated until at least that
    many samples are produced, then truncated exactly to target_num_samples.
    """
    if (message is None) == (random_bits is None):
        raise ValueError("Provide exactly one of message or random_bits")

    if random_bits is not None:
        payload_crc_bytes = build_payload_bytes_from_random_bits(random_bits, seed=random_seed)
        payload_source = f"random_bits:{random_bits}"
    else:
        payload_crc_bytes = build_payload_bytes_from_message(message)
        payload_source = "message"

    fec_cfg = FECConfig(mode=fec)
    frame_bits, payload_len, payload_plain_bits, payload_coded_bits = build_frame_bits(
        payload_crc_bytes=payload_crc_bytes,
        fec_cfg=fec_cfg,
        interleave=interleave,
        interleave_rows=interleave_rows,
        idle_gap_bits=idle_gap_bits,
    )

    one_burst_iq = tx_waveform(frame_bits, sps=sps, beta=beta, span=span)

    if target_num_samples is None:
        iq = one_burst_iq
        n_repeats = 1
    else:
        if target_num_samples <= 0:
            raise ValueError("target_num_samples must be > 0")
        n_repeats = int(math.ceil(target_num_samples / len(one_burst_iq)))
        iq = np.tile(one_burst_iq, n_repeats)[:target_num_samples].astype(np.complex64)

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
        "random_bits": random_bits,
        "random_seed": random_seed if random_bits is not None else None,
        "payload_len_bytes": payload_len,
        "target_num_samples": target_num_samples,
        "actual_num_samples": int(len(iq)),
        "frame_payload_plain_bits": payload_plain_bits,
        "frame_payload_coded_bits": payload_coded_bits,
        "n_repeats": n_repeats,
        "idle_gap_bits": idle_gap_bits,
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

    return TXBuildResult(iq=iq.astype(np.complex64), metadata=metadata)


def save_tx_iq_object(result: TXBuildResult, iq_path: Union[str, Path], metadata_path: Optional[Union[str, Path]] = None) -> Tuple[str, str]:
    iq_path = str(iq_path)
    if metadata_path is None:
        metadata_path = str(Path(iq_path).with_suffix(".json"))
    else:
        metadata_path = str(metadata_path)

    np.save(iq_path, result.iq)
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


def parse_multipath_taps_arg(multipath_taps: Optional[str]) -> Optional[List[complex]]:
    if not multipath_taps:
        return None
    return [complex(t) for t in multipath_taps.split(",")]


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    tx = sub.add_parser("tx")
    tx.add_argument("--message", type=str)
    tx.add_argument("--message-file", type=str)
    tx.add_argument("--random-bits", type=int, default=None)
    tx.add_argument("--random-seed", type=int, default=1)
    tx.add_argument("--output", type=str, default="tx_iq.npy")
    tx.add_argument("--metadata-path", type=str, default=None)
    tx.add_argument("--target-num-samples", type=int, default=None)
    tx.add_argument("--idle-gap-bits", type=int, default=0)

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

    return parser


def tx_command(args):
    if args.random_bits is None and args.message is None and args.message_file is None:
        raise SystemExit("tx requires one of --message, --message-file, or --random-bits")
    if args.random_bits is not None and (args.message is not None or args.message_file is not None):
        raise SystemExit("Use either random bits or message input, not both.")

    message = None if args.random_bits is not None else read_message_arg(args)
    multipath_taps = parse_multipath_taps_arg(args.multipath_taps)

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
        idle_gap_bits=args.idle_gap_bits,
        seed=args.seed,
    )

    iq_path, meta_path = save_tx_iq_object(result, iq_path=args.output, metadata_path=args.metadata_path)

    print(f"Saved IQ: {iq_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Samples: {result.metadata['actual_num_samples']}")
    print(f"Payload source: {result.metadata['payload_source']}")
    return {"iq_path": iq_path, "metadata_path": meta_path, **result.metadata}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "tx":
        return tx_command(args)
    raise SystemExit("Only tx is implemented in this flexible TX module.")


if __name__ == "__main__":
    main()