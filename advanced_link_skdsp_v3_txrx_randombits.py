#!/usr/bin/env python3
"""
advanced_link_skdsp_v3_txrx_randombits.py

Robust offline DBPSK packet link with:
- RF/sample-rate metadata
- coarse frequency acquisition
- training-based channel estimation
- symbol-spaced equalization
- protected header
- optional payload FEC/interleaving
- TX can send either:
    * a message
    * a random bit string of chosen length

Dependencies:
    numpy
    scipy
    scikit-dsp-comm
"""

import argparse
import binascii
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from sk_dsp_comm import fec_conv


PREAMBLE_HALF_LEN_BITS = 64
SYNC_WORD = b"\xD3\x91\xC5\x7A"
TRAINING_LEN_BITS = 128
POSTAMBLE_BITS = 256

HEADER_MAGIC = 0xA55A
MAX_PAYLOAD_BYTES = 10_000_000

FEC_NONE = "none"
FEC_REP3 = "rep3"
FEC_CONV = "conv"


def bytes_to_bits_msb(data: bytes) -> List[int]:
    out = []
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

PREAMBLE_BITS_LEN = len(PREAMBLE_BITS)
SYNC_BITS_LEN = len(SYNC_BITS)
TRAINING_BITS_LEN = len(TRAINING_BITS)
ACCESS_BITS_LEN = len(ACCESS_BITS)

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


def agc_rms(x: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    rms = np.sqrt(np.mean(np.abs(x) ** 2) + 1e-15)
    return (x * (target_rms / rms)).astype(np.complex64)


def measure_power(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x) ** 2)) if len(x) else 0.0


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


def apply_frequency_offset(iq: np.ndarray, freq_offset: float) -> np.ndarray:
    if freq_offset == 0.0:
        return iq.astype(np.complex64)
    n = np.arange(len(iq), dtype=np.float64)
    rot = np.exp(1j * 2.0 * np.pi * freq_offset * n)
    return (iq * rot).astype(np.complex64)


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


def resample_iq(iq: np.ndarray, fs_in_hz: float, fs_out_hz: float) -> np.ndarray:
    if fs_in_hz <= 0 or fs_out_hz <= 0:
        raise ValueError("Sample rates must be positive")
    if np.isclose(fs_in_hz, fs_out_hz):
        return iq.astype(np.complex64)
    new_len = max(1, int(round(len(iq) * fs_out_hz / fs_in_hz)))
    y = signal.resample(iq, new_len)
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


def save_iq(path: str, iq: np.ndarray) -> None:
    np.asarray(iq, dtype=np.complex64).tofile(path)


def load_iq(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.complex64)


def default_metadata_path(iq_path: str) -> str:
    p = Path(iq_path)
    return str(p.with_suffix(p.suffix + ".json"))


def save_iq_metadata(
    iq_path: str,
    sample_rate_hz: float,
    rf_center_hz: float,
    carrier_hz: float,
    metadata_path: Optional[str] = None,
) -> str:
    if metadata_path is None:
        metadata_path = default_metadata_path(iq_path)

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


def load_iq_metadata(iq_path: str, metadata_path: Optional[str] = None) -> Optional[dict]:
    if metadata_path is None:
        metadata_path = default_metadata_path(iq_path)

    p = Path(metadata_path)
    if not p.exists():
        return None

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


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


def build_header_bytes(payload_len: int) -> bytes:
    if payload_len <= 0:
        raise ValueError("payload_len must be > 0")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError("payload_len too large")

    magic = HEADER_MAGIC
    payload_len_inv = payload_len ^ 0xFFFFFFFF

    prefix = struct.pack(">HII", magic, payload_len, payload_len_inv)
    hdr_crc = binascii.crc32(prefix) & 0xFFFFFFFF
    header = prefix + struct.pack(">I", hdr_crc)
    return header


def parse_header_bytes(header: bytes) -> Optional[int]:
    if len(header) != HEADER_BYTES_LEN:
        return None

    magic, payload_len, payload_len_inv, hdr_crc = struct.unpack(">HIII", header)
    prefix = struct.pack(">HII", magic, payload_len, payload_len_inv)
    calc_crc = binascii.crc32(prefix) & 0xFFFFFFFF

    if magic != HEADER_MAGIC:
        return None
    if payload_len == 0:
        return None
    if payload_len > MAX_PAYLOAD_BYTES:
        return None
    if payload_len_inv != (payload_len ^ 0xFFFFFFFF):
        return None
    if hdr_crc != calc_crc:
        return None

    return payload_len


def build_payload_bytes_from_message(message: Union[str, bytes]) -> bytes:
    payload = message.encode("utf-8") if isinstance(message, str) else bytes(message)
    if len(payload) == 0:
        raise ValueError("Empty payloads are not supported by this framing")
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
            self._conv_polys = ("111", "101")
            self._conv_depth = 40
            self._conv_k = len(self._conv_polys[0])
            self._conv = fec_conv.FECConv(self._conv_polys, Depth=self._conv_depth)

    def _conv_tail_bits(self) -> int:
        return self._conv_k - 1

    def _conv_flush_bits(self) -> int:
        return self._conv_depth - 1

    def encoded_length(self, input_bit_len: int) -> int:
        if self.cfg.mode == FEC_NONE:
            return input_bit_len
        if self.cfg.mode == FEC_REP3:
            return 3 * input_bit_len
        if self.cfg.mode == FEC_CONV:
            total_uncoded = input_bit_len + self._conv_tail_bits() + self._conv_flush_bits()
            return 2 * total_uncoded
        raise ValueError("Unsupported FEC mode")

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

    def decode_soft(self, soft_values: np.ndarray) -> List[int]:
        if self.cfg.mode == FEC_NONE:
            return (soft_values > 0).astype(np.int8).tolist()

        if self.cfg.mode == FEC_REP3:
            return rep3_decode_soft(soft_values)

        if self.cfg.mode == FEC_CONV:
            x = np.asarray(soft_values, dtype=np.float64)
            usable = (len(x) // 2) * 2
            x = x[:usable]
            if len(x) == 0:
                return []

            x = np.clip(x, -10.0, 10.0)
            xmax = np.max(np.abs(x))
            if xmax > 0:
                x = x / xmax

            q = np.rint((x + 1.0) * 3.5).astype(int)
            q = np.clip(q, 0, 7)

            decoded = self._conv.viterbi_decoder(q, metric_type="soft", quant_level=3)
            decoded = np.asarray(decoded).astype(np.int8).flatten().tolist()

            strip = self._conv_tail_bits()
            if strip > 0 and len(decoded) >= strip:
                decoded = decoded[:-strip]

            return [int(v) & 1 for v in decoded]

        raise ValueError("Unsupported FEC mode")


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


def upsample_and_shape(symbols: np.ndarray, sps: int, taps: np.ndarray) -> np.ndarray:
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    y = signal.lfilter(taps, [1.0], up)
    return y.astype(np.complex64)


def build_payload_section(args) -> Tuple[bytes, str]:
    """
    Returns payload+crc bytes and a description string of source.
    """
    if args.random_bits is not None:
        payload_crc = build_payload_bytes_from_random_bits(args.random_bits, seed=args.random_seed)
        return payload_crc, f"random_bits:{args.random_bits}"
    else:
        payload_crc = build_payload_bytes_from_message(read_message_arg(args))
        return payload_crc, "message"


def build_tx_bitstream(
    payload_crc_bytes: bytes,
    fec_cfg: FECConfig,
    interleave: bool,
    interleave_rows: int,
) -> Tuple[List[int], int, int, int]:
    access_bits = ACCESS_BITS

    payload_len = len(payload_crc_bytes) - 4
    header_bytes = build_header_bytes(payload_len)
    header_bits = bytes_to_bits_msb(header_bytes)
    header_bits_prot = rep3_encode_bits(header_bits)

    payload_bits_plain = bytes_to_bits_msb(payload_crc_bytes)

    fec = FECCodec(fec_cfg)
    payload_bits_coded = fec.encode_bits(payload_bits_plain)

    if interleave:
        payload_bits_coded = block_interleave_bits(payload_bits_coded, rows=interleave_rows)

    tx_bits = access_bits + header_bits_prot + payload_bits_coded + [0] * POSTAMBLE_BITS
    return tx_bits, payload_len, len(payload_bits_plain), len(payload_bits_coded)


def tx_waveform(tx_bits: List[int], sps: int, beta: float, span: int) -> np.ndarray:
    syms = dbpsk_symbols_from_bits(tx_bits, prepend_ref=True)
    taps = rrc_taps(beta=beta, sps=sps, span=span)
    return upsample_and_shape(syms, sps=sps, taps=taps)


def build_access_reference_waveform(sps: int, beta: float, span: int) -> np.ndarray:
    return tx_waveform(ACCESS_BITS, sps=sps, beta=beta, span=span)


def build_access_reference_symbols() -> np.ndarray:
    return dbpsk_symbols_from_bits(ACCESS_BITS, prepend_ref=True)


def coarse_frequency_acquire(
    iq: np.ndarray,
    ref_waveform: np.ndarray,
    sample_rate_hz: float,
    search_hz: float,
    n_bins: int,
) -> Tuple[int, float, float]:
    if len(iq) < len(ref_waveform):
        raise RuntimeError("IQ shorter than reference waveform")

    best_metric = -1.0
    best_start = 0
    best_cfo = 0.0

    n = np.arange(len(iq), dtype=np.float64)
    bins = np.linspace(-search_hz, search_hz, n_bins)
    ref = np.asarray(ref_waveform, dtype=np.complex64)

    for f_hz in bins:
        rot = np.exp(-1j * 2.0 * np.pi * f_hz * n / sample_rate_hz)
        y = iq * rot
        corr = signal.correlate(y, ref, mode="valid", method="fft")
        mag = np.abs(corr)
        idx = int(np.argmax(mag))
        metric = float(mag[idx])

        if metric > best_metric:
            best_metric = metric
            best_start = idx
            best_cfo = float(f_hz)

    return best_start, best_cfo, best_metric


def matched_filter(iq: np.ndarray, sps: int, beta: float, span: int) -> np.ndarray:
    taps = rrc_taps(beta=beta, sps=sps, span=span)
    return signal.lfilter(taps, [1.0], iq).astype(np.complex64)


def extract_symbols_from_start(
    mf: np.ndarray,
    start_index_samples: int,
    sps: int,
    span: int,
    sample_delta: int = 0,
) -> np.ndarray:
    first = start_index_samples + 2 * span * sps + sample_delta
    if first < 0 or first >= len(mf):
        return np.array([], dtype=np.complex64)
    return np.asarray(mf[first::sps], dtype=np.complex64)


def estimate_residual_cfo_from_preamble(
    symbols: np.ndarray,
    symbol_rate_hz: float,
) -> float:
    half = PREAMBLE_HALF_LEN_BITS
    start1 = 1
    start2 = 1 + half

    if len(symbols) < start2 + half:
        return 0.0

    s1 = symbols[start1:start1 + half]
    s2 = symbols[start2:start2 + half]
    ph = np.angle(np.sum(s2 * np.conj(s1)))
    sep_symbols = float(half)

    if sep_symbols <= 0:
        return 0.0

    cfo_hz = (ph / (2.0 * np.pi * sep_symbols)) * symbol_rate_hz
    return float(cfo_hz)


def apply_symbol_rate_cfo(symbols: np.ndarray, cfo_hz: float, symbol_rate_hz: float) -> np.ndarray:
    if cfo_hz == 0.0:
        return symbols.astype(np.complex64)
    n = np.arange(len(symbols), dtype=np.float64)
    rot = np.exp(-1j * 2.0 * np.pi * cfo_hz * n / symbol_rate_hz)
    return (symbols * rot).astype(np.complex64)


def design_symbol_equalizer_ls(
    rx_train: np.ndarray,
    tx_train: np.ndarray,
    ntaps: int = 7,
    ridge: float = 1e-3,
) -> np.ndarray:
    if ntaps % 2 == 0:
        raise ValueError("ntaps must be odd")
    if len(rx_train) != len(tx_train):
        raise ValueError("Training sequences must have same length")
    if len(rx_train) < ntaps:
        return np.array([1.0 + 0.0j], dtype=np.complex64)

    half = ntaps // 2
    rpad = np.pad(rx_train, (half, half), mode="constant")
    X = np.stack([rpad[i:i + ntaps] for i in range(len(rx_train))], axis=0)

    A = X.conj().T @ X + ridge * np.eye(ntaps, dtype=np.complex128)
    b = X.conj().T @ tx_train
    w = np.linalg.solve(A, b)
    return w.astype(np.complex64)


def apply_symbol_equalizer(symbols: np.ndarray, w: np.ndarray) -> np.ndarray:
    ntaps = len(w)
    if ntaps == 1:
        return (symbols * w[0]).astype(np.complex64)

    half = ntaps // 2
    spad = np.pad(symbols, (half, half), mode="constant")
    out = np.empty(len(symbols), dtype=np.complex64)

    for i in range(len(symbols)):
        x = spad[i:i + ntaps]
        out[i] = np.dot(x, w)

    return out


def fine_phase_correct_from_training(
    symbols: np.ndarray,
    tx_access_symbols: np.ndarray,
) -> np.ndarray:
    train_start = 1 + PREAMBLE_BITS_LEN + SYNC_BITS_LEN
    train_end = train_start + TRAINING_BITS_LEN

    if len(symbols) < train_end:
        return symbols.astype(np.complex64)

    rx_train = symbols[train_start:train_end]
    tx_train = tx_access_symbols[train_start:train_end]

    ph = np.angle(np.sum(rx_train * np.conj(tx_train)))
    rot = np.exp(-1j * ph)
    return (symbols * rot).astype(np.complex64)


def differential_soft_demod(symbols: np.ndarray) -> np.ndarray:
    if len(symbols) < 2:
        return np.array([], dtype=np.float64)
    z = symbols[1:] * np.conj(symbols[:-1])
    return (-np.real(z)).astype(np.float64)


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
    fec_cfg = FECConfig(mode=args.fec)

    payload_crc_bytes, payload_source = build_payload_section(args)
    tx_bits, payload_len, payload_plain_bits, payload_coded_bits = build_tx_bitstream(
        payload_crc_bytes=payload_crc_bytes,
        fec_cfg=fec_cfg,
        interleave=args.interleave,
        interleave_rows=args.interleave_rows,
    )

    iq = tx_waveform(tx_bits, sps=args.sps, beta=args.beta, span=args.span)

    iq = apply_carrier_frequency(
        iq,
        carrier_hz=args.carrier_hz,
        sample_rate_hz=args.sample_rate_hz,
    )

    multipath_taps = None
    if args.multipath_taps:
        if isinstance(args.multipath_taps, str):
            multipath_taps = [complex(t) for t in args.multipath_taps.split(",")]
        else:
            multipath_taps = args.multipath_taps

    iq = impair_iq(
        iq,
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

    save_iq(args.output, iq)

    metadata_path = save_iq_metadata(
        iq_path=args.output,
        sample_rate_hz=args.sample_rate_hz,
        rf_center_hz=args.rf_center_hz,
        carrier_hz=args.carrier_hz,
        metadata_path=args.metadata_path,
    )

    result = {
        "output": args.output,
        "metadata_path": metadata_path,
        "sample_rate_hz": float(args.sample_rate_hz),
        "rf_center_hz": float(args.rf_center_hz),
        "carrier_hz": float(args.carrier_hz),
        "absolute_rf_hz": float(args.rf_center_hz + args.carrier_hz),
        "payload_source": payload_source,
        "payload_len": payload_len,
        "fec": args.fec,
        "interleave": args.interleave,
        "interleave_rows": args.interleave_rows,
        "payload_plain_bits": payload_plain_bits,
        "payload_coded_bits": payload_coded_bits,
        "snr_db": args.snr_db,
        "noise_color": args.noise_color,
        "fading_mode": args.fading_mode,
        "num_iq_samples": int(len(iq)),
    }

    print(f"Wrote IQ: {args.output}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Sample rate (Hz): {args.sample_rate_hz}")
    print(f"RF center (Hz): {args.rf_center_hz}")
    print(f"Digital carrier offset (Hz): {args.carrier_hz}")
    print(f"Absolute RF (Hz): {args.rf_center_hz + args.carrier_hz}")
    print(f"Payload source: {payload_source}")
    print(f"Payload bytes: {payload_len}")
    print(f"FEC: {args.fec}")
    print(f"Interleave: {args.interleave} (rows={args.interleave_rows})")
    print(f"Payload+CRC bits before FEC: {payload_plain_bits}")
    print(f"Payload+CRC bits after FEC: {payload_coded_bits}")
    print(f"SNR(dB): {args.snr_db}")
    print(f"Noise color: {args.noise_color}")
    print(f"Fading mode: {args.fading_mode}")

    return result


def try_decode_from_symbols(
    symbols: np.ndarray,
    fec_cfg: FECConfig,
    interleave: bool,
    interleave_rows: int,
    symbol_rate_hz: float,
    eq_taps: int,
    access_ref_symbols: np.ndarray,
) -> Optional[bytes]:
    if len(symbols) < len(access_ref_symbols) + 16:
        return None

    residual_cfo_hz = estimate_residual_cfo_from_preamble(symbols, symbol_rate_hz)
    symbols = apply_symbol_rate_cfo(symbols, residual_cfo_hz, symbol_rate_hz)

    train_start = 1 + PREAMBLE_BITS_LEN + SYNC_BITS_LEN
    train_end = train_start + TRAINING_BITS_LEN

    if len(symbols) < train_end:
        return None

    rx_train = symbols[train_start:train_end]
    tx_train = access_ref_symbols[train_start:train_end]

    w = design_symbol_equalizer_ls(rx_train, tx_train, ntaps=eq_taps, ridge=1e-3)
    eq_symbols = apply_symbol_equalizer(symbols, w)
    eq_symbols = fine_phase_correct_from_training(eq_symbols, access_ref_symbols)

    soft_bits = np.clip(differential_soft_demod(eq_symbols), -10.0, 10.0)
    if len(soft_bits) < ACCESS_BITS_LEN + HEADER_PROT_BITS_LEN:
        return None

    hdr_start = ACCESS_BITS_LEN
    hdr_end = hdr_start + HEADER_PROT_BITS_LEN
    header_soft = soft_bits[hdr_start:hdr_end]
    header_bits = rep3_decode_soft(header_soft)
    if len(header_bits) < HEADER_BITS_LEN:
        return None

    header_bytes = bits_to_bytes_msb(header_bits[:HEADER_BITS_LEN])
    payload_len = parse_header_bytes(header_bytes)
    if payload_len is None:
        return None

    payload_plain_bits = (payload_len + 4) * 8
    fec = FECCodec(fec_cfg)
    payload_coded_bits = fec.encoded_length(payload_plain_bits)

    pay_start = hdr_end
    pay_end = pay_start + payload_coded_bits
    if pay_end > len(soft_bits):
        return None

    payload_soft = soft_bits[pay_start:pay_end]
    if interleave:
        payload_soft = block_deinterleave_soft(payload_soft, rows=interleave_rows)

    payload_bits = fec.decode_soft(payload_soft)
    if len(payload_bits) < payload_plain_bits:
        return None

    payload_bits = payload_bits[:payload_plain_bits]
    payload_bytes = bits_to_bytes_msb(payload_bits)
    payload = parse_payload_bytes(payload_bytes, payload_len)
    if payload is None:
        return None

    return payload


def rx_command(args):
    fec_cfg = FECConfig(mode=args.fec)
    iq = load_iq(args.input)

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

    baseband_offset_hz = tx_absolute_rf_hz - rx_rf_center_hz
    if abs(baseband_offset_hz) >= tx_sample_rate_hz / 2:
        raise ValueError(
            f"Requested RX RF center gives baseband offset {baseband_offset_hz} Hz, "
            f"outside Nyquist for TX/file sample rate {tx_sample_rate_hz} Hz"
        )

    iq = apply_carrier_frequency(iq, carrier_hz=-baseband_offset_hz, sample_rate_hz=tx_sample_rate_hz)

    if not np.isclose(rx_sample_rate_hz, tx_sample_rate_hz):
        iq = resample_iq(iq, fs_in_hz=tx_sample_rate_hz, fs_out_hz=rx_sample_rate_hz)

    dsp_sample_rate_hz = tx_sample_rate_hz
    if not np.isclose(rx_sample_rate_hz, dsp_sample_rate_hz):
        iq = resample_iq(iq, fs_in_hz=rx_sample_rate_hz, fs_out_hz=dsp_sample_rate_hz)

    access_ref_waveform = build_access_reference_waveform(sps=sps, beta=args.beta, span=args.span)
    access_ref_symbols = build_access_reference_symbols()

    coarse_start, coarse_cfo_hz, coarse_metric = coarse_frequency_acquire(
        iq=iq,
        ref_waveform=access_ref_waveform,
        sample_rate_hz=dsp_sample_rate_hz,
        search_hz=args.coarse_freq_search_hz,
        n_bins=args.coarse_freq_bins,
    )

    iq = apply_carrier_frequency(iq, carrier_hz=-coarse_cfo_hz, sample_rate_hz=dsp_sample_rate_hz)
    mf = matched_filter(iq, sps=sps, beta=args.beta, span=args.span)

    payload = None
    sample_offset_used = None

    for sample_delta in range(-args.sample_phase_search, args.sample_phase_search + 1):
        symbols = extract_symbols_from_start(
            mf=mf,
            start_index_samples=coarse_start,
            sps=sps,
            span=args.span,
            sample_delta=sample_delta,
        )
        payload = try_decode_from_symbols(
            symbols=symbols,
            fec_cfg=fec_cfg,
            interleave=args.interleave,
            interleave_rows=args.interleave_rows,
            symbol_rate_hz=symbol_rate_hz,
            eq_taps=args.eq_taps,
            access_ref_symbols=access_ref_symbols,
        )
        if payload is not None:
            sample_offset_used = sample_delta
            break

    if payload is None:
        raise RuntimeError("No valid packet found after acquisition, header decode, FEC decode, and CRC")

    try:
        message_text = payload.decode("utf-8")
    except UnicodeDecodeError:
        message_text = None

    result = {
        "input": args.input,
        "metadata_path": args.metadata_path if args.metadata_path else default_metadata_path(args.input),
        "tx_sample_rate_hz": tx_sample_rate_hz,
        "rx_sample_rate_hz": rx_sample_rate_hz,
        "dsp_sample_rate_hz": dsp_sample_rate_hz,
        "tx_rf_center_hz": tx_rf_center_hz,
        "tx_carrier_hz": tx_carrier_hz,
        "tx_absolute_rf_hz": tx_absolute_rf_hz,
        "rx_rf_center_hz": rx_rf_center_hz,
        "baseband_offset_hz": baseband_offset_hz,
        "coarse_cfo_hz": coarse_cfo_hz,
        "coarse_metric": coarse_metric,
        "sample_offset_used": sample_offset_used,
        "effective_sps": sps,
        "fec": args.fec,
        "interleave": args.interleave,
        "interleave_rows": args.interleave_rows,
        "payload_bytes": payload,
        "payload_len": len(payload),
        "message": message_text,
        "output_file": args.output_file,
    }

    # print("Recovered payload length:", len(payload))
    # print(f"TX/file sample rate (Hz): {tx_sample_rate_hz}")
    # print(f"RX front-end sample rate (Hz): {rx_sample_rate_hz}")
    # print(f"DSP sample rate (Hz): {dsp_sample_rate_hz}")
    # print(f"TX absolute RF (Hz): {tx_absolute_rf_hz}")
    # print(f"RX RF center (Hz): {rx_rf_center_hz}")
    # print(f"Baseband offset corrected (Hz): {baseband_offset_hz}")
    # print(f"Coarse CFO estimate (Hz): {coarse_cfo_hz}")
    # print(f"Coarse acquisition metric: {coarse_metric:.3f}")
    # print(f"Sample offset used: {sample_offset_used}")

    if message_text is not None:
        print("Recovered message:")
        print(message_text)
    else:
        print("Recovered payload is binary/non-UTF8.")

    if args.output_file:
        mode = "w" if message_text is not None else "wb"
        with open(args.output_file, mode) as f:
            if message_text is not None:
                f.write(message_text)
            else:
                f.write(payload)
        print(f"Saved recovered payload to: {args.output_file}")

    return result


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    tx = sub.add_parser("tx")
    tx.add_argument("--message", type=str)
    tx.add_argument("--message-file", type=str)
    tx.add_argument("--random-bits", type=int, default=None,
                    help="Transmit a random bit string of this length instead of a message.")
    tx.add_argument("--random-seed", type=int, default=1,
                    help="Seed for random bit-string generation.")
    tx.add_argument("--output", type=str, default="tx_skdsp_v3.iq")
    tx.add_argument("--metadata-path", type=str, default=None)

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
    rx.add_argument("--input", type=str, default="tx_skdsp_v3.iq")
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

    rx.add_argument("--coarse-freq-search-hz", type=float, default=20_000.0)
    rx.add_argument("--coarse-freq-bins", type=int, default=81)
    rx.add_argument("--sample-phase-search", type=int, default=2)
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