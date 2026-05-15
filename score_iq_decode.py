#!/usr/bin/env python3
"""
score_iq_decode.py

Load an IQ file and metadata, attempt decode with advanced_link_skdsp_v7_robust,
and output an error score:

- Pre-FEC coded BER from payload soft decisions after pilot removal/deinterleaving.
- Plus post-FEC payload+CRC BER after FEC decode and descrambling, before CRC rejection.
- Plus 1.0 when the expected decoded payload/message is missing.

Lower scores are better; a clean decode with no bit errors scores 0.0.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

import advanced_link_skdsp_v7_robust as link


def load_iq_file(iq_path: Path) -> np.ndarray:
    suffix = iq_path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(iq_path), dtype=np.complex64)
    return np.fromfile(iq_path, dtype=np.complex64)


def write_temp_raw_iq(iq: np.ndarray) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".iq", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    np.asarray(iq, dtype=np.complex64).tofile(tmp_path)
    return tmp_path


def reconstruct_expected_random_payload(random_bits: int, random_seed: int) -> bytes:
    bits = link.prbs_bits(random_bits, seed=random_seed)
    pad = (-random_bits) % 8
    if pad:
        bits = bits + [0] * pad
    return link.bits_to_bytes_msb(bits)


def bytes_to_bitstring(data: bytes, n_bits: Optional[int] = None) -> str:
    s = "".join(f"{b:08b}" for b in data)
    if n_bits is not None:
        s = s[:n_bits]
    return s


def decode_iq(iq_raw_path: Path, metadata: dict) -> Optional[dict]:
    rx_args = [
        "rx",
        "--input", str(iq_raw_path),
        "--sps", str(metadata.get("sps", 8)),
        "--beta", str(metadata.get("beta", 0.35)),
        "--span", str(metadata.get("span", 6)),
        "--sample-rate-hz", str(metadata.get("sample_rate_hz", 1_000_000.0)),
        "--rf-center-hz", str(metadata.get("rf_center_hz", 0.0)),
        "--tx-sample-rate-hz", str(metadata.get("sample_rate_hz", 1_000_000.0)),
        "--tx-rf-center-hz", str(metadata.get("rf_center_hz", 0.0)),
        "--tx-carrier-hz", str(metadata.get("carrier_hz", 0.0)),
        "--fec", str(metadata.get("fec", "none")),
        "--coarse-freq-search-hz", "25000",
        "--coarse-freq-bins", "101",
        "--sample-phase-search", "3",
        "--eq-taps", "7",
    ]

    if metadata.get("interleave", False):
        rx_args.append("--interleave")

    try:
        return link.main(rx_args)
    except Exception:
        return None


def expected_payload_crc_bytes(metadata: dict) -> Optional[bytes]:
    payload_source = metadata.get("payload_source")
    payload_desc = metadata.get("payload_desc", {})

    if payload_source == "message" or payload_desc.get("mode") == "message":
        message = metadata.get("message")
        if message is None:
            message = payload_desc.get("message")
        if message is None:
            return None
        return link.build_payload_bytes_from_message(message)

    if (
        isinstance(payload_source, str)
        and payload_source.startswith("random_bits:")
    ) or payload_desc.get("mode") == "random_bits":
        random_bits = metadata.get("random_bits")
        random_seed = metadata.get("random_seed")
        if random_bits is None:
            random_bits = payload_desc.get("random_bits")
        if random_seed is None:
            random_seed = payload_desc.get("random_seed")
        if random_bits is None or random_seed is None:
            return None
        return link.build_payload_bytes_from_random_bits(int(random_bits), int(random_seed))

    return None


def bit_error_rate(rx_bits, tx_bits) -> float:
    if rx_bits is None or tx_bits is None:
        return 0.0
    tx = np.asarray(tx_bits, dtype=np.int8)
    rx = np.asarray(rx_bits, dtype=np.int8)
    if tx.size == 0:
        return 0.0
    n = min(rx.size, tx.size)
    errors = int(np.count_nonzero(rx[:n] != tx[:n]))
    errors += max(0, int(tx.size) - int(rx.size))
    return float(errors) / float(tx.size)


def expected_coded_bits(payload_crc_bytes: bytes, fec_mode: str) -> list[int]:
    payload_plain_bits = link.bytes_to_bits_msb(payload_crc_bytes)
    payload_scrambled = link.scramble_bits(payload_plain_bits, seed=0x5D)
    return link.FECCodec(fec_mode).encode_bits(payload_scrambled)


def decoded_payload_missing(rx_result: Optional[dict], metadata: dict) -> bool:
    if rx_result is None:
        return True

    payload_source = metadata.get("payload_source")
    payload_desc = metadata.get("payload_desc", {})
    if payload_source == "message" or payload_desc.get("mode") == "message":
        return rx_result.get("message") is None

    if (
        isinstance(payload_source, str)
        and payload_source.startswith("random_bits:")
    ) or payload_desc.get("mode") == "random_bits":
        return rx_result.get("payload_bytes") is None

    return rx_result.get("payload_bytes") is None and rx_result.get("message") is None


def score_decode(rx_result: Optional[dict], metadata: dict) -> float:
    expected_payload_crc = expected_payload_crc_bytes(metadata)
    missing_penalty = 2.0 if decoded_payload_missing(rx_result, metadata) else 0.0

    # if rx_result is None or expected_payload_crc is None:
    #     return missing_penalty

    diagnostics = rx_result.get("decode_diagnostics") or {}
    fec_mode = str(metadata.get("fec", "none"))

    tx_payload_crc_bits = link.bytes_to_bits_msb(expected_payload_crc)
    tx_coded_bits = expected_coded_bits(expected_payload_crc, fec_mode)

    pre_fec_coded_ber = bit_error_rate(
        diagnostics.get("rx_coded_bits_hard"),
        tx_coded_bits,
    )
    post_fec_payload_crc_ber = bit_error_rate(
        diagnostics.get("rx_fec_decoded_bits"),
        tx_payload_crc_bits,
    )

    return pre_fec_coded_ber + post_fec_payload_crc_ber + missing_penalty


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iq", type=str, required=True, help="Path to IQ file (.npy or raw complex64)")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata JSON")
    return parser


def main(argv=None) -> float:
    parser = build_parser()
    args = parser.parse_args(argv)

    iq_path = Path(args.iq)
    meta_path = Path(args.metadata)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    iq = load_iq_file(iq_path)
    temp_raw_path = write_temp_raw_iq(iq)

    try:
        rx_result = decode_iq(temp_raw_path, metadata)
        score = score_decode(rx_result, metadata)
        print(score)
        return score
    finally:
        try:
            temp_raw_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()