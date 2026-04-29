#!/usr/bin/env python3
"""
score_iq_decode.py

Load an IQ file and metadata, attempt decode with advanced_link_skdsp_v4_robust,
and output a score:

- 0.0 if decode fails or raises an error
- 1.0 if decode matches the original message/bit string perfectly
- 1 / Levenshtein_distance if decode succeeds but does not match perfectly

Matching logic
--------------
Message mode:
- compare decoded UTF-8 message with original message

Random-bits mode:
- reconstruct original random bit string from metadata
- compare against decoded payload converted to bit string and truncated to original length
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

import advanced_link_skdsp_v4_robust as link


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


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


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


def score_from_strings(expected: str, decoded: Optional[str]) -> float:
    if decoded is None:
        return 1.0
    if decoded == expected:
        return 0.0
    dist = levenshtein_distance(expected, decoded)
    if dist <= 0:
        return 0.0
    return float(dist) / (1.0 + float(dist))


def score_decode(rx_result: Optional[dict], metadata: dict) -> float:
    if rx_result is None:
        return 1.0

    payload_source = metadata.get("payload_source")
    payload_desc = metadata.get("payload_desc", {})

    # Message mode
    if payload_source == "message" or payload_desc.get("mode") == "message":
        expected_message = metadata.get("message")

        if expected_message is None:
            expected_message = payload_desc.get("message")

        if expected_message is None:
            preview = payload_desc.get("message_preview")
            decoded_message = rx_result.get("message")
            if decoded_message is None:
                return 1.0
            if preview is not None and not decoded_message.startswith(preview):
                return 1.0
            expected_len = payload_desc.get("message_length")
            if expected_len is not None and len(decoded_message) != expected_len:
                return 1.0
            return 0.0

        return score_from_strings(expected_message, rx_result.get("message"))

    # Random-bit mode
    if (isinstance(payload_source, str) and payload_source.startswith("random_bits:")) or payload_desc.get("mode") == "random_bits":
        random_bits = metadata.get("random_bits")
        random_seed = metadata.get("random_seed")

        if random_bits is None:
            random_bits = payload_desc.get("random_bits")
        if random_seed is None:
            random_seed = payload_desc.get("random_seed")

        if random_bits is None or random_seed is None:
            return 1.0

        expected_payload = reconstruct_expected_random_payload(int(random_bits), int(random_seed))
        expected_bitstring = bytes_to_bitstring(expected_payload, n_bits=int(random_bits))

        decoded_payload = rx_result.get("payload_bytes")
        if decoded_payload is None:
            return 1.0

        decoded_bitstring = bytes_to_bitstring(decoded_payload, n_bits=int(random_bits))
        if decoded_bitstring == expected_bitstring:
            return 0.0

        dist = levenshtein_distance(expected_bitstring, decoded_bitstring)
        if dist <= 0:
            return 0.0
        return float(dist)  / (1 + float(dist))

    # Unknown mode fallback
    if rx_result.get("payload_len", 0) <= 0:
        return 1.0
    return 0.0


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