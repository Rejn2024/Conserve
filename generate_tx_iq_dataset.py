#!/usr/bin/env python3
"""
generate_tx_iq_dataset.py

Generate an arbitrary number of TX IQ outputs using realistic parameters,
save whole IQ outputs plus 3 random 1024-sample subsections from each output,
and save all metadata.

This version uses advanced_link_skdsp_v4_robust.build_tx_iq_object(...)
and guarantees that each WHOLE transmission kept in the dataset is decodable
by advanced_link_skdsp_v4_robust. If a candidate is not decodable, it is
discarded and replaced by a new one.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import tempfile
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import advanced_link_skdsp_v6_robust as link


ALPHANUM_SPACE = string.ascii_letters + string.digits + "     "


def random_phrase(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(ALPHANUM_SPACE) for _ in range(length)).strip() or "X"


def realistic_params(rng: random.Random) -> Dict:
    """
    Realistic but still decodable region for current v4 RX.
    """

    sr = rng.choice([(500_000.0, 48_000.0), 
                     (1_000_000.0, 200_000.0),
                     (1_000_000_000.0, 433_920_000.0),
                     (2_000_000_000.0, 915_000_000.0),
                     (5_000_000_000.0, 2_400_000_000.0)])
    
    sample_rate_hz = sr[0]
    rf_center_hz = sr[1]

    # Keep digital carrier at zero for reliable decode with current v4 RX.
    carrier_hz = 0.0

    fec = rng.choice(["conv", "conv", "conv", "rep3"])
    interleave = True

    noise_color = rng.choice(["white", "white", "pink"])
    snr_db = rng.uniform(28.0, 42.0)

    fading_mode = rng.choices(
        ["none", "rician_block", "multipath_static"],
        weights=[0.50, 0.35, 0.15],
        k=1,
    )[0]

    multipath_taps = None
    if fading_mode == "multipath_static":
        multipath_taps = [
            1.0 + 0.0j,
            complex(rng.uniform(0.02, 0.10), rng.uniform(-0.04, 0.04)),
            complex(rng.uniform(0.00, 0.04), rng.uniform(-0.03, 0.03)),
        ]

    return {
        "sample_rate_hz": sample_rate_hz,
        "rf_center_hz": rf_center_hz,
        "carrier_hz": carrier_hz,
        "fec": fec,
        "interleave": interleave,
        "snr_db": snr_db,
        "noise_color": noise_color,
        "freq_offset": rng.uniform(-2e-5, 2e-5),
        "timing_offset": rng.uniform(0.99999, 1.00001),
        "fading_mode": fading_mode,
        "fading_block_len": rng.choice([512, 1024, 2048]),
        "rician_k_db": rng.uniform(10.0, 18.0),
        "multipath_taps": multipath_taps,
        "burst_probability": rng.choice([0.0, 0.0, 0.0, 1e-5]),
        "burst_len_min": 16,
        "burst_len_max": 64,
        "burst_power_ratio_db": rng.uniform(8.0, 10.0),
        "burst_color": "white",
    }


def conservative_fallback_params(rng: random.Random) -> Dict:
    """
    Easier channel used after repeated failures.
    """
    print("resorted to fallback")
    sample_rate_hz = rng.choice([1_000_000_000.0, 2_000_000_000.0, 4_000_000_000.0])
    rf_center_hz = rng.choice([433_920_000.0, 915_000_000.0, 2_400_000_000.0])

    return {
        "sample_rate_hz": sample_rate_hz,
        "rf_center_hz": rf_center_hz,
        "carrier_hz": 0.0,
        "fec": "conv",
        "interleave": True,
        "snr_db": rng.uniform(38.0, 55.0),
        "noise_color": "white",
        "freq_offset": rng.uniform(-5e-6, 5e-6),
        "timing_offset": rng.uniform(0.999998, 1.000002),
        "fading_mode": "none",
        "fading_block_len": 1024,
        "rician_k_db": 14.0,
        "multipath_taps": None,
        "burst_probability": 0.0,
        "burst_len_min": 16,
        "burst_len_max": 64,
        "burst_power_ratio_db": 8.0,
        "burst_color": "white",
    }


def cut_random_sections(
    iq: np.ndarray,
    num_sections: int,
    section_len: int,
    seed: int,
    lengthening_factor: int = 5 # present to provide a hedge against lower jammer sampling frequencies 
) -> Dict:
    
    lengthened_section = lengthening_factor * section_len

    if len(iq) < section_len:
        raise ValueError("IQ shorter than section length")

    rng = np.random.default_rng(seed)
    max_start = len(iq) - 5*section_len
    starts = [int(rng.integers(0, max_start + 1)) for _ in range(num_sections)]
    sections = np.stack([iq[s:s + lengthened_section] for s in starts], axis=0)

    return {
        "sections": sections.astype(np.complex64),
        "starts": starts,
        "section_len": section_len,
        "num_sections": num_sections,
        "lengthening_factor": lengthening_factor
    }


def save_sample_bundle(
    out_dir: Path,
    whole_iq: np.ndarray,
    whole_meta: Dict,
    sections: np.ndarray,
    sections_meta: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "whole_iq.npy", whole_iq.astype(np.complex64))
    with open(out_dir / "whole_meta.json", "w", encoding="utf-8") as f:
        json.dump(whole_meta, f, indent=2)

    np.save(out_dir / "sections.npy", sections.astype(np.complex64))
    with open(out_dir / "sections_meta.json", "w", encoding="utf-8") as f:
        json.dump(sections_meta, f, indent=2)


def _build_payload_description(
    rng: random.Random,
    idx: int,
    random_payload_probability: float,
    min_chars: int = 500,
    max_chars: int = 1000,
) -> Tuple[Optional[str], Optional[int], int, Dict]:
    if rng.random() < random_payload_probability:
        n_bits = rng.randint(min_chars * 8, max_chars * 8)
        random_seed = 10_000 + idx
        payload_desc = {
            "mode": "random_bits",
            "random_bits": n_bits,
            "random_seed": random_seed,
        }
        return None, n_bits, random_seed, payload_desc

    phrase_len = rng.randint(min_chars, max_chars)
    phrase = random_phrase(phrase_len, rng)
    payload_desc = {
        "mode": "message",
        "message_length": len(phrase),
        "message_preview": phrase[:120],
        "message": phrase,
    }
    return phrase, None, 1, payload_desc


def _decode_candidate_with_v4(
    iq: np.ndarray,
    metadata: Dict,
) -> Optional[Dict]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        iq_path = td_path / "candidate.iq"
        meta_path = td_path / "candidate.iq.json"

        np.asarray(iq.detach().cpu(), dtype=np.complex64).tofile(iq_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sample_rate_hz": float(metadata["sample_rate_hz"]),
                    "rf_center_hz": float(metadata["rf_center_hz"]),
                    "carrier_hz": float(metadata["carrier_hz"]),
                    "absolute_rf_hz": float(metadata["absolute_rf_hz"]),
                },
                f,
                indent=2,
            )

        rx_args = [
            "rx",
            "--input", str(iq_path),
            "--metadata-path", str(meta_path),
            "--fec", str(metadata.get("fec", "none")),
            "--sps", str(metadata.get("sps", 8)),
            "--beta", str(metadata.get("beta", 0.35)),
            "--span", str(metadata.get("span", 6)),
            "--sample-rate-hz", str(metadata.get("sample_rate_hz", 1_000_000_000.0)),
            "--rf-center-hz", str(metadata.get("rf_center_hz", 0.0)),
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


def _candidate_matches_payload(rx_result: Optional[Dict], payload_desc: Dict) -> bool:
    if rx_result is None:
        return False

    mode = payload_desc.get("mode")
    if mode == "message":
        expected = payload_desc.get("message")
        return expected is not None and rx_result.get("message") == expected

    if mode == "random_bits":
        n_bits = int(payload_desc["random_bits"])
        seed = int(payload_desc["random_seed"])
        expected_payload = link.build_payload_bytes_from_random_bits(n_bits, seed)[:-4]
        return rx_result.get("payload_bytes") == expected_payload

    return False


def _build_candidate_iq_with_full_packet(
    *,
    message: Optional[str],
    random_bits: Optional[int],
    random_seed: int,
    requested_target_num_samples: int,
    params: Dict,
) -> link.TXBuildResult:
    """
    Ensure the candidate contains at least one full packet.

    We first build the natural full packet with target_num_samples=None.
    Then we enlarge to max(requested_target_num_samples, full_packet_len).
    """
    base_result = link.build_tx_iq_object(
        message=message,
        random_bits=random_bits,
        random_seed=random_seed,
        target_num_samples=None,
        **params,
    )

    min_required = len(base_result.iq)
    final_target = max(int(requested_target_num_samples), int(min_required))

    if final_target == min_required:
        return base_result

    return link.build_tx_iq_object(
        message=message,
        random_bits=random_bits,
        random_seed=random_seed,
        target_num_samples=final_target,
        **params,
    )


def build_decodable_sample(
    *,
    dataset_index: int,
    target_num_samples: int,
    rng: random.Random,
    random_payload_probability: float,
    max_attempts_per_sample: int = 100,
) -> Tuple[np.ndarray, Dict]:
    for attempt in range(max_attempts_per_sample):
        params = realistic_params(rng) if attempt < max_attempts_per_sample // 2 else conservative_fallback_params(rng)

        message, random_bits, random_seed, payload_desc = _build_payload_description(
            rng=rng,
            idx=dataset_index * 1000 + attempt,
            random_payload_probability=random_payload_probability,
        )

        tx_result = _build_candidate_iq_with_full_packet(
            message=message,
            random_bits=random_bits,
            random_seed=random_seed,
            requested_target_num_samples=target_num_samples,
            params=params,
        )

        whole_meta = {
            **tx_result.metadata,
            "dataset_index": dataset_index,
            "payload_desc": payload_desc,
            "generation_attempt": attempt + 1,
            "requested_target_num_samples": int(target_num_samples),
        }

        rx_result = _decode_candidate_with_v4(tx_result.iq, whole_meta)
        if _candidate_matches_payload(rx_result, payload_desc):
            whole_meta["decode_verified"] = True
            whole_meta["decoder_result_summary"] = {
                "payload_len": int(rx_result.get("payload_len", 0)),
                "message_is_text": rx_result.get("message") is not None,
            }

            print("sucessfully generated data")
            
            if isinstance(tx_result.iq, np.ndarray):
                return tx_result.iq.astype(np.complex64), whole_meta
            elif isinstance(tx_result.iq, torch.Tensor):
                tri = tx_result.iq.detach().cpu().numpy()
                return tri.astype(np.complex64), whole_meta


    raise RuntimeError(
        f"Failed to generate a decodable transmission after {max_attempts_per_sample} attempts "
        f"for dataset index {dataset_index}."
    )


def generate_dataset(
    output_root: Path,
    num_outputs: int,
    min_total_samples: int,
    max_total_samples: int,
    section_len: int = 1024,
    num_sections: int = 3,
    seed: int = 1,
    random_payload_probability: float = 0.5,
    max_attempts_per_sample: int = 100,
    start_index: int = -1,
) -> List[Path]:
    rng = random.Random(seed)
    output_root.mkdir(parents=True, exist_ok=True)

    produced_dirs: List[Path] = []

    for i in range(num_outputs):
        sample_index = start_index + i + 1
        requested_target_num_samples = rng.randint(min_total_samples, max_total_samples)

        whole_iq, whole_meta = build_decodable_sample(
            dataset_index=sample_index,
            target_num_samples=requested_target_num_samples,
            rng=rng,
            random_payload_probability=random_payload_probability,
            max_attempts_per_sample=max_attempts_per_sample,
        )

        cuts = cut_random_sections(
            iq=whole_iq,
            num_sections=num_sections,
            section_len=section_len,
            seed=seed + 10_000 + sample_index,
        )

        sample_dir = output_root / f"sample_{sample_index:06d}"
        sections_meta = {
            "dataset_index": sample_index,
            "section_len": section_len,
            "num_sections": num_sections,
            "starts": cuts["starts"],
            "whole_num_samples": int(len(whole_iq)),
        }

        save_sample_bundle(
            out_dir=sample_dir,
            whole_iq=whole_iq,
            whole_meta=whole_meta,
            sections=cuts["sections"],
            sections_meta=sections_meta,
        )
        produced_dirs.append(sample_dir)

    return produced_dirs


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--num-outputs", type=int, default=10)
    parser.add_argument("--min-total-samples", type=int, default=8192)
    parser.add_argument("--max-total-samples", type=int, default=65536)
    parser.add_argument("--section-len", type=int, default=1024)
    parser.add_argument("--num-sections", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--random-payload-probability", type=float, default=0.5)
    parser.add_argument("--max-attempts-per-sample", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=-1)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    dirs = generate_dataset(
        output_root=Path(args.output_root),
        num_outputs=args.num_outputs,
        min_total_samples=args.min_total_samples,
        max_total_samples=args.max_total_samples,
        section_len=args.section_len,
        num_sections=args.num_sections,
        seed=args.seed,
        random_payload_probability=args.random_payload_probability,
        max_attempts_per_sample=args.max_attempts_per_sample,
        start_index=args.start_index,
    )

    print(f"Generated {len(dirs)} verified-decodable sample directories under {args.output_root}")
    return {"count": len(dirs), "output_root": str(args.output_root)}