#!/usr/bin/env python3
"""
load_tx_iq_data.py

Loader utilities for:
- whole IQ outputs
- subsection IQ outputs
- combined sample bundles
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_whole_iq(sample_dir: Path) -> Dict:
    sample_dir = Path(sample_dir)
    iq = np.load(sample_dir / "whole_iq.npy")
    with open(sample_dir / "whole_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "iq": iq,
        "meta": meta,
    }


def load_sections(sample_dir: Path) -> Dict:
    sample_dir = Path(sample_dir)
    sections = np.load(sample_dir / "sections.npy")
    with open(sample_dir / "sections_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "sections": sections,
        "meta": meta,
    }


def load_sample_bundle(sample_dir: Path) -> Dict:
    whole = load_whole_iq(sample_dir)
    sections = load_sections(sample_dir)
    return {
        "whole_iq": whole["iq"],
        "whole_meta": whole["meta"],
        "sections": sections["sections"],
        "sections_meta": sections["meta"],
    }


def list_sample_dirs(root: Path) -> List[Path]:
    root = Path(root)
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("sample_")])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=str, required=True)
    args = parser.parse_args()

    bundle = load_sample_bundle(Path(args.sample_dir))
    print("Whole IQ shape:", bundle["whole_iq"].shape)
    print("Sections shape:", bundle["sections"].shape)
    print("Whole metadata keys:", sorted(bundle["whole_meta"].keys()))
    print("Section starts:", bundle["sections_meta"]["starts"])