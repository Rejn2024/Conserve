#!/usr/bin/env python3
"""Build combat-ID training datasets from CMO scenario JSONL exports.

Input is one or more JSONL files produced by cmo_scenario_export.lua from
Command: Modern Operations (CMO). Output is a compact, model-agnostic dataset:

    output_root/
      manifest.json
      label_map.json
      train.jsonl
      val.jsonl
      test.jsonl
      all_examples.csv

Each JSONL example has a categorical `label` suitable for combat-ID training and
numeric/string features derived from CMO contacts or units.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

UNKNOWN_TOKEN = "unknown"
DEFAULT_LABEL_PRIORITY = ("posture", "actual_side", "side")


def _clean_string(value: Any) -> str:
    if value is None:
        return UNKNOWN_TOKEN
    text = str(value).strip()
    return text if text else UNKNOWN_TOKEN


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _stable_id(record: Dict[str, Any], source_path: Path, line_number: int) -> str:
    explicit = _clean_string(record.get("guid"))
    if explicit != UNKNOWN_TOKEN:
        return explicit
    payload = json.dumps(record, sort_keys=True, default=str)
    digest = hashlib.sha1(f"{source_path}:{line_number}:{payload}".encode("utf-8")).hexdigest()
    return digest[:16]


def _choose_label(record: Dict[str, Any], label_fields: Sequence[str]) -> str:
    for field in label_fields:
        label = _clean_string(record.get(field))
        if label != UNKNOWN_TOKEN:
            return label.lower().replace(" ", "_")
    return UNKNOWN_TOKEN


def load_records(paths: Sequence[Path]) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected JSON object in {path} line {line_number}")
                yield path, line_number, record


def build_example(
    record: Dict[str, Any],
    *,
    source_path: Path,
    line_number: int,
    label_fields: Sequence[str],
) -> Dict[str, Any]:
    lat = _to_float(record.get("latitude"))
    lon = _to_float(record.get("longitude"))
    altitude_m = _to_float(record.get("altitude_m"))
    speed_kts = _to_float(record.get("speed_kts"))
    heading_deg = _to_float(record.get("heading_deg"))
    course_deg = _to_float(record.get("course_deg"))

    features = {
        "record_kind": _clean_string(record.get("record_kind")),
        "observer_side": _clean_string(record.get("side")),
        "track_type": _clean_string(record.get("type")),
        "track_subtype": _clean_string(record.get("subtype")),
        "class_name": _clean_string(record.get("class_name")),
        "dbid": _clean_string(record.get("dbid")),
        "latitude": lat,
        "longitude": lon,
        "altitude_m": altitude_m,
        "speed_kts": speed_kts,
        "heading_deg": heading_deg,
        "course_deg": course_deg,
        "identification_status": _clean_string(record.get("identification_status")),
        "detected_by": _clean_string(record.get("detected_by")),
        "has_position": lat is not None and lon is not None,
        "has_kinematics": speed_kts is not None or heading_deg is not None or course_deg is not None,
    }

    return {
        "id": _stable_id(record, source_path, line_number),
        "source_file": str(source_path),
        "source_line": line_number,
        "name": _clean_string(record.get("name")),
        "label": _choose_label(record, label_fields),
        "features": features,
        "raw": record,
    }


def split_examples(
    examples: List[Dict[str, Any]],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_fraction)
    val_end = train_end + int(len(shuffled) * val_fraction)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, examples: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "label",
        "name",
        "source_file",
        "source_line",
        "record_kind",
        "observer_side",
        "track_type",
        "track_subtype",
        "class_name",
        "dbid",
        "latitude",
        "longitude",
        "altitude_m",
        "speed_kts",
        "heading_deg",
        "course_deg",
        "identification_status",
        "detected_by",
        "has_position",
        "has_kinematics",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            row = {key: example.get(key) for key in ("id", "label", "name", "source_file", "source_line")}
            row.update(example["features"])
            writer.writerow(row)


def write_dataset(
    examples: List[Dict[str, Any]],
    *,
    output_root: Path,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    input_paths: Sequence[Path],
    label_fields: Sequence[str],
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    splits = split_examples(examples, train_fraction=train_fraction, val_fraction=val_fraction, seed=seed)

    labels = sorted({example["label"] for example in examples})
    label_map = {label: idx for idx, label in enumerate(labels)}

    for split_name, rows in splits.items():
        write_jsonl(output_root / f"{split_name}.jsonl", rows)
    write_csv(output_root / "all_examples.csv", examples)

    with (output_root / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2, sort_keys=True)

    manifest = {
        "schema": "cmo_combat_id_dataset_v1",
        "input_files": [str(path) for path in input_paths],
        "label_fields_priority": list(label_fields),
        "num_examples": len(examples),
        "num_labels": len(label_map),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "files": {
            "train": "train.jsonl",
            "val": "val.jsonl",
            "test": "test.jsonl",
            "csv": "all_examples.csv",
            "label_map": "label_map.json",
        },
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="CMO JSONL export path(s)")
    parser.add_argument("--output-root", required=True, help="Directory for combat-ID dataset outputs")
    parser.add_argument("--label-field", action="append", dest="label_fields", help="Label field priority; repeatable")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--drop-unknown-labels", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    input_paths = [Path(path) for path in args.input]
    label_fields = tuple(args.label_fields or DEFAULT_LABEL_PRIORITY)

    examples = [
        build_example(record, source_path=source_path, line_number=line_number, label_fields=label_fields)
        for source_path, line_number, record in load_records(input_paths)
    ]
    if args.drop_unknown_labels:
        examples = [example for example in examples if example["label"] != UNKNOWN_TOKEN]
    if not examples:
        raise ValueError("No examples found after loading CMO exports")

    manifest = write_dataset(
        examples,
        output_root=Path(args.output_root),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        input_paths=input_paths,
        label_fields=label_fields,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
