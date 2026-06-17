import json
from pathlib import Path

import extract_cmo_combat_id_dataset as extractor


def test_build_cmo_combat_id_dataset(tmp_path: Path):
    export_path = tmp_path / "snapshot.jsonl"
    records = [
        {
            "record_kind": "contact",
            "side": "Blue",
            "guid": "TRACK-A",
            "name": "Bogey 1",
            "type": "Aircraft",
            "posture": "Hostile",
            "latitude": "12.5",
            "longitude": "-45.0",
            "speed_kts": "420",
        },
        {
            "record_kind": "unit",
            "side": "Blue",
            "guid": "UNIT-B",
            "name": "Blue Ship",
            "type": "Ship",
            "actual_side": "Blue",
            "heading_deg": 180,
        },
    ]
    export_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    manifest = extractor.main(
        [
            "--input",
            str(export_path),
            "--output-root",
            str(tmp_path / "dataset"),
            "--train-fraction",
            "0.5",
            "--val-fraction",
            "0.25",
            "--seed",
            "7",
        ]
    )

    dataset_root = tmp_path / "dataset"
    assert manifest["num_examples"] == 2
    assert manifest["num_labels"] == 2
    assert (dataset_root / "train.jsonl").exists()
    assert (dataset_root / "val.jsonl").exists()
    assert (dataset_root / "test.jsonl").exists()
    assert json.loads((dataset_root / "label_map.json").read_text(encoding="utf-8")) == {
        "blue": 0,
        "hostile": 1,
    }

    csv_text = (dataset_root / "all_examples.csv").read_text(encoding="utf-8")
    assert "TRACK-A" in csv_text
    assert "has_kinematics" in csv_text


def test_drop_unknown_labels(tmp_path: Path):
    export_path = tmp_path / "snapshot.jsonl"
    export_path.write_text(
        "\n".join(
            [
                json.dumps({"guid": "UNKNOWN", "name": "No label"}),
                json.dumps({"guid": "KNOWN", "posture": "Neutral"}),
            ]
        ),
        encoding="utf-8",
    )

    manifest = extractor.main(
        [
            "--input",
            str(export_path),
            "--output-root",
            str(tmp_path / "dataset"),
            "--drop-unknown-labels",
        ]
    )

    assert manifest["num_examples"] == 1
    assert manifest["splits"]["test"] == 1
