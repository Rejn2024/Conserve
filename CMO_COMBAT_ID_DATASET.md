# CMO combat-ID data extraction

This repository includes a two-stage extractor for building combat-ID training
examples from Command: Modern Operations (CMO) scenarios:

1. `cmo_scenario_export.lua` runs inside CMO and appends scenario contacts and
   units to JSONL snapshots.
2. `extract_cmo_combat_id_dataset.py` runs outside CMO and converts one or more
   snapshots into `train.jsonl`, `val.jsonl`, `test.jsonl`, `label_map.json`,
   `manifest.json`, and `all_examples.csv`.

## Detailed Lua exporter workflow

### 1. Pick a snapshot folder outside the repository

Create a writable folder for CMO exports, for example `C:/cmo_exports`. Keep
these raw scenario snapshots outside Git because large scenarios and repeated
time steps can produce many records.

### 2. Copy or reference the Lua script from CMO

CMO executes Lua through its built-in Lua console or event/action system. The
simplest manual workflow is to reference this repository script directly with
`dofile`:

```lua
dofile('C:/path/to/Conserve/cmo_scenario_export.lua')
```

Use forward slashes in Windows paths, or escape backslashes (`C:\\path\\file.lua`).

### 3. Export one scenario time step

Open the scenario, pause it at the time step you want to sample, open the Lua
console, set `CMO_COMBAT_ID_EXPORT`, and run the exporter:

```lua
CMO_COMBAT_ID_EXPORT = 'C:/cmo_exports/scenario_001_t0000.jsonl'
dofile('C:/path/to/Conserve/cmo_scenario_export.lua')
```

The exporter appends to the file named by `CMO_COMBAT_ID_EXPORT`. If the file
does not exist, CMO creates it. If it already exists, new records are added to
the end, so delete the file first when you want a clean re-export.

### 4. Capture multiple time steps

For temporal coverage, advance the scenario clock, change the output filename,
and rerun the same two Lua lines:

```lua
CMO_COMBAT_ID_EXPORT = 'C:/cmo_exports/scenario_001_t0010.jsonl'
dofile('C:/path/to/Conserve/cmo_scenario_export.lua')
```

Recommended naming is `scenario_<id>_t<minutes-or-seconds>.jsonl` so each file
maps cleanly back to a scenario and time offset. Capturing separate files per
time step makes it easier to exclude bad snapshots, balance scenarios, and audit
training examples.

### 5. What the Lua exporter writes

For every side, the script attempts to export both:

- `contact` records from `ScenEdit_GetContacts(...)`, which are the preferred
  combat-ID observations because they represent what an observing side currently
  knows.
- `unit` records from `ScenEdit_GetUnits(...)`, which are useful for ground
  truth, debugging, and label QA.

Each output line is one JSON object. Important fields include `record_kind`,
`side`, `guid`, `name`, `type`, `subtype`, `class_name`, `dbid`, `latitude`,
`longitude`, `altitude_m`, `speed_kts`, `heading_deg`, `course_deg`, `posture`,
`actual_side`, `identification_status`, and `detected_by`.

### 6. Verify the export before conversion

After running the Lua script, confirm that the CMO Lua console prints a message
like:

```text
CMO combat-ID export appended to C:/cmo_exports/scenario_001_t0000.jsonl
```

Open the JSONL file in a text editor and verify that it contains one JSON object
per line. Empty files usually mean the scenario has no sides/contacts visible at
that time step, the output folder is not writable, or the Lua console ran from a
different scenario state than expected.

### 7. Optional: automate snapshot collection in CMO

For larger data-generation runs, attach the same Lua command to a CMO event or
run it at regular manual pause points. Use a unique output path for each time
step or scenario branch. If you intentionally append multiple time steps to the
same file, keep an external log of which scenario times were appended because
the exporter only records the track/contact attributes it receives from CMO.

## Convert snapshots into combat-ID splits

Run the Python converter against one or more CMO JSONL exports:

```bash
python extract_cmo_combat_id_dataset.py \
  --input C:/cmo_exports/scenario_001_t0000.jsonl C:/cmo_exports/scenario_001_t0010.jsonl \
  --output-root C:/combat_id_dataset/scenario_001 \
  --drop-unknown-labels
```

By default, labels are selected from CMO fields in this priority order:
`posture`, `actual_side`, then `side`. Override that order when your combat-ID
pathway expects a different target, for example:

```bash
python extract_cmo_combat_id_dataset.py \
  --input C:/cmo_exports/*.jsonl \
  --output-root C:/combat_id_dataset/by_side \
  --label-field actual_side \
  --label-field posture
```

Each JSONL example contains:

- `label`: the combat-ID class selected from the configured label fields.
- `features`: normalized CMO attributes such as observer side, track type,
  class, DBID, latitude, longitude, altitude, speed, heading, course, and
  identification status.
- `raw`: the original exported CMO record for traceability and later feature
  expansion.

Use `label_map.json` to map class strings to integer model targets, and use the
split JSONL files as the direct inputs for the combat-ID training pathway.
