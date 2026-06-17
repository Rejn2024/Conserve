-- cmo_scenario_export.lua
-- Command: Modern Operations (CMO) Lua export helper for combat-ID datasets.
--
-- Usage from CMO:
--   1. Open the scenario at the time step you want to sample.
--   2. Open the Lua console and run: dofile('C:/path/to/cmo_scenario_export.lua')
--   3. Optionally set CMO_COMBAT_ID_EXPORT before running this file:
--        CMO_COMBAT_ID_EXPORT = 'C:/path/to/export.jsonl'
--   4. Advance scenario time and run again with another output path for more snapshots.
--
-- The script writes one JSON object per detected/contact-or-unit track. The companion
-- Python script extract_cmo_combat_id_dataset.py converts those JSONL snapshots into
-- train/validation/test splits for the combat ID training pathway.

local output_path = CMO_COMBAT_ID_EXPORT or 'cmo_combat_id_export.jsonl'

local function json_escape(value)
  if value == nil then return '' end
  value = tostring(value)
  value = value:gsub('\\', '\\\\')
  value = value:gsub('"', '\\"')
  value = value:gsub('\n', '\\n')
  value = value:gsub('\r', '\\r')
  value = value:gsub('\t', '\\t')
  return value
end

local function json_value(value)
  if value == nil then
    return 'null'
  end
  local value_type = type(value)
  if value_type == 'number' then
    return tostring(value)
  end
  if value_type == 'boolean' then
    return value and 'true' or 'false'
  end
  return '"' .. json_escape(value) .. '"'
end

local function field(object, ...)
  for _, key in ipairs({...}) do
    local ok, value = pcall(function() return object[key] end)
    if ok and value ~= nil and value ~= '' then
      return value
    end
  end
  return nil
end

local function emit_record(handle, kind, side_name, object)
  local record = {
    export_schema = 'cmo_combat_id_v1',
    source = 'cmo_lua',
    record_kind = kind,
    side = side_name,
    guid = field(object, 'guid', 'Guid', 'GUID'),
    name = field(object, 'name', 'Name'),
    type = field(object, 'type', 'Type'),
    subtype = field(object, 'subtype', 'SubType', 'unitType'),
    class_name = field(object, 'class', 'Class', 'classname'),
    dbid = field(object, 'dbid', 'DBID'),
    latitude = field(object, 'latitude', 'Latitude', 'lat'),
    longitude = field(object, 'longitude', 'Longitude', 'lon'),
    altitude_m = field(object, 'altitude', 'Altitude', 'alt'),
    speed_kts = field(object, 'speed', 'Speed'),
    heading_deg = field(object, 'heading', 'Heading'),
    course_deg = field(object, 'course', 'Course'),
    posture = field(object, 'posture', 'Posture'),
    actual_side = field(object, 'actualside', 'ActualSide', 'actual_side'),
    identification_status = field(object, 'identificationstatus', 'IdentificationStatus', 'identstatus'),
    detected_by = field(object, 'detectedby', 'DetectedBy'),
  }

  local parts = {}
  for key, value in pairs(record) do
    table.insert(parts, '"' .. key .. '":' .. json_value(value))
  end
  handle:write('{' .. table.concat(parts, ',') .. '}\n')
end

local function export_side(handle, side)
  local side_name = field(side, 'name', 'Name') or tostring(side)

  local ok_units, units = pcall(function() return ScenEdit_GetUnits({side=side_name}) end)
  if ok_units and units ~= nil then
    for _, unit in pairs(units) do
      emit_record(handle, 'unit', side_name, unit)
    end
  end

  -- Contacts are the preferred combat-ID observations: they preserve the observer's
  -- current identification state instead of only the ground-truth unit attributes.
  local ok_contacts, contacts = pcall(function() return ScenEdit_GetContacts(side_name) end)
  if ok_contacts and contacts ~= nil then
    for _, contact in pairs(contacts) do
      emit_record(handle, 'contact', side_name, contact)
    end
  end
end

local handle, err = io.open(output_path, 'a')
if handle == nil then
  error('Unable to open CMO export path: ' .. tostring(err))
end

local ok_sides, sides = pcall(function() return VP_GetSides() end)
if ok_sides and sides ~= nil then
  for _, side in pairs(sides) do
    export_side(handle, side)
  end
else
  error('Unable to enumerate CMO sides with VP_GetSides()')
end

handle:close()
print('CMO combat-ID export appended to ' .. output_path)
