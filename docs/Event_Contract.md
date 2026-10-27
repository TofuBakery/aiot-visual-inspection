# Common event contract v1

Draft, being agreed with the group this week. It applies to every block that writes events into the shared system: image inspection, the aiot-stream-cv video block, sensors and workflows. It merges the image model output with the video block contract (`aiot-stream-cv/docs/data-contract.md`, Phạm Anh Quân, 2026-10-22). **Needs review by the whole group.**

## Fields

| Field | Required | Meaning |
|---|---|---|
| `event_id` | recommended | Unique ID set by the source block. Re-sending the same ID does not create a new record (HTTP 200, `duplicate: true`) |
| `source_id` | yes | Camera, sensor or service that produced the event |
| `event_type` | yes | See the event types below |
| `occurred_at` | recommended | When the measurement or frame was taken (ISO-8601 with time zone). Cameras without a reliable clock use the frame capture time |
| `received_at` | set by the system | When the shared system received the event |
| `status` | yes (default `ok`) | `ok` or `error`. `error` means processing failed or the source is unusable; it **never** means "normal" |
| `result` | yes | Short outcome: `normal`, `anomaly`, `alert`, `cleared`, `offline`, `online`, `ended`, or an error code |
| `details` | no | Object specific to the event type (score, zone_id, track_id, value…) |
| `evidence` | no | `{image_url, heatmap_url, clip_url}`, absolute or relative to the source block |
| `model_version` | when a model is used | e.g. `patch_knn-pcb1-v2`, `yolov8n-coco+bytetrack` |
| `error` | when `status=error` | `{code, message}` |

## Event types

| `event_type` | Block | `result` | `details` |
|---|---|---|---|
| `image_inspection` | Image model | `normal` / `anomaly` | anomaly_score, threshold |
| `zone_intrusion` | Video block | `alert` | zone_id, track_id, class, confidence, dwell_s, bbox |
| `sensor_alert` | Sensor flow | `alert` / `cleared` | value, unit, measured_at, rule |
| `data_rejected` | Sensor flow | error code (`invalid_unit`, `missing_value`, `stale_reading`, `invalid_payload`) | raw payload |
| `source_offline` | Any source | `offline` (`status=error`) | state, last_frame_age_s or last_seen, reconnects, last_error |
| `source_recovered` | Any source | `online` | same as above |
| `source_ended` | Video block (file) | `ended` | same as above |

`source_offline` / `source_recovered` follow the video block naming.

## Mapping from the video block

The bridge belongs to the workflow part. Every 2 s it calls `GET :8080/events?limit=50` on the video block, converts the events to this format and sends them to the shared system.

| aiot-stream-cv | Common contract |
|---|---|
| `event_id`, `source_id`, `event_type`, `occurred_at` | unchanged |
| `created_at` | `origin.created_at` |
| `result` (object) | `details` |
| (by type) | `result` = `alert` / `offline` / `online` / `ended` |
| `evidence_path`, `evidence_clip` | `evidence.image_url`, `evidence.clip_url` (prefixed with `http://<cv-host>:8080/`) |

When the video block does not respond, the bridge sends **one** `source_offline` with `source_id=cv_api`, and a `source_recovered` when it is back.

## Suggestions for the video block (to confirm with Quân)

1. Add `frame_index` or the video timestamp to `result`, so accuracy on labelled video can be measured without manual time alignment.
2. Use H.264 clips if the web needs to play them inline.
3. Add `status` to each event; the bridge currently infers `status=error` for `source_offline`.
