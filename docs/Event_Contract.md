# Common event contract v1

Draft,

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


