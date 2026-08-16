# Tracking hardening (Phases 8-13)

`src/tracking/tracker.py` (`IoUTracker`) was rebuilt from IoU-only
matching into a stable multi-object tracker.  Tests in
`tests/test_tracking_stability.py` and `tests/test_label_stability.py`
prove the behaviour without hardware.

## Phase 8 — association

Old: greedy match by IoU alone.  A detection whose box overlapped
another track's box could steal that track's ID even for a different
class.

New **affinity score** combines three signals:

```
affinity = w_iou*IoU + w_center*centre-proximity + w_size*size-similarity
```

- `w_iou=0.6`, `w_center=0.25`, `w_size=0.15` (weights configurable).
- Centre proximity = 1 − (centre distance / track-box diagonal),
  clamped ≥ 0 — keeps IDs during fast movement between detections.
- Size similarity = min/max ratio on width × height.

## Phase 10 — class consistency

When `class_consistent=true` (default), a match whose class differs from
the track's is multiplied by **0.05** — effectively rejected.  A chair
box sitting exactly on a person never takes the person's identity; it
creates a new track instead.  Proved by
`test_different_class_never_steals_identity`.

## Phase 11 — box smoothing (EMA)

On every match, the displayed box is an exponential moving average:

```
box = alpha*raw + (1-alpha)*box     (alpha=0.4 default; 0 = raw)
```

This removes per-frame jitter.  The last raw detection is kept in
`raw_box` for diagnostics (the desktop app draws it in cyan in debug
mode).  Proved by `test_smoothed_box_oscillates_less_than_raw`.

## Phase 12 — label stability (temporal voting)

Each track keeps a rolling window of recent labels (default 5).  The
displayed label only switches when a label holds a clear majority
(`label_vote_ratio` 0.6).  A single spurious "chair" frame cannot flip
"person"; four consistent frames do.  Proved by
`tests/test_label_stability.py`.

## Phase 13 — confidence smoothing (EMA)

`confidence` is smoothed the same way (`conf_smoothing` 0.5 default) so
a single weak detection doesn't make the spoken/highlighted confidence
collapse.  `raw_confidence` is kept for diagnostics.

## Phase 9 — per-track fields

`TrackedObject` exposes: `track_id`, `label`, `confidence` (smoothed),
`box` (smoothed), `center`, `width`, `height`, `area`, `age`, `missed`
(consecutive unmatched frames), `first_seen`, `last_seen`, `timestamp`,
`direction` / `distance_m` (cached by downstream geometry), plus
`raw_box` / `raw_confidence`.

## Persistence

Tracks survive short detection gaps: `missed` counts unmatched frames
and the track is only dropped after `max_missed` (default 8).  When an
object is briefly not detected it keeps its identity (it reappears with
the same ID); a truly new object gets a new ID.  Proved by
`test_id_survives_brief_gap` / `test_lost_then_new_object_gets_new_id`.