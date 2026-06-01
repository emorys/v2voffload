# highD SUMO Segment Configs

These configs separate highD replay from the Beijing G6 demo map.

Use them in two ways:

```powershell
# Replay highD trajectories for a selected highD location.
.\.venv\Scripts\python.exe vanet_offload.py --mode highd --config configs\highway\highd_segments\highd_location_2.json

# Build and run the matching SUMO-style Autobahn segment.
.\.venv\Scripts\python.exe scripts\build_highway_scenario.py --config configs\highway\highd_segments\highd_location_2.json
.\.venv\Scripts\python.exe vanet_offload.py --mode sumo --config configs\highway\highd_segments\highd_location_2.json
```

Important:

- highD public CSV metadata gives `locationId`, but not a direct GPS bbox.
- The bboxes here are editable OSM Autobahn corridor approximations for SUMO-side consistency.
- If you have exact highD georeference information, replace the `map.bbox` values and keep `dataset.location_id`.
- `dataset.recording_id: 0` means: scan `data/highD/*_recordingMeta.csv` and use the first recording whose `locationId` matches `dataset.location_id`.
