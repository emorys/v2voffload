# Highway VANET Offloading with SUMO and OpenStreetMap

This workspace is now organized around a modular highway VANET pipeline:

- `map`: download a real OpenStreetMap highway corridor and convert it into SUMO assets
- `state`: maintain per-slot vehicle state, compute capacity, task load, willingness, and role
- `clustering`: identify slow vehicles and build dynamically expanding helper clusters
- `tasks`: generate sourced autonomous-driving task components
- `channel`: compute V2V transmission rates
- `delay`: compute local and V2V branch delays
- `safety`: map decision latency to safe speed and acceleration-limited speed updates
- `mobility`: vehicle dynamics and state aggregation
- `optimization`: Stage-I paid baseline optimization, Stage-II residual-resource boosting, and incentive coordination
- `simulation`: end-to-end orchestration for demo mode and SUMO mode
- `datasets`: replay external trajectory datasets such as highD

## Project Layout

- `src/vanetsim/config.py`: scenario and module configuration
- `src/vanetsim/domain/models.py`: vehicle state, task, cluster, stage output, and metrics data structures
- `src/vanetsim/map/osm_pipeline.py`: OSM download, SUMO net build, route build, config generation
- `src/vanetsim/datasets/highd.py`: highD CSV loader and trajectory replay adapter
- `src/vanetsim/state/vehicle_state.py`: `VehicleStateManager`
- `src/vanetsim/clustering/cluster_manager.py`: `ClusterManager`
- `src/vanetsim/tasks/catalog.py`: sourced perception, prediction, and planning task profiles
- `src/vanetsim/tasks/generator.py`: `TaskGenerator`
- `src/vanetsim/channel/model.py`: `ChannelModel`
- `src/vanetsim/delay/model.py`: `DelayModel`
- `src/vanetsim/safety/model.py`: `SafetyModel`
- `src/vanetsim/optimization/stage_i.py`: `StageIOptimizer`
- `src/vanetsim/optimization/stage_ii.py`: `StageIIOptimizer`
- `src/vanetsim/optimization/incentive_manager.py`: `IncentiveManager`
- `src/vanetsim/optimization/baselines.py`: comparison baselines and literature-adapted proxies
- `src/vanetsim/metrics/collector.py`: output metrics aggregation
- `src/vanetsim/simulation/simulator.py`: slot loop and state-speed feedback
- `src/vanetsim/simulation/orchestrator.py`: simulation entry point
- `configs/highway/beijing_g6_demo.json`: sample real-map highway scenario
- `configs/highway/highd_segments/`: Germany Autobahn SUMO segment configs for highD `locationId` switching
- `scripts/build_highway_scenario.py`: helper entry for map/scenario construction
- `docs/project_flow.md`: execution flow, Stage-II skip reasons, and parameter notes
- `paper/README.md`: paper-oriented experiment and artifact workspace

## Default Real-Map Scenario

The bundled sample configuration uses a real OpenStreetMap corridor around the Beijing-Tibet Expressway (G6):

- config: `configs/highway/beijing_g6_demo.json`
- default bbox: `116.3450,39.9900,116.3710,40.0260`
- generated artifacts: `scenarios/beijing_g6_demo/`

You can later replace the bbox or place query with your own highway segment.

## Build a Real Highway Scenario

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\build_highway_scenario.py --config d:\vscodeworkspace\configs\highway\beijing_g6_demo.json
```

This downloads OSM data, generates the SUMO network, generates traffic flows, and writes `highway.sumocfg`.

## Run

Demo mode:

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\vanet_offload.py --mode demo --config d:\vscodeworkspace\configs\highway\beijing_g6_demo.json
```

The demo places slow vehicles ahead of willing helper vehicles, so both Stage-I and Stage-II can be observed quickly.

SUMO mode using the generated real-map scenario:

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\vanet_offload.py --mode sumo --config d:\vscodeworkspace\configs\highway\beijing_g6_demo.json --steps 30
```

highD replay mode:

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\vanet_offload.py --mode highd --config d:\vscodeworkspace\configs\highway\highd_segments\highd_location_2.json --steps 50
```

Build or run the matching SUMO-side Autobahn segment:

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\build_highway_scenario.py --config d:\vscodeworkspace\configs\highway\highd_segments\highd_location_2.json
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\vanet_offload.py --mode sumo --config d:\vscodeworkspace\configs\highway\highd_segments\highd_location_2.json
```

Put the highD files under `data/highD/` by default:

```text
data/highD/01_tracks.csv
data/highD/01_tracksMeta.csv
data/highD/01_recordingMeta.csv
```

List local highD recordings before switching segments:

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\list_highd_recordings.py
```

For `configs/highway/highd_segments/highd_location_*.json`, `dataset.recording_id: 0` means the loader scans `data/highD` and selects the first recording whose `recordingMeta.locationId` matches `dataset.location_id`.

For paper reproducibility assets, use the `paper/` workspace:

- fixed configs: `paper/configs/`
- run outputs: `paper/results/`
- publish figures: `paper/figures/`
- final numeric tables: `paper/tables/`

Open the GUI:

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\vanet_offload.py --mode sumo --config d:\vscodeworkspace\configs\highway\beijing_g6_demo.json --gui
```

## Baseline Comparisons

Select a single method or baseline with `--policy`:

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\vanet_offload.py --mode demo --policy delay-greedy-v2v --steps 20
```

Implemented policies:

- `ours`
- `local-only`
- `equal-split-v2v`
- `delay-greedy-v2v`
- `stage-i-only`
- `no-baseline-maintenance`
- `no-incentive`
- `static-cluster`
- `fan-2023`
- `nan-2023`
- `kumar-2023`

Batch-run the comparison table and write a CSV:

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\run_baselines.py --steps 30 --quiet --output d:\vscodeworkspace\results\baseline_comparison.csv
```

Export the full plotting table for the paper experiments into one CSV:

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\export_experiment_results.py --steps 20 --output d:\vscodeworkspace\results\paper_experiment_results.csv
```

The exported file is a long table. Use `record_type=summary` for aggregate bar/line plots, `record_type=slot` for time-series plots, and `record_type=contrast` for direct `ours - stage-i-only` marginal gains.

Draw all current paper figures:

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\plot_experiment_results.py --input d:\vscodeworkspace\results\paper_experiment_results.csv
```

Draw individual figure groups:

```powershell
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\plot_method_comparison.py --input d:\vscodeworkspace\results\paper_experiment_results.csv
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\plot_density_figures.py --input d:\vscodeworkspace\results\paper_experiment_results.csv
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\plot_task_load_figures.py --input d:\vscodeworkspace\results\paper_experiment_results.csv
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\plot_time_series_figures.py --input d:\vscodeworkspace\results\paper_experiment_results.csv
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\plot_communication_figures.py --input d:\vscodeworkspace\results\paper_experiment_results.csv
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\plot_stage2_ablation_figures.py --input d:\vscodeworkspace\results\paper_experiment_results.csv
d:\vscodeworkspace\.venv\Scripts\python.exe d:\vscodeworkspace\scripts\plot_safety_incentive_figures.py --input d:\vscodeworkspace\results\paper_experiment_results.csv
```

The literature-named policies are same-resource-pool adapted proxies, intended for controlled experiments rather than full faithful reimplementations of external training pipelines.

## Notes

- If the SUMO scenario has not been built yet, `--mode sumo` will build it automatically from OSM first.
- Stage-I minimizes paid helper cost while satisfying complete task allocation, communication, compute capacity, and safety-delay constraints.
- Stage-II uses residual resources after baseline reservation, searches feasible speed targets under speed/safety limits, and chooses the lowest-cost allocation for the selected target while filtering helpers by individual rationality.
- Vehicle speed updates are acceleration-limited; optimized target speed is not applied as an instantaneous jump.
- Default tasks are no longer synthetic arrays. They are sourced real-scene autonomous-driving representatives: camera object detection on GPU, LiDAR point-cloud processing on accelerator resources, and Apollo-style EM motion planning on CPU. `task_load_scale` scales the sourced full workload before optimization so small examples remain tractable.

## highD Field Mapping

highD is a trajectory dataset, so the cleanest simulation path is replay mode: each highD frame becomes the vehicle state table for one simulation slot.

For SUMO-side map consistency, use the German Autobahn configs under `configs/highway/highd_segments/` rather than the Beijing G6 scenario. The public highD CSV metadata does not provide a direct GPS bbox, so the provided SUMO bboxes are editable OSM Autobahn corridor approximations. If you have exact highD georeference information, replace only `map.bbox` and keep `dataset.location_id`.

| Project field | highD source |
| --- | --- |
| `vehicle_id` | `id` |
| `position` | `x`, `y`, optionally shifted from bounding-box corner to center |
| `speed` | `sqrt(xVelocity^2 + yVelocity^2)` |
| `acceleration` | `sqrt(xAcceleration^2 + yAcceleration^2)` |
| `lane_id` | `laneId` |
| `progress` | `x` normalized by `drivingDirection` |
| `time_slot` | `frame / frameRate` |
| `compute_capacity` | deterministic value from config and vehicle ID |
| `task_load` | deterministic scaled load from the sourced autonomous-driving task catalog and vehicle ID |
| `willingness` | deterministic binary value from vehicle ID |
| `role` | `slow` if `speed < v_l`, otherwise `helper` if willing |

The dataset does not contain compute resources, task load, price, or willingness, so those remain model parameters. That keeps the physical traffic motion grounded in highD while still letting the VANET resource model be controlled experimentally.
