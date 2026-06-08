from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vanetsim.config import (  # noqa: E402
    DatasetConfig,
    IncentiveConfig,
    MobilityConfig,
    OffloadingConfig,
    SimulationConfig,
    load_scenario_config,
)
from vanetsim.datasets import HighDLoader  # noqa: E402
from vanetsim.state import VehicleStateManager  # noqa: E402


SUMMARY_COLUMNS = [
    "recording_id",
    "location_id",
    "window_index",
    "window_start",
    "window_end",
    "frame_start",
    "frame_end",
    "frame_count",
    "avg_vehicle_count",
    "min_vehicle_count",
    "max_vehicle_count",
    "avg_speed",
    "min_speed",
    "slow_frame_ratio",
    "avg_slow_count",
    "avg_helper_count",
    "avg_neighbor_count",
    "triggerable_frame_ratio",
]


def summarize_recording(
    *,
    dataset: DatasetConfig,
    mobility: MobilityConfig,
    communication_max_distance: float,
    offloading: OffloadingConfig,
    incentive: IncentiveConfig,
    simulation: SimulationConfig,
    workspace_root: str | Path,
) -> dict[str, float | int | str]:
    loader = HighDLoader(dataset, workspace_root=workspace_root)
    frames = loader.load_frames()
    return _summarize_frames(
        loader=loader,
        frames=frames,
        dataset=dataset,
        mobility=mobility,
        communication_max_distance=communication_max_distance,
        offloading=offloading,
        incentive=incentive,
        simulation=simulation,
        window_index=0,
        window_start=frames[0].frame if frames else "",
        window_end=frames[-1].frame if frames else "",
    )


def summarize_recording_windows(
    *,
    dataset: DatasetConfig,
    mobility: MobilityConfig,
    communication_max_distance: float,
    offloading: OffloadingConfig,
    incentive: IncentiveConfig,
    simulation: SimulationConfig,
    workspace_root: str | Path,
    window_size_frames: int,
    window_stride_frames: int,
) -> list[dict[str, float | int | str]]:
    loader = HighDLoader(dataset, workspace_root=workspace_root)
    frames = loader.load_frames()
    if not frames:
        return []

    rows = []
    size = max(1, int(window_size_frames))
    stride = max(1, int(window_stride_frames))
    for window_index, start in enumerate(range(0, len(frames), stride)):
        window_frames = frames[start : start + size]
        if len(window_frames) < size:
            break
        rows.append(
            _summarize_frames(
                loader=loader,
                frames=window_frames,
                dataset=dataset,
                mobility=mobility,
                communication_max_distance=communication_max_distance,
                offloading=offloading,
                incentive=incentive,
                simulation=simulation,
                window_index=window_index,
                window_start=window_frames[0].frame,
                window_end=window_frames[-1].frame,
            )
        )
    return rows


def _summarize_frames(
    *,
    loader: HighDLoader,
    frames,
    dataset: DatasetConfig,
    mobility: MobilityConfig,
    communication_max_distance: float,
    offloading: OffloadingConfig,
    incentive: IncentiveConfig,
    simulation: SimulationConfig,
    window_index: int,
    window_start: int | str,
    window_end: int | str,
) -> dict[str, float | int | str]:
    state_manager = VehicleStateManager(
        mobility=mobility,
        offloading=offloading,
        incentive=incentive,
        simulation=simulation,
    )
    recording_meta = loader._read_recording_meta()
    location_id = int(float(recording_meta.get("locationId", -1)))

    vehicle_counts: list[int] = []
    speeds: list[float] = []
    min_speeds: list[float] = []
    slow_counts: list[int] = []
    helper_counts: list[int] = []
    neighbor_counts: list[float] = []
    triggerable = 0

    for frame in frames:
        vehicles = state_manager.build_from_snapshots(frame.snapshots, max_vehicles=dataset.max_vehicles)
        vehicle_counts.append(len(vehicles))
        if not vehicles:
            min_speeds.append(0.0)
            slow_counts.append(0)
            helper_counts.append(0)
            neighbor_counts.append(0.0)
            continue

        frame_speeds = [float(vehicle.speed) for vehicle in vehicles]
        speeds.extend(frame_speeds)
        min_speeds.append(min(frame_speeds))
        slow_indices = [index for index, vehicle in enumerate(vehicles) if vehicle.role == "slow"]
        helper_indices = [index for index, vehicle in enumerate(vehicles) if vehicle.role == "helper"]
        slow_counts.append(len(slow_indices))
        helper_counts.append(len(helper_indices))
        neighbor_counts.append(_avg_neighbor_count(vehicles, communication_max_distance))
        if _has_triggerable_slow_vehicle(vehicles, slow_indices, helper_indices, communication_max_distance):
            triggerable += 1

    frame_count = len(frames)
    frame_start = frames[0].frame if frames else ""
    frame_end = frames[-1].frame if frames else ""
    return {
        "recording_id": loader.recording_id,
        "location_id": location_id,
        "window_index": window_index,
        "window_start": window_start,
        "window_end": window_end,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": frame_count,
        "avg_vehicle_count": _mean(vehicle_counts),
        "min_vehicle_count": min(vehicle_counts) if vehicle_counts else 0,
        "max_vehicle_count": max(vehicle_counts) if vehicle_counts else 0,
        "avg_speed": _mean(speeds),
        "min_speed": min(min_speeds) if min_speeds else 0.0,
        "slow_frame_ratio": _ratio(sum(1 for count in slow_counts if count > 0), frame_count),
        "avg_slow_count": _mean(slow_counts),
        "avg_helper_count": _mean(helper_counts),
        "avg_neighbor_count": _mean(neighbor_counts),
        "triggerable_frame_ratio": _ratio(triggerable, frame_count),
    }


def summarize_dataset(
    *,
    config_path: str | Path,
    workspace_root: str | Path,
    recording_ids: list[int] | None = None,
    location_ids: list[int] | None = None,
    start_frame: int | None = None,
    end_frame: int | None | str = "config",
    frame_step: int | None = None,
    max_vehicles: int | None = None,
    window_size_frames: int | None = None,
    window_stride_frames: int | None = None,
) -> list[dict[str, float | int | str]]:
    scenario = load_scenario_config(config_path)
    root = _resolve_root(scenario.dataset.root, workspace_root)
    recording_ids = recording_ids or _recording_ids_for_locations(root, location_ids)
    rows = []
    for recording_id in recording_ids:
        recording_location = _recording_location(root, recording_id)
        if location_ids is not None and recording_location not in location_ids:
            continue
        dataset = replace(scenario.dataset, recording_id=recording_id, location_id=recording_location)
        if start_frame is not None:
            dataset.start_frame = start_frame
        if end_frame != "config":
            dataset.end_frame = end_frame if isinstance(end_frame, int) else None
        if frame_step is not None:
            dataset.frame_step = frame_step
        if max_vehicles is not None:
            dataset.max_vehicles = max_vehicles
        if window_size_frames is not None:
            rows.extend(
                summarize_recording_windows(
                    dataset=dataset,
                    mobility=scenario.mobility,
                    communication_max_distance=scenario.communication.max_distance,
                    offloading=scenario.offloading,
                    incentive=scenario.incentive,
                    simulation=scenario.simulation,
                    workspace_root=workspace_root,
                    window_size_frames=window_size_frames,
                    window_stride_frames=window_stride_frames or window_size_frames,
                )
            )
        else:
            rows.append(
                summarize_recording(
                    dataset=dataset,
                    mobility=scenario.mobility,
                    communication_max_distance=scenario.communication.max_distance,
                    offloading=scenario.offloading,
                    incentive=scenario.incentive,
                    simulation=scenario.simulation,
                    workspace_root=workspace_root,
                )
            )
    return rows


def _has_triggerable_slow_vehicle(vehicles, slow_indices: list[int], helper_indices: list[int], max_distance: float) -> bool:
    for slow_index in slow_indices:
        slow = vehicles[slow_index]
        for helper_index in helper_indices:
            helper = vehicles[helper_index]
            if _distance(slow, helper) <= max_distance:
                return True
    return False


def _avg_neighbor_count(vehicles, max_distance: float) -> float:
    if not vehicles:
        return 0.0
    counts = []
    for index, vehicle in enumerate(vehicles):
        count = 0
        for other_index, other in enumerate(vehicles):
            if index == other_index:
                continue
            if _distance(vehicle, other) <= max_distance:
                count += 1
        counts.append(count)
    return float(np.mean(counts)) if counts else 0.0


def _distance(first, second) -> float:
    return float(math.hypot(first.position[0] - second.position[0], first.position[1] - second.position[1]))


def _mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _resolve_root(root: str, workspace_root: str | Path) -> Path:
    path = Path(root)
    if not path.is_absolute():
        path = Path(workspace_root) / path
    return path.resolve()


def _recording_ids_for_locations(root: Path, location_ids: list[int] | None) -> list[int]:
    ids = []
    for path in sorted(root.glob("*_recordingMeta.csv")):
        recording_id = int(path.name.split("_", 1)[0])
        location_id = _recording_location(root, recording_id)
        if location_ids is None or location_id in location_ids:
            ids.append(recording_id)
    return ids


def _recording_location(root: Path, recording_id: int) -> int:
    path = root / f"{recording_id:02d}_recordingMeta.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        row = next(csv.DictReader(file), {})
    return int(float(row.get("locationId", -1)))


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize highD recordings for VANET experiment scenario selection.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "highway" / "highd_segments" / "highd_location_2.json")
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--recording-ids", default=None, help="Comma-separated recording ids, e.g. 1,2,3.")
    parser.add_argument("--location-ids", default=None, help="Comma-separated highD location ids, e.g. 1,2,5.")
    parser.add_argument("--start-frame", type=int, default=None, help="Override dataset.start_frame.")
    parser.add_argument("--end-frame", type=int, default=None, help="Override dataset.end_frame.")
    parser.add_argument("--full-recording", action="store_true", help="Ignore dataset.end_frame and scan to the end of each recording.")
    parser.add_argument("--frame-step", type=int, default=None, help="Override dataset.frame_step.")
    parser.add_argument("--max-vehicles", type=int, default=None, help="Override dataset.max_vehicles.")
    parser.add_argument("--window-size-frames", type=int, default=None, help="Number of sampled highD frames per summary window.")
    parser.add_argument("--window-stride-frames", type=int, default=None, help="Stride in sampled highD frames between windows.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "highd_recording_summary.csv")
    return parser.parse_args()


def write_rows(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = summarize_dataset(
        config_path=args.config,
        workspace_root=args.workspace_root,
        recording_ids=_parse_int_list(args.recording_ids),
        location_ids=_parse_int_list(args.location_ids),
        start_frame=args.start_frame,
        end_frame=None if args.full_recording else (args.end_frame if args.end_frame is not None else "config"),
        frame_step=args.frame_step,
        max_vehicles=args.max_vehicles,
        window_size_frames=args.window_size_frames,
        window_stride_frames=args.window_stride_frames,
    )
    if not rows:
        raise RuntimeError("No highD recordings matched the selected filters.")
    write_rows(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
