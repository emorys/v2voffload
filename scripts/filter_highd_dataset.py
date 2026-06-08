from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACK_COLUMNS = [
    "frame",
    "id",
    "x",
    "y",
    "width",
    "height",
    "xVelocity",
    "yVelocity",
    "xAcceleration",
    "yAcceleration",
    "laneId",
]
TRACK_META_COLUMNS = ["id", "width", "height", "drivingDirection"]
RECORDING_META_COLUMNS = ["id", "frameRate", "locationId", "speedLimit", "duration", "numVehicles"]


@dataclass(frozen=True)
class FilterSummary:
    recording_id: int
    location_id: int | None
    kept_track_rows: int
    kept_track_meta_rows: int
    output_dir: Path


def filter_highd_dataset(
    *,
    input_root: str | Path,
    output_root: str | Path,
    recording_ids: list[int] | None = None,
    location_ids: list[int] | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    frame_step: int = 1,
    driving_direction: int | None = None,
    lane_ids: list[int] | None = None,
) -> list[FilterSummary]:
    input_path = Path(input_root).resolve()
    output_path = Path(output_root).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    summaries: list[FilterSummary] = []
    for recording_meta_path in sorted(input_path.glob("*_recordingMeta.csv")):
        recording_id = int(recording_meta_path.name.split("_", 1)[0])
        if recording_ids is not None and recording_id not in recording_ids:
            continue

        recording_meta = _read_single_row(recording_meta_path)
        location_id = _optional_int(recording_meta.get("locationId"))
        if location_ids is not None and location_id not in location_ids:
            continue

        prefix = f"{recording_id:02d}"
        tracks_path = input_path / f"{prefix}_tracks.csv"
        tracks_meta_path = input_path / f"{prefix}_tracksMeta.csv"
        if not tracks_path.exists() or not tracks_meta_path.exists():
            continue

        tracks_meta = _read_meta_by_vehicle_id(tracks_meta_path)
        kept_vehicle_ids = _filter_tracks(
            source=tracks_path,
            target=output_path / f"{prefix}_tracks.csv",
            tracks_meta=tracks_meta,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
            driving_direction=driving_direction,
            lane_ids=lane_ids,
        )
        kept_meta_rows = _filter_tracks_meta(
            source=tracks_meta_path,
            target=output_path / f"{prefix}_tracksMeta.csv",
            kept_vehicle_ids=kept_vehicle_ids,
        )
        _write_selected_row(output_path / f"{prefix}_recordingMeta.csv", RECORDING_META_COLUMNS, recording_meta)
        summaries.append(
            FilterSummary(
                recording_id=recording_id,
                location_id=location_id,
                kept_track_rows=kept_vehicle_ids.row_count,
                kept_track_meta_rows=kept_meta_rows,
                output_dir=output_path,
            )
        )

    return summaries


@dataclass(frozen=True)
class _KeptVehicles:
    ids: set[int]
    row_count: int


def _filter_tracks(
    *,
    source: Path,
    target: Path,
    tracks_meta: dict[int, dict[str, str]],
    start_frame: int | None,
    end_frame: int | None,
    frame_step: int,
    driving_direction: int | None,
    lane_ids: list[int] | None,
) -> _KeptVehicles:
    kept_ids: set[int] = set()
    kept_count = 0
    lane_set = set(lane_ids) if lane_ids is not None else None
    step = max(1, int(frame_step))

    with source.open("r", encoding="utf-8-sig", newline="") as input_file, target.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        reader = csv.DictReader(input_file)
        writer = csv.DictWriter(output_file, fieldnames=TRACK_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            frame = int(row["frame"])
            if start_frame is not None and frame < start_frame:
                continue
            if end_frame is not None and frame > end_frame:
                continue
            if start_frame is not None and (frame - start_frame) % step != 0:
                continue

            vehicle_id = int(row["id"])
            meta = tracks_meta.get(vehicle_id, {})
            direction = _optional_int(meta.get("drivingDirection") or row.get("drivingDirection"))
            if driving_direction is not None and direction != driving_direction:
                continue

            lane_id = _optional_int(row.get("laneId"))
            if lane_set is not None and lane_id not in lane_set:
                continue

            writer.writerow({column: row.get(column, "") for column in TRACK_COLUMNS})
            kept_ids.add(vehicle_id)
            kept_count += 1

    return _KeptVehicles(ids=kept_ids, row_count=kept_count)


def _filter_tracks_meta(*, source: Path, target: Path, kept_vehicle_ids: _KeptVehicles) -> int:
    kept_count = 0
    with source.open("r", encoding="utf-8-sig", newline="") as input_file, target.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        reader = csv.DictReader(input_file)
        writer = csv.DictWriter(output_file, fieldnames=TRACK_META_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            vehicle_id = int(row["id"])
            if vehicle_id not in kept_vehicle_ids.ids:
                continue
            writer.writerow({column: row.get(column, "") for column in TRACK_META_COLUMNS})
            kept_count += 1
    return kept_count


def _read_meta_by_vehicle_id(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {int(row["id"]): row for row in csv.DictReader(file)}


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return next(csv.DictReader(file), {})


def _write_selected_row(path: Path, columns: list[str], row: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(float(str(value)))


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter highD CSV files down to the fields used by VANET replay.")
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "highD")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "highD_filtered")
    parser.add_argument("--recording-ids", default=None, help="Comma-separated recording ids, e.g. 1,2,3.")
    parser.add_argument("--location-ids", default=None, help="Comma-separated highD location ids, e.g. 1,2.")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--driving-direction", type=int, default=None)
    parser.add_argument("--lane-ids", default=None, help="Comma-separated lane ids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = filter_highd_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        recording_ids=_parse_int_list(args.recording_ids),
        location_ids=_parse_int_list(args.location_ids),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_step=args.frame_step,
        driving_direction=args.driving_direction,
        lane_ids=_parse_int_list(args.lane_ids),
    )
    print("recording_id,location_id,kept_track_rows,kept_track_meta_rows,output_root")
    for summary in summaries:
        location = "" if summary.location_id is None else summary.location_id
        print(
            f"{summary.recording_id:02d},{location},{summary.kept_track_rows},"
            f"{summary.kept_track_meta_rows},{summary.output_dir}"
        )


if __name__ == "__main__":
    main()
