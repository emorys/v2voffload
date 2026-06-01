from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List local highD recordings and their locationId values.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "highD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"highD root does not exist: {root}")

    print("recording_id,location_id,frame_rate,speed_limit,num_vehicles,duration")
    for path in sorted(root.glob("*_recordingMeta.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            row = next(csv.DictReader(file), None)
        if row is None:
            continue
        recording_id = int(path.name.split("_", 1)[0])
        print(
            f"{recording_id:02d},"
            f"{row.get('locationId', '')},"
            f"{row.get('frameRate', '')},"
            f"{row.get('speedLimit', '')},"
            f"{row.get('numVehicles', '')},"
            f"{row.get('duration', '')}"
        )


if __name__ == "__main__":
    main()
