from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vanetsim.config import load_scenario_config  # noqa: E402


NUMERIC_FLOAT_FIELDS = {
    "avg_vehicle_count",
    "min_speed",
    "slow_frame_ratio",
    "avg_slow_count",
    "avg_helper_count",
    "avg_neighbor_count",
    "triggerable_frame_ratio",
}
NUMERIC_INT_FIELDS = {
    "recording_id",
    "location_id",
    "window_index",
    "window_start",
    "window_end",
    "frame_count",
    "min_vehicle_count",
    "max_vehicle_count",
}


def load_summary(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in NUMERIC_FLOAT_FIELDS:
            row[field] = float(row[field])
        for field in NUMERIC_INT_FIELDS:
            row[field] = int(float(row[field]))
    return rows


def select_scenarios(rows: list[dict]) -> list[tuple[str, dict]]:
    selected: list[tuple[str, dict]] = []

    def add(name: str, candidates: list[dict], key) -> None:
        if not candidates:
            return
        selected.append((name, sorted(candidates, key=key, reverse=True)[0]))

    triggerable = [row for row in rows if row["triggerable_frame_ratio"] >= 0.8]
    add(
        "highd_win_l1_severe_congestion",
        [row for row in triggerable if row["location_id"] == 1],
        lambda row: (row["avg_slow_count"], -row["avg_helper_count"], row["avg_vehicle_count"]),
    )
    add(
        "highd_win_l2_mild_bottleneck",
        [row for row in triggerable if row["location_id"] == 2],
        lambda row: (row["avg_helper_count"], row["avg_slow_count"]),
    )
    add(
        "highd_win_l4_sparse_bottleneck",
        [row for row in triggerable if row["location_id"] == 4],
        lambda row: (-row["avg_vehicle_count"], row["avg_slow_count"], row["avg_helper_count"]),
    )
    add(
        "highd_win_l5_dense_bottleneck",
        [row for row in triggerable if row["location_id"] == 5],
        lambda row: (row["avg_vehicle_count"], row["avg_helper_count"], row["avg_slow_count"]),
    )
    add(
        "highd_win_l6_generalization",
        [row for row in triggerable if row["location_id"] == 6],
        lambda row: (row["avg_vehicle_count"], row["avg_slow_count"], row["avg_helper_count"]),
    )
    add(
        "highd_win_freeflow_control",
        [row for row in rows if row["slow_frame_ratio"] == 0.0 and row["avg_vehicle_count"] >= 20.0],
        lambda row: (row["avg_vehicle_count"], row["avg_helper_count"], row["min_speed"]),
    )

    seen: set[tuple[int, int, int]] = set()
    unique = []
    for name, row in selected:
        identity = (row["recording_id"], row["location_id"], row["window_index"])
        if identity in seen:
            continue
        seen.add(identity)
        unique.append((name, row))
    return unique


def build_config(base_config: dict, name: str, row: dict) -> dict:
    config = json.loads(json.dumps(base_config))
    config["name"] = name
    config.setdefault("dataset", {})
    config["dataset"].update(
        {
            "type": "highd",
            "root": "data/highD_filtered",
            "location_id": row["location_id"],
            "recording_id": row["recording_id"],
            "start_frame": row["window_start"],
            "end_frame": row["window_end"],
            "frame_step": 5,
            "max_vehicles": 24,
            "center_positions": True,
        }
    )
    config.setdefault("simulation", {})
    config["simulation"].update({"mode": "highd", "steps": row["frame_count"], "max_vehicles": 24})
    config.setdefault("map", {})
    config["map"]["highd_location_id"] = row["location_id"]
    config["map"]["output_dir"] = f"scenarios/highd/location_{row['location_id']}"
    config["map"]["prefix"] = f"highd_l{row['location_id']}"
    config["map"]["place_query"] = f"highD location {row['location_id']} approximate Autobahn corridor"
    config["map"]["segment_label"] = f"highD location {row['location_id']} selected replay window"
    config["map"]["segment_note"] = (
        f"Selected from highD window summary: recording {row['recording_id']:02d}, "
        f"frames {row['window_start']}-{row['window_end']}, "
        f"slow_frame_ratio={row['slow_frame_ratio']:.2f}, "
        f"triggerable_frame_ratio={row['triggerable_frame_ratio']:.2f}."
    )
    return config


def write_selection(
    *,
    selections: list[tuple[str, dict]],
    base_config_path: Path,
    output_dir: Path,
    selected_csv: Path,
) -> None:
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["scenario_name", *list(selections[0][1].keys())] if selections else ["scenario_name"]
    with selected_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in selections:
            writer.writerow({"scenario_name": name, **row})
            config = build_config(base_config, name, row)
            target = output_dir / f"{name}.json"
            target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select representative highD window scenarios and write configs.")
    parser.add_argument("--summary", type=Path, default=PROJECT_ROOT / "results" / "highd_window_summary.csv")
    parser.add_argument("--base-config", type=Path, default=PROJECT_ROOT / "paper" / "configs" / "highd_bottleneck_balanced.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "paper" / "configs" / "highd_windows")
    parser.add_argument("--selected-csv", type=Path, default=PROJECT_ROOT / "results" / "highd_selected_windows.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selections = select_scenarios(load_summary(args.summary))
    if not selections:
        raise RuntimeError("No highD windows matched the selection rules.")
    write_selection(
        selections=selections,
        base_config_path=args.base_config,
        output_dir=args.output_dir,
        selected_csv=args.selected_csv,
    )
    print(f"Wrote {len(selections)} scenario configs to {args.output_dir}")
    print(f"Wrote selected window table to {args.selected_csv}")


if __name__ == "__main__":
    main()
