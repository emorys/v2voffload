from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from vanetsim.config import DatasetConfig
from vanetsim.domain import VehicleSnapshot


@dataclass(frozen=True)
class HighDFrame:
    """表示 highD 数据集中一个帧号对应的车辆快照集合。"""

    frame: int
    time: float
    snapshots: list[VehicleSnapshot]


class HighDLoader:
    """将 highD CSV 轨迹文件加载为仿真器使用的车辆快照格式。

    highD 原始位置是车辆包围盒左上角坐标；默认会转换到车辆中心点，
    便于后续计算 V2V 距离。数据集只提供交通运动状态，VANET 资源字段
    会在 VehicleStateManager 中根据确定性模型参数补齐。
    """

    def __init__(self, dataset: DatasetConfig, workspace_root: str | Path):
        """初始化数据集路径、记录编号和三个 highD CSV 文件路径。"""
        self.dataset = dataset
        self.workspace_root = Path(workspace_root).resolve()
        self.root = self._resolve_root(dataset.root)
        self.recording_id = self._resolve_recording_id()
        self.recording_prefix = f"{self.recording_id:02d}"
        self.tracks_path = self.root / f"{self.recording_prefix}_tracks.csv"
        self.tracks_meta_path = self.root / f"{self.recording_prefix}_tracksMeta.csv"
        self.recording_meta_path = self.root / f"{self.recording_prefix}_recordingMeta.csv"

    def load_frames(self) -> list[HighDFrame]:
        """读取筛选后的 highD 行数据，并按帧组装为 HighDFrame 列表。"""
        self._assert_files_exist()
        tracks_meta = self._read_tracks_meta()
        recording_meta = self._read_recording_meta()
        self._assert_location_matches(recording_meta)
        frame_rate = float(recording_meta.get("frameRate", 25.0))
        rows = self._read_selected_rows(tracks_meta)
        if not rows:
            return []

        min_x = min(row["center_x"] for row in rows)
        max_x = max(row["center_x"] for row in rows)
        by_frame: dict[int, list[dict]] = {}
        for row in rows:
            by_frame.setdefault(row["frame"], []).append(row)

        frames: list[HighDFrame] = []
        for frame in sorted(by_frame):
            snapshots = [self._row_to_snapshot(row, min_x=min_x, max_x=max_x) for row in by_frame[frame]]
            snapshots.sort(key=lambda snapshot: (snapshot.progress, snapshot.vehicle_id))
            if self.dataset.max_vehicles > 0:
                snapshots = snapshots[: self.dataset.max_vehicles]
            frames.append(HighDFrame(frame=frame, time=frame / frame_rate, snapshots=snapshots))
        return frames

    def _resolve_root(self, root: str) -> Path:
        """将相对数据集根目录解析到工作区下的绝对路径。"""
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = self.workspace_root / root_path
        return root_path.resolve()

    def _resolve_recording_id(self) -> int:
        """Resolve recording_id directly or by scanning recordingMeta for location_id."""
        if self.dataset.recording_id > 0:
            return self.dataset.recording_id
        if self.dataset.location_id is None:
            raise ValueError("dataset.recording_id must be positive unless dataset.location_id is provided.")
        if not self.root.exists():
            raise FileNotFoundError(f"highD root does not exist: {self.root}")

        matches: list[int] = []
        for meta_path in sorted(self.root.glob("*_recordingMeta.csv")):
            with meta_path.open("r", encoding="utf-8-sig", newline="") as file:
                row = next(csv.DictReader(file), None)
            if row is None:
                continue
            if int(float(row.get("locationId", -1))) == self.dataset.location_id:
                matches.append(int(meta_path.name.split("_", 1)[0]))

        if not matches:
            raise FileNotFoundError(
                f"No highD recordingMeta file with locationId={self.dataset.location_id} was found under {self.root}."
            )
        return matches[0]

    def _assert_files_exist(self) -> None:
        """检查 tracks、tracksMeta 和 recordingMeta 三个必需文件是否存在。"""
        missing = [path for path in [self.tracks_path, self.tracks_meta_path, self.recording_meta_path] if not path.exists()]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                "highD files were not found. Expected files named "
                f"{self.recording_prefix}_tracks.csv, {self.recording_prefix}_tracksMeta.csv, "
                f"and {self.recording_prefix}_recordingMeta.csv under {self.root}. Missing: {missing_text}"
            )

    def _read_tracks_meta(self) -> dict[int, dict]:
        """读取车辆维度的 highD 元数据并按车辆 ID 建立索引。"""
        with self.tracks_meta_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            return {int(row["id"]): row for row in reader}

    def _read_recording_meta(self) -> dict:
        """读取 recordingMeta 中的全局记录信息，例如帧率。"""
        with self.recording_meta_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            return next(reader, {})

    def _assert_location_matches(self, recording_meta: dict) -> None:
        """Validate that the selected highD recording belongs to the configured location."""
        if self.dataset.location_id is None:
            return
        actual = int(float(recording_meta.get("locationId", -1)))
        if actual != self.dataset.location_id:
            raise ValueError(
                f"Configured highD location_id={self.dataset.location_id}, but "
                f"{self.recording_meta_path.name} has locationId={actual}."
            )

    def _read_selected_rows(self, tracks_meta: dict[int, dict]) -> list[dict]:
        """按帧范围、方向和车道过滤 tracks 行，并转换基础数值字段。"""
        selected_rows: list[dict] = []
        with self.tracks_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for raw in reader:
                frame = int(raw["frame"])
                if frame < self.dataset.start_frame:
                    continue
                if self.dataset.end_frame is not None and frame > self.dataset.end_frame:
                    continue
                if (frame - self.dataset.start_frame) % max(1, self.dataset.frame_step) != 0:
                    continue

                vehicle_id = int(raw["id"])
                meta = tracks_meta.get(vehicle_id, {})
                driving_direction = int(meta.get("drivingDirection", raw.get("drivingDirection", 1)))
                if self.dataset.driving_direction is not None and driving_direction != self.dataset.driving_direction:
                    continue

                lane_id = int(float(raw["laneId"]))
                if self.dataset.lane_ids is not None and lane_id not in self.dataset.lane_ids:
                    continue

                width = float(meta.get("width", raw.get("width", 0.0)) or 0.0)
                height = float(meta.get("height", raw.get("height", 0.0)) or 0.0)
                x = float(raw["x"])
                y = float(raw["y"])
                if self.dataset.center_positions:
                    x += width / 2.0
                    y += height / 2.0

                selected_rows.append(
                    {
                        "frame": frame,
                        "vehicle_id": vehicle_id,
                        "x": x,
                        "y": y,
                        "center_x": x,
                        "x_velocity": float(raw.get("xVelocity", 0.0) or 0.0),
                        "y_velocity": float(raw.get("yVelocity", 0.0) or 0.0),
                        "x_acceleration": float(raw.get("xAcceleration", 0.0) or 0.0),
                        "y_acceleration": float(raw.get("yAcceleration", 0.0) or 0.0),
                        "lane_id": lane_id,
                        "driving_direction": driving_direction,
                    }
                )
        return selected_rows

    def _row_to_snapshot(self, row: dict, *, min_x: float, max_x: float) -> VehicleSnapshot:
        """将一行 highD 轨迹数据转换成统一的 VehicleSnapshot。"""
        speed = math.hypot(row["x_velocity"], row["y_velocity"])
        acceleration = math.hypot(row["x_acceleration"], row["y_acceleration"])
        if row["driving_direction"] == 1:
            progress = max_x - row["center_x"]
        else:
            progress = row["center_x"] - min_x

        return VehicleSnapshot(
            vehicle_id=f"highd_{row['vehicle_id']}",
            x=row["x"],
            y=row["y"],
            progress=float(progress),
            speed=float(speed),
            acceleration=float(acceleration),
            desired_speed=max(float(speed), 33.33),
            lane_id=str(row["lane_id"]),
            allowed_speed=max(float(speed), 33.33),
            max_speed=max(float(speed), 33.33),
        )
