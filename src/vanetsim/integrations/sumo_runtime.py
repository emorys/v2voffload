from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from vanetsim.domain import VehicleSnapshot


DEFAULT_SUMO_HOME_CANDIDATES = (
    os.environ.get("SUMO_HOME"),
    r"C:\Program Files (x86)\Eclipse\Sumo",
    r"C:\Program Files\Eclipse\Sumo",
)

SumoVehicleState = VehicleSnapshot


def _existing_path(path_text: str | None) -> Path | None:
    """将字符串路径转换为已存在的 Path，不存在时返回 None。"""
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.exists():
        return path
    return None


def discover_sumo_home(explicit_home: str | Path | None = None) -> Path:
    """在显式路径、环境变量和常见安装目录中定位 SUMO_HOME。"""
    candidates = [_existing_path(str(explicit_home)) if explicit_home else None]
    candidates.extend(_existing_path(path_text) for path_text in DEFAULT_SUMO_HOME_CANDIDATES)
    for candidate in candidates:
        if candidate and (candidate / "tools").is_dir() and (candidate / "bin").is_dir():
            os.environ.setdefault("SUMO_HOME", str(candidate))
            return candidate
    raise RuntimeError(
        "SUMO installation was not found. Set SUMO_HOME or install SUMO under "
        "'C:\\Program Files (x86)\\Eclipse\\Sumo'."
    )


def ensure_sumo_tools_on_path(explicit_home: str | Path | None = None) -> Path:
    """确保 SUMO 的 tools 目录已加入 Python 导入路径。"""
    sumo_home = discover_sumo_home(explicit_home)
    tools_dir = str(sumo_home / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    return sumo_home


def load_traci(explicit_home: str | Path | None = None):
    """加载 TraCI 与 sumolib 模块，并返回解析到的 SUMO_HOME。"""
    sumo_home = ensure_sumo_tools_on_path(explicit_home)
    traci = importlib.import_module("traci")
    sumolib = importlib.import_module("sumolib")
    return traci, sumolib, sumo_home


def resolve_sumo_binary(gui: bool = False, explicit_home: str | Path | None = None) -> str:
    """根据是否启用 GUI 返回可执行的 SUMO 二进制文件路径。"""
    sumo_home = discover_sumo_home(explicit_home)
    binary_name = "sumo-gui.exe" if gui else "sumo.exe"
    binary = sumo_home / "bin" / binary_name
    if not binary.exists():
        raise RuntimeError(f"SUMO binary was not found: {binary}")
    return str(binary)


def build_sumo_command(
    sumocfg_path: Path,
    step_length: float,
    seed: int,
    gui: bool = False,
    explicit_home: str | Path | None = None,
) -> list[str]:
    """构建启动 SUMO 或 sumo-gui 的命令行参数列表。"""
    command = [
        resolve_sumo_binary(gui=gui, explicit_home=explicit_home),
        "-c",
        str(sumocfg_path),
        "--step-length",
        str(step_length),
        "--seed",
        str(seed),
        "--no-step-log",
        "true",
    ]
    if gui:
        command.extend(["--quit-on-end", "true"])
    return command


def collect_vehicle_snapshots(traci_module) -> list[VehicleSnapshot]:
    """从 TraCI 当前仿真步收集车辆位置、速度和车道信息。"""
    snapshots: list[VehicleSnapshot] = []
    for vehicle_id in traci_module.vehicle.getIDList():
        x, y = traci_module.vehicle.getPosition(vehicle_id)
        speed = float(traci_module.vehicle.getSpeed(vehicle_id))
        try:
            acceleration = float(traci_module.vehicle.getAcceleration(vehicle_id))
        except Exception:
            acceleration = 0.0
        max_speed = float(traci_module.vehicle.getMaxSpeed(vehicle_id))
        allowed_speed = float(traci_module.vehicle.getAllowedSpeed(vehicle_id))
        lane_id = traci_module.vehicle.getLaneID(vehicle_id)
        route_distance = float(traci_module.vehicle.getDistance(vehicle_id))
        try:
            lane_position = float(traci_module.vehicle.getLanePosition(vehicle_id))
        except Exception:
            lane_position = route_distance
        progress = max(route_distance, lane_position)
        desired_speed = max(speed, min(max_speed, allowed_speed))

        snapshots.append(
            VehicleSnapshot(
                vehicle_id=vehicle_id,
                x=float(x),
                y=float(y),
                progress=progress,
                speed=speed,
                acceleration=acceleration,
                desired_speed=desired_speed,
                lane_id=lane_id,
                allowed_speed=allowed_speed,
                max_speed=max_speed,
            )
        )

    snapshots.sort(key=lambda item: (item.progress, item.vehicle_id))
    return snapshots


def collect_vehicle_states(traci_module) -> list[VehicleSnapshot]:
    """兼容旧接口，返回当前仿真步的车辆快照列表。"""
    return collect_vehicle_snapshots(traci_module)


def apply_target_speeds(traci_module, vehicle_ids: list[str], target_speeds) -> None:
    """将优化后的目标速度写回 TraCI 中对应的车辆。"""
    for vehicle_id, target_speed in zip(vehicle_ids, target_speeds):
        traci_module.vehicle.setSpeed(vehicle_id, float(max(0.0, target_speed)))
