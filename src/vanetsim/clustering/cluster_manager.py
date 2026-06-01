from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vanetsim.config import CommunicationConfig, MobilityConfig
from vanetsim.domain import ClusterRound, VehicleState


@dataclass(frozen=True)
class ClusterState:
    """保存慢车索引集合及其对应的动态聚类轮次。"""

    slow_indices: tuple[int, ...]
    rounds: list[ClusterRound]


class ClusterManager:
    """构建 B_t 和 C_t^(ell) 动态车辆簇。

    慢车按速度从低到高排序形成 b1, b2, ..., bM；每一轮增加一辆慢车，
    再寻找通信半径 r 内愿意参与的非慢车作为辅助车辆。
    """

    def __init__(self, mobility: MobilityConfig, communication: CommunicationConfig):
        """保存慢车阈值和通信半径等聚类参数。"""
        self.mobility = mobility
        self.communication = communication

    def identify_slow_vehicles(self, vehicles: list[VehicleState]) -> list[int]:
        """找出低于最小速度阈值的车辆，并按速度从低到高排序。"""
        slow_indices = [index for index, vehicle in enumerate(vehicles) if vehicle.speed < self.mobility.min_speed]
        slow_indices.sort(key=lambda index: vehicles[index].speed)
        return slow_indices

    def build_dynamic_rounds(self, vehicles: list[VehicleState], slow_indices: list[int]) -> list[ClusterRound]:
        """按慢车扩展顺序构建每一轮可服务车辆簇和目标速度。"""
        rounds: list[ClusterRound] = []
        for ell in range(1, len(slow_indices) + 1):
            served_slow = tuple(slow_indices[:ell])
            helper_indices = []
            for index, vehicle in enumerate(vehicles):
                if index in served_slow:
                    continue
                if vehicle.speed < self.mobility.min_speed or vehicle.willingness != 1:
                    continue
                if any(self.distance(vehicle, vehicles[slow_index]) <= self.communication.max_distance for slow_index in served_slow):
                    helper_indices.append(index)

            if ell < len(slow_indices):
                target_speed = vehicles[slow_indices[ell]].speed
            else:
                target_speed = self.mobility.min_speed

            cluster_indices = tuple(sorted(set(served_slow).union(helper_indices)))
            rounds.append(
                ClusterRound(
                    round_index=ell,
                    target_speed=float(target_speed),
                    slow_vehicle_indices=served_slow,
                    helper_indices=tuple(helper_indices),
                    cluster_indices=cluster_indices,
                )
            )
        return rounds

    def distance(self, first: VehicleState, second: VehicleState) -> float:
        """计算两辆车在二维平面中的欧氏距离。"""
        return float(np.linalg.norm(np.array(first.position) - np.array(second.position)))
