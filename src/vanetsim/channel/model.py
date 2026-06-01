from __future__ import annotations

import math

import numpy as np

from vanetsim.config import CommunicationConfig
from vanetsim.domain import VehicleState


class ChannelModel:
    """根据车辆位置和通信配置计算 V2V 链路速率。"""

    def __init__(self, communication: CommunicationConfig):
        """保存无线信道计算所需的通信参数。"""
        self.communication = communication

    def compute_rates(self, vehicles: list[VehicleState], rng: np.random.Generator) -> np.ndarray:
        """计算每一对车辆之间的香农速率矩阵。"""
        points = np.array([vehicle.position for vehicle in vehicles], dtype=float)
        n = len(points)
        rates = np.zeros((n, n), dtype=float)
        if n == 0:
            return rates

        deltas = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                distance = float(distances[i, j])
                if distance > self.communication.max_distance:
                    continue
                distance = max(distance, 1.0)
                h2 = rng.exponential(1.0)
                gain = self.communication.pathloss_const * (distance ** -self.communication.pathloss_exp) * h2
                interference = self._interference(i, j, distances, rng)
                sinr = (self.communication.tx_power * gain) / (
                    self.communication.noise_density * self.communication.bandwidth + interference
                )
                rates[i, j] = self.communication.bandwidth * math.log2(1.0 + sinr)
        return rates

    def feasible_matrix(self, rates: np.ndarray) -> np.ndarray:
        """根据最小速率阈值判断链路是否可用于任务卸载。"""
        return rates >= self.communication.min_rate

    def _interference(self, source: int, target: int, distances: np.ndarray, rng: np.random.Generator) -> float:
        """估算其他车辆同时传输时对目标接收端造成的干扰功率。"""
        interference = 0.0
        for interferer in range(len(distances)):
            if interferer == source or interferer == target:
                continue
            distance = float(distances[interferer, target])
            if distance > self.communication.max_distance:
                continue
            distance = max(distance, 1.0)
            h2 = rng.exponential(1.0)
            gain = self.communication.pathloss_const * (distance ** -self.communication.pathloss_exp) * h2
            interference += self.communication.tx_power * gain
        return interference
