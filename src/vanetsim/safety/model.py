from __future__ import annotations

import math

from vanetsim.config import MobilityConfig, SimulationConfig
from vanetsim.domain import VehicleState


class SafetyModel:
    """根据安全距离、加速度和系统时延约束计算可行速度。"""

    def __init__(self, mobility: MobilityConfig, simulation: SimulationConfig):
        """保存安全速度计算所需的移动性和仿真步长参数。"""
        self.mobility = mobility
        self.simulation = simulation

    def max_safe_speed(self, delay: float, vehicle: VehicleState | None = None) -> float:
        """根据给定时延上限计算仍能满足安全距离的最大速度。"""
        accel = self.mobility.acceleration
        distance = self.mobility.safe_distance
        disc = accel * accel * delay * delay + 2.0 * accel * distance
        if disc < 0:
            return 0.0
        return max(0.0, -accel * delay + math.sqrt(disc))

    def max_delay_for_speed(self, target_speed: float, vehicle: VehicleState | None = None) -> float:
        """计算达到目标速度时允许的最大端到端任务时延。"""
        accel = self.mobility.acceleration
        distance = self.mobility.safe_distance
        if target_speed <= 0:
            return 0.0
        value = distance / target_speed - target_speed / (2.0 * accel)
        return max(0.0, min(value, self.mobility.system_delay_cap))

    def max_stage2_speed(self, vehicles: list[VehicleState], slow_indices: list[int]) -> float:
        """结合慢车限速和安全距离，估计 Stage-II 可尝试的最高速度。"""
        if not slow_indices:
            return self.mobility.min_speed
        safety_limit = min(math.sqrt(2.0 * self.mobility.acceleration * self.mobility.safe_distance) for _ in slow_indices)
        speed_limit = min([vehicles[index].speed_limit for index in slow_indices] + [self.mobility.speed_limit])
        return max(self.mobility.min_speed, min(speed_limit, safety_limit))

    def update_speed(self, vehicle: VehicleState, delay: float, requested_speed: float) -> float:
        """将优化请求速度裁剪到安全速度、限速和加速度约束范围内。"""
        safe_speed = self.max_safe_speed(delay, vehicle)
        dynamic_speed = vehicle.speed + self.mobility.max_acceleration * self.simulation.step_length
        return min(safe_speed, vehicle.speed_limit, dynamic_speed, requested_speed)
