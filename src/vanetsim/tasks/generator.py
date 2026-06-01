from __future__ import annotations

import numpy as np

from vanetsim.config import MobilityConfig, OffloadingConfig
from vanetsim.domain import TaskComponent, VehicleState


class TaskGenerator:
    """根据车辆任务负载生成可分配到不同资源类型的任务组件。"""

    def __init__(self, offloading: OffloadingConfig, mobility: MobilityConfig):
        """保存任务组件生成所需的卸载配置和时延截止配置。"""
        self.offloading = offloading
        self.mobility = mobility

    def generate_for_vehicle(self, vehicle: VehicleState) -> list[TaskComponent]:
        """为单辆车按资源类型生成感知、预测、规划等任务组件。"""
        components: list[TaskComponent] = []
        for resource_index, component_type in enumerate(self.offloading.component_types):
            compute_load = float(vehicle.task_load[resource_index])
            if vehicle.task_input_bits is not None and resource_index < len(vehicle.task_input_bits):
                input_size = float(vehicle.task_input_bits[resource_index])
            else:
                base_cycles = np.array(self.offloading.base_cycles_per_bit, dtype=float)
                input_size = compute_load / max(base_cycles[resource_index], 1.0)
            if vehicle.task_output_bits is not None and resource_index < len(vehicle.task_output_bits):
                output_size = float(vehicle.task_output_bits[resource_index])
            else:
                output_size = input_size * self.offloading.output_size_ratio
            if vehicle.task_deadlines is not None and resource_index < len(vehicle.task_deadlines):
                deadline = min(self.mobility.system_delay_cap, float(vehicle.task_deadlines[resource_index]))
            else:
                deadline = self.mobility.system_delay_cap
            components.append(
                TaskComponent(
                    task_id=f"{vehicle.vehicle_id}:{component_type}",
                    vehicle_id=vehicle.vehicle_id,
                    component_type=component_type,
                    resource_index=resource_index,
                    compute_load=float(compute_load),
                    input_size=float(input_size),
                    output_size=float(output_size),
                    deadline=deadline,
                    splitable=True,
                )
            )
        return components

    def generate_for_slow_vehicles(
        self, vehicles: list[VehicleState], slow_indices: list[int]
    ) -> dict[str, list[TaskComponent]]:
        """为所有慢车批量生成任务组件，并按车辆 ID 建立映射。"""
        return {vehicles[index].vehicle_id: self.generate_for_vehicle(vehicles[index]) for index in slow_indices}
