from __future__ import annotations

import math

import numpy as np

from vanetsim.domain import TaskComponent, VehicleState


class DelayModel:
    """计算本地执行或 V2V 卸载分支上的任务时延系数。"""

    def branch_delay_coefficient(
        self,
        source: VehicleState,
        helper: VehicleState,
        component: TaskComponent,
        uplink_rate: float,
        downlink_rate: float,
    ) -> float:
        """计算某个任务组件分配到某个辅助车辆时产生的总时延。"""
        resource_index = component.resource_index
        if source.vehicle_id == helper.vehicle_id:
            return component.compute_load / max(source.compute_capacity[resource_index], 1.0)

        if uplink_rate <= 0 or downlink_rate <= 0:
            return math.inf

        tx_delay = component.input_size / uplink_rate
        compute_delay = component.compute_load / max(helper.compute_capacity[resource_index], 1.0)
        rx_delay = component.output_size / downlink_rate
        return tx_delay + compute_delay + rx_delay

    def delay_coefficients(
        self,
        vehicles: list[VehicleState],
        source_indices: list[int],
        helper_indices: list[int],
        tasks_by_vehicle: dict[str, list[TaskComponent]],
        rates: np.ndarray,
    ) -> np.ndarray:
        """批量生成源车辆、辅助车辆和资源组件之间的时延系数张量。"""
        source_count = len(source_indices)
        helper_count = len(helper_indices)
        resource_count = len(next(iter(tasks_by_vehicle.values()))) if tasks_by_vehicle else 0
        coeffs = np.zeros((source_count, helper_count, resource_count), dtype=float)

        for source_pos, source_index in enumerate(source_indices):
            source = vehicles[source_index]
            for helper_pos, helper_index in enumerate(helper_indices):
                helper = vehicles[helper_index]
                for component in tasks_by_vehicle[source.vehicle_id]:
                    coeffs[source_pos, helper_pos, component.resource_index] = self.branch_delay_coefficient(
                        source=source,
                        helper=helper,
                        component=component,
                        uplink_rate=rates[source_index, helper_index],
                        downlink_rate=rates[helper_index, source_index],
                    )
        return coeffs
