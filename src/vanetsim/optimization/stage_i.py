from __future__ import annotations

import math

import cvxpy as cp
import numpy as np

from vanetsim.config import CommunicationConfig, MobilityConfig, OffloadingConfig, SimulationConfig
from vanetsim.delay import DelayModel
from vanetsim.domain import ClusterRound, StageIOutput, TaskComponent, VehicleState
from vanetsim.safety import SafetyModel


class StageIOptimizer:
    """Stage-I：通过付费资源预留建立慢车基线速度。

    对每个动态聚类轮次 ell，在满足通信可行性、辅助车辆算力容量和
    目标速度时延上限的前提下，为慢车任务组件寻找成本最小的分配方案。
    """

    def __init__(
        self,
        mobility: MobilityConfig,
        communication: CommunicationConfig,
        offloading: OffloadingConfig,
        simulation: SimulationConfig,
        delay_model: DelayModel,
        safety_model: SafetyModel,
    ):
        """保存 Stage-I 优化需要的配置、时延模型和安全速度模型。"""
        self.mobility = mobility
        self.communication = communication
        self.offloading = offloading
        self.simulation = simulation
        self.delay_model = delay_model
        self.safety_model = safety_model

    def optimize(
        self,
        vehicles: list[VehicleState],
        rounds: list[ClusterRound],
        tasks_by_vehicle: dict[str, list[TaskComponent]],
        rates: np.ndarray,
    ) -> StageIOutput:
        """遍历动态聚类轮次，求解慢车达到基线目标速度所需的资源分配。"""
        n = len(vehicles)
        k_count = self.offloading.resource_count
        total_capacity = np.array([vehicle.compute_capacity for vehicle in vehicles], dtype=float) * self.simulation.step_length
        baseline_reservation = np.zeros((n, k_count), dtype=float)
        full_allocation = np.zeros((n, n, k_count), dtype=float)
        delay_by_vehicle: dict[str, float] = {}
        target_speeds: dict[str, float] = {}
        payment = 0.0
        baseline_feasible = bool(rounds)

        for round_state in rounds:
            # Round target follows the requirement:
            # b1 reaches b2 speed, then b1/b2 reach b3 speed, ..., final round reaches v_l.
            source_indices = list(round_state.slow_vehicle_indices)
            helper_indices = list(round_state.cluster_indices)
            result = self.solve_allocation(
                vehicles=vehicles,
                source_indices=source_indices,
                helper_indices=helper_indices,
                tasks_by_vehicle=tasks_by_vehicle,
                rates=rates,
                capacity=total_capacity,
                target_speed=round_state.target_speed,
                prices=np.array([vehicle.price for vehicle in vehicles], dtype=float),
                objective="payment",
            )

            if result is None:
                baseline_feasible = False
                break

            full_allocation = result["allocation"]
            baseline_reservation = result["resource_used"]
            delay_by_vehicle = result["delay_by_vehicle"]
            payment = result["cost"]
            for source_index in source_indices:
                target_speeds[vehicles[source_index].vehicle_id] = round_state.target_speed

        completion_rate = self._completion_rate(vehicles, list(tasks_by_vehicle), delay_by_vehicle, baseline_feasible)
        return StageIOutput(
            allocation=full_allocation,
            delay_by_vehicle=delay_by_vehicle,
            payment=float(payment),
            baseline_reservation=baseline_reservation,
            baseline_feasible=baseline_feasible,
            target_speeds=target_speeds,
            rounds=rounds,
            task_completion_rate=completion_rate,
        )

    def solve_allocation(
        self,
        *,
        vehicles: list[VehicleState],
        source_indices: list[int],
        helper_indices: list[int],
        tasks_by_vehicle: dict[str, list[TaskComponent]],
        rates: np.ndarray,
        capacity: np.ndarray,
        target_speed: float | None,
        prices: np.ndarray,
        objective: str,
    ):
        """构建并求解指定慢车集合和辅助车辆集合上的任务分配问题。"""
        if not source_indices or not helper_indices:
            return None

        source_count = len(source_indices)
        helper_count = len(helper_indices)
        k_count = self.offloading.resource_count
        alpha = cp.Variable((source_count * helper_count, k_count), nonneg=True)
        component_delay = cp.Variable((source_count, k_count), nonneg=True)
        constraints = []

        compute_loads = np.zeros((source_count, k_count), dtype=float)
        delay_coeffs = np.zeros((source_count, helper_count, k_count), dtype=float)
        for s_pos, source_index in enumerate(source_indices):
            source = vehicles[source_index]
            for component in tasks_by_vehicle[source.vehicle_id]:
                compute_loads[s_pos, component.resource_index] = component.compute_load
                for h_pos, helper_index in enumerate(helper_indices):
                    helper = vehicles[helper_index]
                    delay_coeffs[s_pos, h_pos, component.resource_index] = self.delay_model.branch_delay_coefficient(
                        source=source,
                        helper=helper,
                        component=component,
                        uplink_rate=rates[source_index, helper_index],
                        downlink_rate=rates[helper_index, source_index],
                    )

        for s_pos, source_index in enumerate(source_indices):
            for k in range(k_count):
                rows = [s_pos * helper_count + h_pos for h_pos in range(helper_count)]
                constraints.append(cp.sum(alpha[rows, k]) == 1.0)

                for h_pos, helper_index in enumerate(helper_indices):
                    row = s_pos * helper_count + h_pos
                    coeff = delay_coeffs[s_pos, h_pos, k]
                    if helper_index != source_index and (
                        rates[source_index, helper_index] < self.communication.min_rate or not math.isfinite(coeff)
                    ):
                        constraints.append(alpha[row, k] == 0.0)
                    else:
                        constraints.append(component_delay[s_pos, k] >= coeff * alpha[row, k])

            if target_speed is not None:
                t_max = self.safety_model.max_delay_for_speed(target_speed, vehicles[source_index])
                constraints.append(cp.sum(component_delay[s_pos, :]) <= t_max)

        for h_pos, helper_index in enumerate(helper_indices):
            for k in range(k_count):
                usage = 0
                for s_pos in range(source_count):
                    row = s_pos * helper_count + h_pos
                    usage += alpha[row, k] * compute_loads[s_pos, k]
                constraints.append(usage <= capacity[helper_index, k])

        cost_expr = 0
        max_delay = cp.Variable(nonneg=True)
        for s_pos in range(source_count):
            constraints.append(max_delay >= cp.sum(component_delay[s_pos, :]))
            for h_pos, helper_index in enumerate(helper_indices):
                row = s_pos * helper_count + h_pos
                for k in range(k_count):
                    cost_expr += prices[helper_index, k] * alpha[row, k] * compute_loads[s_pos, k]

        if objective == "max_delay":
            objective_expr = max_delay
        else:
            objective_expr = cost_expr

        problem = cp.Problem(cp.Minimize(objective_expr), constraints)
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            problem.solve(solver=cp.SCS, verbose=False)

        if alpha.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return None

        allocation = np.zeros((len(vehicles), len(vehicles), k_count), dtype=float)
        resource_used = np.zeros((len(vehicles), k_count), dtype=float)
        delay_by_vehicle: dict[str, float] = {}
        for s_pos, source_index in enumerate(source_indices):
            source_id = vehicles[source_index].vehicle_id
            realized_components = np.zeros(k_count, dtype=float)
            for h_pos, helper_index in enumerate(helper_indices):
                row = s_pos * helper_count + h_pos
                for k in range(k_count):
                    share = max(0.0, float(alpha.value[row, k]))
                    allocation[source_index, helper_index, k] = share
                    resource_used[helper_index, k] += share * compute_loads[s_pos, k]
                    coeff = delay_coeffs[s_pos, h_pos, k]
                    if share <= 1e-8 or not math.isfinite(coeff):
                        continue
                    realized_components[k] = max(
                        realized_components[k],
                        coeff * share,
                    )
            delay_by_vehicle[source_id] = float(np.sum(realized_components))

        return {
            "allocation": allocation,
            "resource_used": resource_used,
            "delay_by_vehicle": delay_by_vehicle,
            "cost": float(cost_expr.value),
        }

    def _completion_rate(
        self,
        vehicles: list[VehicleState],
        task_vehicle_ids: list[str],
        delay_by_vehicle: dict[str, float],
        feasible: bool,
    ) -> float:
        """根据可行性和已产生时延结果计算慢车任务完成率。"""
        if not task_vehicle_ids:
            return 1.0
        completed = sum(1 for vehicle_id in task_vehicle_ids if feasible and vehicle_id in delay_by_vehicle)
        return completed / len(task_vehicle_ids)
