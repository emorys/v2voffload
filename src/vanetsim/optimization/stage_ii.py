from __future__ import annotations

import numpy as np

from vanetsim.config import IncentiveConfig, MobilityConfig, OffloadingConfig, SimulationConfig
from vanetsim.domain import StageIIOutput, StageIOutput, TaskComponent, VehicleState
from vanetsim.optimization.stage_i import StageIOptimizer
from vanetsim.safety import SafetyModel


class StageIIOptimizer:
    """利用 Stage-I 之后的剩余资源，评估进一步提升瓶颈慢车速度的收益。"""

    def __init__(
        self,
        mobility: MobilityConfig,
        offloading: OffloadingConfig,
        incentive: IncentiveConfig,
        simulation: SimulationConfig,
        safety_model: SafetyModel,
        stage_i_optimizer: StageIOptimizer,
    ):
        """保存 Stage-II 收益成本评估和复用 Stage-I 分配模型所需依赖。"""
        self.mobility = mobility
        self.offloading = offloading
        self.incentive = incentive
        self.simulation = simulation
        self.safety_model = safety_model
        self.stage_i_optimizer = stage_i_optimizer

    def optimize(
        self,
        vehicles: list[VehicleState],
        slow_indices: list[int],
        tasks_by_vehicle: dict[str, list[TaskComponent]],
        rates: np.ndarray,
        stage_i: StageIOutput,
    ) -> StageIIOutput:
        """在候选速度网格上搜索剩余资源分配，并返回收益最大的可行方案。"""
        n = len(vehicles)
        k_count = self.offloading.resource_count
        empty_allocation = np.zeros((n, n, k_count), dtype=float)
        if not slow_indices:
            return self._empty_output(empty_allocation, status="no_slow_vehicle")
        if not stage_i.baseline_feasible:
            return self._empty_output(empty_allocation, status="stage_i_infeasible")

        total_capacity = np.array([vehicle.compute_capacity for vehicle in vehicles], dtype=float) * self.simulation.step_length
        residual_capacity = np.maximum(total_capacity - stage_i.baseline_reservation, 0.0)
        bottleneck = vehicles[slow_indices[0]]
        potential_helpers = self._participating_helper_candidates(vehicles, slow_indices, residual_capacity, bottleneck)
        candidate_helper_ids = tuple(vehicles[index].vehicle_id for index in potential_helpers)
        if not potential_helpers:
            return self._empty_output(empty_allocation, boosted_speed=self.mobility.min_speed, status="no_helper_behind_with_residual_resource")

        max_speed = self.safety_model.max_stage2_speed(vehicles, slow_indices)
        candidate_speeds = np.linspace(self.mobility.min_speed, max_speed, max(2, self.offloading.stage2_speed_grid))
        best = None
        prices = np.array([vehicle.price for vehicle in vehicles], dtype=float)
        feasible_speed_count = 0

        # Stage-II is evaluated over possible bottleneck speeds under the safety and
        # road speed limits. For every feasible speed, the allocation LP minimizes
        # the actual helper payment on the currently available residual resource pool.
        # The selected policy is lexicographic: highest feasible speed first, then
        # minimum residual cost among candidates at that speed.
        for boosted_speed in candidate_speeds:
            if boosted_speed <= self.mobility.min_speed + 1e-9:
                continue
            helper_indices = sorted(set(slow_indices).union(potential_helpers))
            result = self.stage_i_optimizer.solve_allocation(
                vehicles=vehicles,
                source_indices=slow_indices,
                helper_indices=helper_indices,
                tasks_by_vehicle=tasks_by_vehicle,
                rates=rates,
                capacity=residual_capacity + stage_i.baseline_reservation,
                target_speed=float(boosted_speed),
                prices=prices,
                objective="payment",
            )
            if result is None:
                continue
            feasible_speed_count += 1

            extra_used = np.maximum(result["resource_used"] - stage_i.baseline_reservation, 0.0)
            cost = float(np.sum(extra_used * prices))
            benefit = self._traffic_benefit(vehicles, bottleneck, boosted_speed)
            utility = benefit - self.incentive.opportunity_cost_lambda * cost
            participating_helpers, helper_utilities = self._filter_individually_rational_helpers(
                vehicles=vehicles,
                helper_indices=potential_helpers,
                bottleneck=bottleneck,
                boosted_speed=float(boosted_speed),
                extra_used=extra_used,
                prices=prices,
            )
            if not participating_helpers:
                continue
            if best is None or self._is_better_candidate(float(boosted_speed), cost, best):
                best = {
                    "objective": utility,
                    "speed": float(boosted_speed),
                    "result": result,
                    "extra_used": extra_used,
                    "benefit": benefit,
                    "cost": cost,
                    "participating_helpers": participating_helpers,
                    "helper_utilities": helper_utilities,
                }

        if best is None:
            status = "no_helper_passed_individual_rationality" if feasible_speed_count else "no_feasible_residual_allocation"
            return self._empty_output(
                empty_allocation,
                boosted_speed=self.mobility.min_speed,
                status=status,
                candidate_helpers=candidate_helper_ids,
                evaluated_speed_count=len(candidate_speeds),
                feasible_speed_count=feasible_speed_count,
            )

        target_speeds = {vehicles[index].vehicle_id: best["speed"] for index in slow_indices}
        return StageIIOutput(
            boosted_speed=best["speed"],
            allocation=best["result"]["allocation"],
            residual_used=best["extra_used"],
            benefit=best["benefit"],
            cost=best["cost"],
            participating_helpers=tuple(best["participating_helpers"]),
            target_speeds=target_speeds,
            status="ok",
            objective_value=float(best["objective"]),
            candidate_helpers=candidate_helper_ids,
            helper_utilities=best["helper_utilities"],
            evaluated_speed_count=len(candidate_speeds),
            feasible_speed_count=feasible_speed_count,
        )

    def _is_better_candidate(self, boosted_speed: float, cost: float, best: dict) -> bool:
        """Prefer higher feasible speed; break ties by lower residual cost."""
        if boosted_speed > best["speed"] + 1e-9:
            return True
        if abs(boosted_speed - best["speed"]) <= 1e-9 and cost < best["cost"] - 1e-9:
            return True
        return False

    def _participating_helper_candidates(
        self,
        vehicles: list[VehicleState],
        slow_indices: list[int],
        residual_capacity: np.ndarray,
        bottleneck: VehicleState,
    ) -> list[int]:
        """筛选位于瓶颈车后方、愿意参与且仍有剩余资源的辅助车辆。"""
        slow_set = set(slow_indices)
        helpers = []
        for index, vehicle in enumerate(vehicles):
            if index in slow_set:
                continue
            if vehicle.speed < self.mobility.min_speed or vehicle.willingness != 1:
                continue
            if not np.any(residual_capacity[index, :] > 0.0):
                continue
            # In this coordinate system, smaller progress means the vehicle is behind
            # the bottleneck. Only those vehicles receive travel-time benefit from
            # improving the bottleneck speed.
            if vehicle.progress < bottleneck.progress and vehicle.desired_speed > self.mobility.min_speed:
                helpers.append(index)
        return helpers

    def _traffic_benefit(self, vehicles: list[VehicleState], bottleneck: VehicleState, boosted_speed: float) -> float:
        """估计瓶颈车提速后给后方车辆节省的旅行时间收益。"""
        benefit = 0.0
        for vehicle in vehicles:
            if vehicle.progress >= bottleneck.progress or vehicle.desired_speed <= self.mobility.min_speed:
                continue
            base_time = self.mobility.segment_length / max(min(vehicle.desired_speed, self.mobility.min_speed), 1e-6)
            boosted_time = self.mobility.segment_length / max(min(vehicle.desired_speed, boosted_speed), 1e-6)
            benefit += vehicle.value_of_time * max(0.0, base_time - boosted_time)
        return float(benefit)

    def _filter_individually_rational_helpers(
        self,
        *,
        vehicles: list[VehicleState],
        helper_indices: list[int],
        bottleneck: VehicleState,
        boosted_speed: float,
        extra_used: np.ndarray,
        prices: np.ndarray,
    ) -> tuple[list[str], dict[str, float]]:
        """保留个人收益不低于机会成本的 Stage-II 参与辅助车辆。"""
        participating = []
        utilities: dict[str, float] = {}
        for index in helper_indices:
            vehicle = vehicles[index]
            base_time = self.mobility.segment_length / max(min(vehicle.desired_speed, self.mobility.min_speed), 1e-6)
            boosted_time = self.mobility.segment_length / max(min(vehicle.desired_speed, boosted_speed), 1e-6)
            helper_benefit = vehicle.value_of_time * max(0.0, base_time - boosted_time)
            helper_cost = self.incentive.opportunity_cost_lambda * float(np.sum(extra_used[index, :] * prices[index, :]))
            helper_utility = float(helper_benefit - helper_cost)
            utilities[vehicle.vehicle_id] = helper_utility
            if helper_utility >= 0.0 and vehicle.progress < bottleneck.progress:
                participating.append(vehicle.vehicle_id)
        return participating, utilities

    def _empty_output(
        self,
        allocation: np.ndarray,
        boosted_speed: float = 0.0,
        status: str = "skipped",
        *,
        objective_value: float = 0.0,
        candidate_helpers: tuple[str, ...] = (),
        helper_utilities: dict[str, float] | None = None,
        evaluated_speed_count: int = 0,
        feasible_speed_count: int = 0,
    ) -> StageIIOutput:
        """生成无可行 Stage-II 方案时使用的空结果对象。"""
        return StageIIOutput(
            boosted_speed=boosted_speed,
            allocation=allocation,
            residual_used=np.zeros((allocation.shape[0], allocation.shape[2]), dtype=float)
            if allocation.ndim == 3
            else np.array([]),
            benefit=0.0,
            cost=0.0,
            participating_helpers=(),
            target_speeds={},
            status=status,
            objective_value=objective_value,
            candidate_helpers=candidate_helpers,
            helper_utilities=helper_utilities or {},
            evaluated_speed_count=evaluated_speed_count,
            feasible_speed_count=feasible_speed_count,
        )
