from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from vanetsim.channel import ChannelModel
from vanetsim.clustering import ClusterManager
from vanetsim.config import CommunicationConfig, IncentiveConfig, MobilityConfig, OffloadingConfig, SimulationConfig
from vanetsim.delay import DelayModel
from vanetsim.domain import (
    AllocationResult,
    ClusterRound,
    StageDecision,
    StageIOutput,
    StageIIOutput,
    TaskComponent,
    VehicleState,
)
from vanetsim.optimization.stage_i import StageIOptimizer
from vanetsim.optimization.stage_ii import StageIIOptimizer
from vanetsim.safety import SafetyModel
from vanetsim.tasks import TaskGenerator


BASELINE_POLICIES = (
    "local-only",
    "equal-split-v2v",
    "delay-greedy-v2v",
    "stage-i-only",
    "no-baseline-maintenance",
    "no-incentive",
    "static-cluster",
    "fan-2023",
    "nan-2023",
    "kumar-2023",
)


class ForwardOnlyDelayModel(DelayModel):
    """Delay model for traditional offloading baselines without result feedback."""

    def branch_delay_coefficient(
        self,
        source: VehicleState,
        helper: VehicleState,
        component: TaskComponent,
        uplink_rate: float,
        downlink_rate: float,
    ) -> float:
        resource_index = component.resource_index
        if source.vehicle_id == helper.vehicle_id:
            return component.compute_load / max(source.compute_capacity[resource_index], 1.0)
        if uplink_rate <= 0:
            return math.inf
        tx_delay = component.input_size / uplink_rate
        compute_delay = component.compute_load / max(helper.compute_capacity[resource_index], 1.0)
        return tx_delay + compute_delay


class BaselinePolicyRunner:
    """Runs comparison policies under the same vehicle state and resource pool."""

    def __init__(
        self,
        policy: str,
        mobility: MobilityConfig,
        communication: CommunicationConfig,
        offloading: OffloadingConfig,
        incentive: IncentiveConfig,
        simulation: SimulationConfig,
    ):
        self.policy = policy
        self.mobility = mobility
        self.communication = communication
        self.offloading = offloading
        self.incentive = incentive
        self.simulation = simulation
        self.cluster_manager = ClusterManager(mobility, communication)
        self.channel_model = ChannelModel(communication)
        self.task_generator = TaskGenerator(offloading, mobility)
        self.delay_model = DelayModel()
        self.forward_only_delay_model = ForwardOnlyDelayModel()
        self.safety_model = SafetyModel(mobility, simulation)
        self.stage_i_optimizer = StageIOptimizer(
            mobility=mobility,
            communication=communication,
            offloading=offloading,
            simulation=simulation,
            delay_model=self.delay_model,
            safety_model=self.safety_model,
        )
        self.forward_only_stage_i_optimizer = StageIOptimizer(
            mobility=mobility,
            communication=communication,
            offloading=offloading,
            simulation=simulation,
            delay_model=self.forward_only_delay_model,
            safety_model=self.safety_model,
        )
        self.stage_ii_optimizer = StageIIOptimizer(
            mobility=mobility,
            offloading=offloading,
            incentive=incentive,
            simulation=simulation,
            safety_model=self.safety_model,
            stage_i_optimizer=self.stage_i_optimizer,
        )
        self.helper_pressure: dict[str, float] = {}

    def optimize(self, vehicles: list[VehicleState], rng: np.random.Generator) -> AllocationResult:
        """Dispatch the current time slot to the selected comparison policy."""
        if not vehicles:
            return AllocationResult(target_speeds=np.array([], dtype=float), stage_decisions=[])

        if self.policy == "local-only":
            return self._optimize_local_only(vehicles)
        if self.policy == "equal-split-v2v":
            rates = self.channel_model.compute_rates(vehicles, rng)
            return self._optimize_heuristic(vehicles, rates, mode="equal", stage_name="equal-split")
        if self.policy == "delay-greedy-v2v":
            rates = self.channel_model.compute_rates(vehicles, rng)
            return self._optimize_heuristic(vehicles, rates, mode="greedy", stage_name="delay-greedy")
        if self.policy == "stage-i-only":
            rates = self.channel_model.compute_rates(vehicles, rng)
            return self._optimize_stage_i_only(vehicles, rates)
        if self.policy == "no-baseline-maintenance":
            rates = self.channel_model.compute_rates(vehicles, rng)
            return self._optimize_no_baseline_maintenance(vehicles, rates)
        if self.policy == "no-incentive":
            rates = self.channel_model.compute_rates(vehicles, rng)
            return self._optimize_no_incentive(vehicles, rates)
        if self.policy == "static-cluster":
            rates = self.channel_model.compute_rates(vehicles, rng)
            return self._optimize_static_cluster(vehicles, rates)
        if self.policy == "fan-2023":
            rates = self.channel_model.compute_rates(vehicles, rng)
            return self._optimize_delay_min_lp(
                vehicles,
                rates,
                optimizer=self.forward_only_stage_i_optimizer,
                stage_name="fan-2023",
            )
        if self.policy == "nan-2023":
            rates = self.channel_model.compute_rates(vehicles, rng)
            return self._optimize_delay_min_lp(
                vehicles,
                rates,
                optimizer=self.stage_i_optimizer,
                stage_name="nan-2023",
            )
        if self.policy == "kumar-2023":
            rates = self.channel_model.compute_rates(vehicles, rng)
            result = self._optimize_heuristic(
                vehicles,
                rates,
                mode="greedy",
                stage_name="kumar-2023",
                load_balance_weight=0.35,
            )
            self._update_helper_pressure(vehicles, result.stage_i)
            return result

        raise ValueError(f"Unsupported baseline policy: {self.policy}")

    def _optimize_local_only(self, vehicles: list[VehicleState]) -> AllocationResult:
        slow_indices = self.cluster_manager.identify_slow_vehicles(vehicles)
        if not slow_indices:
            return self._no_slow_result(vehicles)

        n = len(vehicles)
        k_count = self.offloading.resource_count
        allocation = np.zeros((n, n, k_count), dtype=float)
        resource_used = np.zeros((n, k_count), dtype=float)
        delay_by_vehicle: dict[str, float] = {}
        target_by_id = self._current_speed_map(vehicles)

        for source_index in slow_indices:
            vehicle = vehicles[source_index]
            total_delay = 0.0
            complete = True
            for k in range(k_count):
                compute_load = float(vehicle.task_load[k])
                slot_capacity = vehicle.compute_capacity[k] * self.simulation.step_length
                if compute_load > slot_capacity + 1e-9:
                    complete = False
                allocation[source_index, source_index, k] = 1.0 if complete else 0.0
                resource_used[source_index, k] += min(compute_load, slot_capacity)
                total_delay += compute_load / max(vehicle.compute_capacity[k], 1.0)

            if complete:
                delay_by_vehicle[vehicle.vehicle_id] = total_delay
                target_by_id[vehicle.vehicle_id] = self._speed_from_delay(vehicle, total_delay)

        stage_i = self._stage_i_output(
            vehicles=vehicles,
            slow_indices=slow_indices,
            allocation=allocation,
            resource_used=resource_used,
            delay_by_vehicle=delay_by_vehicle,
            target_by_id=target_by_id,
            payment=0.0,
            rounds=[],
        )
        decisions = self._single_stage_decisions(vehicles, slow_indices, stage_i, "local")
        return AllocationResult(
            target_speeds=self._target_array(vehicles, target_by_id),
            stage_decisions=decisions,
            stage_i=stage_i,
            stage_ii=self._empty_stage_ii(n, k_count, "disabled"),
        )

    def _optimize_heuristic(
        self,
        vehicles: list[VehicleState],
        rates: np.ndarray,
        *,
        mode: str,
        stage_name: str,
        load_balance_weight: float = 0.0,
    ) -> AllocationResult:
        slow_indices = self.cluster_manager.identify_slow_vehicles(vehicles)
        if not slow_indices:
            return self._no_slow_result(vehicles)

        tasks_by_vehicle = self.task_generator.generate_for_slow_vehicles(vehicles, slow_indices)
        n = len(vehicles)
        k_count = self.offloading.resource_count
        allocation = np.zeros((n, n, k_count), dtype=float)
        total_capacity = np.array([vehicle.compute_capacity for vehicle in vehicles], dtype=float) * self.simulation.step_length
        remaining_capacity = total_capacity.copy()
        resource_used = np.zeros((n, k_count), dtype=float)
        delay_by_vehicle: dict[str, float] = {}
        payment = 0.0
        target_by_id = self._current_speed_map(vehicles)

        for source_index in slow_indices:
            source = vehicles[source_index]
            total_delay = 0.0
            source_complete = True
            for component in tasks_by_vehicle[source.vehicle_id]:
                allocations = self._allocate_component(
                    vehicles=vehicles,
                    source_index=source_index,
                    component=component,
                    rates=rates,
                    remaining_capacity=remaining_capacity,
                    mode=mode,
                    load_balance_weight=load_balance_weight,
                )
                if sum(item[1] for item in allocations) < 1.0 - 1e-6:
                    source_complete = False

                component_delay = 0.0
                for helper_index, share, coeff in allocations:
                    k = component.resource_index
                    compute_load = share * component.compute_load
                    allocation[source_index, helper_index, k] += share
                    remaining_capacity[helper_index, k] = max(remaining_capacity[helper_index, k] - compute_load, 0.0)
                    resource_used[helper_index, k] += compute_load
                    if helper_index != source_index:
                        payment += compute_load * vehicles[helper_index].price[k]
                    component_delay = max(component_delay, coeff * share)
                total_delay += component_delay if allocations else self.mobility.system_delay_cap

            if source_complete:
                delay_by_vehicle[source.vehicle_id] = total_delay
                target_by_id[source.vehicle_id] = self._speed_from_delay(source, total_delay)

        rounds = self.cluster_manager.build_dynamic_rounds(vehicles, slow_indices)
        stage_i = self._stage_i_output(
            vehicles=vehicles,
            slow_indices=slow_indices,
            allocation=allocation,
            resource_used=resource_used,
            delay_by_vehicle=delay_by_vehicle,
            target_by_id=target_by_id,
            payment=payment,
            rounds=rounds,
        )
        decisions = self._single_stage_decisions(vehicles, slow_indices, stage_i, stage_name)
        return AllocationResult(
            target_speeds=self._target_array(vehicles, target_by_id),
            stage_decisions=decisions,
            stage_i=stage_i,
            stage_ii=self._empty_stage_ii(n, k_count, "disabled"),
        )

    def _optimize_stage_i_only(self, vehicles: list[VehicleState], rates: np.ndarray) -> AllocationResult:
        slow_indices = self.cluster_manager.identify_slow_vehicles(vehicles)
        if not slow_indices:
            return self._no_slow_result(vehicles)

        rounds = self.cluster_manager.build_dynamic_rounds(vehicles, slow_indices)
        tasks_by_vehicle = self.task_generator.generate_for_slow_vehicles(vehicles, slow_indices)
        stage_i = self.stage_i_optimizer.optimize(vehicles, rounds, tasks_by_vehicle, rates)
        target_by_id = self._current_speed_map(vehicles)
        target_by_id.update(stage_i.target_speeds)
        n = len(vehicles)
        k_count = self.offloading.resource_count
        return AllocationResult(
            target_speeds=self._target_array(vehicles, target_by_id),
            stage_decisions=self._single_stage_decisions(vehicles, slow_indices, stage_i, "stage1"),
            stage_i=stage_i,
            stage_ii=self._empty_stage_ii(n, k_count, "stage_ii_disabled"),
        )

    def _optimize_no_baseline_maintenance(self, vehicles: list[VehicleState], rates: np.ndarray) -> AllocationResult:
        slow_indices = self.cluster_manager.identify_slow_vehicles(vehicles)
        if not slow_indices:
            return self._no_slow_result(vehicles)

        n = len(vehicles)
        k_count = self.offloading.resource_count
        helper_indices = sorted(set(slow_indices).union(self._helpers_around_any_slow(vehicles, slow_indices, rates)))
        tasks_by_vehicle = self.task_generator.generate_for_slow_vehicles(vehicles, slow_indices)
        capacity = np.array([vehicle.compute_capacity for vehicle in vehicles], dtype=float) * self.simulation.step_length
        prices = np.array([vehicle.price for vehicle in vehicles], dtype=float)
        result = self.stage_i_optimizer.solve_allocation(
            vehicles=vehicles,
            source_indices=slow_indices,
            helper_indices=helper_indices,
            tasks_by_vehicle=tasks_by_vehicle,
            rates=rates,
            capacity=capacity,
            target_speed=self.mobility.min_speed,
            prices=prices,
            objective="payment",
        )

        target_by_id = self._current_speed_map(vehicles)
        if result is None:
            allocation = np.zeros((n, n, k_count), dtype=float)
            delay_by_vehicle: dict[str, float] = {}
            payment = 0.0
            baseline_feasible = False
        else:
            allocation = result["allocation"]
            delay_by_vehicle = result["delay_by_vehicle"]
            payment = result["cost"]
            baseline_feasible = True
            for index in slow_indices:
                target_by_id[vehicles[index].vehicle_id] = self.mobility.min_speed

        zero_reservation = np.zeros((n, k_count), dtype=float)
        rounds = [
            ClusterRound(
                round_index=1,
                target_speed=float(self.mobility.min_speed),
                slow_vehicle_indices=tuple(slow_indices),
                helper_indices=tuple(index for index in helper_indices if index not in slow_indices),
                cluster_indices=tuple(helper_indices),
            )
        ]
        stage_i = StageIOutput(
            allocation=allocation,
            delay_by_vehicle=delay_by_vehicle,
            payment=float(payment),
            baseline_reservation=zero_reservation,
            baseline_feasible=baseline_feasible,
            target_speeds={
                vehicles[index].vehicle_id: target_by_id[vehicles[index].vehicle_id]
                for index in slow_indices
                if vehicles[index].vehicle_id in delay_by_vehicle
            },
            rounds=rounds,
            task_completion_rate=0.0 if not slow_indices else len(delay_by_vehicle) / len(slow_indices),
        )
        stage_ii = self.stage_ii_optimizer.optimize(vehicles, slow_indices, tasks_by_vehicle, rates, stage_i)
        target_by_id.update(stage_ii.target_speeds)
        return AllocationResult(
            target_speeds=self._target_array(vehicles, target_by_id),
            stage_decisions=self._two_stage_decisions(vehicles, slow_indices, stage_i, stage_ii),
            stage_i=stage_i,
            stage_ii=stage_ii,
        )

    def _optimize_no_incentive(self, vehicles: list[VehicleState], rates: np.ndarray) -> AllocationResult:
        cloned = []
        for vehicle in vehicles:
            role = "slow" if vehicle.speed < self.mobility.min_speed else "helper"
            cloned.append(
                replace(
                    vehicle,
                    willingness=1,
                    role=role,
                    price=np.zeros_like(vehicle.price, dtype=float),
                )
            )

        slow_indices = self.cluster_manager.identify_slow_vehicles(cloned)
        if not slow_indices:
            return self._no_slow_result(vehicles)

        rounds = self.cluster_manager.build_dynamic_rounds(cloned, slow_indices)
        tasks_by_vehicle = self.task_generator.generate_for_slow_vehicles(cloned, slow_indices)
        stage_i = self.stage_i_optimizer.optimize(cloned, rounds, tasks_by_vehicle, rates)
        stage_ii = self.stage_ii_optimizer.optimize(cloned, slow_indices, tasks_by_vehicle, rates, stage_i)
        target_by_id = self._current_speed_map(vehicles)
        target_by_id.update(stage_i.target_speeds)
        target_by_id.update(stage_ii.target_speeds)
        decisions = self._two_stage_decisions(vehicles, slow_indices, stage_i, stage_ii)
        return AllocationResult(
            target_speeds=self._target_array(vehicles, target_by_id),
            stage_decisions=decisions,
            stage_i=stage_i,
            stage_ii=stage_ii,
        )

    def _optimize_static_cluster(self, vehicles: list[VehicleState], rates: np.ndarray) -> AllocationResult:
        slow_indices = self.cluster_manager.identify_slow_vehicles(vehicles)
        if not slow_indices:
            return self._no_slow_result(vehicles)

        helper_indices = self._helpers_around_any_slow(vehicles, slow_indices, rates)
        cluster_indices = tuple(sorted(set(slow_indices).union(helper_indices)))
        rounds = [
            ClusterRound(
                round_index=1,
                target_speed=float(self.mobility.min_speed),
                slow_vehicle_indices=tuple(slow_indices),
                helper_indices=tuple(helper_indices),
                cluster_indices=cluster_indices,
            )
        ]
        tasks_by_vehicle = self.task_generator.generate_for_slow_vehicles(vehicles, slow_indices)
        stage_i = self.stage_i_optimizer.optimize(vehicles, rounds, tasks_by_vehicle, rates)
        target_by_id = self._current_speed_map(vehicles)
        target_by_id.update(stage_i.target_speeds)
        n = len(vehicles)
        k_count = self.offloading.resource_count
        return AllocationResult(
            target_speeds=self._target_array(vehicles, target_by_id),
            stage_decisions=self._single_stage_decisions(vehicles, slow_indices, stage_i, "static-cluster"),
            stage_i=stage_i,
            stage_ii=self._empty_stage_ii(n, k_count, "disabled"),
        )

    def _optimize_delay_min_lp(
        self,
        vehicles: list[VehicleState],
        rates: np.ndarray,
        *,
        optimizer: StageIOptimizer,
        stage_name: str,
    ) -> AllocationResult:
        slow_indices = self.cluster_manager.identify_slow_vehicles(vehicles)
        if not slow_indices:
            return self._no_slow_result(vehicles)

        n = len(vehicles)
        k_count = self.offloading.resource_count
        helper_indices = sorted(
            set(slow_indices).union(
                self._helpers_around_any_slow(
                    vehicles,
                    slow_indices,
                    rates,
                    require_feedback=optimizer is self.stage_i_optimizer,
                )
            )
        )
        tasks_by_vehicle = self.task_generator.generate_for_slow_vehicles(vehicles, slow_indices)
        capacity = np.array([vehicle.compute_capacity for vehicle in vehicles], dtype=float) * self.simulation.step_length
        prices = np.array([vehicle.price for vehicle in vehicles], dtype=float)
        result = optimizer.solve_allocation(
            vehicles=vehicles,
            source_indices=slow_indices,
            helper_indices=helper_indices,
            tasks_by_vehicle=tasks_by_vehicle,
            rates=rates,
            capacity=capacity,
            target_speed=None,
            prices=prices,
            objective="max_delay",
        )

        target_by_id = self._current_speed_map(vehicles)
        if result is None:
            allocation = np.zeros((n, n, k_count), dtype=float)
            resource_used = np.zeros((n, k_count), dtype=float)
            delay_by_vehicle: dict[str, float] = {}
            payment = 0.0
        else:
            allocation = result["allocation"]
            resource_used = result["resource_used"]
            delay_by_vehicle = result["delay_by_vehicle"]
            payment = result["cost"]
            for vehicle in vehicles:
                if vehicle.vehicle_id in delay_by_vehicle:
                    target_by_id[vehicle.vehicle_id] = self._speed_from_delay(vehicle, delay_by_vehicle[vehicle.vehicle_id])

        rounds = [
            ClusterRound(
                round_index=1,
                target_speed=0.0,
                slow_vehicle_indices=tuple(slow_indices),
                helper_indices=tuple(index for index in helper_indices if index not in slow_indices),
                cluster_indices=tuple(helper_indices),
            )
        ]
        stage_i = self._stage_i_output(
            vehicles=vehicles,
            slow_indices=slow_indices,
            allocation=allocation,
            resource_used=resource_used,
            delay_by_vehicle=delay_by_vehicle,
            target_by_id=target_by_id,
            payment=payment,
            rounds=rounds,
        )
        return AllocationResult(
            target_speeds=self._target_array(vehicles, target_by_id),
            stage_decisions=self._single_stage_decisions(vehicles, slow_indices, stage_i, stage_name),
            stage_i=stage_i,
            stage_ii=self._empty_stage_ii(n, k_count, "disabled"),
        )

    def _allocate_component(
        self,
        *,
        vehicles: list[VehicleState],
        source_index: int,
        component: TaskComponent,
        rates: np.ndarray,
        remaining_capacity: np.ndarray,
        mode: str,
        load_balance_weight: float,
    ) -> list[tuple[int, float, float]]:
        k = component.resource_index
        candidates = self._component_candidates(vehicles, source_index, component, rates, remaining_capacity)
        if not candidates:
            return []

        if mode == "equal":
            return self._equal_split(component, candidates, remaining_capacity)

        def score(item):
            helper_index, coeff = item
            capacity = max(remaining_capacity[helper_index, k], 1.0)
            pressure = self.helper_pressure.get(vehicles[helper_index].vehicle_id, 0.0)
            load_term = load_balance_weight * (component.compute_load / capacity + pressure)
            return coeff + self.mobility.system_delay_cap * load_term

        remaining_share = 1.0
        allocations: list[tuple[int, float, float]] = []
        for helper_index, coeff in sorted(candidates, key=score):
            if remaining_share <= 1e-9:
                break
            share = min(remaining_share, remaining_capacity[helper_index, k] / max(component.compute_load, 1.0))
            if share <= 1e-9:
                continue
            allocations.append((helper_index, float(share), float(coeff)))
            remaining_share -= share
        return allocations

    def _component_candidates(
        self,
        vehicles: list[VehicleState],
        source_index: int,
        component: TaskComponent,
        rates: np.ndarray,
        remaining_capacity: np.ndarray,
    ) -> list[tuple[int, float]]:
        source = vehicles[source_index]
        candidates: list[tuple[int, float]] = []
        for helper_index, helper in enumerate(vehicles):
            if remaining_capacity[helper_index, component.resource_index] <= 1e-9:
                continue
            if helper_index != source_index:
                if helper.speed < self.mobility.min_speed or helper.willingness != 1:
                    continue
                if rates[source_index, helper_index] < self.communication.min_rate:
                    continue
                if rates[helper_index, source_index] < self.communication.min_rate:
                    continue
            coeff = self.delay_model.branch_delay_coefficient(
                source=source,
                helper=helper,
                component=component,
                uplink_rate=rates[source_index, helper_index],
                downlink_rate=rates[helper_index, source_index],
            )
            if math.isfinite(coeff):
                candidates.append((helper_index, coeff))
        return candidates

    def _equal_split(
        self,
        component: TaskComponent,
        candidates: list[tuple[int, float]],
        remaining_capacity: np.ndarray,
    ) -> list[tuple[int, float, float]]:
        k = component.resource_index
        remaining_share = 1.0
        active = list(candidates)
        allocations: list[tuple[int, float, float]] = []
        while active and remaining_share > 1e-9:
            share_per_helper = remaining_share / len(active)
            next_active = []
            allocated_this_round = 0.0
            for helper_index, coeff in active:
                capacity_share = remaining_capacity[helper_index, k] / max(component.compute_load, 1.0)
                share = min(share_per_helper, capacity_share)
                if share <= 1e-9:
                    continue
                allocations.append((helper_index, float(share), float(coeff)))
                remaining_capacity[helper_index, k] = max(
                    remaining_capacity[helper_index, k] - share * component.compute_load,
                    0.0,
                )
                allocated_this_round += share
                if capacity_share > share + 1e-9:
                    next_active.append((helper_index, coeff))
            remaining_share -= allocated_this_round
            if allocated_this_round <= 1e-9:
                break
            active = next_active

        for helper_index, share, _ in allocations:
            remaining_capacity[helper_index, k] += share * component.compute_load
        return allocations

    def _helpers_around_any_slow(
        self,
        vehicles: list[VehicleState],
        slow_indices: list[int],
        rates: np.ndarray,
        *,
        require_feedback: bool = True,
    ) -> list[int]:
        helpers = []
        slow_set = set(slow_indices)
        for index, vehicle in enumerate(vehicles):
            if index in slow_set:
                continue
            if vehicle.speed < self.mobility.min_speed or vehicle.willingness != 1:
                continue
            if any(self._link_ok(rates, slow_index, index, require_feedback=require_feedback) for slow_index in slow_indices):
                helpers.append(index)
        return helpers

    def _link_ok(self, rates: np.ndarray, source_index: int, helper_index: int, *, require_feedback: bool) -> bool:
        if rates[source_index, helper_index] < self.communication.min_rate:
            return False
        if require_feedback and rates[helper_index, source_index] < self.communication.min_rate:
            return False
        return True

    def _stage_i_output(
        self,
        *,
        vehicles: list[VehicleState],
        slow_indices: list[int],
        allocation: np.ndarray,
        resource_used: np.ndarray,
        delay_by_vehicle: dict[str, float],
        target_by_id: dict[str, float],
        payment: float,
        rounds: list[ClusterRound],
    ) -> StageIOutput:
        task_count = len(slow_indices)
        completed = sum(1 for index in slow_indices if vehicles[index].vehicle_id in delay_by_vehicle)
        completion_rate = 1.0 if task_count == 0 else completed / task_count
        target_speeds = {
            vehicles[index].vehicle_id: target_by_id.get(vehicles[index].vehicle_id, vehicles[index].speed)
            for index in slow_indices
            if vehicles[index].vehicle_id in delay_by_vehicle
        }
        return StageIOutput(
            allocation=allocation,
            delay_by_vehicle=delay_by_vehicle,
            payment=float(payment),
            baseline_reservation=resource_used,
            baseline_feasible=completed == task_count,
            target_speeds=target_speeds,
            rounds=rounds,
            task_completion_rate=completion_rate,
        )

    def _single_stage_decisions(
        self,
        vehicles: list[VehicleState],
        slow_indices: list[int],
        stage_i: StageIOutput,
        stage_name: str,
    ) -> list[StageDecision]:
        decisions: list[StageDecision] = []
        for index in slow_indices:
            vehicle = vehicles[index]
            delay = stage_i.delay_by_vehicle.get(vehicle.vehicle_id, self.mobility.system_delay_cap)
            decisions.append(
                StageDecision(
                    vehicle_id=vehicle.vehicle_id,
                    stage=stage_name,
                    delay=delay,
                    target_speed=stage_i.target_speeds.get(vehicle.vehicle_id, vehicle.speed),
                    cost=stage_i.payment,
                    slack=0.0 if vehicle.vehicle_id in stage_i.delay_by_vehicle else 1.0,
                    slack_t=0.0,
                )
            )
        return decisions

    def _two_stage_decisions(
        self,
        vehicles: list[VehicleState],
        slow_indices: list[int],
        stage_i: StageIOutput,
        stage_ii: StageIIOutput,
    ) -> list[StageDecision]:
        decisions = self._single_stage_decisions(vehicles, slow_indices, stage_i, "stage1")
        if not stage_ii.participating_helpers:
            return decisions
        for index in slow_indices:
            vehicle = vehicles[index]
            decisions.append(
                StageDecision(
                    vehicle_id=vehicle.vehicle_id,
                    stage="stage2",
                    delay=stage_i.delay_by_vehicle.get(vehicle.vehicle_id, self.mobility.system_delay_cap),
                    target_speed=stage_ii.target_speeds.get(vehicle.vehicle_id, vehicle.speed),
                    cost=stage_ii.cost,
                    slack=0.0,
                    slack_t=0.0,
                    donors=stage_ii.participating_helpers,
                )
            )
        return decisions

    def _empty_stage_ii(self, vehicle_count: int, resource_count: int, status: str) -> StageIIOutput:
        return StageIIOutput(
            boosted_speed=0.0,
            allocation=np.zeros((vehicle_count, vehicle_count, resource_count), dtype=float),
            residual_used=np.zeros((vehicle_count, resource_count), dtype=float),
            benefit=0.0,
            cost=0.0,
            participating_helpers=(),
            target_speeds={},
            status=status,
        )

    def _no_slow_result(self, vehicles: list[VehicleState]) -> AllocationResult:
        return AllocationResult(
            target_speeds=np.array([vehicle.speed for vehicle in vehicles], dtype=float),
            stage_decisions=[],
        )

    def _current_speed_map(self, vehicles: list[VehicleState]) -> dict[str, float]:
        return {vehicle.vehicle_id: vehicle.speed for vehicle in vehicles}

    def _target_array(self, vehicles: list[VehicleState], target_by_id: dict[str, float]) -> np.ndarray:
        return np.array([target_by_id.get(vehicle.vehicle_id, vehicle.speed) for vehicle in vehicles], dtype=float)

    def _speed_from_delay(self, vehicle: VehicleState, delay: float) -> float:
        safe_speed = self.safety_model.max_safe_speed(delay, vehicle)
        return max(vehicle.speed, min(vehicle.desired_speed, vehicle.speed_limit, safe_speed))

    def _update_helper_pressure(self, vehicles: list[VehicleState], stage_i: StageIOutput | None) -> None:
        if stage_i is None:
            return
        total_capacity = np.array([vehicle.compute_capacity for vehicle in vehicles], dtype=float) * self.simulation.step_length
        for index, vehicle in enumerate(vehicles):
            usage = float(np.sum(stage_i.baseline_reservation[index, :]))
            capacity = float(np.sum(total_capacity[index, :]))
            utilization = usage / max(capacity, 1.0)
            old_value = self.helper_pressure.get(vehicle.vehicle_id, 0.0)
            self.helper_pressure[vehicle.vehicle_id] = 0.7 * old_value + 0.3 * utilization
