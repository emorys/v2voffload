from __future__ import annotations

import numpy as np

from vanetsim.channel import ChannelModel
from vanetsim.clustering import ClusterManager
from vanetsim.config import CommunicationConfig, IncentiveConfig, MobilityConfig, OffloadingConfig, SimulationConfig
from vanetsim.delay import DelayModel
from vanetsim.domain import AllocationResult, StageDecision, VehicleState
from vanetsim.optimization.stage_i import StageIOptimizer
from vanetsim.optimization.stage_ii import StageIIOptimizer
from vanetsim.safety import SafetyModel
from vanetsim.tasks import TaskGenerator


class IncentiveManager:
    """协调论文式需求中描述的两阶段激励优化模型。

    该类是主要算法入口：识别慢车、构建动态簇、生成任务、计算 V2V 速率、
    求解 Stage-I，并在剩余资源上尝试 Stage-II。
    """

    def __init__(
        self,
        mobility: MobilityConfig,
        communication: CommunicationConfig,
        offloading: OffloadingConfig,
        incentive: IncentiveConfig,
        simulation: SimulationConfig,
    ):
        """初始化聚类、信道、任务生成、时延、安全和两阶段优化组件。"""
        self.mobility = mobility
        self.communication = communication
        self.offloading = offloading
        self.simulation = simulation
        self.cluster_manager = ClusterManager(mobility, communication)
        self.channel_model = ChannelModel(communication)
        self.task_generator = TaskGenerator(offloading, mobility)
        self.delay_model = DelayModel()
        self.safety_model = SafetyModel(mobility, simulation)
        self.stage_i_optimizer = StageIOptimizer(
            mobility=mobility,
            communication=communication,
            offloading=offloading,
            simulation=simulation,
            delay_model=self.delay_model,
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

    def optimize(self, vehicles: list[VehicleState], rng: np.random.Generator) -> AllocationResult:
        """对当前时间片车辆状态执行完整两阶段激励优化流程。"""
        if not vehicles:
            return AllocationResult(target_speeds=np.array([], dtype=float), stage_decisions=[])

        slow_indices = self.cluster_manager.identify_slow_vehicles(vehicles)
        if not slow_indices:
            return AllocationResult(
                target_speeds=np.array([vehicle.speed for vehicle in vehicles], dtype=float),
                stage_decisions=[],
            )

        # b1, b2, ..., bM are encoded by slow_indices sorted from lowest speed upward.
        rounds = self.cluster_manager.build_dynamic_rounds(vehicles, slow_indices)
        tasks_by_vehicle = self.task_generator.generate_for_slow_vehicles(vehicles, slow_indices)
        rates = self.channel_model.compute_rates(vehicles, rng)
        stage_i = self.stage_i_optimizer.optimize(vehicles, rounds, tasks_by_vehicle, rates)
        stage_ii = self.stage_ii_optimizer.optimize(vehicles, slow_indices, tasks_by_vehicle, rates, stage_i)

        target_by_id = {vehicle.vehicle_id: vehicle.speed for vehicle in vehicles}
        for vehicle_id, target_speed in stage_i.target_speeds.items():
            target_by_id[vehicle_id] = max(target_by_id[vehicle_id], target_speed)
        # Stage-II target speeds are applied only when at least one helper passes
        # the individual-rationality check. Otherwise Stage-II is reported as skipped.
        for vehicle_id, target_speed in stage_ii.target_speeds.items():
            target_by_id[vehicle_id] = max(target_by_id[vehicle_id], target_speed)

        delay_by_vehicle = dict(stage_i.delay_by_vehicle)
        stage_decisions = self._build_stage_decisions(vehicles, slow_indices, stage_i, stage_ii, delay_by_vehicle)
        target_speeds = np.array([target_by_id[vehicle.vehicle_id] for vehicle in vehicles], dtype=float)
        return AllocationResult(
            target_speeds=target_speeds,
            stage_decisions=stage_decisions,
            stage_i=stage_i,
            stage_ii=stage_ii,
        )

    def _build_stage_decisions(self, vehicles, slow_indices, stage_i, stage_ii, delay_by_vehicle):
        """将 Stage-I/Stage-II 输出整理成便于打印和统计的车辆决策列表。"""
        decisions: list[StageDecision] = []
        for index in slow_indices:
            vehicle = vehicles[index]
            delay = delay_by_vehicle.get(vehicle.vehicle_id, self.mobility.system_delay_cap)
            decisions.append(
                StageDecision(
                    vehicle_id=vehicle.vehicle_id,
                    stage="stage1",
                    delay=delay,
                    target_speed=stage_i.target_speeds.get(vehicle.vehicle_id, vehicle.speed),
                    cost=stage_i.payment,
                    slack=0.0 if stage_i.baseline_feasible else 1.0,
                    slack_t=0.0,
                )
            )

        if stage_ii.participating_helpers:
            for index in slow_indices:
                vehicle = vehicles[index]
                decisions.append(
                    StageDecision(
                        vehicle_id=vehicle.vehicle_id,
                        stage="stage2",
                        delay=delay_by_vehicle.get(vehicle.vehicle_id, self.mobility.system_delay_cap),
                        target_speed=stage_ii.target_speeds.get(vehicle.vehicle_id, vehicle.speed),
                        cost=stage_ii.cost,
                        slack=0.0,
                        slack_t=0.0,
                        donors=stage_ii.participating_helpers,
                    )
                )
        return decisions
