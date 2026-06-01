from __future__ import annotations

from pathlib import Path

import numpy as np

from vanetsim.config import ScenarioConfig
from vanetsim.datasets import HighDLoader
from vanetsim.integrations import (
    apply_target_speeds,
    build_sumo_command,
    collect_vehicle_snapshots,
    load_traci,
)
from vanetsim.metrics import MetricsCollector
from vanetsim.optimization import build_policy_manager, normalize_policy
from vanetsim.safety import SafetyModel
from vanetsim.state import VehicleStateManager


class Simulator:
    """负责各运行模式下的时间片循环。

    demo、highD 和 SUMO 的差别主要在车辆状态来源；后续的慢车识别、
    激励优化、速度限制和指标采集流程保持一致。
    """

    def __init__(self, scenario: ScenarioConfig):
        """根据场景配置初始化状态管理器、激励优化器、安全模型和指标采集器。"""
        self.scenario = scenario
        self.policy = normalize_policy(getattr(scenario.simulation, "policy", "ours"))
        self.scenario.simulation.policy = self.policy
        self.vehicle_state_manager = VehicleStateManager(
            mobility=scenario.mobility,
            offloading=scenario.offloading,
            incentive=scenario.incentive,
            simulation=scenario.simulation,
        )
        self.incentive_manager = build_policy_manager(
            self.policy,
            mobility=scenario.mobility,
            communication=scenario.communication,
            offloading=scenario.offloading,
            incentive=scenario.incentive,
            simulation=scenario.simulation,
        )
        self.safety_model = SafetyModel(scenario.mobility, scenario.simulation)
        self.metrics = MetricsCollector()

    def run_demo(self, steps: int | None = None) -> None:
        """使用内置合成车辆状态运行闭环演示仿真。"""
        steps = self.scenario.simulation.steps if steps is None else steps
        rng = np.random.default_rng(self.scenario.simulation.seed)
        vehicles = self.vehicle_state_manager.build_demo_states()

        for step in range(steps):
            # One slot of the closed loop:
            # state table -> slow vehicles/clusters -> Stage-I/Stage-II -> speed update.
            result = self.incentive_manager.optimize(vehicles, rng)
            target_by_id = self._limited_target_speeds(vehicles, result)
            metrics = self.metrics.collect(step, vehicles, result)
            self._print_step(step, vehicles, result, metrics)
            vehicles = self.vehicle_state_manager.advance_demo_states(vehicles, target_by_id)

        self._print_summary()

    def run_highd(self, steps: int | None = None, workspace_root: str | Path | None = None) -> None:
        """读取 highD 轨迹帧并在每个帧上运行资源激励优化。"""
        loader = HighDLoader(self.scenario.dataset, workspace_root=workspace_root or Path.cwd())
        frames = loader.load_frames()
        if steps is not None:
            frames = frames[:steps]

        rng = np.random.default_rng(self.scenario.simulation.seed)
        for slot, highd_frame in enumerate(frames):
            # highD provides measured trajectories. We use them as the physical
            # state source and run the VANET resource/incentive model on top.
            vehicles = self.vehicle_state_manager.build_from_snapshots(
                highd_frame.snapshots,
                max_vehicles=self.scenario.dataset.max_vehicles,
            )
            result = self.incentive_manager.optimize(vehicles, rng)
            metrics = self.metrics.collect(slot, vehicles, result)
            self._print_step(slot, vehicles, result, metrics)

        if not frames:
            print("No highD frames matched the configured frame range, direction, or lanes.")
        self._print_summary()

    def run_sumo(
        self,
        *,
        sumocfg_path: str | Path,
        gui: bool = False,
        max_vehicles: int = 0,
        explicit_sumo_home: str | Path | None = None,
        steps: int | None = None,
    ) -> None:
        """启动 SUMO/TraCI 联动仿真，并将优化速度实时写回车辆。"""
        steps = self.scenario.simulation.steps if steps is None else steps
        traci, _, _ = load_traci(explicit_sumo_home)
        command = build_sumo_command(
            sumocfg_path=Path(sumocfg_path).resolve(),
            step_length=self.scenario.simulation.step_length,
            seed=self.scenario.simulation.seed,
            gui=gui,
            explicit_home=explicit_sumo_home,
        )

        rng = np.random.default_rng(self.scenario.simulation.seed)
        traci.start(command)
        try:
            for step in range(steps):
                traci.simulationStep()
                snapshots = collect_vehicle_snapshots(traci)
                if not snapshots and traci.simulation.getMinExpectedNumber() <= 0:
                    print(f"\n=== Time slot {step} ===")
                    print("No active vehicles. SUMO simulation finished.")
                    break

                vehicles = self.vehicle_state_manager.build_from_snapshots(snapshots, max_vehicles=max_vehicles)
                # SUMO mode is closed-loop: optimized speeds are written back to TraCI
                # after safety and acceleration limits are applied.
                result = self.incentive_manager.optimize(vehicles, rng)
                target_by_id = self._limited_target_speeds(vehicles, result)
                metrics = self.metrics.collect(step, vehicles, result)
                self._print_step(step, vehicles, result, metrics)
                apply_target_speeds(traci, list(target_by_id.keys()), list(target_by_id.values()))
        finally:
            traci.close()

        self._print_summary()

    def _limited_target_speeds(self, vehicles, result) -> dict[str, float]:
        """结合安全时延和加速度限制，得到每辆车可执行的目标速度。"""
        targets: dict[str, float] = {}
        for index, vehicle in enumerate(vehicles):
            requested_speed = float(result.target_speeds[index]) if index < len(result.target_speeds) else vehicle.speed
            delay = self.scenario.mobility.system_delay_cap
            if result.stage_i and vehicle.vehicle_id in result.stage_i.delay_by_vehicle:
                delay = result.stage_i.delay_by_vehicle[vehicle.vehicle_id]
            if vehicle.role == "slow":
                targets[vehicle.vehicle_id] = self.safety_model.update_speed(vehicle, delay, requested_speed)
            else:
                targets[vehicle.vehicle_id] = min(vehicle.speed_limit, max(vehicle.speed, requested_speed))
        return targets

    def _print_step(self, step: int, vehicles, result, metrics) -> None:
        """打印单个时间片的阶段决策和核心指标。"""
        print(f"\n=== Time slot {step} ===")
        slow_ids = [vehicle.vehicle_id for vehicle in vehicles if vehicle.role == "slow"]
        helper_ids = [vehicle.vehicle_id for vehicle in vehicles if vehicle.role == "helper"]
        print(
            f"iteration setup: policy={self.policy}, vehicles={len(vehicles)}, "
            f"slow={slow_ids or []}, helpers={len(helper_ids)}, "
            f"v_l={self.scenario.mobility.min_speed:.2f}, "
            f"lambda={self.scenario.incentive.opportunity_cost_lambda:.2f}, "
            f"stage2_grid={self.scenario.offloading.stage2_speed_grid}"
        )
        print("optimization objectives: stage1=min payment under delay/capacity constraints, stage2=max benefit-lambda*residual_cost")
        if not result.stage_decisions:
            print("No slow vehicles below v_l in this step.")
        self._print_rounds(vehicles, result)
        for decision in result.stage_decisions:
            if decision.stage == "stage1":
                feasible = result.stage_i.baseline_feasible if result.stage_i else False
                print(
                    f"{decision.vehicle_id} stage1: target={decision.target_speed:.2f} m/s, "
                    f"T={decision.delay:.4f}s, payment={decision.cost:.4e}, feasible={feasible}"
                )
            elif decision.stage == "stage2" and result.stage_ii:
                print(
                    f"{decision.vehicle_id} stage2: boosted={decision.target_speed:.2f} m/s, "
                    f"helpers={list(decision.donors)}, benefit={result.stage_ii.benefit:.4e}"
                )
            else:
                print(
                    f"{decision.vehicle_id} {decision.stage}: target={decision.target_speed:.2f} m/s, "
                    f"T={decision.delay:.4f}s, cost={decision.cost:.4e}, slack={decision.slack:.2f}"
                )
        if (
            result.stage_i
            and result.stage_i.baseline_feasible
            and result.stage_ii
            and result.stage_ii.status not in {"ok", "disabled", "stage_ii_disabled"}
        ):
            print(f"stage2 skipped: {result.stage_ii.status}")
        self._print_allocation("stage1", vehicles, result.stage_i.allocation if result.stage_i else None)
        self._print_resource_usage(
            "stage1",
            vehicles,
            result.stage_i.baseline_reservation if result.stage_i else None,
        )
        self._print_stage2_detail(vehicles, result)
        self._print_speed_gains(vehicles, result)
        print(
            f"metrics: min_v={metrics.min_speed:.2f}, avg_v={metrics.avg_speed:.2f}, "
            f"target_min_v={metrics.min_target_speed:.2f}, avg_gain={metrics.avg_speed_gain:.2f}, "
            f"completion={metrics.task_completion_rate:.2f}, util={metrics.resource_utilization:.3f}, "
            f"cost={metrics.total_cost:.4e}, welfare={metrics.social_welfare:.4e}, "
            f"unit_gain_cost={metrics.unit_speed_gain_cost:.4e}"
        )

    def _print_rounds(self, vehicles, result) -> None:
        stage_i = result.stage_i
        if not stage_i or not stage_i.rounds:
            print("task assignment rounds: none")
            return
        print("task assignment rounds:")
        for round_state in stage_i.rounds:
            slow_ids = [vehicles[index].vehicle_id for index in round_state.slow_vehicle_indices]
            helper_ids = [vehicles[index].vehicle_id for index in round_state.helper_indices]
            cluster_ids = [vehicles[index].vehicle_id for index in round_state.cluster_indices]
            print(
                f"  round {round_state.round_index}: target={round_state.target_speed:.2f} m/s, "
                f"slow={slow_ids}, helpers={helper_ids}, cluster={cluster_ids}"
            )

    def _print_allocation(self, label: str, vehicles, allocation) -> None:
        if allocation is None or getattr(allocation, "size", 0) == 0:
            print(f"{label} allocation: none")
            return
        printed = False
        print(f"{label} allocation ratios by source task component:")
        for source_index, source in enumerate(vehicles):
            for resource_index in range(allocation.shape[2]):
                entries = []
                for helper_index, helper in enumerate(vehicles):
                    share = float(allocation[source_index, helper_index, resource_index])
                    if share <= 1e-4:
                        continue
                    load = 0.0
                    if resource_index < len(source.task_load):
                        load = share * float(source.task_load[resource_index])
                    entries.append(f"{helper.vehicle_id}:{share:.2f}/{load:.2e}")
                if entries:
                    printed = True
                    print(f"  {source.vehicle_id} r{resource_index}: " + ", ".join(entries))
        if not printed:
            print("  no positive allocation")

    def _print_resource_usage(self, label: str, vehicles, resource_used) -> None:
        if resource_used is None or getattr(resource_used, "size", 0) == 0:
            print(f"{label} resource use: none")
            return
        total_capacity = np.array([vehicle.compute_capacity for vehicle in vehicles], dtype=float) * self.scenario.simulation.step_length
        printed = False
        print(f"{label} helper resource use and ratios:")
        for index, vehicle in enumerate(vehicles):
            used_total = float(np.sum(resource_used[index, :]))
            if used_total <= 1e-6:
                continue
            cap_total = float(np.sum(total_capacity[index, :]))
            resource_parts = []
            for resource_index in range(resource_used.shape[1]):
                used = float(resource_used[index, resource_index])
                cap = float(total_capacity[index, resource_index])
                if used > 1e-6:
                    resource_parts.append(f"r{resource_index}={used:.2e}/{cap:.2e}({used / max(cap, 1.0):.2%})")
            printed = True
            print(
                f"  {vehicle.vehicle_id}: total={used_total:.2e}/{cap_total:.2e}"
                f"({used_total / max(cap_total, 1.0):.2%}), " + ", ".join(resource_parts)
            )
        if not printed:
            print("  no resource consumed")

    def _print_stage2_detail(self, vehicles, result) -> None:
        stage_ii = result.stage_ii
        if not stage_ii:
            print("stage2 incentive: not run")
            return
        print(
            f"stage2 incentive: status={stage_ii.status}, candidates={list(stage_ii.candidate_helpers)}, "
            f"speed_grid={stage_ii.feasible_speed_count}/{stage_ii.evaluated_speed_count}, "
            f"boosted={stage_ii.boosted_speed:.2f} m/s, benefit={stage_ii.benefit:.4e}, "
            f"residual_cost={stage_ii.cost:.4e}, utility={stage_ii.objective_value:.4e}, "
            f"participants={list(stage_ii.participating_helpers)}"
        )
        if stage_ii.helper_utilities:
            utilities = ", ".join(
                f"{vehicle_id}:{utility:.4e}"
                for vehicle_id, utility in sorted(stage_ii.helper_utilities.items())
            )
            print(f"stage2 individual utilities: {utilities}")
        self._print_allocation("stage2", vehicles, stage_ii.allocation)
        self._print_resource_usage("stage2 residual", vehicles, stage_ii.residual_used)

    def _print_speed_gains(self, vehicles, result) -> None:
        if len(result.target_speeds) != len(vehicles):
            print("speed gains: unavailable")
            return
        entries = []
        for index, vehicle in enumerate(vehicles):
            target = float(result.target_speeds[index])
            gain = max(target - vehicle.speed, 0.0)
            if gain > 1e-6 or vehicle.role == "slow":
                entries.append(f"{vehicle.vehicle_id}:{vehicle.speed:.2f}->{target:.2f}(+{gain:.2f})")
        print("speed gains: " + (", ".join(entries) if entries else "none"))

    def _print_summary(self) -> None:
        """打印整个仿真运行结束后的平均指标摘要。"""
        summary = self.metrics.summary()
        if not summary:
            return
        print(f"\n=== Summary ({self.policy}) ===")
        print(
            f"avg_speed={summary['avg_speed']:.2f}, min_speed={summary['min_speed']:.2f}, "
            f"payment={summary['total_payment']:.4e}, stageII_benefit={summary['stageII_benefit']:.4e}, "
            f"avg_latency={summary['avg_latency']:.4f}, completion={summary['task_completion_rate']:.2f}, "
            f"target_min_v={summary['min_target_speed']:.2f}, welfare={summary['social_welfare']:.4e}"
        )
