from __future__ import annotations

import numpy as np

from vanetsim.domain import AllocationResult, SimulationMetrics, VehicleState


class MetricsCollector:
    """收集每个时间片的仿真指标，并提供跨时间片平均摘要。"""

    def __init__(self):
        """初始化指标历史缓存。"""
        self.history: list[SimulationMetrics] = []

    def collect(self, time_slot: int, vehicles: list[VehicleState], result: AllocationResult) -> SimulationMetrics:
        """从车辆状态和优化结果中计算当前时间片的核心性能指标。"""
        speeds = np.array([vehicle.speed for vehicle in vehicles], dtype=float)
        target_speeds = (
            np.array(result.target_speeds, dtype=float)
            if len(result.target_speeds) == len(vehicles)
            else speeds.copy()
        )
        speed_gains = np.maximum(target_speeds - speeds, 0.0)
        stage_i = result.stage_i
        stage_ii = result.stage_ii

        latencies = np.array(list(stage_i.delay_by_vehicle.values()), dtype=float) if stage_i else np.array([])
        total_capacity = np.sum([np.sum(vehicle.compute_capacity) for vehicle in vehicles])
        used_resource = 0.0
        if stage_i is not None:
            used_resource += float(np.sum(stage_i.baseline_reservation))
        if stage_ii is not None:
            used_resource += float(np.sum(stage_ii.residual_used))
        per_vehicle_usage = np.zeros(len(vehicles), dtype=float)
        if stage_i is not None:
            per_vehicle_usage += np.sum(stage_i.baseline_reservation, axis=1)
        if stage_ii is not None:
            per_vehicle_usage += np.sum(stage_ii.residual_used, axis=1)
        if len(per_vehicle_usage) and np.sum(per_vehicle_usage * per_vehicle_usage) > 0:
            helper_load_jain_index = float(np.sum(per_vehicle_usage) ** 2 / (len(per_vehicle_usage) * np.sum(per_vehicle_usage * per_vehicle_usage)))
        else:
            helper_load_jain_index = 1.0

        helper_count = sum(1 for vehicle in vehicles if vehicle.role == "helper")
        participating_count = len(stage_ii.participating_helpers) if stage_ii else 0
        total_payment = stage_i.payment if stage_i else 0.0
        stage_ii_benefit = stage_ii.benefit if stage_ii else 0.0
        stage_ii_cost = stage_ii.cost if stage_ii else 0.0
        total_cost = total_payment + stage_ii_cost
        total_speed_gain = float(np.sum(speed_gains))
        metrics = SimulationMetrics(
            time_slot=time_slot,
            min_speed=float(np.min(speeds)) if len(speeds) else 0.0,
            avg_speed=float(np.mean(speeds)) if len(speeds) else 0.0,
            throughput=len(vehicles),
            total_payment=total_payment,
            stageII_benefit=stage_ii_benefit,
            task_completion_rate=stage_i.task_completion_rate if stage_i else 1.0,
            avg_latency=float(np.mean(latencies)) if len(latencies) else 0.0,
            max_latency=float(np.max(latencies)) if len(latencies) else 0.0,
            p95_latency=float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
            resource_utilization=float(used_resource / max(total_capacity, 1.0)),
            helper_participation_rate=float(participating_count / max(helper_count, 1)),
            cluster_reconfiguration_count=len(stage_i.rounds) if stage_i else 0,
            min_target_speed=float(np.min(target_speeds)) if len(target_speeds) else 0.0,
            avg_target_speed=float(np.mean(target_speeds)) if len(target_speeds) else 0.0,
            avg_speed_gain=float(np.mean(speed_gains)) if len(speed_gains) else 0.0,
            total_cost=float(total_cost),
            social_welfare=float(stage_ii_benefit - total_cost),
            unit_speed_gain_cost=float(total_cost / max(total_speed_gain, 1e-9)),
            stageII_cost=float(stage_ii_cost),
            helper_load_jain_index=helper_load_jain_index,
            traffic_flow_proxy=float(len(vehicles) * np.mean(speeds)) if len(speeds) else 0.0,
        )
        self.history.append(metrics)
        return metrics

    def summary(self) -> dict[str, float]:
        """对已收集的时间片指标按字段求平均，生成仿真摘要。"""
        if not self.history:
            return {}
        fields = [
            "min_speed",
            "avg_speed",
            "throughput",
            "total_payment",
            "stageII_benefit",
            "task_completion_rate",
            "avg_latency",
            "max_latency",
            "p95_latency",
            "resource_utilization",
            "helper_participation_rate",
            "cluster_reconfiguration_count",
            "min_target_speed",
            "avg_target_speed",
            "avg_speed_gain",
            "total_cost",
            "social_welfare",
            "unit_speed_gain_cost",
            "stageII_cost",
            "helper_load_jain_index",
            "traffic_flow_proxy",
        ]
        return {
            field: float(np.mean([getattr(metrics, field) for metrics in self.history]))
            for field in fields
        }
