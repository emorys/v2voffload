from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TaskComponent:
    """One resource-specific component of a vehicle task."""

    task_id: str
    vehicle_id: str
    component_type: str
    resource_index: int
    compute_load: float
    input_size: float
    output_size: float
    deadline: float
    splitable: bool = True


@dataclass(frozen=True)
class VehicleState:
    """Per-slot vehicle mobility, resource, price, and participation state."""

    vehicle_id: str
    position: tuple[float, float]
    progress: float
    speed: float
    acceleration: float
    lane_id: str
    compute_capacity: np.ndarray
    task_load: np.ndarray
    willingness: int
    role: str
    desired_speed: float
    speed_limit: float
    price: np.ndarray
    value_of_time: float
    task_input_bits: np.ndarray | None = None
    task_output_bits: np.ndarray | None = None
    task_deadlines: np.ndarray | None = None
    task_profile_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class VehicleSnapshot:
    """Raw vehicle mobility snapshot loaded from SUMO or highD."""

    vehicle_id: str
    x: float
    y: float
    progress: float
    speed: float
    desired_speed: float
    lane_id: str
    allowed_speed: float
    max_speed: float
    acceleration: float = 0.0


@dataclass(frozen=True)
class StageDecision:
    """Decision result for one slow vehicle in Stage-I or Stage-II."""

    vehicle_id: str
    stage: str
    delay: float
    target_speed: float
    cost: float
    slack: float
    slack_t: float
    donors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClusterRound:
    """One dynamic clustering round with served slow vehicles and helpers."""

    round_index: int
    target_speed: float
    slow_vehicle_indices: tuple[int, ...]
    helper_indices: tuple[int, ...]
    cluster_indices: tuple[int, ...]


@dataclass(frozen=True)
class StageIOutput:
    """Stage-I baseline reservation, payment, delay, and target speed output."""

    allocation: np.ndarray
    delay_by_vehicle: dict[str, float]
    payment: float
    baseline_reservation: np.ndarray
    baseline_feasible: bool
    target_speeds: dict[str, float]
    rounds: list[ClusterRound]
    task_completion_rate: float


@dataclass(frozen=True)
class StageIIOutput:
    """Stage-II residual-resource boosting result and incentive accounting."""

    boosted_speed: float
    allocation: np.ndarray
    residual_used: np.ndarray
    benefit: float
    cost: float
    participating_helpers: tuple[str, ...]
    target_speeds: dict[str, float]
    status: str = "ok"
    objective_value: float = 0.0
    candidate_helpers: tuple[str, ...] = ()
    helper_utilities: dict[str, float] = field(default_factory=dict)
    evaluated_speed_count: int = 0
    feasible_speed_count: int = 0


@dataclass(frozen=True)
class SimulationMetrics:
    """Metrics collected for one simulation time slot."""

    time_slot: int
    min_speed: float
    avg_speed: float
    throughput: int
    total_payment: float
    stageII_benefit: float
    task_completion_rate: float
    avg_latency: float
    max_latency: float
    p95_latency: float
    resource_utilization: float
    helper_participation_rate: float
    cluster_reconfiguration_count: int
    min_target_speed: float = 0.0
    avg_target_speed: float = 0.0
    avg_speed_gain: float = 0.0
    total_cost: float = 0.0
    social_welfare: float = 0.0
    unit_speed_gain_cost: float = 0.0
    stageII_cost: float = 0.0
    helper_load_jain_index: float = 0.0
    traffic_flow_proxy: float = 0.0


@dataclass(frozen=True)
class AllocationResult:
    """Optimization output containing target speeds, decisions, and metrics."""

    target_speeds: np.ndarray
    stage_decisions: list[StageDecision]
    stage_i: StageIOutput | None = None
    stage_ii: StageIIOutput | None = None
    metrics: SimulationMetrics | None = None
