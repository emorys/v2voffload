from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def _default_bbox() -> list[float]:
    # Demo corridor around the Beijing-Tibet Expressway (G6).
    return [116.3450, 39.9900, 116.3710, 40.0260]


def _default_road_types() -> dict[str, list[str]]:
    return {"highway": ["motorway", "motorway_link", "trunk", "trunk_link"]}


def _default_keep_edge_types() -> list[str]:
    return [
        "highway.motorway",
        "highway.motorway_link",
        "highway.trunk",
        "highway.trunk_link",
    ]


@dataclass
class MapBuildConfig:
    output_dir: str = "scenarios/beijing_g6_demo"
    prefix: str = "beijing_g6"
    place_query: str | None = "G6 Expressway Beijing"
    bbox: list[float] | None = field(default_factory=_default_bbox)
    highd_location_id: int | None = None
    segment_label: str | None = None
    segment_note: str | None = None
    road_types: dict[str, list[str]] = field(default_factory=_default_road_types)
    keep_edge_types: list[str] = field(default_factory=_default_keep_edge_types)
    remove_geometry: bool = True
    join_junctions: bool = True
    guess_ramps: bool = True


@dataclass
class RouteBuildConfig:
    begin: float = 0.0
    end: float = 600.0
    period: float = 2.0
    fringe_factor: float = 20.0
    seed: int = 7
    validate: bool = True


@dataclass
class DatasetConfig:
    # highD is used as trajectory replay input. It provides motion states, but
    # does not provide compute capacity, task load, resource prices, or willingness.
    type: str = "none"
    root: str = "data/highD"
    location_id: int | None = None
    recording_id: int = 1
    start_frame: int = 0
    end_frame: int | None = 250
    frame_step: int = 5
    driving_direction: int | None = None
    lane_ids: list[int] | None = None
    max_vehicles: int = 0
    center_positions: bool = True


@dataclass
class MobilityConfig:
    # Speeds are in m/s. min_speed is v_l in the model; 20 m/s is about 72 km/h.
    # speed_limit defaults to 33.33 m/s, about 120 km/h for a highway scenario.
    safe_distance: float = 80.0
    acceleration: float = 7.5
    system_delay_cap: float = 0.25
    min_speed: float = 16.0
    speed_limit: float = 33.33
    # max_acceleration prevents optimized speeds from being applied as an
    # instantaneous jump in SUMO/demo speed updates.
    max_acceleration: float = 2.5
    segment_length: float = 1000.0


@dataclass
class CommunicationConfig:
    # A compact V2V wireless model. max_distance is also the clustering radius r.
    bandwidth: float = 10e6
    tx_power: float = 0.1
    noise_density: float = 1e-9
    pathloss_exp: float = 2.5
    pathloss_const: float = 1e-3
    max_distance: float = 200.0
    min_rate: float = 1e5


@dataclass
class OffloadingConfig:
    resource_names: list[str] = field(default_factory=lambda: ["GPU", "TPU", "CPU"])
    component_types: list[str] = field(default_factory=lambda: ["perception", "processing", "planning"])
    base_compute: list[float] = field(default_factory=lambda: [12e9, 10e9, 6e9])
    base_fraction: float = 0.5
    base_price: list[float] = field(default_factory=lambda: [2.0e-9, 2.5e-9, 1.0e-9])
    base_lambda_bits: list[float] = field(default_factory=lambda: [40e6, 10e6, 5e6])
    base_cycles_per_bit: list[float] = field(default_factory=lambda: [800.0, 1500.0, 2200.0])
    # task_load_scale keeps the sample problem numerically small enough for quick
    # cvxpy solves while preserving relative workload differences across tasks.
    task_load_scale: float = 0.005
    output_size_ratio: float = 0.05
    penalty_s: float = 1e6
    penalty_t: float = 1e4
    stage2_speed_grid: int = 8

    @property
    def resource_count(self) -> int:
        return len(self.resource_names)


@dataclass
class IncentiveConfig:
    # Stage-II compares travel-time savings against residual-resource opportunity cost.
    value_of_time: float = 1.0
    opportunity_cost_lambda: float = 1.0


@dataclass
class SimulationConfig:
    mode: str = "sumo"
    policy: str = "ours"
    steps: int = 60
    seed: int = 7
    step_length: float = 0.1
    max_vehicles: int = 0


@dataclass
class ScenarioConfig:
    name: str = "beijing_g6_highway_demo"
    map: MapBuildConfig = field(default_factory=MapBuildConfig)
    route: RouteBuildConfig = field(default_factory=RouteBuildConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    offloading: OffloadingConfig = field(default_factory=OffloadingConfig)
    incentive: IncentiveConfig = field(default_factory=IncentiveConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    source_path: Path | None = None

    def artifact_dir(self, workspace_root: Path) -> Path:
        return (workspace_root / self.map.output_dir).resolve()

    def default_sumocfg_path(self, workspace_root: Path) -> Path:
        return self.artifact_dir(workspace_root) / "highway.sumocfg"


def _coerce_map_config(data: dict) -> MapBuildConfig:
    config = MapBuildConfig()
    for key, value in data.items():
        setattr(config, key, value)
    return config


def _coerce_route_config(data: dict) -> RouteBuildConfig:
    config = RouteBuildConfig()
    for key, value in data.items():
        setattr(config, key, value)
    return config


def _coerce_dataset_config(data: dict) -> DatasetConfig:
    config = DatasetConfig()
    for key, value in data.items():
        setattr(config, key, value)
    return config


def _coerce_mobility_config(data: dict) -> MobilityConfig:
    config = MobilityConfig()
    for key, value in data.items():
        setattr(config, key, value)
    return config


def _coerce_communication_config(data: dict) -> CommunicationConfig:
    config = CommunicationConfig()
    for key, value in data.items():
        setattr(config, key, value)
    return config


def _coerce_offloading_config(data: dict) -> OffloadingConfig:
    config = OffloadingConfig()
    for key, value in data.items():
        setattr(config, key, value)
    return config


def _coerce_incentive_config(data: dict) -> IncentiveConfig:
    config = IncentiveConfig()
    for key, value in data.items():
        setattr(config, key, value)
    return config


def _coerce_simulation_config(data: dict) -> SimulationConfig:
    config = SimulationConfig()
    for key, value in data.items():
        setattr(config, key, value)
    return config


def load_scenario_config(config_path: str | Path) -> ScenarioConfig:
    path = Path(config_path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))

    scenario = ScenarioConfig(
        name=data.get("name", ScenarioConfig().name),
        map=_coerce_map_config(data.get("map", {})),
        route=_coerce_route_config(data.get("route", {})),
        dataset=_coerce_dataset_config(data.get("dataset", {})),
        mobility=_coerce_mobility_config(data.get("mobility", {})),
        communication=_coerce_communication_config(data.get("communication", {})),
        offloading=_coerce_offloading_config(data.get("offloading", {})),
        incentive=_coerce_incentive_config(data.get("incentive", {})),
        simulation=_coerce_simulation_config(data.get("simulation", {})),
        source_path=path,
    )
    return scenario
