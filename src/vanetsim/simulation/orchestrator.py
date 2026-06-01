from __future__ import annotations

from pathlib import Path

from vanetsim.config import ScenarioConfig
from vanetsim.map import build_highway_scenario
from vanetsim.simulation.simulator import Simulator


class HighwaySimulationOrchestrator:
    """连接场景构建、轨迹回放和仿真运行的高层编排器。"""

    def __init__(self, scenario: ScenarioConfig, workspace_root: str | Path):
        """保存场景配置和工作区根目录，并创建底层仿真器。"""
        self.scenario = scenario
        self.workspace_root = Path(workspace_root).resolve()
        self.simulator = Simulator(scenario)

    def build_scenario(self, explicit_sumo_home: str | Path | None = None):
        """根据当前场景配置生成 OSM/SUMO 路网、路线和配置文件。"""
        return build_highway_scenario(self.scenario, self.workspace_root, explicit_sumo_home)

    def run_demo(self, steps: int | None = None) -> None:
        """运行内置的合成车辆状态演示模式。"""
        self.simulator.run_demo(steps=steps)

    def run_highd(self, steps: int | None = None) -> None:
        """运行 highD 轨迹数据回放模式。"""
        self.simulator.run_highd(steps=steps, workspace_root=self.workspace_root)

    def run_sumo(
        self,
        *,
        sumocfg_path: str | Path | None = None,
        gui: bool = False,
        max_vehicles: int | None = None,
        explicit_sumo_home: str | Path | None = None,
        steps: int | None = None,
    ) -> None:
        """运行 SUMO 联动模式，必要时先自动构建默认场景文件。"""
        max_vehicles = self.scenario.simulation.max_vehicles if max_vehicles is None else max_vehicles
        if sumocfg_path is None:
            default_sumocfg = self.scenario.default_sumocfg_path(self.workspace_root)
            if not default_sumocfg.exists():
                self.build_scenario(explicit_sumo_home)
            sumocfg_path = self.scenario.default_sumocfg_path(self.workspace_root)

        self.simulator.run_sumo(
            sumocfg_path=sumocfg_path,
            gui=gui,
            max_vehicles=max_vehicles,
            explicit_sumo_home=explicit_sumo_home,
            steps=steps,
        )
