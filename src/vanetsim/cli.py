from __future__ import annotations

import argparse
from pathlib import Path

from vanetsim import HighwaySimulationOrchestrator, load_scenario_config
from vanetsim.optimization import POLICY_CHOICES, normalize_policy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "highway" / "beijing_g6_demo.json"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，得到场景配置、运行模式和 SUMO 选项。"""
    parser = argparse.ArgumentParser(description="Highway VANET offloading with SUMO and OpenStreetMap support.")
    parser.add_argument("--mode", choices=("demo", "sumo", "highd", "build-scenario"), default="sumo")
    parser.add_argument("--policy", default=None, help=f"Optimization policy/baseline. Choices: {', '.join(POLICY_CHOICES)}")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sumocfg", type=Path, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--max-vehicles", type=int, default=None)
    parser.add_argument("--sumo-home", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> None:
    """命令行入口：加载配置、覆盖参数并分发到对应运行模式。"""
    args = parse_args()
    scenario = load_scenario_config(args.config)

    if args.seed is not None:
        scenario.simulation.seed = args.seed
        scenario.route.seed = args.seed
    if args.steps is not None:
        scenario.simulation.steps = args.steps
    if args.max_vehicles is not None:
        scenario.simulation.max_vehicles = args.max_vehicles
    if args.policy is not None:
        scenario.simulation.policy = normalize_policy(args.policy)
    if args.mode != "build-scenario":
        scenario.simulation.mode = args.mode

    orchestrator = HighwaySimulationOrchestrator(scenario, workspace_root=args.workspace_root)

    if args.mode == "build-scenario":
        artifacts = orchestrator.build_scenario(explicit_sumo_home=args.sumo_home)
        print(f"Scenario built under: {artifacts.output_dir}")
        print(f"Resolved bbox: {artifacts.resolved_bbox}")
        if artifacts.source_display_name:
            print(f"OSM source: {artifacts.source_display_name}")
        print(f"SUMO config: {artifacts.sumocfg_path}")
        return

    if args.mode == "sumo":
        orchestrator.run_sumo(
            sumocfg_path=args.sumocfg,
            gui=args.gui,
            max_vehicles=args.max_vehicles,
            explicit_sumo_home=args.sumo_home,
            steps=args.steps,
        )
        return

    if args.mode == "highd":
        orchestrator.run_highd(steps=args.steps)
        return

    orchestrator.run_demo(steps=args.steps)
