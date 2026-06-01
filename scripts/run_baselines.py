from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vanetsim import HighwaySimulationOrchestrator, load_scenario_config  # noqa: E402
from vanetsim.optimization import POLICY_CHOICES, normalize_policy  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "highway" / "beijing_g6_demo.json"
DEFAULT_POLICIES = (
    "ours",
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


def parse_csv_values(text: str) -> list[str]:
    """Parse comma-separated CLI values while ignoring empty items."""
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Parse arguments for the baseline comparison runner."""
    parser = argparse.ArgumentParser(description="Run VANET offloading baselines and write a CSV summary.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("demo", "highd"), default="demo")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "baseline_comparison.csv")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def run_once(base_scenario, *, policy: str, seed: int, args: argparse.Namespace) -> dict[str, float | int | str]:
    """Run one policy/seed pair and return a flat summary row."""
    scenario = copy.deepcopy(base_scenario)
    scenario.simulation.mode = args.mode
    scenario.simulation.policy = policy
    scenario.simulation.seed = seed
    scenario.route.seed = seed
    if args.steps is not None:
        scenario.simulation.steps = args.steps

    orchestrator = HighwaySimulationOrchestrator(scenario, workspace_root=args.workspace_root)
    sink = io.StringIO()
    output_context = contextlib.redirect_stdout(sink) if args.quiet else contextlib.nullcontext()
    with output_context:
        if args.mode == "highd":
            orchestrator.run_highd(steps=args.steps)
        else:
            orchestrator.run_demo(steps=args.steps)

    summary = orchestrator.simulator.metrics.summary()
    return {
        "policy": policy,
        "seed": seed,
        "mode": args.mode,
        **summary,
    }


def write_rows(rows: list[dict[str, float | int | str]], output: Path) -> None:
    """Write comparison rows to CSV with a stable column order."""
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = [key for key in rows[0] if key not in {"policy", "seed", "mode"}]
    fieldnames = ["policy", "seed", "mode", *metric_fields]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, float | int | str]], output: Path) -> None:
    """Print a compact terminal table for the most important comparison metrics."""
    print(f"\nWrote: {output}")
    print("policy, seed, min_target_speed, avg_speed_gain, total_cost, social_welfare")
    for row in rows:
        print(
            f"{row['policy']}, {row['seed']}, "
            f"{row.get('min_target_speed', 0.0):.3f}, "
            f"{row.get('avg_speed_gain', 0.0):.3f}, "
            f"{row.get('total_cost', 0.0):.4e}, "
            f"{row.get('social_welfare', 0.0):.4e}"
        )


def main() -> None:
    """Run the configured policy list and persist the aggregate comparison table."""
    args = parse_args()
    base_scenario = load_scenario_config(args.config)
    policies = [normalize_policy(policy) for policy in parse_csv_values(args.policies)]
    seeds = [int(seed) for seed in parse_csv_values(args.seeds)]

    invalid = [policy for policy in policies if policy not in POLICY_CHOICES]
    if invalid:
        choices = ", ".join(POLICY_CHOICES)
        raise ValueError(f"Unknown policies: {invalid}. Available policies: {choices}")

    rows = []
    for seed in seeds:
        for policy in policies:
            print(f"Running policy={policy}, seed={seed}")
            rows.append(run_once(base_scenario, policy=policy, seed=seed, args=args))

    if not rows:
        raise RuntimeError("No baseline runs were executed.")
    write_rows(rows, args.output)
    print_rows(rows, args.output)


if __name__ == "__main__":
    main()
