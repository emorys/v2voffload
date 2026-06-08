from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vanetsim import HighwaySimulationOrchestrator, load_scenario_config  # noqa: E402
from vanetsim.optimization import normalize_policy  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "highway" / "beijing_g6_demo.json"
MAIN_POLICIES = (
    "local-only",
    "equal-split-v2v",
    "delay-greedy-v2v",
    "stage-i-only",
    "fan-2023",
    "kumar-2023",
    "ours",
)
ABLATION_POLICIES = (
    "ours",
    "stage-i-only",
    "no-baseline-maintenance",
    "no-incentive",
    "static-cluster",
    "delay-greedy-v2v",
)
FIGURE_HINTS = {
    "method_comparison": "main_table_task_offloading_effectiveness",
    "density": "fig1_density_min_speed_fig2_density_throughput",
    "task_load": "fig3_load_delay_fig4_load_bottleneck_speed",
    "communication_range": "fig6_v2v_range_sensitivity",
    "safety_speed": "safe_speed_threshold_sensitivity",
    "incentive_lambda": "fig10_cost_welfare_incentive_sensitivity",
    "stage2_gain": "fig7_stage_i_vs_ours_incremental_gain",
    "ablation": "fig8_ablation_mechanism_necessity",
    "time_series": "fig5_bottleneck_speed_over_slots",
}
METRIC_COLUMNS = [
    "min_speed",
    "avg_speed",
    "throughput",
    "traffic_flow_proxy",
    "total_payment",
    "stageII_cost",
    "stageII_benefit",
    "total_cost",
    "social_welfare",
    "unit_speed_gain_cost",
    "task_completion_rate",
    "avg_latency",
    "max_latency",
    "p95_latency",
    "resource_utilization",
    "helper_participation_rate",
    "helper_load_jain_index",
    "baseline_feasibility",
    "stage2_activation",
    "offloading_ratio",
    "runtime_per_slot",
    "cluster_reconfiguration_count",
    "min_target_speed",
    "avg_target_speed",
    "avg_speed_gain",
    "estimated_avg_travel_time",
    "baseline_maintenance_ratio",
    "actual_min_speed_ratio",
    "speed_drop_probability",
    "bottleneck_recovery_slot",
]
BASE_COLUMNS = [
    "record_type",
    "experiment",
    "figure_hint",
    "policy",
    "seed",
    "mode",
    "slot",
    "x_name",
    "x_value",
    "x_unit",
    "density_max_vehicles",
    "task_load_scale",
    "communication_range_m",
    "min_speed_mps",
    "safe_distance_m",
    "acceleration_mps2",
    "opportunity_cost_lambda",
    "value_of_time",
    "base_price_scale",
]


def parse_csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_float_values(text: str) -> list[float]:
    return [float(item) for item in parse_csv_values(text)]


def parse_int_values(text: str) -> list[int]:
    return [int(item) for item in parse_csv_values(text)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all VANET experiment results into one plotting-ready CSV.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("demo", "highd"), default="demo")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--policies", default=",".join(MAIN_POLICIES))
    parser.add_argument(
        "--experiments",
        default="method_comparison,density,task_load,communication_range,safety_speed,incentive_lambda,stage2_gain,ablation,time_series",
    )
    parser.add_argument("--density-levels", default="6,8,10,12")
    parser.add_argument("--task-load-multipliers", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--communication-ranges", default="100,150,200,250,300")
    parser.add_argument("--safety-speeds", default="12,16,20,24")
    parser.add_argument("--incentive-lambdas", default="0.0,0.5,1.0,2.0,4.0")
    parser.add_argument("--stage2-load-multipliers", default="0.75,1.0,1.25,1.5")
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "paper_experiment_results.csv")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def apply_common_run_settings(scenario, *, mode: str, seed: int, steps: int, policy: str) -> None:
    scenario.simulation.mode = mode
    scenario.simulation.policy = policy
    scenario.simulation.seed = seed
    scenario.simulation.steps = steps
    scenario.route.seed = seed


def run_scenario(base_scenario, *, args: argparse.Namespace, policy: str, seed: int, mutate: Callable | None):
    scenario = copy.deepcopy(base_scenario)
    policy = normalize_policy(policy)
    apply_common_run_settings(scenario, mode=args.mode, seed=seed, steps=args.steps, policy=policy)
    if mutate is not None:
        mutate(scenario)

    orchestrator = HighwaySimulationOrchestrator(scenario, workspace_root=args.workspace_root)
    output_context = contextlib.nullcontext() if args.verbose else contextlib.redirect_stdout(io.StringIO())
    with output_context:
        if args.mode == "highd":
            orchestrator.run_highd(steps=args.steps)
        else:
            orchestrator.run_demo(steps=args.steps)
    return scenario, orchestrator.simulator.metrics.history, orchestrator.simulator.metrics.summary()


def scenario_context(scenario) -> dict[str, float | int | str]:
    base_prices = [float(value) for value in scenario.offloading.base_price]
    price_scale = sum(base_prices) / max(len(base_prices), 1)
    return {
        "density_max_vehicles": scenario.simulation.max_vehicles,
        "task_load_scale": scenario.offloading.task_load_scale,
        "communication_range_m": scenario.communication.max_distance,
        "min_speed_mps": scenario.mobility.min_speed,
        "safe_distance_m": scenario.mobility.safe_distance,
        "acceleration_mps2": scenario.mobility.acceleration,
        "opportunity_cost_lambda": scenario.incentive.opportunity_cost_lambda,
        "value_of_time": scenario.incentive.value_of_time,
        "base_price_scale": price_scale,
    }


def derived_summary_metrics(history, summary: dict[str, float], scenario) -> dict[str, float | int | str]:
    min_speed_threshold = float(scenario.mobility.min_speed)
    min_speeds = [float(item.min_speed) for item in history]
    min_targets = [float(item.min_target_speed) for item in history]
    if not history:
        return {
            "estimated_avg_travel_time": "",
            "baseline_maintenance_ratio": "",
            "actual_min_speed_ratio": "",
            "speed_drop_probability": "",
            "bottleneck_recovery_slot": "",
        }

    avg_speed = max(float(summary.get("avg_speed", 0.0)), 1e-9)
    speed_drops = [
        1.0 if min_speeds[index] < min_speeds[index - 1] - 1e-6 else 0.0
        for index in range(1, len(min_speeds))
    ]
    recovery_slot = ""
    for item in history:
        if item.min_speed >= min_speed_threshold:
            recovery_slot = item.time_slot
            break
    return {
        "estimated_avg_travel_time": float(scenario.mobility.segment_length / avg_speed),
        "baseline_maintenance_ratio": float(sum(value >= min_speed_threshold for value in min_targets) / len(min_targets)),
        "actual_min_speed_ratio": float(sum(value >= min_speed_threshold for value in min_speeds) / len(min_speeds)),
        "speed_drop_probability": float(sum(speed_drops) / max(len(speed_drops), 1)),
        "bottleneck_recovery_slot": recovery_slot,
    }


def base_row(
    *,
    record_type: str,
    experiment: str,
    policy: str,
    seed: int,
    mode: str,
    slot: int | str,
    x_name: str,
    x_value: str | int | float,
    x_unit: str,
    scenario,
) -> dict[str, float | int | str]:
    row = {
        "record_type": record_type,
        "experiment": experiment,
        "figure_hint": FIGURE_HINTS.get(experiment, experiment),
        "policy": policy,
        "seed": seed,
        "mode": mode,
        "slot": slot,
        "x_name": x_name,
        "x_value": x_value,
        "x_unit": x_unit,
    }
    row.update(scenario_context(scenario))
    return row


def rows_for_run(
    *,
    experiment: str,
    x_name: str,
    x_value: str | int | float,
    x_unit: str,
    policy: str,
    seed: int,
    args: argparse.Namespace,
    scenario,
    history,
    summary: dict[str, float],
) -> list[dict[str, float | int | str]]:
    rows = []
    summary_row = base_row(
        record_type="summary",
        experiment=experiment,
        policy=policy,
        seed=seed,
        mode=args.mode,
        slot="",
        x_name=x_name,
        x_value=x_value,
        x_unit=x_unit,
        scenario=scenario,
    )
    summary_row.update(summary)
    summary_row.update(derived_summary_metrics(history, summary, scenario))
    rows.append(summary_row)

    if args.summary_only:
        return rows

    for item in history:
        slot_row = base_row(
            record_type="slot",
            experiment=experiment,
            policy=policy,
            seed=seed,
            mode=args.mode,
            slot=item.time_slot,
            x_name=x_name,
            x_value=x_value,
            x_unit=x_unit,
            scenario=scenario,
        )
        slot_row.update(asdict(item))
        slot_row["estimated_avg_travel_time"] = (
            float(scenario.mobility.segment_length / max(item.avg_speed, 1e-9)) if item.avg_speed > 0 else ""
        )
        slot_row["baseline_maintenance_ratio"] = 1.0 if item.min_target_speed >= scenario.mobility.min_speed else 0.0
        slot_row["actual_min_speed_ratio"] = 1.0 if item.min_speed >= scenario.mobility.min_speed else 0.0
        slot_row["speed_drop_probability"] = ""
        slot_row["bottleneck_recovery_slot"] = ""
        rows.append(slot_row)
    return rows


def add_run(
    rows: list[dict[str, float | int | str]],
    base_scenario,
    *,
    args: argparse.Namespace,
    experiment: str,
    x_name: str,
    x_value: str | int | float,
    x_unit: str,
    policy: str,
    seed: int,
    mutate: Callable | None = None,
) -> None:
    print(f"Running experiment={experiment}, x={x_value}, policy={policy}, seed={seed}")
    scenario, history, summary = run_scenario(base_scenario, args=args, policy=policy, seed=seed, mutate=mutate)
    rows.extend(
        rows_for_run(
            experiment=experiment,
            x_name=x_name,
            x_value=x_value,
            x_unit=x_unit,
            policy=normalize_policy(policy),
            seed=seed,
            args=args,
            scenario=scenario,
            history=history,
            summary=summary,
        )
    )


def set_density(value: int) -> Callable:
    def mutate(scenario) -> None:
        scenario.simulation.max_vehicles = value
        scenario.dataset.max_vehicles = value

    return mutate


def set_task_load(base_value: float, multiplier: float) -> Callable:
    def mutate(scenario) -> None:
        scenario.offloading.task_load_scale = base_value * multiplier

    return mutate


def set_communication_range(value: float) -> Callable:
    def mutate(scenario) -> None:
        scenario.communication.max_distance = value

    return mutate


def set_safety_speed(value: float) -> Callable:
    def mutate(scenario) -> None:
        scenario.mobility.min_speed = value

    return mutate


def set_incentive_lambda(value: float) -> Callable:
    def mutate(scenario) -> None:
        scenario.incentive.opportunity_cost_lambda = value

    return mutate


def add_contrast_rows(rows: list[dict[str, float | int | str]]) -> None:
    summary_rows = [row for row in rows if row.get("record_type") == "summary"]
    index = {
        (row["experiment"], row["x_name"], str(row["x_value"]), row["seed"], row["policy"]): row
        for row in summary_rows
    }
    contrast_rows = []
    for key, ours in index.items():
        experiment, x_name, x_value, seed, policy = key
        if policy != "ours":
            continue
        stage_i = index.get((experiment, x_name, x_value, seed, "stage-i-only"))
        if not stage_i:
            continue
        contrast = {
            key_name: ours.get(key_name, "")
            for key_name in BASE_COLUMNS
        }
        contrast["record_type"] = "contrast"
        contrast["policy"] = "ours_minus_stage-i-only"
        contrast["slot"] = ""
        for metric in METRIC_COLUMNS:
            ours_value = ours.get(metric, "")
            stage_i_value = stage_i.get(metric, "")
            if isinstance(ours_value, (int, float)) and isinstance(stage_i_value, (int, float)):
                contrast[metric] = float(ours_value) - float(stage_i_value)
            else:
                contrast[metric] = ""
        contrast_rows.append(contrast)
    rows.extend(contrast_rows)


def build_rows(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    base_scenario = load_scenario_config(args.config)
    policies = [normalize_policy(policy) for policy in parse_csv_values(args.policies)]
    experiments = set(parse_csv_values(args.experiments))
    seeds = parse_int_values(args.seeds)
    rows: list[dict[str, float | int | str]] = []
    base_task_load = float(base_scenario.offloading.task_load_scale)

    for seed in seeds:
        if "method_comparison" in experiments:
            for policy in policies:
                add_run(
                    rows,
                    base_scenario,
                    args=args,
                    experiment="method_comparison",
                    x_name="method",
                    x_value=policy,
                    x_unit="",
                    policy=policy,
                    seed=seed,
                )

        if "density" in experiments:
            for density in parse_int_values(args.density_levels):
                for policy in policies:
                    add_run(
                        rows,
                        base_scenario,
                        args=args,
                        experiment="density",
                        x_name="max_vehicles",
                        x_value=density,
                        x_unit="vehicles",
                        policy=policy,
                        seed=seed,
                        mutate=set_density(density),
                    )

        if "task_load" in experiments:
            for multiplier in parse_float_values(args.task_load_multipliers):
                for policy in policies:
                    add_run(
                        rows,
                        base_scenario,
                        args=args,
                        experiment="task_load",
                        x_name="task_load_multiplier",
                        x_value=multiplier,
                        x_unit="x",
                        policy=policy,
                        seed=seed,
                        mutate=set_task_load(base_task_load, multiplier),
                    )

        if "communication_range" in experiments:
            for value in parse_float_values(args.communication_ranges):
                for policy in policies:
                    add_run(
                        rows,
                        base_scenario,
                        args=args,
                        experiment="communication_range",
                        x_name="communication_range",
                        x_value=value,
                        x_unit="m",
                        policy=policy,
                        seed=seed,
                        mutate=set_communication_range(value),
                    )

        if "safety_speed" in experiments:
            for value in parse_float_values(args.safety_speeds):
                for policy in policies:
                    add_run(
                        rows,
                        base_scenario,
                        args=args,
                        experiment="safety_speed",
                        x_name="min_safe_speed",
                        x_value=value,
                        x_unit="m/s",
                        policy=policy,
                        seed=seed,
                        mutate=set_safety_speed(value),
                    )

        if "incentive_lambda" in experiments:
            for value in parse_float_values(args.incentive_lambdas):
                for policy in ("ours", "no-incentive", "stage-i-only"):
                    add_run(
                        rows,
                        base_scenario,
                        args=args,
                        experiment="incentive_lambda",
                        x_name="opportunity_cost_lambda",
                        x_value=value,
                        x_unit="",
                        policy=policy,
                        seed=seed,
                        mutate=set_incentive_lambda(value),
                    )

        if "stage2_gain" in experiments:
            for multiplier in parse_float_values(args.stage2_load_multipliers):
                for policy in ("stage-i-only", "ours"):
                    add_run(
                        rows,
                        base_scenario,
                        args=args,
                        experiment="stage2_gain",
                        x_name="task_load_multiplier",
                        x_value=multiplier,
                        x_unit="x",
                        policy=policy,
                        seed=seed,
                        mutate=set_task_load(base_task_load, multiplier),
                    )

        if "ablation" in experiments:
            for policy in ABLATION_POLICIES:
                add_run(
                    rows,
                    base_scenario,
                    args=args,
                    experiment="ablation",
                    x_name="ablation",
                    x_value=policy,
                    x_unit="",
                    policy=policy,
                    seed=seed,
                )

        if "time_series" in experiments:
            for policy in policies:
                add_run(
                    rows,
                    base_scenario,
                    args=args,
                    experiment="time_series",
                    x_name="slot_index",
                    x_value="all",
                    x_unit="slot",
                    policy=policy,
                    seed=seed,
                )

    add_contrast_rows(rows)
    return rows


def write_rows(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*BASE_COLUMNS, *METRIC_COLUMNS]
    extra_fields = sorted({key for row in rows for key in row if key not in fieldnames})
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*fieldnames, *extra_fields], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    if not rows:
        raise RuntimeError("No experiment rows were generated.")
    write_rows(rows, args.output)
    summary_count = sum(1 for row in rows if row.get("record_type") == "summary")
    slot_count = sum(1 for row in rows if row.get("record_type") == "slot")
    contrast_count = sum(1 for row in rows if row.get("record_type") == "contrast")
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"summary={summary_count}, slot={slot_count}, contrast={contrast_count}")


if __name__ == "__main__":
    main()
