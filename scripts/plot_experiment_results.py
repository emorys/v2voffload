from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "paper_experiment_results.csv"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "paper" / "tables"

POLICY_ORDER = [
    "local-only",
    "equal-split-v2v",
    "delay-greedy-v2v",
    "stage-i-only",
    "fan-2023",
    "kumar-2023",
    "ours",
    "no-baseline-maintenance",
    "no-incentive",
    "static-cluster",
]
POLICY_LABELS = {
    "local-only": "Local-only",
    "equal-split-v2v": "Equal-split",
    "delay-greedy-v2v": "Delay-greedy",
    "stage-i-only": "Stage-I only",
    "fan-2023": "Fan-2023",
    "kumar-2023": "Kumar-2023",
    "ours": "Ours",
    "no-baseline-maintenance": "No-baseline",
    "no-incentive": "No-incentive",
    "static-cluster": "Static-cluster",
    "ours_minus_stage-i-only": "Ours - Stage-I",
}
METRIC_LABELS = {
    "min_speed": "Cluster minimum speed (m/s)",
    "avg_speed": "Cluster average speed (m/s)",
    "min_target_speed": "Target minimum speed (m/s)",
    "avg_target_speed": "Average target speed (m/s)",
    "avg_speed_gain": "Average speed gain (m/s)",
    "traffic_flow_proxy": "Road throughput proxy (veh*m/s)",
    "throughput": "Vehicles in slot",
    "avg_latency": "Average task delay (s)",
    "p95_latency": "95th-percentile delay (s)",
    "task_completion_rate": "Task completion ratio",
    "resource_utilization": "Helper resource utilization",
    "baseline_maintenance_ratio": "Baseline maintenance ratio",
    "actual_min_speed_ratio": "Actual min-speed ratio",
    "cluster_reconfiguration_count": "Reconfiguration count",
    "total_payment": "Stage-I payment cost",
    "stageII_cost": "Stage-II helper cost",
    "total_cost": "Total cost",
    "social_welfare": "Social welfare",
    "unit_speed_gain_cost": "Unit speed-gain cost",
    "helper_load_jain_index": "Helper load Jain index",
    "estimated_avg_travel_time": "Average travel time (s)",
}
SUMMARY_METRICS = [
    "min_speed",
    "avg_speed",
    "min_target_speed",
    "avg_target_speed",
    "avg_speed_gain",
    "traffic_flow_proxy",
    "avg_latency",
    "p95_latency",
    "task_completion_rate",
    "resource_utilization",
    "baseline_maintenance_ratio",
    "actual_min_speed_ratio",
    "cluster_reconfiguration_count",
    "total_payment",
    "stageII_cost",
    "total_cost",
    "social_welfare",
    "unit_speed_gain_cost",
    "helper_load_jain_index",
    "estimated_avg_travel_time",
]
COLOR_CYCLE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#000000",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw paper-ready figures and tables from paper_experiment_results.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--formats", default="png,svg", help="Comma-separated figure formats, e.g. png,svg,pdf.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--policies", default=",".join(POLICY_ORDER[:7]))
    parser.add_argument("--include-extended", action="store_true", help="Include ablation-only policies in main line plots.")
    return parser.parse_args()


def import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install matplotlib\n"
            "or update dependencies with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc
    return plt


def split_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def to_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def policy_sort_key(policy: str) -> tuple[int, str]:
    if policy in POLICY_ORDER:
        return POLICY_ORDER.index(policy), policy
    return len(POLICY_ORDER), policy


def selected_policies(args: argparse.Namespace) -> list[str]:
    policies = split_csv(args.policies)
    if args.include_extended:
        for policy in POLICY_ORDER:
            if policy not in policies:
                policies.append(policy)
    return policies


def mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None, int]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None, 0
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std, len(clean)


def aggregate(
    rows: list[dict[str, str]],
    *,
    record_type: str,
    experiment: str | None = None,
    policies: list[str] | None = None,
) -> dict[tuple[str, str, str], dict[str, dict[str, float | int | None]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    policy_set = set(policies) if policies else None
    for row in rows:
        if row.get("record_type") != record_type:
            continue
        if experiment is not None and row.get("experiment") != experiment:
            continue
        policy = row.get("policy", "")
        if policy_set is not None and policy not in policy_set:
            continue
        key = (policy, row.get("x_name", ""), row.get("x_value", ""))
        groups[key].append(row)

    result: dict[tuple[str, str, str], dict[str, dict[str, float | int | None]]] = {}
    for key, items in groups.items():
        metric_stats = {}
        for metric in SUMMARY_METRICS:
            mean, std, count = mean_std(to_float(item.get(metric)) for item in items)
            metric_stats[metric] = {"mean": mean, "std": std, "count": count}
        result[key] = metric_stats
    return result


def numeric_sort_value(text: str) -> tuple[int, float | str]:
    value = to_float(text)
    if value is not None:
        return 0, value
    return 1, text


def save_figure(fig, args: argparse.Namespace, stem: str, manifest: list[str]) -> None:
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for fmt in split_csv(args.formats):
        path = args.figure_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        manifest.append(str(path))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_aggregated_tables(rows: list[dict[str, str]], args: argparse.Namespace) -> list[str]:
    args.table_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    summary_rows = [row for row in rows if row.get("record_type") == "summary"]
    table_rows = []
    group_keys = sorted(
        {(row.get("experiment", ""), row.get("policy", ""), row.get("x_name", ""), row.get("x_value", "")) for row in summary_rows}
    )
    for experiment, policy, x_name, x_value in group_keys:
        items = [
            row
            for row in summary_rows
            if row.get("experiment") == experiment
            and row.get("policy") == policy
            and row.get("x_name") == x_name
            and row.get("x_value") == x_value
        ]
        out = {"experiment": experiment, "policy": policy, "x_name": x_name, "x_value": x_value}
        for metric in SUMMARY_METRICS:
            mean, std, count = mean_std(to_float(item.get(metric)) for item in items)
            out[f"{metric}_mean"] = "" if mean is None else mean
            out[f"{metric}_std"] = "" if std is None else std
            out[f"{metric}_n"] = count
        table_rows.append(out)

    fields = ["experiment", "policy", "x_name", "x_value"]
    for metric in SUMMARY_METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])
    path = args.table_dir / "all_summary_aggregated.csv"
    write_csv(path, table_rows, fields)
    created.append(str(path))

    for experiment in ["method_comparison", "ablation", "stage2_gain", "incentive_lambda"]:
        subset = [row for row in table_rows if row["experiment"] == experiment]
        if not subset:
            continue
        path = args.table_dir / f"{experiment}_table.csv"
        write_csv(path, subset, fields)
        created.append(str(path))
    return created


def bar_panel(plt, rows, args, *, experiment: str, metrics: list[str], stem: str, title: str, policies: list[str]) -> list[str]:
    agg = aggregate(rows, record_type="summary", experiment=experiment, policies=policies)
    available = sorted({key[0] for key in agg}, key=policy_sort_key)
    if not available:
        return []

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4), squeeze=False)
    axes_flat = axes[0]
    for axis, metric in zip(axes_flat, metrics):
        values = []
        labels = []
        errors = []
        for policy in available:
            key_candidates = [key for key in agg if key[0] == policy]
            if not key_candidates:
                continue
            stats = agg[key_candidates[0]].get(metric, {})
            mean = stats.get("mean")
            if mean is None:
                continue
            labels.append(POLICY_LABELS.get(policy, policy))
            values.append(mean)
            errors.append(stats.get("std") or 0.0)
        positions = range(len(values))
        axis.bar(positions, values, yerr=errors, capsize=3, color=[COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in positions])
        axis.set_title(METRIC_LABELS.get(metric, metric))
        axis.set_xticks(list(positions))
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    manifest: list[str] = []
    save_figure(fig, args, stem, manifest)
    plt.close(fig)
    return manifest


def line_plot(
    plt,
    rows,
    args,
    *,
    experiment: str,
    metric: str,
    stem: str,
    title: str,
    xlabel: str,
    policies: list[str],
    record_type: str = "summary",
) -> list[str]:
    agg = aggregate(rows, record_type=record_type, experiment=experiment, policies=policies)
    if not agg:
        return []

    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    any_series = False
    for policy_index, policy in enumerate(sorted({key[0] for key in agg}, key=policy_sort_key)):
        points = []
        for key, stats_by_metric in agg.items():
            key_policy, _, x_value = key
            if key_policy != policy:
                continue
            stats = stats_by_metric.get(metric, {})
            mean = stats.get("mean")
            if mean is None:
                continue
            points.append((x_value, mean, stats.get("std") or 0.0))
        points.sort(key=lambda item: numeric_sort_value(item[0]))
        if not points:
            continue
        xs = [to_float(item[0]) if to_float(item[0]) is not None else item[0] for item in points]
        ys = [item[1] for item in points]
        yerr = [item[2] for item in points]
        axis.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker=MARKERS[policy_index % len(MARKERS)],
            linewidth=1.8,
            capsize=3,
            label=POLICY_LABELS.get(policy, policy),
            color=COLOR_CYCLE[policy_index % len(COLOR_CYCLE)],
        )
        any_series = True
    if not any_series:
        plt.close(fig)
        return []
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(METRIC_LABELS.get(metric, metric))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    manifest: list[str] = []
    save_figure(fig, args, stem, manifest)
    plt.close(fig)
    return manifest


def time_series_plot(plt, rows, args, *, metric: str, stem: str, title: str, policies: list[str]) -> list[str]:
    slot_rows = [
        row
        for row in rows
        if row.get("record_type") == "slot" and row.get("experiment") == "time_series" and row.get("policy") in policies
    ]
    if not slot_rows:
        return []
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in slot_rows:
        slot = to_float(row.get("slot"))
        value = to_float(row.get(metric))
        if slot is None or value is None:
            continue
        grouped[row.get("policy", "")][slot].append(value)

    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    any_series = False
    for policy_index, policy in enumerate(sorted(grouped, key=policy_sort_key)):
        slots = sorted(grouped[policy])
        ys = [statistics.fmean(grouped[policy][slot]) for slot in slots]
        if not slots:
            continue
        axis.plot(
            slots,
            ys,
            marker=MARKERS[policy_index % len(MARKERS)],
            linewidth=1.8,
            label=POLICY_LABELS.get(policy, policy),
            color=COLOR_CYCLE[policy_index % len(COLOR_CYCLE)],
        )
        any_series = True
    if not any_series:
        plt.close(fig)
        return []
    axis.set_title(title)
    axis.set_xlabel("Slot index")
    axis.set_ylabel(METRIC_LABELS.get(metric, metric))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    manifest: list[str] = []
    save_figure(fig, args, stem, manifest)
    plt.close(fig)
    return manifest


def stage2_contrast_plot(plt, rows, args) -> list[str]:
    contrast_rows = [
        row
        for row in rows
        if row.get("record_type") == "contrast"
        and row.get("experiment") == "stage2_gain"
        and row.get("policy") == "ours_minus_stage-i-only"
    ]
    if not contrast_rows:
        return []
    metrics = ["avg_speed_gain", "traffic_flow_proxy", "total_cost"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 3.4), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in contrast_rows:
            value = to_float(row.get(metric))
            if value is not None:
                grouped[row.get("x_value", "")].append(value)
        points = []
        for x_value, values in grouped.items():
            points.append((x_value, statistics.fmean(values)))
        points.sort(key=lambda item: numeric_sort_value(item[0]))
        xs = [to_float(item[0]) if to_float(item[0]) is not None else item[0] for item in points]
        ys = [item[1] for item in points]
        axis.bar(xs, ys, color="#000000", alpha=0.78)
        axis.axhline(0.0, color="#666666", linewidth=0.9)
        axis.set_xlabel("Task load multiplier")
        axis.set_title(METRIC_LABELS.get(metric, metric))
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Stage-II incremental gain: Ours minus Stage-I only")
    fig.tight_layout()
    manifest: list[str] = []
    save_figure(fig, args, "fig07_stage2_incremental_gain", manifest)
    plt.close(fig)
    return manifest


def build_figures(rows: list[dict[str, str]], args: argparse.Namespace) -> list[str]:
    plt = import_matplotlib()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "savefig.facecolor": "white",
        }
    )
    policies = selected_policies(args)
    created: list[str] = []

    created += bar_panel(
        plt,
        rows,
        args,
        experiment="method_comparison",
        metrics=["min_target_speed", "traffic_flow_proxy", "avg_latency", "social_welfare"],
        stem="fig00_method_comparison_panel",
        title="Task offloading effectiveness across methods",
        policies=policies,
    )
    created += line_plot(
        plt,
        rows,
        args,
        experiment="density",
        metric="min_target_speed",
        stem="fig01_density_min_target_speed",
        title="Cluster minimum target speed under traffic density",
        xlabel="Traffic density proxy: max vehicles",
        policies=policies,
    )
    created += line_plot(
        plt,
        rows,
        args,
        experiment="density",
        metric="traffic_flow_proxy",
        stem="fig02_density_throughput_proxy",
        title="Road throughput proxy under traffic density",
        xlabel="Traffic density proxy: max vehicles",
        policies=policies,
    )
    created += line_plot(
        plt,
        rows,
        args,
        experiment="task_load",
        metric="avg_latency",
        stem="fig03_task_load_avg_latency",
        title="Task delay under task load",
        xlabel="Task load multiplier",
        policies=policies,
    )
    created += line_plot(
        plt,
        rows,
        args,
        experiment="task_load",
        metric="min_target_speed",
        stem="fig04_task_load_bottleneck_speed",
        title="Bottleneck speed under task load",
        xlabel="Task load multiplier",
        policies=policies,
    )
    created += time_series_plot(
        plt,
        rows,
        args,
        metric="min_speed",
        stem="fig05_bottleneck_speed_time_series",
        title="Bottleneck speed over time slots",
        policies=policies,
    )
    created += line_plot(
        plt,
        rows,
        args,
        experiment="communication_range",
        metric="min_target_speed",
        stem="fig06_communication_range_min_target_speed",
        title="Performance under V2V communication range",
        xlabel="Communication range (m)",
        policies=policies,
    )
    created += stage2_contrast_plot(plt, rows, args)
    created += bar_panel(
        plt,
        rows,
        args,
        experiment="ablation",
        metrics=["min_target_speed", "baseline_maintenance_ratio", "cluster_reconfiguration_count", "social_welfare"],
        stem="fig08_ablation_panel",
        title="Ablation study",
        policies=POLICY_ORDER,
    )
    created += line_plot(
        plt,
        rows,
        args,
        experiment="safety_speed",
        metric="baseline_maintenance_ratio",
        stem="fig09_safety_speed_maintenance_ratio",
        title="Baseline maintenance under safety-speed threshold",
        xlabel="Minimum safe speed threshold (m/s)",
        policies=policies,
    )
    created += line_plot(
        plt,
        rows,
        args,
        experiment="incentive_lambda",
        metric="social_welfare",
        stem="fig10_incentive_lambda_social_welfare",
        title="Economic outcome under opportunity-cost weight",
        xlabel="Opportunity-cost lambda",
        policies=["ours", "stage-i-only", "no-incentive"],
    )
    created += line_plot(
        plt,
        rows,
        args,
        experiment="incentive_lambda",
        metric="total_cost",
        stem="fig10b_incentive_lambda_total_cost",
        title="Total cost under opportunity-cost weight",
        xlabel="Opportunity-cost lambda",
        policies=["ours", "stage-i-only", "no-incentive"],
    )
    return created


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    created_tables = write_aggregated_tables(rows, args)
    created_figures = build_figures(rows, args)

    manifest_path = args.figure_dir / "plot_manifest.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        file.write("Figures:\n")
        for path in created_figures:
            file.write(f"{path}\n")
        file.write("\nTables:\n")
        for path in created_tables:
            file.write(f"{path}\n")

    print(f"Input: {args.input}")
    print(f"Figures written: {len(created_figures)}")
    print(f"Tables written: {len(created_tables)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
