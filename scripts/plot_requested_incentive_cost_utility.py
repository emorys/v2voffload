from __future__ import annotations

import argparse
import csv
from pathlib import Path

from plot_experiment_results import import_matplotlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"

POLICY_LABELS = {
    "ours": "Ours",
    "stage-i-only": "Stage-I only",
    "no-incentive": "No-incentive",
    "no-baseline-maintenance": "No-baseline",
    "static-cluster": "Static-cluster",
    "fan-2023": "Fan-2023",
    "kumar-2023": "Kumar-2023",
    "delay-greedy-v2v": "Delay-greedy",
    "equal-split-v2v": "Equal-split",
    "local-only": "Local-only",
}

COLORS = {
    "ours": "#111111",
    "stage-i-only": "#d55e00",
    "no-incentive": "#0072b2",
    "no-baseline-maintenance": "#009e73",
    "static-cluster": "#cc79a7",
    "fan-2023": "#e69f00",
    "kumar-2023": "#56b4e9",
    "delay-greedy-v2v": "#999999",
    "equal-split-v2v": "#666666",
    "local-only": "#bbbbbb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot requested incentive validation and cost-utility figures.")
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--formats", default="png,svg")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def save(fig, args: argparse.Namespace, stem: str) -> list[Path]:
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in [item.strip() for item in args.formats.split(",") if item.strip()]:
        path = args.figure_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def plot_incentive_validation(plt, args: argparse.Namespace) -> list[Path]:
    rows = read_table(args.table_dir / "ablation_table.csv")
    policy_order = ["stage-i-only", "ours", "no-incentive", "no-baseline-maintenance", "static-cluster"]
    selected = [next(row for row in rows if row["policy"] == policy) for policy in policy_order]
    labels = [POLICY_LABELS[row["policy"]] for row in selected]
    colors = [COLORS[row["policy"]] for row in selected]

    metrics = [
        ("min_target_speed_mean", "Target minimum speed (m/s)"),
        ("avg_speed_gain_mean", "Average speed gain (m/s)"),
        ("baseline_maintenance_ratio_mean", "Baseline maintenance ratio"),
        ("social_welfare_mean", "Social welfare"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(13.8, 3.6), squeeze=False)
    for axis, (metric, title) in zip(axes[0], metrics):
        values = [number(row, metric) for row in selected]
        axis.bar(range(len(values)), values, color=colors, alpha=0.88)
        axis.set_title(title)
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels(labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Incentive mechanism validation")
    fig.tight_layout()
    paths = save(fig, args, "fig11_incentive_mechanism_validation")
    plt.close(fig)
    return paths


def plot_cost_utility(plt, args: argparse.Namespace) -> list[Path]:
    rows = read_table(args.table_dir / "method_comparison_table.csv")
    policy_order = ["local-only", "delay-greedy-v2v", "fan-2023", "kumar-2023", "stage-i-only", "ours"]
    selected = [next(row for row in rows if row["policy"] == policy) for policy in policy_order]
    labels = [POLICY_LABELS[row["policy"]] for row in selected]
    colors = [COLORS[row["policy"]] for row in selected]
    x = list(range(len(selected)))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), squeeze=False)
    cost_axis, utility_axis = axes[0]

    stage_i_cost = [number(row, "total_payment_mean") for row in selected]
    stage_ii_cost = [number(row, "stageII_cost_mean") for row in selected]
    social_welfare = [number(row, "social_welfare_mean") for row in selected]
    unit_cost = [number(row, "unit_speed_gain_cost_mean") for row in selected]

    cost_axis.bar(x, stage_i_cost, label="Stage-I payment", color="#4c78a8", alpha=0.9)
    cost_axis.bar(x, stage_ii_cost, bottom=stage_i_cost, label="Stage-II helper cost", color="#f58518", alpha=0.9)
    cost_axis.set_title("Cost composition")
    cost_axis.set_ylabel("Cost")
    cost_axis.set_xticks(x)
    cost_axis.set_xticklabels(labels, rotation=28, ha="right")
    cost_axis.grid(axis="y", alpha=0.25)
    cost_axis.legend(fontsize=8)

    utility_axis.bar([item - 0.18 for item in x], social_welfare, width=0.36, label="Social welfare", color=colors, alpha=0.88)
    utility_axis_twin = utility_axis.twinx()
    utility_axis_twin.plot([item + 0.18 for item in x], unit_cost, marker="o", linewidth=1.8, color="#333333", label="Unit speed-gain cost")
    utility_axis.axhline(0.0, color="#666666", linewidth=0.9)
    utility_axis.set_title("Utility and cost efficiency")
    utility_axis.set_ylabel("Social welfare")
    utility_axis_twin.set_ylabel("Unit speed-gain cost")
    utility_axis.set_xticks(x)
    utility_axis.set_xticklabels(labels, rotation=28, ha="right")
    utility_axis.grid(axis="y", alpha=0.25)

    handles_left, labels_left = utility_axis.get_legend_handles_labels()
    handles_right, labels_right = utility_axis_twin.get_legend_handles_labels()
    utility_axis.legend(handles_left + handles_right, labels_left + labels_right, fontsize=8, loc="upper left")

    fig.suptitle("Cost-utility results")
    fig.tight_layout()
    paths = save(fig, args, "fig12_cost_utility_results")
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
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
    created = []
    created.extend(plot_incentive_validation(plt, args))
    created.extend(plot_cost_utility(plt, args))
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
