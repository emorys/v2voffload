from __future__ import annotations

import argparse
from pathlib import Path

from plot_experiment_results import DEFAULT_FIGURE_DIR, DEFAULT_INPUT, import_matplotlib, line_plot, load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot safety-threshold and incentive-sensitivity figures.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--formats", default="png,svg")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt = import_matplotlib()
    rows = load_rows(args.input)
    created = []
    created += line_plot(
        plt,
        rows,
        args,
        experiment="safety_speed",
        metric="baseline_maintenance_ratio",
        stem="fig09_safety_speed_maintenance_ratio",
        title="Baseline maintenance under safety-speed threshold",
        xlabel="Minimum safe speed threshold (m/s)",
        policies=["local-only", "delay-greedy-v2v", "stage-i-only", "fan-2023", "kumar-2023", "ours"],
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
    print(f"Figures written: {len(created)}")


if __name__ == "__main__":
    main()
