from __future__ import annotations

import argparse
from pathlib import Path

from plot_experiment_results import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_INPUT,
    DEFAULT_TABLE_DIR,
    POLICY_ORDER,
    bar_panel,
    import_matplotlib,
    load_rows,
    selected_policies,
    write_aggregated_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot method-comparison figures for the paper.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--formats", default="png,svg")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--policies", default=",".join(POLICY_ORDER[:7]))
    parser.add_argument("--include-extended", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt = import_matplotlib()
    rows = load_rows(args.input)
    policies = selected_policies(args)
    created = bar_panel(
        plt,
        rows,
        args,
        experiment="method_comparison",
        metrics=["min_target_speed", "traffic_flow_proxy", "avg_latency", "social_welfare"],
        stem="fig00_method_comparison_panel",
        title="Task offloading effectiveness across methods",
        policies=policies,
    )
    write_aggregated_tables(rows, args)
    print(f"Figures written: {len(created)}")


if __name__ == "__main__":
    main()
