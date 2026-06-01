from __future__ import annotations

import argparse
from pathlib import Path

from plot_experiment_results import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_INPUT,
    POLICY_ORDER,
    import_matplotlib,
    line_plot,
    load_rows,
    selected_policies,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot traffic-density experiment figures.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
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
    created = []
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
    print(f"Figures written: {len(created)}")


if __name__ == "__main__":
    main()
