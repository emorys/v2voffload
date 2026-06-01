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
    parser = argparse.ArgumentParser(description="Plot communication-range experiment figures.")
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
    created = line_plot(
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
    print(f"Figures written: {len(created)}")


if __name__ == "__main__":
    main()
