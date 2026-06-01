from __future__ import annotations

import argparse
from pathlib import Path

from plot_experiment_results import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_INPUT,
    POLICY_ORDER,
    bar_panel,
    import_matplotlib,
    load_rows,
    stage2_contrast_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Stage-II gain and ablation figures.")
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
    print(f"Figures written: {len(created)}")


if __name__ == "__main__":
    main()
