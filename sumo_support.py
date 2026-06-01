from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vanetsim.integrations.sumo_runtime import (  # noqa: E402
    SumoVehicleState,
    apply_target_speeds,
    build_sumo_command,
    collect_vehicle_snapshots,
    collect_vehicle_states,
    discover_sumo_home,
    ensure_sumo_tools_on_path,
    load_traci,
    resolve_sumo_binary,
)

__all__ = [
    "SumoVehicleState",
    "apply_target_speeds",
    "build_sumo_command",
    "collect_vehicle_snapshots",
    "collect_vehicle_states",
    "discover_sumo_home",
    "ensure_sumo_tools_on_path",
    "load_traci",
    "resolve_sumo_binary",
]
