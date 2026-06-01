from __future__ import annotations

from vanetsim.config import CommunicationConfig, IncentiveConfig, MobilityConfig, OffloadingConfig, SimulationConfig
from vanetsim.optimization.baselines import BASELINE_POLICIES, BaselinePolicyRunner
from vanetsim.optimization.incentive_manager import IncentiveManager


OURS_POLICY = "ours"
POLICY_ALIASES = {
    "our": OURS_POLICY,
    "proposed": OURS_POLICY,
    "local": "local-only",
    "equal": "equal-split-v2v",
    "equal-split": "equal-split-v2v",
    "delay-greedy": "delay-greedy-v2v",
    "no-baseline": "no-baseline-maintenance",
    "no-maintenance": "no-baseline-maintenance",
    "stage1-only": "stage-i-only",
    "stage-i": "stage-i-only",
    "no-price": "no-incentive",
    "static": "static-cluster",
    "fan2023": "fan-2023",
    "nan2023": "nan-2023",
    "kumar2023": "kumar-2023",
}
POLICY_CHOICES = (OURS_POLICY, *BASELINE_POLICIES)


def normalize_policy(policy: str | None) -> str:
    """Normalize CLI/config policy names to the canonical policy id."""
    value = (policy or OURS_POLICY).strip().lower()
    value = POLICY_ALIASES.get(value, value)
    if value not in POLICY_CHOICES:
        choices = ", ".join(POLICY_CHOICES)
        raise ValueError(f"Unknown policy '{policy}'. Available policies: {choices}")
    return value


def build_policy_manager(
    policy: str | None,
    *,
    mobility: MobilityConfig,
    communication: CommunicationConfig,
    offloading: OffloadingConfig,
    incentive: IncentiveConfig,
    simulation: SimulationConfig,
):
    """Build the optimizer/coordinator for the selected method or baseline."""
    normalized = normalize_policy(policy)
    if normalized == OURS_POLICY:
        return IncentiveManager(
            mobility=mobility,
            communication=communication,
            offloading=offloading,
            incentive=incentive,
            simulation=simulation,
        )
    return BaselinePolicyRunner(
        policy=normalized,
        mobility=mobility,
        communication=communication,
        offloading=offloading,
        incentive=incentive,
        simulation=simulation,
    )
