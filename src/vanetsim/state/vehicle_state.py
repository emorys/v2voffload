from __future__ import annotations

import numpy as np

from vanetsim.config import IncentiveConfig, MobilityConfig, OffloadingConfig, SimulationConfig
from vanetsim.domain import VehicleSnapshot, VehicleState
from vanetsim.tasks.catalog import build_task_vectors


class VehicleStateManager:
    """Build complete per-slot vehicle states from external snapshots or demos."""

    def __init__(
        self,
        mobility: MobilityConfig,
        offloading: OffloadingConfig,
        incentive: IncentiveConfig,
        simulation: SimulationConfig,
    ):
        self.mobility = mobility
        self.offloading = offloading
        self.incentive = incentive
        self.simulation = simulation

    def stable_seed(self, vehicle_id: str, base_seed: int) -> int:
        """Generate a deterministic seed from vehicle id and base seed."""
        value = base_seed & 0xFFFFFFFF
        for byte in vehicle_id.encode("utf-8"):
            value = ((value * 1664525) + byte + 1013904223) & 0xFFFFFFFF
        return value

    def build_from_snapshots(self, snapshots: list[VehicleSnapshot], max_vehicles: int = 0) -> list[VehicleState]:
        """Convert SUMO or highD snapshots into resource-aware vehicle states."""
        if max_vehicles > 0:
            snapshots = snapshots[:max_vehicles]

        states = []
        for snapshot in snapshots:
            states.append(
                self._build_vehicle_state(
                    vehicle_id=snapshot.vehicle_id,
                    position=(snapshot.x, snapshot.y),
                    progress=snapshot.progress,
                    speed=snapshot.speed,
                    acceleration=snapshot.acceleration,
                    lane_id=snapshot.lane_id,
                    desired_speed=snapshot.desired_speed,
                    speed_limit=min(snapshot.allowed_speed, snapshot.max_speed, self.mobility.speed_limit),
                )
            )
        return states

    def build_demo_states(self) -> list[VehicleState]:
        """Construct a deterministic synthetic demo vehicle state list."""
        progress = [220.0, 285.0, 0.0, 75.0, 145.0, 360.0]
        speeds = [16.5, 18.2, 24.0, 25.0, 23.0, 26.5]
        desired = [27.0, 28.0, 30.0, 31.0, 29.0, 32.0]
        vehicle_count = self.simulation.max_vehicles if self.simulation.max_vehicles > 0 else len(progress)
        while len(progress) < vehicle_count:
            index = len(progress)
            progress.append(420.0 + 58.0 * (index - 6))
            if index % 5 == 0:
                speeds.append(max(0.0, self.mobility.min_speed - 1.5))
            else:
                speeds.append(min(self.mobility.speed_limit, 22.0 + 0.9 * (index % 7)))
            desired.append(min(self.mobility.speed_limit, 28.0 + 0.6 * (index % 6)))
        vehicle_count = min(vehicle_count, len(progress))
        return [
            self._build_vehicle_state(
                vehicle_id=f"V{index}",
                position=(progress[index], 0.0),
                progress=progress[index],
                speed=speeds[index],
                acceleration=0.0,
                lane_id="demo_lane",
                desired_speed=desired[index],
                speed_limit=self.mobility.speed_limit,
            )
            for index in range(vehicle_count)
        ]

    def advance_demo_states(self, states: list[VehicleState], target_speeds: dict[str, float]) -> list[VehicleState]:
        """Advance demo vehicle positions using already safety-limited target speeds."""
        advanced = []
        for state in states:
            next_speed = target_speeds.get(state.vehicle_id, state.speed)
            next_progress = state.progress + next_speed * self.simulation.step_length
            acceleration = (next_speed - state.speed) / self.simulation.step_length
            role = "slow" if next_speed < self.mobility.min_speed else "helper" if state.willingness else "inactive"
            advanced.append(
                VehicleState(
                    vehicle_id=state.vehicle_id,
                    position=(next_progress, state.position[1]),
                    progress=next_progress,
                    speed=next_speed,
                    acceleration=acceleration,
                    lane_id=state.lane_id,
                    compute_capacity=state.compute_capacity,
                    task_load=state.task_load,
                    willingness=state.willingness,
                    role=role,
                    desired_speed=state.desired_speed,
                    speed_limit=state.speed_limit,
                    price=state.price,
                    value_of_time=state.value_of_time,
                    task_input_bits=state.task_input_bits,
                    task_output_bits=state.task_output_bits,
                    task_deadlines=state.task_deadlines,
                    task_profile_ids=state.task_profile_ids,
                )
            )
        return advanced

    def _build_vehicle_state(
        self,
        *,
        vehicle_id: str,
        position: tuple[float, float],
        progress: float,
        speed: float,
        acceleration: float,
        lane_id: str,
        desired_speed: float,
        speed_limit: float,
    ) -> VehicleState:
        """Fill resource, workload, price, willingness, and role fields for one vehicle."""
        rng = np.random.default_rng(self.stable_seed(vehicle_id, self.simulation.seed))
        base_compute = np.array(self.offloading.base_compute, dtype=float)
        base_price = np.array(self.offloading.base_price, dtype=float)
        task_load, input_bits, output_bits, deadlines, profile_ids = build_task_vectors(
            self.offloading.component_types,
            self.offloading.task_load_scale,
        )

        compute_capacity = rng.uniform(0.85, 1.15) * base_compute * self.offloading.base_fraction
        task_load = task_load * rng.uniform(
            0.9,
            1.1,
            size=len(base_compute),
        )
        price = rng.uniform(0.85, 1.15) * base_price
        willingness = int(rng.random() < 0.75)
        role = "slow" if speed < self.mobility.min_speed else "helper" if willingness else "inactive"

        return VehicleState(
            vehicle_id=vehicle_id,
            position=position,
            progress=progress,
            speed=speed,
            acceleration=acceleration,
            lane_id=lane_id,
            compute_capacity=compute_capacity,
            task_load=task_load,
            willingness=willingness,
            role=role,
            desired_speed=desired_speed,
            speed_limit=speed_limit,
            price=price,
            value_of_time=self.incentive.value_of_time,
            task_input_bits=input_bits,
            task_output_bits=output_bits,
            task_deadlines=deadlines,
            task_profile_ids=tuple(profile_ids),
        )
