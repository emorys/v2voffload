from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskProfile:
    """Representative autonomous-driving task profile with source metadata."""

    task_id: str
    component_type: str
    resource_name: str
    compute_ops: float
    input_bits: float
    output_bits: float
    deadline: float
    source: str
    source_url: str
    note: str


# The workload values are full-frame representative operations before
# OffloadingConfig.task_load_scale is applied for tractable optimization.
AUTONOMOUS_DRIVING_TASKS: tuple[TaskProfile, ...] = (
    TaskProfile(
        task_id="camera_yolov3_object_detection",
        component_type="perception",
        resource_name="GPU",
        compute_ops=65.86e9,
        input_bits=416.0 * 416.0 * 3.0 * 8.0,
        output_bits=100.0 * 7.0 * 32.0,
        deadline=0.10,
        source="YOLOv3 / Darknet-53 object detection, 65.86 BFLOPs reported for YOLOv3-416",
        source_url="https://pjreddie.com/darknet/yolo/",
        note="Camera object detection is a representative GPU-heavy perception task.",
    ),
    TaskProfile(
        task_id="pointpillars_lidar_point_cloud_processing",
        component_type="processing",
        resource_name="TPU",
        compute_ops=32.2e9,
        input_bits=120000.0 * 4.0 * 32.0,
        output_bits=100.0 * 8.0 * 32.0,
        deadline=1.0 / 62.0,
        source="PointPillars LiDAR point-cloud object detection reports 62 Hz on KITTI",
        source_url="https://arxiv.org/abs/1812.05784",
        note="LiDAR point cloud processing is represented by a real-time pillar encoder and 3D detection head.",
    ),
    TaskProfile(
        task_id="apollo_em_motion_planning",
        component_type="planning",
        resource_name="CPU",
        compute_ops=5.0e9,
        input_bits=256.0 * 64.0,
        output_bits=80.0 * 5.0 * 32.0,
        deadline=0.10,
        source="Baidu Apollo EM planner: DP/QP path-speed optimization deployed in L4 Apollo vehicles",
        source_url="https://arxiv.org/abs/1807.08048",
        note="CPU-oriented planning workload representing EM-style path and speed optimization.",
    ),
)


def task_profiles_by_component() -> dict[str, TaskProfile]:
    """Return the default real-scene task catalog keyed by component type."""
    return {profile.component_type: profile for profile in AUTONOMOUS_DRIVING_TASKS}


def default_resource_names() -> list[str]:
    """Resource dimension labels aligned with the default task catalog."""
    return [profile.resource_name for profile in AUTONOMOUS_DRIVING_TASKS]


def default_component_types() -> list[str]:
    """Component dimension labels aligned with the default task catalog."""
    return [profile.component_type for profile in AUTONOMOUS_DRIVING_TASKS]


def build_task_vectors(
    component_types: list[str],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build compute, input, output, deadline, and profile-id vectors for components."""
    catalog = task_profiles_by_component()
    compute_load = []
    input_bits = []
    output_bits = []
    deadlines = []
    profile_ids = []
    for component_type in component_types:
        profile = catalog.get(component_type)
        if profile is None:
            raise ValueError(
                f"No autonomous-driving task profile is defined for component_type={component_type!r}. "
                f"Available profiles: {', '.join(sorted(catalog))}"
            )
        compute_load.append(profile.compute_ops * scale)
        input_bits.append(profile.input_bits)
        output_bits.append(profile.output_bits)
        deadlines.append(profile.deadline)
        profile_ids.append(profile.task_id)
    return (
        np.array(compute_load, dtype=float),
        np.array(input_bits, dtype=float),
        np.array(output_bits, dtype=float),
        np.array(deadlines, dtype=float),
        profile_ids,
    )
