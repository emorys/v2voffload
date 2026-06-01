from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from vanetsim.config import ScenarioConfig
from vanetsim.integrations import discover_sumo_home


@dataclass(frozen=True)
class ScenarioArtifacts:
    """记录从 OSM 构建高速 SUMO 场景后生成的关键文件路径和来源信息。"""

    output_dir: Path
    osm_path: Path
    net_path: Path
    route_path: Path
    sumocfg_path: Path
    resolved_bbox: list[float]
    source_display_name: str | None


def _run(command: list[str], cwd: Path) -> None:
    """在指定工作目录执行外部命令，失败时抛出异常。"""
    subprocess.run(command, cwd=str(cwd), check=True)


def geocode_place(place_query: str) -> tuple[list[float], str]:
    """调用 OpenStreetMap Nominatim，将地点查询解析为边界框和显示名称。"""
    params = urllib.parse.urlencode(
        {
            "format": "jsonv2",
            "limit": 1,
            "q": place_query,
        }
    )
    request = urllib.request.Request(
        f"https://nominatim.openstreetmap.org/search?{params}",
        headers={"User-Agent": "Codex-VANET-Project/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload:
        raise RuntimeError(f"OpenStreetMap geocoding returned no result for '{place_query}'.")

    match = payload[0]
    south, north, west, east = [float(value) for value in match["boundingbox"]]
    return [west, south, east, north], match.get("display_name", place_query)


def resolve_bbox(scenario: ScenarioConfig) -> tuple[list[float], str | None]:
    """优先使用配置中的 bbox，否则通过地点名称解析地图范围。"""
    if scenario.map.bbox:
        return list(scenario.map.bbox), scenario.map.place_query
    if scenario.map.place_query:
        return geocode_place(scenario.map.place_query)
    raise RuntimeError("Map configuration must provide either 'bbox' or 'place_query'.")


def _write_sumocfg(output_dir: Path, net_path: Path, route_path: Path, step_length: float) -> Path:
    """写入 SUMO 配置文件，关联生成的路网文件和路线文件。"""
    sumocfg_path = output_dir / "highway.sumocfg"
    sumocfg_text = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net_path.name}"/>
        <route-files value="{route_path.name}"/>
    </input>

    <time>
        <begin value="0.0"/>
        <end value="2000.0"/>
        <step-length value="{step_length}"/>
    </time>

    <processing>
        <time-to-teleport value="-1"/>
    </processing>

    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <duration-log.disable value="true"/>
    </report>
</configuration>
"""
    sumocfg_path.write_text(sumocfg_text, encoding="utf-8")
    return sumocfg_path


def _write_metadata(output_dir: Path, scenario: ScenarioConfig, resolved_bbox: list[float], display_name: str | None) -> None:
    """写入场景元数据，记录来源地点、边界框和保留道路类型。"""
    metadata = {
        "scenario_name": scenario.name,
        "place_query": scenario.map.place_query,
        "source_display_name": display_name,
        "resolved_bbox": resolved_bbox,
        "highd_location_id": scenario.map.highd_location_id,
        "segment_label": scenario.map.segment_label,
        "segment_note": scenario.map.segment_note,
        "road_types": scenario.map.road_types,
        "keep_edge_types": scenario.map.keep_edge_types,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_highway_scenario(
    scenario: ScenarioConfig,
    workspace_root: str | Path,
    explicit_sumo_home: str | Path | None = None,
) -> ScenarioArtifacts:
    """从 OSM 下载地图，转换 SUMO 路网，生成随机路线并返回产物路径。"""
    workspace_root = Path(workspace_root).resolve()
    output_dir = scenario.artifact_dir(workspace_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    sumo_home = discover_sumo_home(explicit_sumo_home)
    resolved_bbox, display_name = resolve_bbox(scenario)

    bbox_text = ",".join(f"{value:.6f}" for value in resolved_bbox)
    prefix_path = output_dir / scenario.map.prefix
    osm_path = output_dir / f"{scenario.map.prefix}_bbox.osm.xml"
    net_path = output_dir / "highway.net.xml"
    route_path = output_dir / "highway.rou.xml"

    road_types_json = json.dumps(scenario.map.road_types)
    _run(
        [
            sys.executable,
            str(sumo_home / "tools" / "osmGet.py"),
            "--bbox",
            bbox_text,
            "--prefix",
            str(prefix_path),
            "--road-types",
            road_types_json,
        ],
        cwd=workspace_root,
    )

    netconvert_command = [
        str(sumo_home / "bin" / "netconvert.exe"),
        "--osm-files",
        str(osm_path),
        "--output-file",
        str(net_path),
        "--tls.discard-simple",
        "--keep-edges.by-type",
        ",".join(scenario.map.keep_edge_types),
    ]
    if scenario.map.remove_geometry:
        netconvert_command.append("--geometry.remove")
    if scenario.map.guess_ramps:
        netconvert_command.append("--ramps.guess")
    if scenario.map.join_junctions:
        netconvert_command.append("--junctions.join")
    _run(netconvert_command, cwd=workspace_root)

    random_trips_command = [
        sys.executable,
        str(sumo_home / "tools" / "randomTrips.py"),
        "-n",
        str(net_path),
        "-r",
        str(route_path),
        "--begin",
        str(scenario.route.begin),
        "--end",
        str(scenario.route.end),
        "--period",
        str(scenario.route.period),
        "--fringe-factor",
        str(scenario.route.fringe_factor),
        "--seed",
        str(scenario.route.seed),
    ]
    if scenario.route.validate:
        random_trips_command.append("--validate")
    _run(random_trips_command, cwd=workspace_root)

    sumocfg_path = _write_sumocfg(output_dir, net_path, route_path, scenario.simulation.step_length)
    _write_metadata(output_dir, scenario, resolved_bbox, display_name)

    return ScenarioArtifacts(
        output_dir=output_dir,
        osm_path=osm_path,
        net_path=net_path,
        route_path=route_path,
        sumocfg_path=sumocfg_path,
        resolved_bbox=resolved_bbox,
        source_display_name=display_name,
    )
