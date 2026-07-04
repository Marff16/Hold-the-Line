"""JSON instance loading for Hold The Line.

Instance file naming convention: the filename (minus ``.json``) becomes the
instance's stable id and must be ``lower_snake_case``, e.g. ``test1.json``
-> id ``test1``. The JSON ``name`` field is the Title Case display name
shown in the UI dropdown and should read as the Title Case of the filename
(e.g. "Test 1"), so id and name always stay recognizably in sync.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.geometry import Circle, Rect
from src.map_config import BlueDroneConfig, FixedMapConfig, RedDroneConfig


INSTANCE_DIR = Path(__file__).resolve().parents[1] / "instances"


def list_instances(instance_dir: Path = INSTANCE_DIR) -> list[dict[str, str]]:
    if not instance_dir.exists():
        return []

    instances = []
    for path in sorted(instance_dir.glob("*.json")):
        try:
            data = _read_json(path)
        except (OSError, ValueError):
            continue
        instances.append(
            {
                "id": path.stem,
                "name": str(data.get("name", path.stem)),
                "path": path.name,
            }
        )
    return instances


def load_instance(instance_id: str, instance_dir: Path = INSTANCE_DIR) -> FixedMapConfig:
    safe_id = Path(instance_id).stem
    path = instance_dir / f"{safe_id}.json"
    data = _read_json(path)
    return parse_instance(data, base_dir=instance_dir, fallback_name=safe_id)


def parse_instance(data: dict[str, Any], base_dir: Path | None = None, fallback_name: str = "instance") -> FixedMapConfig:
    world_n = float(data.get("world_size", data.get("world_n", 100.0)))
    if isinstance(data.get("world_size"), list):
        world_size = (float(data["world_size"][0]), float(data["world_size"][1]))
    else:
        world_size = (world_n, world_n)

    terrain_image = _terrain_image(data.get("terrain"), base_dir)
    drones = data.get("drones", {})
    blue = drones.get("blue", {})
    red = drones.get("red", {})

    objectives = data.get("objectives", data.get("assets", []))
    if not objectives:
        raise ValueError("instance must define at least one objective")

    red_spawn = data.get("red_spawn_zone")
    red_spawns = data.get("red_spawn_zones", [red_spawn] if red_spawn else [])
    if not red_spawns:
        raise ValueError("instance must define red_spawn_zone or red_spawn_zones")

    return FixedMapConfig(
        name=str(data.get("name", fallback_name)),
        world_size=world_size,
        buildings=[_rect(item) for item in data.get("buildings", [])],
        protected_zone=_rect(data["protected_zone"]),
        assets=[_circle(item, default_radius=1.5) for item in objectives],
        red_spawn_zones=[_rect(item) for item in red_spawns],
        blue_spawn_zone=_rect(data["blue_spawn_zone"]),
        terrain_image=terrain_image,
        front_line_y=float(data["front_line_y"]) if "front_line_y" in data else None,
        blue_drones=BlueDroneConfig(
            count=int(blue.get("count", 2)),
            radius=float(blue.get("radius", 1.0)),
            max_speed=float(blue.get("max_speed", 18.0)),
            detection_radius=float(blue.get("detection_radius", 18.0)),
            intercept_radius=float(blue.get("intercept_radius", 2.0)),
            destroy_time=float(blue.get("destroy_time", 3.0)),
        ),
        red_drones=RedDroneConfig(
            count=int(red.get("count", 2)),
            radius=float(red.get("radius", 1.0)),
            max_speed=float(red.get("max_speed", 15.0)),
            scouting_radius=float(red.get("scouting_radius", 10.0)),
            info_threshold=float(red.get("info_threshold", 5.0)),
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _rect(data: dict[str, Any]) -> Rect:
    return Rect(float(data["x"]), float(data["y"]), float(data["w"]), float(data["h"]))


def _circle(data: dict[str, Any], default_radius: float) -> Circle:
    if "center" in data:
        center = data["center"]
        return Circle((float(center[0]), float(center[1])), float(data.get("radius", default_radius)))
    return Circle((float(data["x"]), float(data["y"])), float(data.get("radius", default_radius)))


def _terrain_image(terrain: Any, base_dir: Path | None) -> str | None:
    if terrain is None:
        return None
    image = terrain.get("image") if isinstance(terrain, dict) else terrain
    if not image:
        return None
    image_path = Path(str(image))
    if image_path.is_absolute():
        return str(image_path)
    if base_dir is None:
        return str(image_path)
    return str((base_dir / image_path).resolve())
