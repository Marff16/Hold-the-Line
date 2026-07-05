"""Fixed map definition for the Hold The Line MVP."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.geometry import Circle, Obstacle, Rect, obstacle_contains_point


@dataclass(frozen=True)
class BlueDroneConfig:
    count: int = 2
    radius: float = 1.0
    max_speed: float = 18.0
    detection_radius: float = 18.0
    intercept_radius: float = 2.0
    # Seconds a red drone must stay continuously tethered (inside detection_radius
    # with clear line of sight) before it's destroyed. Breaking the tether at any
    # point resets the exposure clock to zero.
    destroy_time: float = 3.0


@dataclass(frozen=True)
class RedDroneConfig:
    count: int = 2
    radius: float = 1.0
    max_speed: float = 15.0
    scouting_radius: float = 10.0
    info_threshold: float = 5.0


@dataclass(frozen=True)
class FixedMapConfig:
    world_size: tuple[float, float]
    buildings: list[Obstacle]
    protected_zone: Rect
    assets: list[Circle]
    red_spawn_zones: list[Rect]
    blue_spawn_zone: Rect
    name: str = "fixed_top_bottom"
    terrain_image: str | None = None
    blue_drones: BlueDroneConfig = field(default_factory=BlueDroneConfig)
    red_drones: RedDroneConfig = field(default_factory=RedDroneConfig)
    # World-y of the front line separating the teams' "known" territory from
    # fogged territory in the UI. None means "not set by this instance" -
    # callers fall back to the midpoint between the spawn zones.
    front_line_y: float | None = None


def default_fixed_map(world_n: float = 100.0) -> FixedMapConfig:
    """Create the fixed top-vs-bottom starter map."""

    return FixedMapConfig(
        name="fixed_top_bottom",
        world_size=(world_n, world_n),
        buildings=[],
        protected_zone=Rect(34.0, 8.0, 32.0, 18.0),
        assets=[
            Circle((44.0, 17.0), 1.5),
            Circle((58.0, 18.0), 1.5),
        ],
        red_spawn_zones=[
            Rect(4.0, 90.0, 92.0, 8.0),
        ],
        blue_spawn_zone=Rect(4.0, 4.0, 92.0, 8.0),
    )


def sample_point_in_rect(
    rng: np.random.Generator,
    rect: Rect,
    blocked_rects: list[Obstacle],
    max_attempts: int = 200,
    margin: float = 0.0,
) -> np.ndarray:
    """Sample a free point in a rectangle, retrying if the point hits an obstacle."""

    for _ in range(max_attempts):
        point = np.array(
            [
                rng.uniform(rect.min_x, rect.max_x),
                rng.uniform(rect.min_y, rect.max_y),
            ],
            dtype=np.float32,
        )
        if not any(obstacle_contains_point(point, blocker, margin=margin) for blocker in blocked_rects):
            return point

    raise RuntimeError(f"could not sample a free point inside spawn zone {rect}")
