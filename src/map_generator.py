"""Procedural map generation and validation for Hold The Line."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.geometry import Circle, Rect, distance, point_in_any_rect
from src.map_config import FixedMapConfig


@dataclass(frozen=True)
class MapValidationResult:
    valid: bool
    reasons: list[str]
    approach_routes: int


def generate_valid_map(
    seed: int | None = None,
    world_size: tuple[float, float] = (100.0, 100.0),
    building_count: int = 7,
    max_attempts: int = 250,
) -> FixedMapConfig:
    """Generate a validated fixed-size urban map candidate."""

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        candidate = generate_candidate_map(rng, world_size=world_size, building_count=building_count)
        if validate_map(candidate).valid:
            return candidate

    raise RuntimeError(f"could not generate a valid map after {max_attempts} attempts")


def generate_candidate_map(
    rng: np.random.Generator,
    world_size: tuple[float, float] = (100.0, 100.0),
    building_count: int = 7,
) -> FixedMapConfig:
    """Generate one plausible map candidate without guaranteeing validity."""

    width, height = world_size
    protected_zone = Rect(
        float(rng.uniform(width * 0.28, width * 0.40)),
        float(rng.uniform(height * 0.07, height * 0.14)),
        float(rng.uniform(width * 0.24, width * 0.36)),
        float(rng.uniform(height * 0.14, height * 0.20)),
    )
    protected_point = protected_zone.center

    assets = []
    for _ in range(2):
        point = np.array(
            [
                rng.uniform(protected_zone.min_x + 5.0, protected_zone.max_x - 5.0),
                rng.uniform(protected_zone.min_y + 5.0, protected_zone.max_y - 5.0),
            ],
            dtype=np.float32,
        )
        assets.append(Circle((float(point[0]), float(point[1])), 1.5))

    red_spawn_zones = [
        Rect(
            width * 0.04,
            height - 10.0,
            width * 0.92,
            8.0,
        )
    ]
    blue_spawn_zone = _blue_spawn_near_protected(rng, protected_zone, world_size)

    keepouts = [blue_spawn_zone, *red_spawn_zones]
    buildings: list[Rect] = []
    attempts = 0
    while len(buildings) < building_count and attempts < building_count * 120:
        attempts += 1
        rect = Rect(
            float(rng.uniform(10.0, width - 24.0)),
            float(rng.uniform(height * 0.25, height * 0.78)),
            float(rng.uniform(7.0, 17.0)),
            float(rng.uniform(6.0, 20.0)),
        )
        if not _rect_in_world(rect, world_size):
            continue
        if _rect_overlaps_any_rect(rect, keepouts, margin=2.0):
            continue
        if _rect_overlaps_any_rect(rect, buildings, margin=2.0):
            continue
        if _rects_overlap(rect, protected_zone, margin=4.0):
            continue
        if any(_rect_overlaps_circle(rect, asset, margin=5.0) for asset in assets):
            continue
        buildings.append(rect)

    return FixedMapConfig(
        world_size=world_size,
        buildings=buildings,
        protected_zone=protected_zone,
        assets=assets,
        red_spawn_zones=red_spawn_zones,
        blue_spawn_zone=blue_spawn_zone,
    )


def validate_map(config: FixedMapConfig, cell_size: float = 2.0) -> MapValidationResult:
    """Validate a generated map with a raster reachability check."""

    reasons: list[str] = []
    width, height = config.world_size

    if len(config.buildings) < 5:
        reasons.append("expected at least five buildings")

    protected = config.protected_zone
    if not _rect_in_world(protected, config.world_size):
        reasons.append("protected zone is outside world bounds")

    for index, asset in enumerate(config.assets):
        if not _circle_in_world(asset, config.world_size):
            reasons.append(f"asset {index} is outside world bounds")
        if point_in_any_rect(asset.center_array, config.buildings):
            reasons.append(f"asset {index} is inside a building")

    protected_samples = _sample_rect_perimeter_points(protected, count_per_side=6)
    blocked_protected_samples = sum(point_in_any_rect(point, config.buildings) for point in protected_samples)
    if blocked_protected_samples > 4:
        reasons.append("protected zone perimeter is too blocked")

    red_starts = [
        point
        for zone in config.red_spawn_zones
        for point in _free_rect_points(zone, config.buildings)
    ]
    if not red_starts:
        reasons.append("red spawn zone is blocked")

    blue_center = _free_rect_center(config.blue_spawn_zone, config.buildings)
    if blue_center is None:
        reasons.append("blue spawn zone is blocked")
    elif red_starts and min(distance(blue_center, red_start) for red_start in red_starts) < 45.0:
        reasons.append("blue spawn is too close to a red spawn")

    targets = [asset.center_array for asset in config.assets] + [protected.center]
    route_count = 0
    for red_start in red_starts:
        if all(_path_exists(config, red_start, target, cell_size=cell_size) for target in targets):
            route_count += 1

    if route_count < 2:
        # Require 2+ routes so red always has an alternate path when blue holds a
        # chokepoint - a single-route map would let two blue drones seal the map.
        reasons.append("fewer than two red approach routes reach the assets and protected zone")

    return MapValidationResult(valid=not reasons, reasons=reasons, approach_routes=route_count)


def describe_map(config: FixedMapConfig) -> str:
    """Return a compact, reproducible description for logs and examples."""

    lines = [
        f"world_size={config.world_size}",
        f"protected_zone={_rect_summary(config.protected_zone)}",
        f"blue_spawn={_rect_summary(config.blue_spawn_zone)}",
    ]
    lines.append("red_spawns=" + ", ".join(_rect_summary(zone) for zone in config.red_spawn_zones))
    lines.append("objectives=" + ", ".join(f"center{_round_tuple(asset.center)}" for asset in config.assets))
    lines.append("buildings=" + ", ".join(_rect_summary(building) for building in config.buildings))
    return "\n".join(lines)


def _blue_spawn_near_protected(
    rng: np.random.Generator,
    protected_zone: Rect,
    world_size: tuple[float, float],
) -> Rect:
    width, height = world_size
    rect_width = width * 0.92
    rect_height = 8.0
    x = width * 0.04
    y = 2.0
    return Rect(x, y, rect_width, rect_height)


def _path_exists(config: FixedMapConfig, start: np.ndarray, goal: np.ndarray, cell_size: float) -> bool:
    width, height = config.world_size
    cols = int(np.ceil(width / cell_size))
    rows = int(np.ceil(height / cell_size))
    start_cell = _point_to_cell(start, cell_size, cols, rows)
    goal_cell = _point_to_cell(goal, cell_size, cols, rows)
    if _cell_blocked(start_cell, cell_size, config.buildings) or _cell_blocked(goal_cell, cell_size, config.buildings):
        return False

    queue: deque[tuple[int, int]] = deque([start_cell])
    visited = {start_cell}
    while queue:
        cell = queue.popleft()
        if cell == goal_cell:
            return True
        for neighbor in _neighbors(cell, cols, rows):
            if neighbor in visited or _cell_blocked(neighbor, cell_size, config.buildings):
                continue
            visited.add(neighbor)
            queue.append(neighbor)

    return False


def _neighbors(cell: tuple[int, int], cols: int, rows: int):
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = x + dx
        ny = y + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            yield nx, ny


def _point_to_cell(point: np.ndarray, cell_size: float, cols: int, rows: int) -> tuple[int, int]:
    return (
        int(np.clip(point[0] // cell_size, 0, cols - 1)),
        int(np.clip(point[1] // cell_size, 0, rows - 1)),
    )


def _cell_blocked(cell: tuple[int, int], cell_size: float, buildings: list[Rect]) -> bool:
    center = np.array([(cell[0] + 0.5) * cell_size, (cell[1] + 0.5) * cell_size], dtype=np.float32)
    return point_in_any_rect(center, buildings)


def _free_rect_center(rect: Rect, buildings: list[Rect]) -> np.ndarray | None:
    for candidate in _rect_sample_points(rect):
        if not point_in_any_rect(candidate, buildings):
            return candidate
    return None


def _free_rect_points(rect: Rect, buildings: list[Rect]) -> list[np.ndarray]:
    return [candidate for candidate in _rect_sample_points(rect) if not point_in_any_rect(candidate, buildings)]


def _rect_sample_points(rect: Rect) -> list[np.ndarray]:
    return [
        rect.center,
        np.array([rect.min_x + rect.w * 0.25, rect.min_y + rect.h * 0.25], dtype=np.float32),
        np.array([rect.min_x + rect.w * 0.75, rect.min_y + rect.h * 0.25], dtype=np.float32),
        np.array([rect.min_x + rect.w * 0.25, rect.min_y + rect.h * 0.75], dtype=np.float32),
        np.array([rect.min_x + rect.w * 0.75, rect.min_y + rect.h * 0.75], dtype=np.float32),
    ]


def _sample_rect_perimeter_points(rect: Rect, count_per_side: int) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for t in np.linspace(0.0, 1.0, count_per_side):
        points.extend(
            [
                np.array([rect.min_x + rect.w * t, rect.min_y], dtype=np.float32),
                np.array([rect.min_x + rect.w * t, rect.max_y], dtype=np.float32),
                np.array([rect.min_x, rect.min_y + rect.h * t], dtype=np.float32),
                np.array([rect.max_x, rect.min_y + rect.h * t], dtype=np.float32),
            ]
        )
    return points


def _rect_in_world(rect: Rect, world_size: tuple[float, float]) -> bool:
    width, height = world_size
    return rect.min_x >= 0.0 and rect.min_y >= 0.0 and rect.max_x <= width and rect.max_y <= height


def _circle_in_world(circle: Circle, world_size: tuple[float, float]) -> bool:
    x, y = circle.center
    width, height = world_size
    return circle.radius <= x <= width - circle.radius and circle.radius <= y <= height - circle.radius


def _rect_overlaps_any_rect(rect: Rect, others: list[Rect], margin: float = 0.0) -> bool:
    return any(_rects_overlap(rect, other, margin=margin) for other in others)


def _rects_overlap(a: Rect, b: Rect, margin: float = 0.0) -> bool:
    return not (
        a.max_x + margin < b.min_x
        or a.min_x - margin > b.max_x
        or a.max_y + margin < b.min_y
        or a.min_y - margin > b.max_y
    )


def _rect_overlaps_circle(rect: Rect, circle: Circle, margin: float = 0.0) -> bool:
    center = circle.center_array
    nearest_x = float(np.clip(center[0], rect.min_x, rect.max_x))
    nearest_y = float(np.clip(center[1], rect.min_y, rect.max_y))
    nearest = np.array([nearest_x, nearest_y], dtype=np.float32)
    return distance(center, nearest) <= circle.radius + margin


def _round_tuple(values: tuple[float, float]) -> tuple[float, float]:
    return (round(values[0], 1), round(values[1], 1))


def _rect_summary(rect: Rect) -> str:
    return f"Rect(x={rect.x:.1f}, y={rect.y:.1f}, w={rect.w:.1f}, h={rect.h:.1f})"
