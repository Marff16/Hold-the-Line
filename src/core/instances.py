"""Realistic fixed and procedural instance constructors for Hold The Line.

Each constructor returns a ``FixedMapConfig`` describing a small low-altitude
drone environment in meters (1 simulation unit == 1 meter): buildings and
round obstacles, a protected zone, sensitive assets, a red spawn area near
the top of the map and a blue spawn area near the bottom, all reachable via
at least two approach routes. See ``map_generator.validate_map`` for the
reachability/placement checks these layouts are designed to satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.geometry import Circle, Obstacle, Rect, obstacles_overlap
from src.core.map_config import BlueDroneConfig, FixedMapConfig, RedDroneConfig
from src.core.map_generator import MapValidationResult, validate_map


def create_fixed_industrial_facility() -> FixedMapConfig:
    """Warehouse-district benchmark: rectangular halls plus a few round tanks."""

    return FixedMapConfig(
        name="Test 2",
        world_size=(200.0, 200.0),
        red_spawn_zones=[Rect(80.0, 180.0, 40.0, 15.0)],
        blue_spawn_zone=Rect(80.0, 5.0, 40.0, 15.0),
        protected_zone=Rect(75.0, 55.0, 50.0, 40.0),
        assets=[
            Circle((90.0, 75.0), 1.5),
            Circle((110.0, 75.0), 1.5),
        ],
        blue_drones=BlueDroneConfig(count=2),
        red_drones=RedDroneConfig(count=4),
        buildings=[
            Rect(20.0, 135.0, 50.0, 18.0),
            Rect(130.0, 135.0, 50.0, 18.0),
            Rect(35.0, 95.0, 35.0, 16.0),
            Rect(130.0, 95.0, 35.0, 16.0),
            Rect(85.0, 120.0, 30.0, 18.0),
            Rect(20.0, 45.0, 45.0, 18.0),
            Rect(135.0, 45.0, 45.0, 18.0),
            # Round storage tanks give the plant a less uniform silhouette than
            # an all-rectangle warehouse row, without blocking any corridor.
            Circle((45.0, 80.0), 5.0),
            Circle((155.0, 80.0), 5.0),
            Circle((100.0, 165.0), 8.0),
        ],
    )


def create_random_facility(
    seed: int | None = None,
    world_size: tuple[float, float] | None = None,
    building_count: int | None = None,
    max_attempts: int = 250,
) -> FixedMapConfig:
    """Procedurally generate a validated facility-like map with mixed obstacle shapes."""

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        candidate_world_size = world_size or (
            float(rng.uniform(150.0, 300.0)),
            float(rng.uniform(150.0, 300.0)),
        )
        candidate_building_count = building_count or int(rng.integers(5, 16))
        candidate = _generate_random_facility_candidate(rng, candidate_world_size, candidate_building_count)
        if validate_map(candidate).valid:
            return candidate

    raise RuntimeError(f"could not generate a valid random facility after {max_attempts} attempts")


def _generate_random_facility_candidate(
    rng: np.random.Generator,
    world_size: tuple[float, float],
    building_count: int,
) -> FixedMapConfig:
    width, height = world_size

    protected_zone = Rect(
        float(rng.uniform(width * 0.30, width * 0.42)),
        float(rng.uniform(height * 0.10, height * 0.18)),
        float(rng.uniform(width * 0.22, width * 0.34)),
        float(rng.uniform(height * 0.14, height * 0.20)),
    )

    assets = [
        Circle(
            (
                float(rng.uniform(protected_zone.min_x + 5.0, protected_zone.max_x - 5.0)),
                float(rng.uniform(protected_zone.min_y + 5.0, protected_zone.max_y - 5.0)),
            ),
            1.5,
        )
        for _ in range(2)
    ]

    red_spawn_zones = [Rect(width * 0.04, height * 0.88, width * 0.92, height * 0.08)]
    blue_spawn_zone = Rect(width * 0.04, height * 0.02, width * 0.92, height * 0.08)
    keepouts: list[Obstacle] = [blue_spawn_zone, *red_spawn_zones, protected_zone, *assets]

    buildings: list[Obstacle] = []
    attempts = 0
    while len(buildings) < building_count and attempts < building_count * 150:
        attempts += 1
        # Roughly a third of obstacles are round (tanks/silos/planters) so
        # random facilities aren't just scattered rectangles.
        if rng.uniform() < 0.35:
            candidate: Obstacle = Circle(
                (
                    float(rng.uniform(12.0, width - 12.0)),
                    float(rng.uniform(height * 0.24, height * 0.80)),
                ),
                float(rng.uniform(5.0, 11.0)),
            )
        else:
            candidate = Rect(
                float(rng.uniform(10.0, width - 24.0)),
                float(rng.uniform(height * 0.24, height * 0.80)),
                float(rng.uniform(9.0, 20.0)),
                float(rng.uniform(8.0, 22.0)),
            )

        if not _obstacle_in_world(candidate, world_size):
            continue
        if any(obstacles_overlap(candidate, keepout, margin=4.0) for keepout in keepouts):
            continue
        if any(obstacles_overlap(candidate, other, margin=3.0) for other in buildings):
            continue
        buildings.append(candidate)

    return FixedMapConfig(
        name="random_facility",
        world_size=world_size,
        buildings=buildings,
        protected_zone=protected_zone,
        assets=assets,
        red_spawn_zones=red_spawn_zones,
        blue_spawn_zone=blue_spawn_zone,
    )


def _obstacle_in_world(obstacle: Obstacle, world_size: tuple[float, float]) -> bool:
    width, height = world_size
    if isinstance(obstacle, Circle):
        x, y = obstacle.center
        return obstacle.radius <= x <= width - obstacle.radius and obstacle.radius <= y <= height - obstacle.radius
    return obstacle.min_x >= 0.0 and obstacle.min_y >= 0.0 and obstacle.max_x <= width and obstacle.max_y <= height


@dataclass(frozen=True)
class Route:
    """A single approach corridor: a waypoint polyline and its clear width."""

    points: list[tuple[float, float]]
    width: float


def create_routed_facility(
    seed: int | None = None,
    world_size: tuple[float, float] | None = None,
    num_routes: int | None = None,
    building_count: int | None = None,
    max_attempts: int = 200,
) -> FixedMapConfig:
    """Generate a facility by carving approach routes first, then filling the
    leftover space with obstacles.

    Unlike ``create_random_facility`` (which scatters obstacles and validates
    after the fact), this reserves route corridors as keepouts *before*
    placing any obstacle, so connectivity holds by construction rather than
    by luck. ``validate_map`` is still run as a safety net.
    """

    config, _routes = create_routed_facility_with_routes(
        seed=seed,
        world_size=world_size,
        num_routes=num_routes,
        building_count=building_count,
        max_attempts=max_attempts,
    )
    return config


def create_routed_facility_with_routes(
    seed: int | None = None,
    world_size: tuple[float, float] | None = None,
    num_routes: int | None = None,
    building_count: int | None = None,
    max_attempts: int = 200,
) -> tuple[FixedMapConfig, list[Route]]:
    """Same as ``create_routed_facility`` but also returns the route polylines,
    which are otherwise only an internal generation detail - useful for
    debugging/visualizing the generator itself."""

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        candidate_world_size = world_size or (
            float(rng.uniform(150.0, 300.0)),
            float(rng.uniform(150.0, 300.0)),
        )
        candidate_num_routes = num_routes or int(rng.integers(2, 4))
        candidate_building_count = building_count or int(rng.integers(10, 26))
        candidate, routes = _generate_routed_facility_candidate(
            rng, candidate_world_size, candidate_num_routes, candidate_building_count
        )
        if validate_map(candidate).valid:
            return candidate, routes

    raise RuntimeError(f"could not generate a valid routed facility after {max_attempts} attempts")


def _generate_routes(
    rng: np.random.Generator,
    world_size: tuple[float, float],
    red_spawn_zone: Rect,
    protected_zone: Rect,
    num_routes: int,
    band_count: int = 6,
) -> list[Route]:
    width, _height = world_size
    top_y = red_spawn_zone.min_y
    bottom_y = protected_zone.center[1]
    band_ys = [float(y) for y in np.linspace(top_y, bottom_y, band_count + 1)]

    routes: list[Route] = []
    for _ in range(num_routes):
        start_x = float(rng.uniform(red_spawn_zone.min_x + 5.0, red_spawn_zone.max_x - 5.0))
        target_x = float(rng.uniform(protected_zone.min_x + 5.0, protected_zone.max_x - 5.0))
        route_width = float(rng.uniform(14.0, 26.0))

        xs = [start_x]
        for step in range(1, band_count + 1):
            t = step / band_count
            base_x = start_x + (target_x - start_x) * t
            # Jitter shrinks as routes approach the protected zone so they
            # actually converge on it instead of wandering past it.
            jitter_amp = width * 0.12 * (1.0 - t * 0.5)
            x = float(np.clip(base_x + rng.uniform(-jitter_amp, jitter_amp), route_width, width - route_width))
            xs.append(x)

        points = list(zip(xs, band_ys, strict=True))
        routes.append(Route(points=points, width=route_width))

    if num_routes >= 2:
        # Snap two routes together at one middle band to create a deliberate
        # crossroads instead of hoping obstacle placement leaves one.
        cross_band = int(rng.integers(2, band_count - 1))
        i, j = rng.choice(num_routes, size=2, replace=False)
        shared_x = (routes[i].points[cross_band][0] + routes[j].points[cross_band][0]) / 2.0
        for k in (int(i), int(j)):
            _, y = routes[k].points[cross_band]
            routes[k].points[cross_band] = (shared_x, y)

    return routes


def _route_keepout_circles(route: Route, spacing_factor: float = 0.6) -> list[Circle]:
    step = max(route.width * spacing_factor, 3.0)
    circles: list[Circle] = []
    for start, end in zip(route.points[:-1], route.points[1:], strict=True):
        start_arr = np.array(start, dtype=np.float32)
        end_arr = np.array(end, dtype=np.float32)
        segment_len = float(np.linalg.norm(end_arr - start_arr))
        steps = max(1, int(np.ceil(segment_len / step)))
        for i in range(steps + 1):
            point = start_arr + (end_arr - start_arr) * (i / steps)
            circles.append(Circle((float(point[0]), float(point[1])), route.width / 2.0))
    return circles


def _generate_routed_facility_candidate(
    rng: np.random.Generator,
    world_size: tuple[float, float],
    num_routes: int,
    building_count: int,
) -> tuple[FixedMapConfig, list[Route]]:
    width, height = world_size

    protected_zone = Rect(
        float(rng.uniform(width * 0.30, width * 0.42)),
        float(rng.uniform(height * 0.10, height * 0.18)),
        float(rng.uniform(width * 0.22, width * 0.34)),
        float(rng.uniform(height * 0.14, height * 0.20)),
    )
    assets = [
        Circle(
            (
                float(rng.uniform(protected_zone.min_x + 5.0, protected_zone.max_x - 5.0)),
                float(rng.uniform(protected_zone.min_y + 5.0, protected_zone.max_y - 5.0)),
            ),
            1.5,
        )
        for _ in range(2)
    ]

    red_spawn_zones = [Rect(width * 0.04, height * 0.88, width * 0.92, height * 0.08)]
    blue_spawn_zone = Rect(width * 0.04, height * 0.02, width * 0.92, height * 0.08)

    routes = _generate_routes(rng, world_size, red_spawn_zones[0], protected_zone, num_routes)
    route_keepouts: list[Obstacle] = [
        circle for route in routes for circle in _route_keepout_circles(route)
    ]
    keepouts: list[Obstacle] = [blue_spawn_zone, *red_spawn_zones, protected_zone, *assets, *route_keepouts]

    buildings: list[Obstacle] = []
    attempts = 0
    while len(buildings) < building_count and attempts < building_count * 150:
        attempts += 1
        if rng.uniform() < 0.3:
            candidate: Obstacle = Circle(
                (
                    float(rng.uniform(8.0, width - 8.0)),
                    float(rng.uniform(height * 0.18, height * 0.86)),
                ),
                float(rng.uniform(4.0, 9.0)),
            )
        else:
            candidate = Rect(
                float(rng.uniform(5.0, width - 20.0)),
                float(rng.uniform(height * 0.18, height * 0.86)),
                float(rng.uniform(8.0, 18.0)),
                float(rng.uniform(8.0, 18.0)),
            )

        if not _obstacle_in_world(candidate, world_size):
            continue
        if any(obstacles_overlap(candidate, keepout, margin=2.0) for keepout in keepouts):
            continue
        if any(obstacles_overlap(candidate, other, margin=2.0) for other in buildings):
            continue
        buildings.append(candidate)

    config = FixedMapConfig(
        name="routed_facility",
        world_size=world_size,
        buildings=buildings,
        protected_zone=protected_zone,
        assets=assets,
        red_spawn_zones=red_spawn_zones,
        blue_spawn_zone=blue_spawn_zone,
    )
    return config, routes


FIXED_INSTANCE_FACTORIES = {
    "test2": create_fixed_industrial_facility,
}


def validate_instance(config: FixedMapConfig, min_buildings: int = 0) -> MapValidationResult:
    """Thin wrapper around ``validate_map`` for instance-family callers."""

    return validate_map(config, min_buildings=min_buildings)
