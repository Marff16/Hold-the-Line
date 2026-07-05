"""Geometry primitives for the Hold The Line MVP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle represented by its lower-left corner and size."""

    x: float
    y: float
    w: float
    h: float

    @property
    def min_x(self) -> float:
        return self.x

    @property
    def max_x(self) -> float:
        return self.x + self.w

    @property
    def min_y(self) -> float:
        return self.y

    @property
    def max_y(self) -> float:
        return self.y + self.h

    @property
    def center(self) -> Array:
        return np.array([self.x + self.w * 0.5, self.y + self.h * 0.5], dtype=np.float32)

    def expanded(self, margin: float) -> "Rect":
        return Rect(self.x - margin, self.y - margin, self.w + margin * 2.0, self.h + margin * 2.0)

    def contains_point(self, point: Array, margin: float = 0.0) -> bool:
        px, py = point
        return (
            self.min_x - margin <= px <= self.max_x + margin
            and self.min_y - margin <= py <= self.max_y + margin
        )


@dataclass(frozen=True)
class Circle:
    center: tuple[float, float]
    radius: float

    @property
    def center_array(self) -> Array:
        return np.array(self.center, dtype=np.float32)

    def contains_point(self, point: Array) -> bool:
        return float(np.linalg.norm(point - self.center_array)) <= self.radius


def distance(a: Array, b: Array) -> float:
    return float(np.linalg.norm(a - b))


def clip_norm(vector: Array, max_norm: float) -> Array:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm == 0.0:
        return vector
    return vector * (max_norm / norm)


def segment_intersects_rect(start: Array, end: Array, rect: Rect) -> bool:
    """Return True when a segment intersects a rectangle.

    Liang-Barsky clipping is compact and robust for axis-aligned rectangles. A
    segment that starts inside the rectangle counts as intersecting, which is the
    right behavior for both line-of-sight blocking and movement collision.
    """

    if rect.contains_point(start) or rect.contains_point(end):
        return True

    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    p = [-dx, dx, -dy, dy]
    q = [
        float(start[0] - rect.min_x),
        float(rect.max_x - start[0]),
        float(start[1] - rect.min_y),
        float(rect.max_y - start[1]),
    ]
    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(p, q, strict=True):
        if pi == 0.0:
            if qi < 0.0:
                return False
            continue

        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return False
            u1 = max(u1, t)
        else:
            if t < u1:
                return False
            u2 = min(u2, t)

    return u1 <= u2


def point_in_any_rect(point: Array, rects: list[Rect]) -> bool:
    return any(rect.contains_point(point) for rect in rects)


def circle_intersects_rect(center: Array, radius: float, rect: Rect) -> bool:
    nearest = np.array(
        [
            np.clip(center[0], rect.min_x, rect.max_x),
            np.clip(center[1], rect.min_y, rect.max_y),
        ],
        dtype=np.float32,
    )
    return distance(center, nearest) <= radius


# Obstacle model: buildings are either axis-aligned Rects or round Circles
# (silos, towers, planters, ...). Every hot-path check below dispatches on
# the concrete type so callers can pass a mixed list without caring which
# shape a given obstacle is.
Obstacle = Rect | Circle


def segment_intersects_circle(start: Array, end: Array, circle: Circle) -> bool:
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    center = circle.center_array
    segment = end - start
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq == 0.0:
        return distance(start, center) <= circle.radius
    t = float(np.clip(np.dot(center - start, segment) / seg_len_sq, 0.0, 1.0))
    closest = start + segment * t
    return distance(closest, center) <= circle.radius


def segment_intersects_obstacle(start: Array, end: Array, obstacle: Obstacle) -> bool:
    if isinstance(obstacle, Rect):
        return segment_intersects_rect(start, end, obstacle)
    return segment_intersects_circle(start, end, obstacle)


def circle_intersects_circle(center: Array, radius: float, circle: Circle) -> bool:
    return distance(center, circle.center_array) <= radius + circle.radius


def circle_intersects_obstacle(center: Array, radius: float, obstacle: Obstacle) -> bool:
    if isinstance(obstacle, Rect):
        return circle_intersects_rect(center, radius, obstacle)
    return circle_intersects_circle(center, radius, obstacle)


def expand_obstacle(obstacle: Obstacle, margin: float) -> Obstacle:
    if isinstance(obstacle, Rect):
        return obstacle.expanded(margin)
    return Circle(obstacle.center, obstacle.radius + margin)


def obstacle_contains_point(point: Array, obstacle: Obstacle, margin: float = 0.0) -> bool:
    if isinstance(obstacle, Rect):
        return obstacle.contains_point(point, margin=margin)
    return distance(point, obstacle.center_array) <= obstacle.radius + margin


def point_in_any_obstacle(point: Array, obstacles: list[Obstacle]) -> bool:
    return any(obstacle_contains_point(point, obstacle) for obstacle in obstacles)


def obstacles_overlap(a: Obstacle, b: Obstacle, margin: float = 0.0) -> bool:
    if isinstance(a, Rect) and isinstance(b, Rect):
        return not (
            a.max_x + margin < b.min_x
            or a.min_x - margin > b.max_x
            or a.max_y + margin < b.min_y
            or a.min_y - margin > b.max_y
        )
    if isinstance(a, Circle) and isinstance(b, Circle):
        return circle_intersects_circle(a.center_array, a.radius + margin, b)
    rect, circle = (a, b) if isinstance(a, Rect) else (b, a)
    return circle_intersects_rect(circle.center_array, circle.radius + margin, rect)


def line_of_sight_clear(start: Array, end: Array, blockers: list[Obstacle]) -> bool:
    return not any(segment_intersects_obstacle(start, end, blocker) for blocker in blockers)
