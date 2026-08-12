"""Heuristic baseline policy: wanders with a slowly-drifting heading while
steering away from nearby obstacles and the world boundary, instead of
RandomPolicy's uniform noise (which flails and crashes into buildings
constantly). Has no notion of the objective - it's meant as a sturdier
baseline opponent for training, not a competent player.
"""

from __future__ import annotations

import numpy as np

from src.core.geometry import Obstacle, Rect
from src.core.map_config import FixedMapConfig
from src.policies.base import Policy


class ObstacleAvoidingPolicy(Policy):
    def __init__(
        self,
        map_config: FixedMapConfig,
        seed: int | None = None,
        avoid_radius: float = 10.0,
        edge_margin: float = 6.0,
        heading_drift: float = 0.4,
    ) -> None:
        self.buildings: list[Obstacle] = map_config.buildings
        self.world_size = np.array(map_config.world_size, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.avoid_radius = avoid_radius
        self.edge_margin = edge_margin
        self.heading_drift = heading_drift
        # Per-agent wander heading (radians), persisted across steps so
        # movement reads as a drifting stroll rather than jitter.
        self._headings: dict[str, float] = {}

    def act(self, obs: np.ndarray, agent_id: str) -> np.ndarray:
        # obs[:2] is own position normalized by world_size for both teams
        # (see env.py _observe_blue/_observe_red, both start with _norm_pos).
        pos = obs[:2].astype(np.float32) * self.world_size

        angle = self._headings.get(agent_id)
        if angle is None:
            angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        angle += float(self.rng.normal(0.0, self.heading_drift))
        self._headings[agent_id] = angle
        heading = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

        avoidance = self._obstacle_avoidance(pos) + self._edge_avoidance(pos)

        action = heading + avoidance
        norm = float(np.linalg.norm(action))
        if norm > 1e-6:
            action = action / norm
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _obstacle_avoidance(self, pos: np.ndarray) -> np.ndarray:
        push = np.zeros(2, dtype=np.float32)
        for obstacle in self.buildings:
            nearest = _nearest_point(pos, obstacle)
            delta = pos - nearest
            dist = float(np.linalg.norm(delta))
            if 1e-6 < dist < self.avoid_radius:
                push += (delta / dist) * (self.avoid_radius - dist) / self.avoid_radius
            elif dist <= 1e-6:
                push += self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        return push * 2.5

    def _edge_avoidance(self, pos: np.ndarray) -> np.ndarray:
        push = np.zeros(2, dtype=np.float32)
        margin = self.edge_margin
        width, height = self.world_size
        if pos[0] < margin:
            push[0] += (margin - pos[0]) / margin
        elif pos[0] > width - margin:
            push[0] -= (pos[0] - (width - margin)) / margin
        if pos[1] < margin:
            push[1] += (margin - pos[1]) / margin
        elif pos[1] > height - margin:
            push[1] -= (pos[1] - (height - margin)) / margin
        return push * 2.5


def _nearest_point(pos: np.ndarray, obstacle: Obstacle) -> np.ndarray:
    if isinstance(obstacle, Rect):
        return np.array(
            [
                np.clip(pos[0], obstacle.min_x, obstacle.max_x),
                np.clip(pos[1], obstacle.min_y, obstacle.max_y),
            ],
            dtype=np.float32,
        )
    center = obstacle.center_array
    delta = pos - center
    dist = float(np.linalg.norm(delta))
    if dist <= obstacle.radius:
        return pos.copy()
    return center + (delta / dist) * obstacle.radius
