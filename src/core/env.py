"""PettingZoo-style Parallel API environment for Hold The Line."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from src.core.geometry import (
    Circle,
    Obstacle,
    Rect,
    circle_intersects_obstacle,
    clip_norm,
    distance,
    expand_obstacle,
    line_of_sight_clear,
    obstacle_contains_point,
    segment_intersects_obstacle,
)
from src.core.map_config import FixedMapConfig, default_fixed_map
from src.core.rendering import build_render_state, render_matplotlib
from src.core.spaces import Box


try:  # pragma: no cover - exercised only when PettingZoo is installed.
    from pettingzoo import ParallelEnv
except ModuleNotFoundError:  # pragma: no cover - fallback is covered instead.

    class ParallelEnv:  # type: ignore[no-redef]
        metadata: dict[str, Any] = {}


Array = np.ndarray


@dataclass
class DroneState:
    pos: Array
    vel: Array
    max_speed: float
    radius: float
    alive: bool = True
    info_collected: float = 0.0
    # Seconds a red drone has spent continuously tethered by a blue drone's
    # detection (unused for blue agents). Reset to 0 the instant the tether
    # breaks, so ducking behind a building fully resets the danger.
    exposure: float = 0.0


class HoldTheLineEnv(ParallelEnv):
    """Fixed-map 2D Hold The Line environment.

    Agents use continuous 2D acceleration actions. The environment follows the
    PettingZoo Parallel API shape while remaining runnable without PettingZoo
    installed, which keeps local development lightweight.
    """

    metadata = {
        "name": "HoldTheLine-v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        map_config: FixedMapConfig | None = None,
        dt: float = 1 / 30,
        max_episode_steps: int = 1800,
        acceleration_scale: float = 72.0,
        velocity_damping: float = 0.98,
        movement_penalty: float = 0.001,
    ) -> None:
        self.render_mode = render_mode
        self.map_config = map_config or default_fixed_map()
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.acceleration_scale = acceleration_scale
        self.velocity_damping = velocity_damping
        self.movement_penalty = movement_penalty

        self.num_blue = self.map_config.blue_drones.count
        self.num_red = self.map_config.red_drones.count
        self.blue_agents = [f"blue_{i}" for i in range(self.num_blue)]
        self.red_agents = [f"red_{i}" for i in range(self.num_red)]
        self.possible_agents = self.blue_agents + self.red_agents
        self.agents: list[str] = []

        self.blue_max_speed = self.map_config.blue_drones.max_speed
        self.red_max_speed = self.map_config.red_drones.max_speed
        self.blue_detection_radius = self.map_config.blue_drones.detection_radius
        self.blue_intercept_radius = self.map_config.blue_drones.intercept_radius
        self.blue_destroy_time = self.map_config.blue_drones.destroy_time
        self.red_scouting_radius = self.map_config.red_drones.scouting_radius
        self.red_info_threshold = self.map_config.red_drones.info_threshold
        self.front_line_y = (
            self.map_config.front_line_y
            if self.map_config.front_line_y is not None
            else self._default_front_line_y()
        )

        self._rng = np.random.default_rng()
        self._step_count = 0
        self._states: dict[str, DroneState] = {}
        self._red_success = False
        self._last_detections: dict[str, list[str]] = {}
        self._fig = None
        self._ax = None

    @property
    def world_size(self) -> Array:
        return np.array(self.map_config.world_size, dtype=np.float32)

    @property
    def buildings(self) -> list[Obstacle]:
        return self.map_config.buildings

    @property
    def assets(self) -> list[Circle]:
        return self.map_config.assets

    def _default_front_line_y(self) -> float:
        blue_zone = self.map_config.blue_spawn_zone
        red_edge = min(zone.y for zone in self.map_config.red_spawn_zones)
        return (blue_zone.y + blue_zone.h + red_edge) / 2.0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Array], dict[str, dict[str, Any]]]:
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self._step_count = 0
        self._red_success = False
        self._states = {}
        self._last_detections = {agent: [] for agent in self.blue_agents}

        blue_positions = self._spawn_positions(
            [self.map_config.blue_spawn_zone],
            self.num_blue,
            self.map_config.blue_drones.radius,
        )
        for agent, position in zip(self.blue_agents, blue_positions, strict=True):
            self._states[agent] = DroneState(
                pos=position,
                vel=np.zeros(2, dtype=np.float32),
                max_speed=self.blue_max_speed,
                radius=self.map_config.blue_drones.radius,
            )

        red_positions = self._spawn_positions(
            self.map_config.red_spawn_zones,
            self.num_red,
            self.map_config.red_drones.radius,
        )
        for agent, position in zip(self.red_agents, red_positions, strict=True):
            self._states[agent] = DroneState(
                pos=position,
                vel=np.zeros(2, dtype=np.float32),
                max_speed=self.red_max_speed,
                radius=self.map_config.red_drones.radius,
            )

        observations = self._observe_all()
        infos = {agent: self._agent_info(agent) for agent in self.agents}
        return observations, infos

    def _spawn_positions(self, zones: list[Rect], count: int, radius: float) -> list[Array]:
        if count <= 0:
            return []

        positions: list[Array] = []
        base = count // len(zones)
        remainder = count % len(zones)
        for zone_index, zone in enumerate(zones):
            zone_count = base + (1 if zone_index < remainder else 0)
            if zone_count:
                positions.extend(self._spawn_positions_in_zone(zone, zone_count, radius))
        return positions

    def _spawn_positions_in_zone(self, zone: Rect, count: int, radius: float) -> list[Array]:
        horizontal = zone.w >= zone.h
        usable_min = zone.min_x + radius if horizontal else zone.min_y + radius
        usable_max = zone.max_x - radius if horizontal else zone.max_y - radius
        if count == 1 or usable_max <= usable_min:
            line_values = np.array([(usable_min + usable_max) * 0.5], dtype=np.float32)
        else:
            line_values = np.linspace(usable_min, usable_max, count, dtype=np.float32)

        positions = []
        for value in line_values:
            if horizontal:
                point = np.array(
                    [value, np.clip(zone.center[1], zone.min_y + radius, zone.max_y - radius)],
                    dtype=np.float32,
                )
            else:
                point = np.array(
                    [np.clip(zone.center[0], zone.min_x + radius, zone.max_x - radius), value],
                    dtype=np.float32,
                )
            if any(obstacle_contains_point(point, building, margin=radius) for building in self.buildings):
                raise RuntimeError(f"spawn point {point.tolist()} is blocked by a building")
            positions.append(point)
        return positions

    def step(
        self,
        actions: dict[str, Array],
    ) -> tuple[
        dict[str, Array],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        self._step_count += 1
        rewards = {agent: 0.0 for agent in self.agents}
        rewards = self._apply_movement(actions, rewards)

        detections = self._compute_detections()
        self._last_detections = detections
        intercepted_reds = self._apply_interceptions(detections)
        # Intercept bonus is a team reward: split evenly across blue agents rather
        # than credited to whichever drone made contact, since detection/positioning
        # is a team effort in this MVP (no credit assignment model yet).
        for red_agent in intercepted_reds:
            rewards[red_agent] -= 1.0
            for blue_agent in self.blue_agents:
                rewards[blue_agent] += 1.0 / self.num_blue

        info_delta = self._apply_scouting()
        for red_agent, delta in info_delta.items():
            rewards[red_agent] += delta / self.red_info_threshold

        successful_reds = [
            red_agent
            for red_agent in self.red_agents
            if self._states[red_agent].alive
            and self._states[red_agent].info_collected >= self.red_info_threshold
        ]
        self._red_success = bool(successful_reds)
        if self._red_success:
            for red_agent in successful_reds:
                rewards[red_agent] += 1.0
            for blue_agent in self.blue_agents:
                rewards[blue_agent] -= 1.0

        all_reds_inactive = all(not self._states[red_agent].alive for red_agent in self.red_agents)
        time_limit = self._step_count >= self.max_episode_steps
        episode_done = self._red_success or all_reds_inactive or time_limit
        # Blue only gets the terminal win bonus if red never reached its scouting
        # threshold - running out the clock with reds still alive still counts as a
        # blue win, since red failed its objective either way.
        if episode_done and not self._red_success:
            for blue_agent in self.blue_agents:
                rewards[blue_agent] += 1.0

        terminations = {agent: bool(self._red_success or all_reds_inactive) for agent in self.agents}
        truncations = {agent: bool(time_limit and not (self._red_success or all_reds_inactive)) for agent in self.agents}
        observations = self._observe_all() if not episode_done else {}
        infos = {agent: self._agent_info(agent) for agent in self.agents}

        if episode_done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _apply_movement(self, actions: dict[str, Array], rewards: dict[str, float]) -> dict[str, float]:
        world = self.world_size

        for agent in self.agents:
            state = self._states[agent]
            if not state.alive:
                continue

            raw_action = np.asarray(actions.get(agent, np.zeros(2, dtype=np.float32)), dtype=np.float32)
            if raw_action.shape != (2,):
                raise ValueError(f"action for {agent} must have shape (2,), got {raw_action.shape}")

            action = np.clip(raw_action, -1.0, 1.0)
            old_pos = state.pos.copy()
            state.vel = clip_norm(
                state.vel * self.velocity_damping + action * self.acceleration_scale * self.dt,
                state.max_speed,
            ).astype(np.float32)
            candidate_pos = state.pos + state.vel * self.dt

            boundary_hit = bool(
                candidate_pos[0] - state.radius < 0.0
                or candidate_pos[1] - state.radius < 0.0
                or candidate_pos[0] + state.radius > world[0]
                or candidate_pos[1] + state.radius > world[1]
            )
            obstacle_hit = any(
                circle_intersects_obstacle(candidate_pos, state.radius, building)
                or segment_intersects_obstacle(old_pos, candidate_pos, expand_obstacle(building, state.radius))
                for building in self.buildings
            )
            if boundary_hit or obstacle_hit:
                state.pos = np.clip(old_pos, [state.radius, state.radius], world - state.radius).astype(np.float32)
                state.vel = np.zeros(2, dtype=np.float32)
            else:
                state.pos = candidate_pos.astype(np.float32)

            rewards[agent] -= self.movement_penalty * float(np.linalg.norm(action))

        return rewards

    def _compute_detections(self) -> dict[str, list[str]]:
        detections: dict[str, list[str]] = {blue_agent: [] for blue_agent in self.blue_agents}
        for blue_agent in self.blue_agents:
            for red_agent in self.red_agents:
                if self.blue_can_detect_red(blue_agent, red_agent):
                    detections[blue_agent].append(red_agent)
        return detections

    def blue_can_detect_red(self, blue_agent: str, red_agent: str) -> bool:
        blue_state = self._states[blue_agent]
        red_state = self._states[red_agent]
        if not blue_state.alive or not red_state.alive:
            return False
        in_range = distance(blue_state.pos, red_state.pos) <= self.blue_detection_radius
        return in_range and line_of_sight_clear(blue_state.pos, red_state.pos, self.buildings)

    def red_can_observe_blue(self, red_agent: str, blue_agent: str) -> bool:
        red_state = self._states[red_agent]
        blue_state = self._states[blue_agent]
        if not red_state.alive or not blue_state.alive:
            return False
        in_range = distance(red_state.pos, blue_state.pos) <= self.blue_detection_radius
        return in_range and line_of_sight_clear(red_state.pos, blue_state.pos, self.buildings)

    def _apply_interceptions(self, detections: dict[str, list[str]]) -> list[str]:
        # A red drone isn't destroyed the instant it's spotted - it has to stay
        # tethered (inside detection_radius with clear line of sight, the same
        # condition that draws the detection line in the UI) continuously for
        # destroy_time seconds. Slipping behind a building breaks line of sight,
        # which drops it out of `detections` and resets exposure to zero, so
        # hiding around a corner fully resets the danger rather than just
        # pausing it.
        tethered_reds = {red_agent for red_agents in detections.values() for red_agent in red_agents}
        destroyed: list[str] = []
        for red_agent in self.red_agents:
            red_state = self._states[red_agent]
            if not red_state.alive:
                continue
            if red_agent in tethered_reds:
                red_state.exposure += self.dt
                if red_state.exposure >= self.blue_destroy_time:
                    red_state.alive = False
                    red_state.vel = np.zeros(2, dtype=np.float32)
                    destroyed.append(red_agent)
            else:
                red_state.exposure = 0.0
        return destroyed

    def _apply_scouting(self) -> dict[str, float]:
        deltas: dict[str, float] = {}
        for red_agent in self.red_agents:
            red_state = self._states[red_agent]
            if not red_state.alive:
                continue

            can_scout = any(
                distance(red_state.pos, asset.center_array) <= self.red_scouting_radius
                and line_of_sight_clear(red_state.pos, asset.center_array, self.buildings)
                for asset in self.assets
            )
            if can_scout:
                before = red_state.info_collected
                red_state.info_collected = min(
                    self.red_info_threshold,
                    red_state.info_collected + self.dt,
                )
                deltas[red_agent] = red_state.info_collected - before
            else:
                deltas[red_agent] = 0.0
        return deltas

    def _observe_all(self) -> dict[str, Array]:
        return {agent: self._observe(agent) for agent in self.agents}

    def _observe(self, agent: str) -> Array:
        if agent.startswith("blue_"):
            return self._observe_blue(agent)
        return self._observe_red(agent)

    def _observe_blue(self, agent: str) -> Array:
        # Layout: own pos/vel (4) + per-red [visible, rel_pos(2), rel_vel(2), info_frac, alive] (7 each)
        # + per-other-blue rel_pos/rel_vel (4 each) + per-asset rel_pos (2 each). The shape formula in
        # observation_space() below encodes this same layout - keep both in sync if it changes.
        state = self._states[agent]
        values: list[float] = []
        values.extend(self._norm_pos(state.pos))
        values.extend(self._norm_vel(state.vel, self.blue_max_speed))

        for red_agent in self.red_agents:
            red_state = self._states[red_agent]
            visible = red_agent in self._last_detections.get(agent, [])
            values.append(1.0 if visible else 0.0)
            if visible and red_state.alive:
                values.extend(self._norm_rel_pos(red_state.pos - state.pos))
                values.extend(self._norm_vel(red_state.vel - state.vel, self.blue_max_speed + self.red_max_speed))
                values.append(red_state.info_collected / self.red_info_threshold)
                values.append(1.0)
            else:
                values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for other_agent in self.blue_agents:
            if other_agent == agent:
                continue
            other_state = self._states[other_agent]
            values.extend(self._norm_rel_pos(other_state.pos - state.pos))
            values.extend(self._norm_vel(other_state.vel - state.vel, self.blue_max_speed * 2.0))

        for asset in self.assets:
            values.extend(self._norm_rel_pos(asset.center_array - state.pos))

        return np.array(values, dtype=np.float32)

    def _observe_red(self, agent: str) -> Array:
        # Layout: own pos/vel (4) + per-asset rel_pos (2 each) + per-blue [visible, rel_pos(2), rel_vel(2)]
        # (5 each) + own info fraction (1) + rel_pos to protected zone (2). Mirrors observation_space() below.
        state = self._states[agent]
        values: list[float] = []
        values.extend(self._norm_pos(state.pos))
        values.extend(self._norm_vel(state.vel, self.red_max_speed))

        for asset in self.assets:
            values.extend(self._norm_rel_pos(asset.center_array - state.pos))

        for blue_agent in self.blue_agents:
            blue_state = self._states[blue_agent]
            visible = self.red_can_observe_blue(agent, blue_agent)
            values.append(1.0 if visible else 0.0)
            if visible:
                values.extend(self._norm_rel_pos(blue_state.pos - state.pos))
                values.extend(self._norm_vel(blue_state.vel - state.vel, self.blue_max_speed + self.red_max_speed))
            else:
                values.extend([0.0, 0.0, 0.0, 0.0])

        values.append(state.info_collected / self.red_info_threshold)
        values.extend(self._norm_rel_pos(self.protected_center - state.pos))
        return np.array(values, dtype=np.float32)

    @property
    def protected_center(self) -> Array:
        return self.map_config.protected_zone.center

    def get_observations(self) -> dict[str, Array]:
        return self._observe_all()

    def render_state(self, selected_agent: str | None = None) -> dict[str, Any]:
        """Return GUI-friendly state without importing any GUI framework.

        The actual dict-building lives in ``src.rendering`` to keep this module
        focused on the simulation loop; this method just keeps the public API
        (``env.render_state(...)``) stable for callers like the web app.
        """

        return build_render_state(self, selected_agent)

    def _team_visibility(self) -> dict[str, list[str]]:
        blue_visible_reds = sorted(
            {
                red_agent
                for blue_agent in self.blue_agents
                for red_agent in self.red_agents
                if self.blue_can_detect_red(blue_agent, red_agent)
            }
        )
        red_visible_blues = sorted(
            {
                blue_agent
                for red_agent in self.red_agents
                for blue_agent in self.blue_agents
                if self.red_can_observe_blue(red_agent, blue_agent)
            }
        )
        return {
            "blue_visible_reds": blue_visible_reds,
            "red_visible_blues": red_visible_blues,
        }

    def _agent_visible_in_view(self, agent: str, selected_agent: str | None) -> bool:
        if selected_agent is None or selected_agent == agent:
            return True
        if selected_agent.startswith("blue_") and agent.startswith("red_"):
            return self.blue_can_detect_red(selected_agent, agent)
        if selected_agent.startswith("red_") and agent.startswith("blue_"):
            return self.red_can_observe_blue(selected_agent, agent)
        return True

    def _norm_pos(self, pos: Array) -> list[float]:
        return (pos / self.world_size).astype(np.float32).tolist()

    def _norm_rel_pos(self, rel_pos: Array) -> list[float]:
        return (rel_pos / self.world_size).astype(np.float32).tolist()

    def _norm_vel(self, vel: Array, max_speed: float) -> list[float]:
        if max_speed <= 0.0:
            return [0.0, 0.0]
        return np.clip(vel / max_speed, -1.0, 1.0).astype(np.float32).tolist()

    def _agent_info(self, agent: str) -> dict[str, Any]:
        state = self._states[agent]
        info: dict[str, Any] = {
            "position": state.pos.copy(),
            "velocity": state.vel.copy(),
            "alive": state.alive,
            "team": "blue" if agent.startswith("blue_") else "red",
        }
        if agent.startswith("red_"):
            info["info_collected"] = state.info_collected
            info["info_fraction"] = state.info_collected / self.red_info_threshold
        else:
            info["detected_reds"] = self._last_detections.get(agent, [])[:]
        return info

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Box:
        if agent.startswith("blue_"):
            shape = 4 + self.num_red * 7 + (self.num_blue - 1) * 4 + len(self.assets) * 2
            return Box(low=-1.0, high=1.0, shape=(shape,), dtype=np.float32)
        shape = 4 + len(self.assets) * 2 + self.num_blue * 5 + 3
        return Box(low=-1.0, high=1.0, shape=(shape,), dtype=np.float32)

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Box:
        del agent
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def state(self) -> Array:
        """Global state vector useful for centralized critics."""

        values: list[float] = []
        for agent in self.possible_agents:
            drone = self._states.get(agent)
            if drone is None:
                values.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                max_speed = self.blue_max_speed if agent.startswith("blue_") else self.red_max_speed
                values.extend(self._norm_pos(drone.pos))
                values.extend(self._norm_vel(drone.vel, max_speed))
                values.append(1.0 if drone.alive else 0.0)
        for red_agent in self.red_agents:
            drone = self._states.get(red_agent)
            values.append(0.0 if drone is None else drone.info_collected / self.red_info_threshold)
        return np.array(values, dtype=np.float32)

    def render(self):
        """Draw the current state with matplotlib; body lives in ``src.rendering``."""

        return render_matplotlib(self)

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
        self._fig = None
        self._ax = None


def parallel_env(**kwargs) -> HoldTheLineEnv:
    return HoldTheLineEnv(**kwargs)


def env(**kwargs) -> HoldTheLineEnv:
    """Alias matching common PettingZoo module conventions."""

    return parallel_env(**kwargs)
