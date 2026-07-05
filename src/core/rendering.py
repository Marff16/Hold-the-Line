"""Rendering and state-export helpers for Hold The Line.

Split out of ``env.py`` so the core simulation loop isn't tangled up with
plotting/export code. Both functions take an already-stepped environment
instance and read its state; neither one mutates it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.geometry import Circle, Obstacle, Rect


if TYPE_CHECKING:
    from src.env import HoldTheLineEnv


def build_render_state(env: "HoldTheLineEnv", selected_agent: str | None = None) -> dict[str, Any]:
    """Return GUI-friendly state without importing any GUI framework."""

    agents = {}
    for agent, state in env._states.items():
        agents[agent] = {
            "team": "blue" if agent.startswith("blue_") else "red",
            "position": tuple(float(v) for v in state.pos),
            "velocity": tuple(float(v) for v in state.vel),
            "heading": _heading_from_velocity(state.vel),
            "radius": state.radius,
            "alive": state.alive,
            "max_speed": state.max_speed,
            "info_collected": state.info_collected,
            "info_threshold": env.red_info_threshold if agent.startswith("red_") else None,
            "detection_radius": env.blue_detection_radius if agent.startswith("blue_") else None,
            "intercept_radius": env.blue_intercept_radius if agent.startswith("blue_") else None,
            "scouting_radius": env.red_scouting_radius if agent.startswith("red_") else None,
            "visible": env._agent_visible_in_view(agent, selected_agent),
            "exposure_frac": (
                min(1.0, state.exposure / env.blue_destroy_time)
                if agent.startswith("red_") and env.blue_destroy_time > 0
                else None
            ),
        }

    detection_lines = []
    for blue_agent, red_agents in env._last_detections.items():
        blue_state = env._states.get(blue_agent)
        if blue_state is None:
            continue
        for red_agent in red_agents:
            red_state = env._states.get(red_agent)
            if red_state is None:
                continue
            detection_lines.append(
                {
                    "blue": blue_agent,
                    "red": red_agent,
                    "start": tuple(float(v) for v in blue_state.pos),
                    "end": tuple(float(v) for v in red_state.pos),
                }
            )

    return {
        "step": env._step_count,
        "dt": env.dt,
        "world_size": tuple(float(v) for v in env.world_size),
        "buildings": [_obstacle_to_dict(obstacle) for obstacle in env.buildings],
        "protected_zone": _rect_to_dict(env.map_config.protected_zone),
        "blue_spawn_zone": _rect_to_dict(env.map_config.blue_spawn_zone),
        "front_line_y": env.front_line_y,
        "red_spawn_zones": [_rect_to_dict(rect) for rect in env.map_config.red_spawn_zones],
        "terrain": {
            "image": env.map_config.terrain_image,
            "available": env.map_config.terrain_image is not None,
        },
        "assets": [
            {"center": tuple(float(v) for v in asset.center), "radius": asset.radius} for asset in env.assets
        ],
        "objectives": [
            {"center": tuple(float(v) for v in asset.center), "radius": asset.radius} for asset in env.assets
        ],
        "agents": agents,
        "detections": detection_lines,
        "team_visibility": env._team_visibility(),
        "selected_agent": selected_agent,
        "red_success": env._red_success,
    }


def _heading_from_velocity(velocity: np.ndarray) -> float:
    if float(np.linalg.norm(velocity)) < 1e-6:
        return 0.0
    return float(np.arctan2(velocity[1], velocity[0]))


def _rect_to_dict(rect: Rect) -> dict[str, Any]:
    return {
        "shape": "rect",
        "x": rect.x,
        "y": rect.y,
        "w": rect.w,
        "h": rect.h,
        "center": tuple(float(v) for v in rect.center),
    }


def _circle_to_dict(circle: Circle) -> dict[str, Any]:
    cx, cy = circle.center
    return {
        "shape": "circle",
        "x": cx - circle.radius,
        "y": cy - circle.radius,
        "w": circle.radius * 2.0,
        "h": circle.radius * 2.0,
        "center": (float(cx), float(cy)),
        "radius": float(circle.radius),
    }


def _obstacle_to_dict(obstacle: Obstacle) -> dict[str, Any]:
    if isinstance(obstacle, Circle):
        return _circle_to_dict(obstacle)
    return _rect_to_dict(obstacle)


def render_matplotlib(env: "HoldTheLineEnv"):
    """Draw the current state with matplotlib and return an rgb_array frame (or None for human mode)."""

    if env.render_mode not in {"human", "rgb_array"}:
        return None

    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/hold_the_line_matplotlib")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle as CirclePatch
        from matplotlib.patches import Rectangle
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional install.
        raise RuntimeError("matplotlib is required for rendering") from exc

    if env._fig is None or env._ax is None:
        env._fig, env._ax = plt.subplots(figsize=(7, 7))

    ax = env._ax
    ax.clear()
    ax.set_xlim(0, env.world_size[0])
    ax.set_ylim(0, env.world_size[1])
    ax.set_aspect("equal")
    ax.set_title(f"Hold The Line step {env._step_count}")
    ax.grid(True, color="#dddddd", linewidth=0.5)

    for zone in env.map_config.red_spawn_zones:
        ax.add_patch(
            Rectangle(
                (zone.x, zone.y),
                zone.w,
                zone.h,
                facecolor="#d62728",
                edgecolor="#d62728",
                alpha=0.12,
            )
        )

    blue_zone = env.map_config.blue_spawn_zone
    ax.add_patch(
        Rectangle(
            (blue_zone.x, blue_zone.y),
            blue_zone.w,
            blue_zone.h,
            facecolor="#1f77b4",
            edgecolor="#1f77b4",
            alpha=0.12,
        )
    )

    for building in env.buildings:
        if isinstance(building, Circle):
            ax.add_patch(
                CirclePatch(
                    building.center,
                    building.radius,
                    facecolor="#4c4c4c",
                    edgecolor="#222222",
                    alpha=0.85,
                )
            )
        else:
            ax.add_patch(
                Rectangle(
                    (building.x, building.y),
                    building.w,
                    building.h,
                    facecolor="#4c4c4c",
                    edgecolor="#222222",
                    alpha=0.85,
                )
            )

    protected = env.map_config.protected_zone
    ax.add_patch(
        Rectangle(
            (protected.x, protected.y),
            protected.w,
            protected.h,
            facecolor="#9ad0f5",
            edgecolor="#1f77b4",
            alpha=0.25,
            linewidth=2.0,
        )
    )

    for asset in env.assets:
        ax.add_patch(
            CirclePatch(
                asset.center,
                asset.radius,
                facecolor="#ffd166",
                edgecolor="#9b6b00",
                linewidth=1.5,
            )
        )

    for agent in env.possible_agents:
        state = env._states.get(agent)
        if state is None:
            continue
        color = "#1f77b4" if agent.startswith("blue_") else "#d62728"
        marker = "o" if state.alive else "x"
        alpha = 1.0 if state.alive else 0.35
        ax.scatter(state.pos[0], state.pos[1], c=color, marker=marker, s=70, alpha=alpha)
        ax.text(state.pos[0] + 1.0, state.pos[1] + 1.0, agent, color=color, fontsize=8)
        if agent.startswith("blue_") and state.alive:
            ax.add_patch(
                CirclePatch(
                    tuple(state.pos),
                    env.blue_detection_radius,
                    facecolor="none",
                    edgecolor=color,
                    alpha=0.12,
                )
            )
            ax.add_patch(
                CirclePatch(
                    tuple(state.pos),
                    env.blue_intercept_radius,
                    facecolor="none",
                    edgecolor=color,
                    alpha=0.5,
                    linestyle="--",
                )
            )
        if agent.startswith("red_") and state.alive:
            ax.add_patch(
                CirclePatch(
                    tuple(state.pos),
                    env.red_scouting_radius,
                    facecolor="none",
                    edgecolor=color,
                    alpha=0.12,
                )
            )

    if env.render_mode == "human":
        plt.pause(0.001)
        return None

    env._fig.canvas.draw()
    width, height = env._fig.canvas.get_width_height()
    image = np.asarray(env._fig.canvas.buffer_rgba(), dtype=np.uint8)
    return image.reshape((height, width, 4))[:, :, :3].copy()
