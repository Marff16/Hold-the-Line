"""FastAPI web app for the Hold The Line simulator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import parallel_env
from src.core.instance_loader import INSTANCE_DIR, list_instances, load_instance
from src.policies import LearnedPolicy, ObstacleAvoidingPolicy, RandomPolicy


WEB_DIR = Path(__file__).resolve().parents[2] / "web"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
LEARNED_CHECKPOINTS = {
    "Learned Blue": ("blue", CHECKPOINT_DIR / "blue_actor_critic.pt"),
    "Learned Red": ("red", CHECKPOINT_DIR / "red_actor_critic.pt"),
}


class ControlRequest(BaseModel):
    playing: bool | None = None
    speed: int | None = None
    policy_blue: str | None = None
    policy_red: str | None = None
    selected_agent: str | None = None
    terrain_enabled: bool | None = None


class LoadInstanceRequest(BaseModel):
    instance_id: str


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for websocket in self.connections:
            try:
                await websocket.send_json(message)
            except RuntimeError:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)


class WebSimulation:
    # Policies are chosen per team, not per drone: agents on the same team
    # are homogeneous and share one policy (standard multi-agent parameter
    # sharing). The actual available set is built per-instance in
    # _reset_env() (self.policies) - "Learned Blue" only shows up once a
    # checkpoint actually exists on disk. POLICY_TEAM_RESTRICTIONS keeps a
    # team-specific learned policy (trained on that team's own obs_dim) out
    # of the other team's dropdown, where it would just error out.
    DEFAULT_POLICY = "Random"
    POLICY_TEAM_RESTRICTIONS: dict[str, str] = {name: team for name, (team, _path) in LEARNED_CHECKPOINTS.items()}

    def __init__(self) -> None:
        self.manager = ConnectionManager()
        self.lock = asyncio.Lock()
        self.playing = False
        self.speed = 1
        self.policy_blue = self.DEFAULT_POLICY
        self.policy_red = self.DEFAULT_POLICY
        self.selected_agent: str | None = None
        self.terrain_enabled = False
        available = list_instances()
        self.instance_id = available[0]["id"] if available else "default"
        self._reset_env()

    def _reset_env(self) -> None:
        if self.instance_id == "default":
            self.env = parallel_env()
        else:
            self.env = parallel_env(map_config=load_instance(self.instance_id))
        self.env.reset(seed=7)
        if self.selected_agent not in self.env.possible_agents:
            self.selected_agent = None
        # "Avoider" needs the current instance's buildings, so it's rebuilt
        # here rather than once at startup like RandomPolicy.
        self.policies = {
            "Random": RandomPolicy(seed=11),
            "Avoider": ObstacleAvoidingPolicy(self.env.map_config, seed=11),
        }
        if LearnedPolicy is not None:
            for name, (_team, path) in LEARNED_CHECKPOINTS.items():
                if not path.exists():
                    continue
                try:
                    self.policies[name] = LearnedPolicy(path)
                except Exception:  # noqa: BLE001 - a half-written checkpoint mid-save shouldn't crash the app
                    pass
        self.policy_options_blue = self._options_for_team("blue")
        self.policy_options_red = self._options_for_team("red")
        if self.policy_blue not in self.policy_options_blue:
            self.policy_blue = self.DEFAULT_POLICY
        if self.policy_red not in self.policy_options_red:
            self.policy_red = self.DEFAULT_POLICY

    def _options_for_team(self, team: str) -> list[str]:
        return [
            name
            for name in self.policies
            if self.POLICY_TEAM_RESTRICTIONS.get(name, team) == team
        ]

    def snapshot(self) -> dict[str, Any]:
        state = self.env.render_state(self.selected_agent)
        state["terrain"]["enabled"] = self.terrain_enabled
        state["terrain"]["url"] = self._terrain_url(state["terrain"]["image"])
        state["controls"] = {
            "playing": self.playing,
            "speed": self.speed,
            "policy_blue": self.policy_blue,
            "policy_red": self.policy_red,
            "policy_options_blue": self.policy_options_blue,
            "policy_options_red": self.policy_options_red,
            "selected_agent": self.selected_agent,
            "agents": self.env.possible_agents,
            "terrain_enabled": self.terrain_enabled,
            "instance_id": self.instance_id,
            "instances": list_instances(),
        }
        return state

    def apply_control(self, request: ControlRequest) -> None:
        if request.playing is not None:
            self.playing = request.playing
        if request.speed is not None:
            self.speed = int(np.clip(request.speed, 1, 30))
        if request.policy_blue is not None and request.policy_blue in self.policy_options_blue:
            self.policy_blue = request.policy_blue
        if request.policy_red is not None and request.policy_red in self.policy_options_red:
            self.policy_red = request.policy_red
        if request.selected_agent == "None":
            self.selected_agent = None
        elif request.selected_agent is not None:
            self.selected_agent = request.selected_agent if request.selected_agent in self.env.possible_agents else None
        if request.terrain_enabled is not None:
            self.terrain_enabled = request.terrain_enabled

    def reset(self) -> None:
        self.playing = False
        self._reset_env()

    def load_instance(self, instance_id: str) -> None:
        known = {instance["id"] for instance in list_instances()}
        if instance_id not in known:
            raise ValueError(f"unknown instance: {instance_id}")
        self.instance_id = instance_id
        self.playing = False
        self._reset_env()

    def advance(self, steps: int = 1) -> None:
        for _ in range(max(1, steps)):
            if not self.env.agents:
                self.playing = False
                return
            observations = self.env.get_observations()
            actions = {
                agent: self._policy_for(agent).act(observations[agent], agent)
                for agent in self.env.agents
            }
            self.env.step(actions)

    def _policy_for(self, agent: str):
        policy_name = self.policy_blue if agent.startswith("blue_") else self.policy_red
        return self.policies[policy_name]

    def _terrain_url(self, image: str | None) -> str | None:
        if image is None:
            return None
        path = Path(image)
        try:
            relative = path.resolve().relative_to(INSTANCE_DIR.resolve())
        except ValueError:
            return None
        return f"/instance-files/{relative.as_posix()}"

    async def playback_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0 / 30.0)
            async with self.lock:
                if self.playing:
                    self.advance(self.speed)
                    snapshot = self.snapshot()
                else:
                    snapshot = None
            if snapshot is not None:
                await self.manager.broadcast({"type": "state", "state": snapshot})


simulation = WebSimulation()
app = FastAPI(title="Hold The Line Web")


@app.on_event("startup")
async def start_playback_loop() -> None:
    asyncio.create_task(simulation.playback_loop())


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    async with simulation.lock:
        return simulation.snapshot()


@app.post("/api/control")
async def control(request: ControlRequest) -> dict[str, Any]:
    async with simulation.lock:
        simulation.apply_control(request)
        snapshot = simulation.snapshot()
    await simulation.manager.broadcast({"type": "state", "state": snapshot})
    return snapshot


@app.post("/api/reset")
async def reset() -> dict[str, Any]:
    async with simulation.lock:
        simulation.reset()
        snapshot = simulation.snapshot()
    await simulation.manager.broadcast({"type": "state", "state": snapshot})
    return snapshot


@app.get("/api/instances")
async def instances() -> list[dict[str, str]]:
    return list_instances()


@app.post("/api/load-instance")
async def load_scenario(request: LoadInstanceRequest) -> dict[str, Any]:
    async with simulation.lock:
        simulation.load_instance(request.instance_id)
        snapshot = simulation.snapshot()
    await simulation.manager.broadcast({"type": "state", "state": snapshot})
    return snapshot


@app.post("/api/step")
async def step() -> dict[str, Any]:
    async with simulation.lock:
        simulation.advance(1)
        snapshot = simulation.snapshot()
    await simulation.manager.broadcast({"type": "state", "state": snapshot})
    return snapshot


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await simulation.manager.connect(websocket)
    try:
        async with simulation.lock:
            await websocket.send_json({"type": "state", "state": simulation.snapshot()})
        while True:
            payload = await websocket.receive_json()
            command = payload.get("command")
            async with simulation.lock:
                if command == "reset":
                    simulation.reset()
                elif command == "step":
                    simulation.advance(1)
                elif command == "load_instance":
                    simulation.load_instance(str(payload.get("instance_id")))
                elif command == "control":
                    simulation.apply_control(ControlRequest(**payload.get("data", {})))
                snapshot = simulation.snapshot()
            await simulation.manager.broadcast({"type": "state", "state": snapshot})
    except WebSocketDisconnect:
        simulation.manager.disconnect(websocket)


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/instance-files", StaticFiles(directory=INSTANCE_DIR), name="instance-files")
