"""Uniform-random baseline policy."""

from __future__ import annotations

import numpy as np

from src.policies.base import Policy


class RandomPolicy(Policy):
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, agent_id: str) -> np.ndarray:
        del obs, agent_id
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
