"""Policy interface shared by all teams."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray, agent_id: str) -> np.ndarray:
        """Return a continuous 2D action in [-1, 1]."""
