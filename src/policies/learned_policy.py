"""Wraps a trained ActorCritic checkpoint (see src/training/train.py) as a
Policy for inference - in the web app or anywhere else that only knows the
Policy interface. Always acts greedily (the distribution's mean, no sampling
noise) since this is for watching/evaluating a trained policy, not exploring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.policies.base import Policy
from src.training.networks import ActorCritic


class LearnedPolicy(Policy):
    def __init__(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.obs_dim = checkpoint["obs_dim"]
        self.act_dim = checkpoint["act_dim"]
        self.model = ActorCritic(self.obs_dim, self.act_dim)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def act(self, obs: np.ndarray, agent_id: str) -> np.ndarray:
        del agent_id
        if obs.shape[0] != self.obs_dim:
            raise ValueError(
                f"LearnedPolicy was trained on obs_dim={self.obs_dim}, got {obs.shape[0]} "
                "(this checkpoint was trained for one team only - check which team you assigned it to)"
            )
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean = self.model.policy_net(obs_t)
        return np.clip(mean.squeeze(0).numpy(), -1.0, 1.0).astype(np.float32)
