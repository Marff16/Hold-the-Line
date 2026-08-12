"""Actor-critic network for continuous PPO.

One instance per team (Blue obs_dim=26, Red obs_dim=21 for instance1 - always
read the real shape from env.observation_space(agent) rather than hardcoding
it, since it can change with map_config).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 64) -> None:
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        # Single state-independent learnable log_std, not a second network
        # head - simpler and more stable to train than state-dependent std.
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.policy_net(obs)
        std = self.log_std.exp()
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs).squeeze(-1)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action for rollout collection.

        Returns (action, log_prob, value):
        - action: sampled from Normal(mean, exp(log_std)), NOT yet clipped to
          [-1, 1] - store the raw sample in the buffer; clip only the copy
          you actually send to env.step().
        - log_prob: log_prob of that exact sampled action under the Normal,
          summed over the action dimensions (Normal gives per-dim log_probs).
        - value: critic's V(obs), squeezed to a scalar per env in the batch.
        """
        dist = self._distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, self.value(obs)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used during the PPO update - recompute log_prob/entropy/value for
        actions that were already taken (stored in the rollout buffer), with
        the CURRENT (post-update-step) network parameters.

        Returns (log_prob, entropy, value).
        """
        dist = self._distribution(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, self.value(obs)
