"""Per-agent rollout storage and GAE (generalized advantage estimation) for
PPO. One buffer per agent slot (e.g. one per blue_0/blue_1) - trajectories
are kept separate per agent so GAE never bootstraps across an episode
boundary that belongs to a different agent's sequence, even though the
network itself is parameter-shared across all agents on a team.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Returns (advantages, returns), both the same length as the buffer.

        `last_value` bootstraps the final step: 0.0 if the buffer ends on a
        true episode boundary (done=True), otherwise the critic's value
        estimate of the observation just after the buffer's last step (the
        rollout window cut the episode off mid-flight).
        """
        n = len(self)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            next_non_terminal = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae
            next_value = self.values[t]
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
