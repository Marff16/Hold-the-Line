"""PPO update step: clipped surrogate policy loss + value loss + entropy
bonus, run for a few epochs over shuffled minibatches of one rollout."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.training.networks import ActorCritic


def ppo_update(
    actor_critic: ActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.003,
    epochs: int = 4,
    minibatch_size: int = 64,
    max_grad_norm: float = 0.5,
) -> dict[str, float]:
    obs_t = torch.as_tensor(observations, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.float32)
    old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32)
    returns_t = torch.as_tensor(returns, dtype=torch.float32)

    advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    n = obs_t.shape[0]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}

    for _ in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            batch_obs = obs_t[batch_idx]
            batch_actions = actions_t[batch_idx]
            batch_old_log_probs = old_log_probs_t[batch_idx]
            batch_advantages = advantages_t[batch_idx]
            batch_returns = returns_t[batch_idx]

            log_probs, entropy, values = actor_critic.evaluate(batch_obs, batch_actions)

            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, batch_returns)
            entropy_bonus = entropy.mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy_bonus.item())
            stats["updates"] += 1

    for key in ("policy_loss", "value_loss", "entropy"):
        stats[key] /= max(1, stats["updates"])
    return stats
