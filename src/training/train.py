"""PPO training loop: LEARNING_TEAM learns a parameter-shared ActorCritic
policy; the other team plays a fixed ObstacleAvoidingPolicy opponent (not
learning). This is deliberately the simple "stage 1" version - self-play /
MAPPO (both teams learning, centralized critic) is a follow-up once this
pipeline is verified to actually learn something.

Every episode reset picks a freshly generated map (src.core.instances.
create_packed_facility) from a seeded training pool, instead of training on
one fixed layout - a held-out seeded test pool (disjoint seed range) is used
for periodic no-exploration evaluation, so the eval number reflects
generalization, not memorization of one map.

Early rollouts also get a decaying "residual policy" south-ward action bias
(see SOUTH_BIAS/bias_weight) - with the current sparse discovery/evasion
reward, a freshly initialized (near-zero-mean) policy rarely wanders far
enough south to reach Red's foreign territory on its own, so there's no real
reward signal to learn from for a long time. The bias is added to the action
*after* sampling, only for the copy actually sent to env.step() - the buffer
still stores the network's own unbiased sampled action/log_prob, so PPO is
still training on "what the network itself chose," treating the bias as part
of the environment's dynamics rather than something baked into the policy.
This is standard Residual Policy Learning (Silver et al. 2018).

Run with:
    poetry run python -m src.training.train
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src import parallel_env
from src.core.instances import create_packed_facility
from src.policies import ObstacleAvoidingPolicy
from src.training.buffer import RolloutBuffer
from src.training.networks import ActorCritic
from src.training.ppo import ppo_update

LEARNING_TEAM = "red"  # "blue" or "red" - the other team plays the fixed heuristic opponent
TRAIN_SEEDS = range(1000, 1050)  # 50 generated training maps
TEST_SEEDS = range(2000, 2010)  # 10 held-out maps, only used for eval
MAX_EPISODE_STEPS = 1200  # generous on purpose - don't cut episodes short
ROLLOUT_STEPS = 2048
TOTAL_ITERATIONS = 1000  # a long run on purpose
GAMMA = 0.99
LAM = 0.95
LEARNING_RATE = 3e-4
SEED = 0
EVAL_EVERY = 20
EVAL_EPISODES = 5
CHECKPOINT_EVERY = 10
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / f"{LEARNING_TEAM}_actor_critic.pt"

OPPONENT_TEAM = "red" if LEARNING_TEAM == "blue" else "blue"

# Residual south-ward action bias for LEARNING_TEAM only (Red needs to head
# toward low y / Blue's side to reach discoverable territory - see
# env.py's _default_front_line_y). Linearly fades to 0 by BIAS_DECAY_ITERATIONS
# - long enough to reliably bootstrap real reward signal, short enough to
# leave most of the run for the network to learn unassisted (including
# maneuvering around obstacles, which the bias itself knows nothing about).
SOUTH_BIAS = np.array([0.0, -1.0], dtype=np.float32)
BIAS_DECAY_ITERATIONS = 150


def bias_weight(iteration: int) -> float:
    return max(0.0, 1.0 - iteration / BIAS_DECAY_ITERATIONS)


def is_learning_agent(agent: str) -> bool:
    return agent.startswith(f"{LEARNING_TEAM}_")


def make_episode(seed: int, rng: np.random.Generator) -> tuple:
    """Fresh env + matching opponent for one training map."""
    config = create_packed_facility(seed=seed)
    env = parallel_env(map_config=config, max_episode_steps=MAX_EPISODE_STEPS)
    opponent = ObstacleAvoidingPolicy(config, seed=int(rng.integers(0, 2**31 - 1)))
    return env, opponent


@torch.no_grad()
def evaluate(actor_critic: ActorCritic, seeds: range, episodes_per_seed: int = 1) -> float:
    """Mean episodic return on held-out maps, greedy (mean action, no
    sampling noise) - a rough measure of how well the policy generalizes
    rather than how well it's memorized the training maps."""
    returns = []
    for seed in seeds:
        config = create_packed_facility(seed=seed)
        env = parallel_env(map_config=config, max_episode_steps=MAX_EPISODE_STEPS)
        opponent = ObstacleAvoidingPolicy(config, seed=seed)
        for _ in range(episodes_per_seed):
            observations, _infos = env.reset(seed=seed)
            episode_return = 0.0
            while env.agents:
                actions = {}
                for agent in env.agents:
                    if is_learning_agent(agent):
                        obs_t = torch.as_tensor(observations[agent], dtype=torch.float32).unsqueeze(0)
                        mean = actor_critic.policy_net(obs_t)
                        actions[agent] = np.clip(mean.squeeze(0).numpy(), -1.0, 1.0).astype(np.float32)
                    else:
                        actions[agent] = opponent.act(observations[agent], agent)
                observations, rewards, _terminations, _truncations, _infos = env.step(actions)
                episode_return += sum(reward for agent, reward in rewards.items() if is_learning_agent(agent))
            returns.append(episode_return)
    return float(np.mean(returns))


def main() -> None:
    rng = np.random.default_rng(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    first_seed = int(rng.choice(TRAIN_SEEDS))
    env, opponent = make_episode(first_seed, rng)

    learner_probe = f"{LEARNING_TEAM}_0"
    obs_dim = env.observation_space(learner_probe).shape[0]
    act_dim = env.action_space(learner_probe).shape[0]

    actor_critic = ActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE)
    # Linear LR decay to 10% of the initial rate over the full run - without
    # it, a constant high LR combined with a shrinking entropy bonus let the
    # policy keep taking large, destabilizing steps late in training instead
    # of settling (observed as entropy climbing back up past its starting
    # value from iteration ~150 onward in the previous run).
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=TOTAL_ITERATIONS
    )

    learning_agents = [agent for agent in env.possible_agents if is_learning_agent(agent)]
    buffers = {agent: RolloutBuffer() for agent in learning_agents}

    observations, _infos = env.reset(seed=first_seed)
    episode_return = 0.0
    episode_returns: list[float] = []

    for iteration in range(1, TOTAL_ITERATIONS + 1):
        for buffer in buffers.values():
            buffer.clear()

        bias = SOUTH_BIAS * bias_weight(iteration)

        steps_collected = 0
        while steps_collected < ROLLOUT_STEPS:
            actions: dict[str, np.ndarray] = {}
            step_cache: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}

            for agent in env.agents:
                if is_learning_agent(agent):
                    obs_t = torch.as_tensor(observations[agent], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action, log_prob, value = actor_critic.act(obs_t)
                    # step_cache keeps the network's own unbiased sample (what
                    # PPO trains on); `actions` gets the biased copy that's
                    # actually sent to the env.
                    action_np = action.squeeze(0).numpy().astype(np.float32)
                    step_cache[agent] = (observations[agent], action_np, float(log_prob.item()), float(value.item()))
                    actions[agent] = np.clip(action_np + bias, -1.0, 1.0).astype(np.float32)
                else:
                    actions[agent] = opponent.act(observations[agent], agent)

            observations, rewards, terminations, truncations, _infos = env.step(actions)
            steps_collected += 1

            for agent, (obs, action_np, log_prob, value) in step_cache.items():
                reward = rewards.get(agent, 0.0)
                done = terminations.get(agent, False) or truncations.get(agent, False)
                buffers[agent].add(obs, action_np, log_prob, value, reward, done)
                episode_return += reward

            if not env.agents:
                episode_returns.append(episode_return)
                episode_return = 0.0
                # A fresh episode gets a freshly generated map, not the same
                # layout every time - this is the "training suite" part.
                next_seed = int(rng.choice(TRAIN_SEEDS))
                env, opponent = make_episode(next_seed, rng)
                observations, _infos = env.reset(seed=next_seed)

        all_obs: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_log_probs: list[float] = []
        all_advantages: list[float] = []
        all_returns: list[float] = []

        for agent, buffer in buffers.items():
            if len(buffer) == 0:
                continue
            if buffer.dones[-1]:
                last_value = 0.0
            else:
                obs_t = torch.as_tensor(buffer.observations[-1], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    last_value = float(actor_critic.value(obs_t).item())
            advantages, returns = buffer.compute_gae(last_value, GAMMA, LAM)
            all_obs.extend(buffer.observations)
            all_actions.extend(buffer.actions)
            all_log_probs.extend(buffer.log_probs)
            all_advantages.extend(advantages)
            all_returns.extend(returns)

        stats = ppo_update(
            actor_critic,
            optimizer,
            np.array(all_obs, dtype=np.float32),
            np.array(all_actions, dtype=np.float32),
            np.array(all_log_probs, dtype=np.float32),
            np.array(all_advantages, dtype=np.float32),
            np.array(all_returns, dtype=np.float32),
        )
        scheduler.step()

        recent = episode_returns[-10:]
        mean_return = float(np.mean(recent)) if recent else float("nan")
        message = (
            f"[{LEARNING_TEAM}] iter {iteration:4d} | bias {bias_weight(iteration):.2f} | "
            f"lr {scheduler.get_last_lr()[0]:.6f} | episodes {len(episode_returns):4d} | "
            f"mean_return(last10) {mean_return:7.3f} | policy_loss {stats['policy_loss']:.4f} | "
            f"value_loss {stats['value_loss']:.4f} | entropy {stats['entropy']:.4f}"
        )

        if iteration % EVAL_EVERY == 0:
            eval_return = evaluate(actor_critic, TEST_SEEDS, EVAL_EPISODES)
            message += f" | eval_return(held-out) {eval_return:7.3f}"

        print(message, flush=True)

        if iteration % CHECKPOINT_EVERY == 0 or iteration == TOTAL_ITERATIONS:
            torch.save(
                {"obs_dim": obs_dim, "act_dim": act_dim, "state_dict": actor_critic.state_dict()},
                CHECKPOINT_PATH,
            )
            print(f"saved checkpoint at iter {iteration} -> {CHECKPOINT_PATH}", flush=True)

    print("training complete")


if __name__ == "__main__":
    main()
