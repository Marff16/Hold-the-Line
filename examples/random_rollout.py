"""Run a short random rollout in Hold The Line."""

from __future__ import annotations

from src import parallel_env


def main() -> None:
    env = parallel_env()
    observations, infos = env.reset(seed=7)
    print(f"agents={env.agents}")
    print(f"observation_shapes={ {agent: obs.shape for agent, obs in observations.items()} }")

    total_rewards = {agent: 0.0 for agent in env.possible_agents}
    for step in range(100):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        for agent, reward in rewards.items():
            total_rewards[agent] += reward
        if not env.agents:
            print(f"episode ended at step={step + 1}")
            print(f"terminations={terminations}")
            print(f"truncations={truncations}")
            break

    print(f"total_rewards={total_rewards}")
    env.close()


if __name__ == "__main__":
    main()
