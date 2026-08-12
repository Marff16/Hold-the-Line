import unittest

import numpy as np

from src import parallel_env
from src.core.instance_loader import load_instance
from src.policies import ObstacleAvoidingPolicy, RandomPolicy


class ObstacleAvoidingPolicyTests(unittest.TestCase):
    def test_actions_stay_within_bounds(self):
        config = load_instance("instance1")
        policy = ObstacleAvoidingPolicy(config, seed=1)
        rng = np.random.default_rng(0)
        obs = np.zeros(60, dtype=np.float32)
        for _ in range(50):
            obs[:2] = rng.uniform(0.0, 1.0, size=2)
            action = policy.act(obs, "red_0")
            self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))
            self.assertEqual(action.shape, (2,))

    def test_steers_away_from_a_nearby_building(self):
        config = load_instance("instance1")
        policy = ObstacleAvoidingPolicy(config, seed=1, heading_drift=0.0)
        building = config.buildings[0]
        center = building.center if hasattr(building, "center") else None
        self.assertIsNotNone(center)

        # Place the agent just outside the building's edge, closest point
        # roughly toward its center, and check the resulting action has a
        # positive component pointing away from the obstacle.
        probe_pos = np.array([building.min_x - 2.0, building.min_y + building.h / 2.0], dtype=np.float32)
        probe_pos = np.clip(probe_pos, [0.0, 0.0], config.world_size)
        obs = np.zeros(60, dtype=np.float32)
        obs[:2] = probe_pos / np.array(config.world_size, dtype=np.float32)

        action = policy.act(obs, "red_0")
        away_from_building = probe_pos - np.array([building.min_x, probe_pos[1]], dtype=np.float32)
        if np.linalg.norm(away_from_building) > 1e-6:
            away_from_building = away_from_building / np.linalg.norm(away_from_building)
            self.assertGreater(float(np.dot(action, away_from_building)), 0.0)

    def test_runs_in_env_without_crashing(self):
        config = load_instance("instance1")
        env = parallel_env(map_config=config, max_episode_steps=30)
        red_policy = ObstacleAvoidingPolicy(config, seed=3)
        blue_policy = RandomPolicy(seed=3)
        observations, infos = env.reset(seed=5)

        for _ in range(30):
            if not env.agents:
                break
            actions = {}
            for agent in env.agents:
                policy = blue_policy if agent.startswith("blue_") else red_policy
                actions[agent] = policy.act(observations[agent], agent)
            observations, rewards, terminations, truncations, infos = env.step(actions)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
