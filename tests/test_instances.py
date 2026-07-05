import unittest

import numpy as np

from src import parallel_env
from src.core.geometry import (
    Circle,
    circle_intersects_obstacle,
    line_of_sight_clear,
    point_in_any_obstacle,
    segment_intersects_obstacle,
)
from src.core.instance_loader import load_instance
from src.core.instances import (
    FIXED_INSTANCE_FACTORIES,
    create_random_facility,
)
from src.core.map_generator import validate_map


FIXED_INSTANCE_IDS = list(FIXED_INSTANCE_FACTORIES)


class FixedInstanceTests(unittest.TestCase):
    def test_all_fixed_instances_validate(self):
        for instance_id, factory in FIXED_INSTANCE_FACTORIES.items():
            config = factory()
            with self.subTest(instance=instance_id):
                result = validate_map(config, min_buildings=5)
                self.assertTrue(result.valid, f"{instance_id}: {result.reasons}")
                self.assertGreaterEqual(result.approach_routes, 2)

    def test_all_fixed_instances_load_from_json_and_run(self):
        for instance_id in FIXED_INSTANCE_IDS:
            with self.subTest(instance=instance_id):
                config = load_instance(instance_id)
                env = parallel_env(map_config=config, max_episode_steps=3)
                observations, infos = env.reset(seed=7)
                self.assertEqual(set(observations), set(env.possible_agents))

                actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
                observations, rewards, terminations, truncations, infos = env.step(actions)
                self.assertEqual(set(rewards), set(env.possible_agents))

    def test_assets_not_inside_obstacles(self):
        for instance_id, factory in FIXED_INSTANCE_FACTORIES.items():
            config = factory()
            for index, asset in enumerate(config.assets):
                with self.subTest(instance=instance_id, asset=index):
                    self.assertFalse(point_in_any_obstacle(asset.center_array, config.buildings))

    def test_spawn_zone_centers_not_inside_obstacles(self):
        for instance_id, factory in FIXED_INSTANCE_FACTORIES.items():
            config = factory()
            with self.subTest(instance=instance_id, zone="blue"):
                self.assertFalse(point_in_any_obstacle(config.blue_spawn_zone.center, config.buildings))
            for index, zone in enumerate(config.red_spawn_zones):
                with self.subTest(instance=instance_id, zone=f"red_{index}"):
                    self.assertFalse(point_in_any_obstacle(zone.center, config.buildings))

    def test_obstacles_stay_within_world_bounds(self):
        for instance_id, factory in FIXED_INSTANCE_FACTORIES.items():
            config = factory()
            width, height = config.world_size
            for index, obstacle in enumerate(config.buildings):
                with self.subTest(instance=instance_id, obstacle=index):
                    if isinstance(obstacle, Circle):
                        x, y = obstacle.center
                        self.assertGreaterEqual(x - obstacle.radius, 0.0)
                        self.assertGreaterEqual(y - obstacle.radius, 0.0)
                        self.assertLessEqual(x + obstacle.radius, width)
                        self.assertLessEqual(y + obstacle.radius, height)
                    else:
                        self.assertGreaterEqual(obstacle.min_x, 0.0)
                        self.assertGreaterEqual(obstacle.min_y, 0.0)
                        self.assertLessEqual(obstacle.max_x, width)
                        self.assertLessEqual(obstacle.max_y, height)

    def test_fixed_industrial_facility_has_circular_obstacles(self):
        config = FIXED_INSTANCE_FACTORIES["test2"]()
        self.assertTrue(any(isinstance(obstacle, Circle) for obstacle in config.buildings))

    def test_line_of_sight_blocked_by_circular_obstacle(self):
        blocker = Circle((5.0, 5.0), 2.0)

        self.assertFalse(
            line_of_sight_clear(
                np.array([0.0, 5.0], dtype=np.float32),
                np.array([10.0, 5.0], dtype=np.float32),
                [blocker],
            )
        )
        self.assertTrue(
            line_of_sight_clear(
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([10.0, 0.0], dtype=np.float32),
                [blocker],
            )
        )

    def test_segment_and_circle_intersection_helpers(self):
        blocker = Circle((5.0, 5.0), 2.0)

        self.assertTrue(
            segment_intersects_obstacle(
                np.array([0.0, 5.0], dtype=np.float32),
                np.array([10.0, 5.0], dtype=np.float32),
                blocker,
            )
        )
        self.assertTrue(circle_intersects_obstacle(np.array([5.0, 5.0], dtype=np.float32), 0.5, blocker))
        self.assertFalse(circle_intersects_obstacle(np.array([20.0, 20.0], dtype=np.float32), 0.5, blocker))


class RandomFacilityTests(unittest.TestCase):
    def test_random_instances_validate_for_multiple_seeds(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                config = create_random_facility(seed=seed)
                result = validate_map(config)
                self.assertTrue(result.valid, f"seed {seed}: {result.reasons}")
                self.assertGreaterEqual(result.approach_routes, 2)

    def test_random_instance_world_size_within_spec(self):
        config = create_random_facility(seed=3)
        width, height = config.world_size
        self.assertGreaterEqual(width, 150.0)
        self.assertLessEqual(width, 300.0)
        self.assertGreaterEqual(height, 150.0)
        self.assertLessEqual(height, 300.0)

    def test_random_instance_runs_in_env(self):
        config = create_random_facility(seed=11)
        env = parallel_env(map_config=config, max_episode_steps=3)
        observations, infos = env.reset(seed=1)
        actions = {agent: np.zeros(2, dtype=np.float32) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        self.assertEqual(set(rewards), set(env.possible_agents))


if __name__ == "__main__":
    unittest.main()
